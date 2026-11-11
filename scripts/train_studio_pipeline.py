"""Production Training Engine for SO-101.
Converts studio recorded episodes into LeRobot format, then runs MPS ACT training with real progress streaming!
"""
import sys
import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
TASK = 'Pick up the red cube and place it into the box.'

def normalize_joints(raw, calibration):
    values = []
    for name, val in zip(NAMES, raw):
        c = calibration[name]
        lo, hi = c['range_min'], c['range_max']
        if name == 'gripper':
            values.append((min(hi, max(lo, val))-lo)*100/(hi-lo))
        else:
            values.append((val-(lo+hi)/2)*360/4095)
    return np.asarray(values, dtype=np.float32)

def build_dataset_from_studio(studio_dir: Path, output_root: Path):
    from lerobot.datasets import LeRobotDataset
    calibration = json.loads((ROOT / 'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
    
    sources = [studio_dir] if isinstance(studio_dir, Path) else list(studio_dir)
    ep_files = [f for source in sources for f in sorted(source.glob('episode_*.json'))]
    if not ep_files:
        raise RuntimeError("未在录制目录找到任何 episode_*.json 文件！")
        
    from studio_samples import resample_episode, decode_sample
    # Validate every episode BEFORE creating output; never synthesize images.
    episodes = [resample_episode(json.loads(f.read_text())) for f in ep_files]
    print(f"正在转换 {len(ep_files)} 条真实图像示范为标准 LeRobot 数据集...")
    dataset = LeRobotDataset.create(
        repo_id='local/so101_studio', root=output_root, fps=10,
        robot_type='so101_follower', use_videos=False, image_writer_threads=2,
        features={
            'observation.state': {'dtype':'float32','shape':(6,), 'names':NAMES},
            'action': {'dtype':'float32','shape':(6,), 'names':NAMES},
            'observation.images.external_front': {'dtype':'image','shape':(240,320,3), 'names':['height','width','channels']},
        }
    )
    
    for data in episodes:
        for row in data:
            s_cur = normalize_joints(row['state'], calibration)
            action = normalize_joints(row['action'], calibration)
            dataset.add_frame({
                'observation.state': s_cur,
                'action': action,
                'observation.images.external_front': decode_sample(row),
                'task': TASK
            })
        dataset.save_episode()
        
    dataset.finalize()
    print("LeRobot 格式数据集转换完成！")
    return dataset

def run_training_pipeline(studio_dataset_dir: Path, progress_callback=None,
                          *, chunk_size=20, n_action_steps=10, total_steps=3000):
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    lerobot_data_dir = ROOT / 'datasets' / f'lerobot_{tag}'
    model_output_dir = ROOT / 'outputs' / f'act_{tag}'
    
    # 1. Convert
    build_dataset_from_studio(studio_dataset_dir, lerobot_data_dir)
    
    # 2. Train using train_act_local.py logic
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTConfig, ACTPolicy
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs import FeatureType

    repo_id = 'local/so101_studio'
    meta = LeRobotDatasetMetadata(repo_id, root=lerobot_data_dir)
    features = dataset_to_policy_features(meta.features)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    config = ACTConfig(
        device=device, push_to_hub=False,
        input_features={k: v for k, v in features.items() if v.type is not FeatureType.ACTION},
        output_features={k: v for k, v in features.items() if v.type is FeatureType.ACTION},
        chunk_size=chunk_size, n_action_steps=n_action_steps,
        dim_model=256, dim_feedforward=1024, n_heads=8,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1',
        optimizer_lr=1e-4, optimizer_lr_backbone=1e-5,
    )
    
    torch.manual_seed(42)
    policy = ACTPolicy(config).to(device).train()
    pre, post = make_pre_post_processors(config, dataset_stats=meta.stats)
    
    dataset = LeRobotDataset(repo_id, root=lerobot_data_dir, delta_timestamps={
        'action': [i/meta.fps for i in config.action_delta_indices],
    })
    
    sample = dataset[0]
    if tuple(sample['action'].shape) != (chunk_size, 6):
        raise RuntimeError(f"动作窗口形状错误: {sample['action'].shape}")
    print(f"数据验证通过：{len(dataset)} 帧，action shape={tuple(sample['action'].shape)}", flush=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    
    model_output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    started = time.time()
    sources = [studio_dataset_dir] if isinstance(studio_dataset_dir, Path) else list(studio_dataset_dir)
    manifest = {'source': [str(p.resolve()) for p in sources],
                'dataset': str(lerobot_data_dir), 'model': str(model_output_dir),
                'chunk_size': chunk_size, 'n_action_steps': n_action_steps,
                'total_steps': total_steps, 'device': device, 'status': 'training'}
    (model_output_dir / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"开始训练 ACT ({device})：{total_steps} 步，chunk={chunk_size}，执行窗口={n_action_steps}")
    
    done = False
    while not done:
        for batch in loader:
            batch = pre(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('训练 loss 非有限值，已停止')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            step += 1
            
            if step % 20 == 0 or step == 1:
                cur_loss = float(loss.detach().cpu())
                print(f"Step {step}/{total_steps} - Loss: {cur_loss:.4f}", flush=True)
                (model_output_dir / 'training-status.json').write_text(json.dumps({
                    'step': step, 'loss': cur_loss, 'elapsed_s': time.time()-started}))
                if progress_callback:
                    progress_callback(step, cur_loss)
                    
            if step >= total_steps:
                done = True
                break
                
    # Save final
    final_dir = model_output_dir / 'final'
    policy.save_pretrained(final_dir)
    pre.save_pretrained(final_dir)
    post.save_pretrained(final_dir)
    manifest['status'] = 'completed'
    (model_output_dir / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"🎉 训练完成！模型已保存至: {final_dir}")
    return final_dir

if __name__ == "__main__":
    dirs = sorted(Path(ROOT / "datasets").glob("studio_dataset_*"))
    if dirs:
        run_training_pipeline(dirs[-1])
    else:
        print("无 studio_dataset 目录！")
