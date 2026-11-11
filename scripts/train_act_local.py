"""Train ACT on a real local dataset. No robot/serial/camera access or Hub upload."""
import argparse
import json
from pathlib import Path
import time

import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not (root/'meta/info.json').exists():
        raise RuntimeError('No recorded LeRobot dataset found')
    info = json.loads((root/'meta/info.json').read_text())
    if info.get('total_episodes', 0) < 1:
        raise RuntimeError('Record and save a real episode first')
    repo_id = 'local/so101_pickplace'
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    features = dataset_to_policy_features(meta.features)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    config = ACTConfig(
        device=device, push_to_hub=False,
        input_features={k: v for k, v in features.items() if v.type is not FeatureType.ACTION},
        output_features={k: v for k, v in features.items() if v.type is FeatureType.ACTION},
        chunk_size=20, n_action_steps=10,
        dim_model=256, dim_feedforward=1024, n_heads=8,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
        pretrained_backbone_weights=None if args.no_pretrained else 'ResNet18_Weights.IMAGENET1K_V1',
        optimizer_lr=1e-4, optimizer_lr_backbone=1e-5,
    )
    torch.manual_seed(42)
    policy = ACTPolicy(config).to(device).train()
    pre, post = make_pre_post_processors(config, dataset_stats=meta.stats)
    # ACT takes a single current image [C,H,W], not a length-one image
    # sequence [1,C,H,W]. Only actions receive a temporal chunk axis.
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps={
        'action': [i/meta.fps for i in config.action_delta_indices],
    })
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=0, drop_last=False)
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out/'training-status.json'
    started = time.time()
    step = 0

    def save(label):
        path = out/label
        policy.save_pretrained(path)
        pre.save_pretrained(path)
        post.save_pretrained(path)
        torch.save({'step': step, 'optimizer': optimizer.state_dict()}, path/'optimizer.pt')
        print('CHECKPOINT', path, flush=True)

    print(f'TRAINING REAL DATA: episodes={meta.total_episodes} frames={len(dataset)} device={device}', flush=True)
    try:
        while step < args.steps:
            for batch in loader:
                batch = pre(batch)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = policy.forward(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss; stopping')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                step += 1
                if step == 1 or step % 20 == 0:
                    report = {'step': step, 'steps': args.steps, 'loss': float(loss.detach().cpu()),
                              'elapsed_s': time.time()-started, 'device': device,
                              'episodes': meta.total_episodes, 'dataset': str(root)}
                    status.write_text(json.dumps(report, indent=2))
                    print(json.dumps(report), flush=True)
                if step % 500 == 0:
                    save(f'step-{step:06d}')
                if step >= args.steps:
                    break
        save('final')
        print('TRAINING COMPLETE. No robot inference or motion was started.', flush=True)
    except BaseException:
        if step:
            save('interrupted')
        raise


if __name__ == '__main__':
    main()
