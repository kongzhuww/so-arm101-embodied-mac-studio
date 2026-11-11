"""Offline visual-domain fine-tune: real original trajectories unchanged.
Background normalization + label-preserving translation/light augmentation.
No current diagnostic image is used as an action training label. No hardware.
"""
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from studio_visual import BackgroundView,BatchView
from studio_samples import decode_sample
from train_act_repair import dump,evaluate,validation_samples
from train_studio_pipeline import NAMES
from lerobot.datasets import LeRobotDataset,LeRobotDatasetMetadata
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors

ROOT=Path(__file__).resolve().parents[1]


def source_offsets(view,sources):
    entries=[]
    for src in sources:
        for file in sorted(Path(src).glob('episode_*.json')):
            row=json.loads(file.read_text())[0]
            entries.append({'file':str(file),**view.estimate(decode_sample(row))})
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    base=ROOT/'outputs/act_20261108T151508Z_repair'
    original=json.loads((base/'run-manifest.json').read_text())
    first=json.loads(sorted(Path(original['source'][0]).glob('episode_*.json'))[0].read_text())[0]
    view=BackgroundView.from_rgb(decode_sample(first))
    training=source_offsets(view,original['source']);validation=source_offsets(view,original['validation_source'])
    assert len(training)==23 and len(validation)==6
    print('OFFSETS',json.dumps({'train':training,'validation':validation}),flush=True)
    if args.prepare_only:return
    if args.steps<1000:raise ValueError('Need at least1000 steps')
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');output=ROOT/'outputs'/f'act_{tag}_visual';output.mkdir()
    manifest={**original,'model':str(output),'status':'training','experiment':'canonical-background-and-translation-light-finetune',
              'initial_model':str(base/'final'),'total_steps':args.steps,'learning_rate':1e-5,
              'new_scene_images_used_for_action_training':False,'train_background_offsets':training,'validation_background_offsets':validation}
    manifest.pop('best_step',None);manifest.pop('selected_validation_arm_mae',None)
    dump(output/'run-manifest.json',manifest);dump(ROOT/'reports/act-visual-latest.json',{'output':str(output)})
    device='mps'if torch.backends.mps.is_available()else'cpu';torch.manual_seed(42);np.random.seed(42)
    try:
        policy=ACTPolicy.from_pretrained(str(base/'final')).to(device)
        policy.config.optimizer_lr=1e-5;policy.config.optimizer_lr_backbone=1e-5
        meta=LeRobotDatasetMetadata('local/so101_studio',root=Path(original['dataset']))
        pre,post=make_pre_post_processors(policy.config,dataset_stats=meta.stats)
        delta={'action':[i/10 for i in range(50)]}
        ds=LeRobotDataset('local/so101_studio',root=Path(original['dataset']),delta_timestamps=delta)
        vs=LeRobotDataset('local/so101_studio',root=Path(original['validation_dataset']),delta_timestamps=delta)
        samples=validation_samples(vs,Path(original['validation_dataset']))
        train_view=BatchView(training);val_view=BatchView(validation)
        cal=json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
        raw_baseline=evaluate(policy,pre,post,samples,cal,NAMES)
        aligned_baseline=evaluate(policy,pre,post,samples,cal,NAMES,image_transform=val_view)
        dump(output/'baseline-validation.json',{'original_input':raw_baseline,'aligned_input_before_finetune':aligned_baseline})
        print('BASELINE',json.dumps({'raw':raw_baseline,'aligned':aligned_baseline}),flush=True)
        optimizer=policy.config.get_optimizer_preset().build(policy.parameters())
        loader=torch.utils.data.DataLoader(ds,batch_size=4,shuffle=True)
        history=[];best=float('inf');best_step=None;began=time.time();step=0
        def save(path):
            policy.save_pretrained(path);pre.save_pretrained(path);post.save_pretrained(path);view.save(path/'visual-preprocessing.json')
        while step<args.steps:
            for batch in loader:
                policy.train();optimizer.zero_grad(set_to_none=True)
                batch=train_view(batch,augment=True)
                loss,_=policy.forward(pre(batch))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.);optimizer.step();step+=1
                if step==1 or step%100==0:
                    status={'step':step,'total_steps':args.steps,'loss':float(loss.detach().cpu()),'elapsed_s':time.time()-began,
                            'best_step':best_step,'best_validation_arm_mae':best if best_step else None}
                    dump(output/'training-status.json',status);print('PROGRESS',json.dumps(status),flush=True)
                if step%1000==0 or step==args.steps:
                    metrics=evaluate(policy,pre,post,samples,cal,NAMES,image_transform=val_view)
                    history.append({'step':step,**metrics});save(output/f'checkpoint-{step}')
                    if metrics['selection_arm_mae']<best:
                        best=metrics['selection_arm_mae'];best_step=step;save(output/'best-validation')
                    dump(output/'validation-history.json',history);print('VALIDATION',step,json.dumps(metrics),flush=True)
                if step>=args.steps:break
        # Only candidate export: live three-position checks are a separate gate.
        import shutil
        shutil.copytree(output/'best-validation',output/'final')
        result={'model':str(output/'final'),'best_step':best_step,'baseline_original':raw_baseline,
                'baseline_aligned':aligned_baseline,'selected_validation':next(x for x in history if x['step']==best_step),
                'history':history,'caveat':'Same-setting held-out trajectories; current static scenes are separate diagnostics, not closed-loop success tests.'}
        dump(output/'visual-result.json',result);manifest.update(status='completed',best_step=best_step)
        dump(output/'run-manifest.json',manifest);dump(ROOT/'reports/act-visual-latest.json',{'output':str(output),'result':result})
        print('COMPLETED',json.dumps(result),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));dump(output/'run-manifest.json',manifest);raise

if __name__=='__main__':main()
