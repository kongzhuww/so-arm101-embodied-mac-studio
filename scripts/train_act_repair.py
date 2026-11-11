"""Isolated ACT repair experiment. No camera/serial or Studio configuration writes.

Original v1 episode 0 is excluded because its reset contains a human hand.
V1 uses the successful model's actual cropped source. Episodes 8/9 per
version are held out from NEW training (old baselines have already seen them).
"""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IMAGE='observation.images.external_front'


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def prepare(root,destination):
    from studio_samples import validate_episode,resample_episode
    root=Path(root);destination=Path(destination)
    if destination.exists():raise ValueError('Refuse existing experiment directory')
    old=json.loads((root/'outputs/act_20261106T174035Z/run-manifest.json').read_text())
    new=json.loads((root/'outputs/act_20261107T070132Z/run-manifest.json').read_text())
    plan=[]
    for group,original in enumerate(new['source'],1):
        source=Path(old['source']) if group==1 else Path(original)
        for episode in range(10):
            if group==1 and episode==0:continue
            name=f'episode_{episode:04d}.json';path=source/name;content=path.read_bytes()
            rows=json.loads(content);validate_episode(rows)
            split='validation' if episode in (8,9) else 'train'
            target=destination/split/f'v{group}'/name
            metadata=json.loads((Path(original)/'metadata'/name).read_text())
            metadata.update(derived_from=str(path),source_sha256=hashlib.sha256(content).hexdigest(),split=split)
            plan.append((target,content,metadata,len(resample_episode(rows)),group,episode))
    records=[]
    destination.mkdir(parents=True)
    for path,content,metadata,frames,group,episode in plan:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        dump(path.parent/'metadata'/path.name,metadata)
        records.append({'group':group,'episode':episode,'split':metadata['split'],
            'source':metadata['derived_from'],'sha256':metadata['source_sha256'],'derived':str(path),'frames':frames})
    manifest={'records':records,'excluded':[{'group':1,'episode':0,'reason':'Human reset/incorrect initial block position in recorded startup'}],
        'train_episodes':sum(x['split']=='train' for x in records),'validation_episodes':sum(x['split']=='validation' for x in records),
        'train_frames':sum(x['frames'] for x in records if x['split']=='train'),
        'validation_frames':sum(x['frames'] for x in records if x['split']=='validation'),
        'caveat':'Validation episodes are excluded only from this new training; old baselines saw them. Same setup/near-repeat demonstrations, not independent real-world success tests.'}
    dump(destination/'split-manifest.json',manifest)
    return manifest


def raw(values,cal,names):
    import numpy as np
    a=np.array(values,copy=True,dtype=float)
    for j,name in enumerate(names):
        lo,hi=cal[name]['range_min'],cal[name]['range_max']
        a[...,j]=lo+a[...,j]*(hi-lo)/100 if name=='gripper' else (lo+hi)/2+a[...,j]*4095/360
    return np.clip(np.rint(a),[cal[n]['range_min'] for n in names],[cal[n]['range_max'] for n in names])


def validation_samples(dataset,root):
    import numpy as np
    import pyarrow.parquet as pq
    rows=[]
    for path in sorted((root/'meta/episodes').rglob('*.parquet')):
        for ep in pq.read_table(path,columns=['episode_index','dataset_from_index','dataset_to_index']).to_pylist():
            start,end=ep['dataset_from_index'],ep['dataset_to_index']
            for index in np.unique(np.linspace(start,end-1,24).astype(int)):
                row=dataset[int(index)]
                rows.append((int(ep['episode_index'])//2+1,row))
    return rows


def evaluate(policy,pre,post,samples,cal,names,image_transform=None):
    import numpy as np
    import torch
    policy.eval();errors={g:np.zeros(6) for g in (1,2,3)};counts={g:0 for g in errors}
    with torch.inference_mode():
        for start in range(0,len(samples),4):
            items=samples[start:start+4]
            batch={key:torch.stack([row[key] for _,row in items]) for key in ('observation.state',IMAGE)}
            if image_transform:
                batch['episode_index']=torch.stack([row['episode_index'] for _,row in items])
                batch=image_transform(batch)
                del batch['episode_index']
            pred=raw(post(policy.predict_action_chunk(pre(batch))).detach().cpu().numpy(),cal,names)
            target=raw(torch.stack([row['action'] for _,row in items]).numpy(),cal,names)
            for i,(group,row) in enumerate(items):
                mask=~row['action_is_pad'][:20].numpy().astype(bool)
                errors[group]+=abs(pred[i,:20]-target[i,:20])[mask].sum(axis=0)
                counts[group]+=int(mask.sum())
    per_group={str(g):(errors[g]/max(1,counts[g])).tolist() for g in errors}
    mae=sum(errors.values())/sum(counts.values())
    assert np.isfinite(mae).all()
    return {'selection_arm_mae':float(mae[:5].mean()),'joint_mae':mae.tolist(),'per_group_joint_mae':per_group,
            'observations':len(samples),'valid_future_targets':sum(counts.values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train',action='store_true');parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--prepared',type=Path,help='Reuse a prepared, hash-verified split')
    args=parser.parse_args()
    if args.steps<3000:raise ValueError('Use at least 3000 steps for the planned comparison')
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    derived=args.prepared.resolve() if args.prepared else ROOT/'datasets'/f'act_repair_{tag}'
    if args.prepared:
        split=json.loads((derived/'split-manifest.json').read_text())
        for record in split['records']:
            for key in ('source','derived'):
                if hashlib.sha256(Path(record[key]).read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError(f'Prepared source changed: {record[key]}')
    else:split=prepare(ROOT,derived)
    print('PREPARED',derived,json.dumps({k:v for k,v in split.items() if k not in ('records',)},ensure_ascii=False),flush=True)
    if not args.train:return
    import numpy as np
    import torch
    from train_studio_pipeline import build_dataset_from_studio,NAMES
    from lerobot.datasets import LeRobotDataset,LeRobotDatasetMetadata
    from lerobot.policies.act import ACTConfig,ACTPolicy
    from lerobot.policies import make_pre_post_processors
    training=[derived/'train'/f'v{g}' for g in (1,2,3)];validation=[derived/'validation'/f'v{g}' for g in (1,2,3)]
    train_root=ROOT/'datasets'/f'lerobot_repair_{tag}_train';val_root=ROOT/'datasets'/f'lerobot_repair_{tag}_validation'
    output=ROOT/'outputs'/f'act_{tag}_repair';output.mkdir()
    manifest={'source':[str(p) for p in training],'validation_source':[str(p)for p in validation],
        'dataset':str(train_root),'validation_dataset':str(val_root),'model':str(output),
        'split_manifest':str(derived/'split-manifest.json'),'chunk_size':50,'n_action_steps':20,
        'total_steps':args.steps,'status':'converting','experiment':'repair-v1-source-and-training-budget','caveat':split['caveat']}
    dump(output/'run-manifest.json',manifest)
    dump(ROOT/'reports/act-repair-latest.json',{'output':str(output),'manifest':manifest})
    try:
        build_dataset_from_studio(training,train_root);build_dataset_from_studio(validation,val_root)
        repo='local/so101_studio';meta=LeRobotDatasetMetadata(repo,root=train_root)
        device='mps' if torch.backends.mps.is_available() else 'cpu'
        config=ACTConfig.from_pretrained(ROOT/'outputs/act_20261107T070132Z/final')
        config.device=device;config.push_to_hub=False
        torch.manual_seed(42);np.random.seed(42)
        policy=ACTPolicy(config).to(device).train()
        pre,post=make_pre_post_processors(config,dataset_stats=meta.stats)
        delta={'action':[i/10 for i in range(50)]}
        dataset=LeRobotDataset(repo,root=train_root,delta_timestamps=delta)
        valset=LeRobotDataset(repo,root=val_root,delta_timestamps=delta)
        samples=validation_samples(valset,val_root)
        cal=json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
        baselines={}
        for label,ident in [('old10','act_20261106T174035Z'),('original30','act_20261107T070132Z')]:
            path=ROOT/'outputs'/ident/'final';base=ACTPolicy.from_pretrained(str(path)).to(device).eval()
            bp,bq=make_pre_post_processors(base.config,pretrained_path=path,preprocessor_overrides={'device_processor':{'device':device}})
            baselines[label]=evaluate(base,bp,bq,samples,cal,NAMES);del base
            if device=='mps':torch.mps.empty_cache()
        dump(output/'baseline-validation.json',baselines);print('BASELINES',json.dumps(baselines),flush=True)
        optimizer=config.get_optimizer_preset().build(policy.parameters())
        loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,drop_last=False)
        manifest.update(status='training',device=device,train_frames=len(dataset),validation_frames=len(valset))
        dump(output/'run-manifest.json',manifest)
        started=time.time();step=0;best=float('inf');best_step=None;history=[]
        while step<args.steps:
            for batch in loader:
                policy.train();optimizer.zero_grad(set_to_none=True)
                loss,_=policy.forward(pre(batch))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.0);optimizer.step();step+=1
                if step==1 or step%100==0:
                    status={'step':step,'total_steps':args.steps,'loss':float(loss.detach().cpu()),'elapsed_s':time.time()-started,
                            'best_validation_arm_mae':best if best_step is not None else None,'best_step':best_step}
                    dump(output/'training-status.json',status);print('PROGRESS',json.dumps(status),flush=True)
                if step%1000==0 or step==args.steps:
                    metrics=evaluate(policy,pre,post,samples,cal,NAMES);history.append({'step':step,**metrics})
                    if metrics['selection_arm_mae']<best:
                        best=metrics['selection_arm_mae'];best_step=step
                        checkpoint=output/'best-validation';policy.save_pretrained(checkpoint);pre.save_pretrained(checkpoint);post.save_pretrained(checkpoint)
                    if step==3000:
                        checkpoint=output/'checkpoint-3000';policy.save_pretrained(checkpoint);pre.save_pretrained(checkpoint);post.save_pretrained(checkpoint)
                    dump(output/'validation-history.json',history);print('VALIDATION',step,json.dumps(metrics),flush=True)
                if step>=args.steps:break
        shutil.copytree(output/'best-validation',output/'final')
        result={'model':str(output/'final'),'best_step':best_step,'baseline':baselines,
                'selected_validation':next(x for x in history if x['step']==best_step),'history':history,
                'training_seconds':time.time()-started,'caveat':split['caveat'],
                'not_causal_ablation':'Source correction, one exclusion, split and training budget changed; cannot assign gain to a single factor.'}
        dump(output/'repair-result.json',result)
        manifest.update(status='completed',best_step=best_step,selected_validation_arm_mae=best)
        dump(output/'run-manifest.json',manifest)
        dump(ROOT/'reports/act-repair-latest.json',{'output':str(output),'result':result})
        print('COMPLETED',json.dumps(result),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));dump(output/'run-manifest.json',manifest);raise

if __name__=='__main__':main()
