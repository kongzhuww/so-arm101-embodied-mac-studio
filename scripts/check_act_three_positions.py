"""Offline forensic probe. No camera/serial APIs; does not change datasets or models.
Training-set teacher forcing + image interventions are diagnostics, not success rates.
"""
import io,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from studio_samples import resample_episode,decode_sample
from train_studio_pipeline import normalize_joints,NAMES

ROOT=Path(__file__).resolve().parents[1]
IMAGE='observation.images.external_front'
CAL=json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
MODELS={'old10':'act_20261106T174035Z','new30':'act_20261107T070132Z'}

def raw(values):
    a=np.array(values,copy=True,dtype=float)
    for j,name in enumerate(NAMES):
        c=CAL[name];lo,hi=c['range_min'],c['range_max']
        a[...,j]=lo+a[...,j]*(hi-lo)/100 if name=='gripper' else (lo+hi)/2+a[...,j]*4095/360
    return a


def main():
    manifest=json.loads((ROOT/'outputs'/MODELS['new30']/'run-manifest.json').read_text())
    report={'caveat':'Training observations, teacher-forced; not an independent validation set or physical success rate.',
            'joints':NAMES,'data':[],'models':{},'pipeline':{}}
    samples=[];initial=[];all_sources=[]
    stages={'start':'p1_standby','open':'p2_open','grasp':'p2_grasp','lift':'p3_lift','box':'p4_box','release':'p4_release'}
    for group,source in enumerate(manifest['source']):
        source=Path(source);files=sorted(source.glob('episode_*.json'));all_sources.extend(files)
        source_rows=json.loads(files[0].read_text());meta=json.loads((source/'metadata'/files[0].name).read_text())
        wp=meta['version']['waypoints']
        info={'version':meta['version']['name'],'waypoints':wp,'episodes':len(files),'timing':[]}
        for file in files:
            rows=json.loads(file.read_text());ts=np.array([r['timestamp'] for r in rows]);cmd=np.array([r['command_timestamp']-r['timestamp'] for r in rows]);age=np.array([r['timestamp']-r['image_timestamp'] for r in rows])
            info['timing'].append({'episode':file.name,'median_interval':float(np.median(np.diff(ts))),
                 'max_interval':float(np.max(np.diff(ts))),'median_command_lag':float(np.median(cmd)),
                 'max_command_lag':float(cmd.max()),'max_image_age':float(age.max())})
        for episode in [0,9]:
            rows=resample_episode(json.loads(files[episode].read_text()));acts=np.array([r['action'] for r in rows])
            selection={label:int(np.argmin(np.mean(abs(acts-np.array(wp[key])),axis=1))) for label,key in stages.items()}
            selection['start']=0  # A standby target also occurs at the END; never use nearest-pose for onset.
            selection['early']=min(5,len(rows)-1)
            selection['return']=int(len(rows)*.88)
            for stage,index in selection.items():
                row=rows[index];im=torch.from_numpy(decode_sample(row)).permute(2,0,1).float()/255
                target=acts[index:min(len(rows),index+50)]
                target=np.concatenate([target,np.repeat(target[-1:],50-len(target),axis=0)])
                sample={'group':group+1,'episode':episode,'stage':stage,'index':index,'valid':min(50,len(rows)-index),
                    'state':torch.from_numpy(normalize_joints(row['state'],CAL)),'image':im,'target':target,'raw_state':row['state']}
                samples.append(sample)
                if stage=='start' and episode==0:initial.append(sample)
        report['data'].append(info)
    # Verify actual converted training arrays and RGB bytes, not only matching code.
    table=pq.ParquetFile(sorted((Path(manifest['dataset'])/'data').rglob('*.parquet'))[0]).read().slice(0,1).to_pylist()[0]
    first=resample_episode(json.loads(all_sources[0].read_text()))[0]
    stored=np.array(Image.open(io.BytesIO(table[IMAGE]['bytes'])).convert('RGB'))
    report['pipeline']={'state_max_abs_diff':float(np.max(abs(np.array(table['observation.state'])-normalize_joints(first['state'],CAL)))),
         'action_max_abs_diff':float(np.max(abs(np.array(table['action'])-normalize_joints(first['action'],CAL)))),
         'image_max_abs_diff':int(np.max(abs(stored.astype(int)-decode_sample(first).astype(int))))}
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    for label,ident in MODELS.items():
        path=ROOT/'outputs'/ident/'final'
        policy=ACTPolicy.from_pretrained(str(path)).to(device).eval()
        pre,post=make_pre_post_processors(policy.config,pretrained_path=path,preprocessor_overrides={'device_processor':{'device':device}})
        def predict(items,variant='real',fixed_state=None):
            images=torch.stack([s['image'] if variant=='real' else (initial[0]['image'] if variant=='fixed' else torch.zeros_like(s['image'])) for s in items])
            states=torch.stack([s['state'] if fixed_state is None else fixed_state for s in items])
            with torch.inference_mode():
                out=post(policy.predict_action_chunk(pre({'observation.state':states,IMAGE:images}))).detach().cpu().numpy()
            assert np.isfinite(out).all()
            # Match web execution: calibrated inverse, rounding and per-joint bounds.
            return np.clip(np.rint(raw(out)),[CAL[n]['range_min'] for n in NAMES],
                           [CAL[n]['range_max'] for n in NAMES])
        entries=[]
        for offset in range(0,len(samples),4):
            batch=samples[offset:offset+4];pred=predict(batch);fixed=predict(batch,'fixed');black=predict(batch,'black')
            for s,p,f,b in zip(batch,pred,fixed,black):
                horizon=min(20,s['valid']);truth=s['target']
                entries.append({k:s[k] for k in ('group','episode','stage','index','valid')}|{
                    'raw_state':s['raw_state'],'truth0':truth[0].tolist(),'prediction0':p[0].round(2).tolist(),
                    'first_error':abs(p[0]-truth[0]).round(2).tolist(),
                    'executed20_error':np.mean(abs(p[:horizon]-truth[:horizon]),axis=0).round(2).tolist(),
                    'fixed_image_error':np.mean(abs(f[:horizon]-truth[:horizon]),axis=0).round(2).tolist(),
                    'black_image_error':np.mean(abs(b[:horizon]-truth[:horizon]),axis=0).round(2).tolist(),
                    'fixed_image_shift':np.mean(abs(p[:horizon]-f[:horizon]),axis=0).round(2).tolist()})
            print(label,'samples',min(offset+4,len(samples)),'/',len(samples),flush=True)
        # Counterfactual: three recorded initial images, IDENTICAL joint input.
        common=predict(initial,fixed_state=initial[1]['state'])
        report['models'][label]={'id':ident,'samples':entries,'initial_same_state_image_probe':common[:,:20].round(2).tolist()}
        for group in (1,2,3):
            selected=[x for x in entries if x['group']==group]
            print(label,'version',group,'mean20',np.mean([x['executed20_error'] for x in selected],axis=0).round(1).tolist(),flush=True)
        from lerobot.common.control_utils import predict_action
        check=initial[1]
        expected=predict([check])[0,0]
        policy.reset()
        observation={'observation.state':check['state'].numpy(),
                     IMAGE:(check['image'].permute(1,2,0).numpy()*255).round().astype(np.uint8)}
        actual=predict_action(observation,policy,torch.device(device),pre,post,use_amp=False,
            task='Pick up the red cube and place it into the box.',robot_type='so101_follower')
        actual=np.clip(np.rint(raw(actual.detach().cpu().numpy().reshape(6))),
                       [CAL[n]['range_min'] for n in NAMES],[CAL[n]['range_max'] for n in NAMES])
        difference=float(np.max(abs(expected-actual)))
        report['pipeline'][label+'_official_inference_first_goal_maxdiff']=difference
        assert difference<=1,('Offline vs actual inference mismatch',label,difference)
        del policy
        if device=='mps':torch.mps.empty_cache()
    path=ROOT/'reports/act-three-position-diagnosis.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('SAVED',path,flush=True)

if __name__=='__main__':main()
