"""Offline ACT evaluation on recorded observations. No camera or serial access."""
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'outputs/act_20261106T172457Z/final'
DATA = ROOT / 'datasets/lerobot_20261106T172457Z'
NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']


def main():
    device = 'mps'
    policy = ACTPolicy.from_pretrained(str(MODEL)).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=MODEL,
        preprocessor_overrides={'device_processor': {'device': device}})
    ds = LeRobotDataset('local/so101_studio', root=DATA,
        delta_timestamps={'action': [i/10 for i in range(20)]})
    cal = json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
    def raw(tensor):
        values = tensor.detach().cpu().numpy().copy()
        for i, n in enumerate(NAMES):
            c = cal[n]; lo, hi = c['range_min'],c['range_max']
            values[...,i] = lo + values[...,i]*(hi-lo)/100 if n=='gripper' else (lo+hi)/2+values[...,i]*4095/360
        return values
    report = {'model':str(MODEL), 'dataset':str(DATA), 'stages':[]}
    samples = [0,5,10,15,20,25,30,40,60,80,100,120,140,160,180]
    with torch.inference_mode():
        for i in samples:
            row=ds[i]
            batch={k: v.unsqueeze(0) for k,v in row.items() if torch.is_tensor(v)}
            batch=pre(batch)
            pred=raw(post(policy.predict_action_chunk(batch)))[0]
            truth=raw(row['action'])
            state=raw(row['observation.state'])
            item={'index':i,'time':float(row['timestamp']), 'state':state.round(1).tolist(),
                  'target0':truth[0].round(1).tolist(), 'pred0':pred[0].round(1).tolist(),
                  'pred9':pred[9].round(1).tolist(), 'pred19':pred[19].round(1).tolist(),
                  'target19':truth[19].round(1).tolist(),
                  'chunk_mae':np.mean(abs(pred-truth),axis=0).round(2).tolist()}
            report['stages'].append(item)
            print(json.dumps(item),flush=True)
        # Teacher-forced evaluation is not a rollout success rate.
        errors=[]
        for start in range(0,185,16):
            indices=range(start,min(start+16,185))
            rs=[ds[i] for i in indices]
            batch={k: torch.stack([r[k] for r in rs]) for k,v in rs[0].items() if torch.is_tensor(v)}
            pred=raw(post(policy.predict_action_chunk(pre(batch))))
            target=raw(torch.stack([r['action'] for r in rs]))
            errors.extend(abs(pred[:,0]-target[:,0]).tolist())
        report['episode0_first_action_mae']=np.mean(errors,axis=0).tolist()
    path=ROOT/'reports/act-offline-diagnosis.json'
    path.write_text(json.dumps(report,indent=2))
    print('Report:',path)
    print('First action MAE:',report['episode0_first_action_mae'])

if __name__=='__main__':
    main()
