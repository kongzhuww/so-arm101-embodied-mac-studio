"""Offline prediction of saved current start observation; never opens camera or bus."""
import json
from pathlib import Path
import numpy as np
import cv2
from studio_act import PolicyBackend
from studio_samples import decode_sample,resample_episode

ROOT=Path(__file__).resolve().parents[1]

def main():
    state=json.loads((ROOT/'reports/act-start-stall-current.json').read_text())
    path=ROOT/'outputs'/state['model']/'final';backend=PolicyBackend(path,ROOT)
    from run_act_inference import normalize_joints,denormalize_joints
    import torch
    live=cv2.cvtColor(cv2.imread(str(ROOT/'reports/act-start-stall-current.jpg')),cv2.COLOR_BGR2RGB)
    manifest=json.loads((path.parent/'run-manifest.json').read_text());scenarios=[]
    for g,source in enumerate(manifest['source'],1):
        row=resample_episode(json.loads(sorted(Path(source).glob('episode_*.json'))[0].read_text()))[0]
        scenarios.extend([(f'recorded{g}',row['state'],decode_sample(row)),(f'current_state_recorded_image{g}',state['positions'],decode_sample(row))])
    scenarios.insert(0,('current',state['positions'],live))
    result={'model':str(path),'current_state':state['positions'],'scenarios':{}}
    with torch.inference_mode():
        for label,positions,rgb in scenarios:
            batch={'observation.state':torch.from_numpy(normalize_joints(positions,backend.calibration)).unsqueeze(0),
                   'observation.images.external_front':torch.from_numpy(rgb.copy()).permute(2,0,1).float().unsqueeze(0)/255}
            out=backend.post(backend.policy.predict_action_chunk(backend.pre(batch))).cpu().numpy()[0]
            goals=np.array([denormalize_joints(x,backend.calibration)for x in out])
            result['scenarios'][label]={'state':positions,'goals':goals.tolist(),
                'first20_shoulder_max':int(goals[:20,1].max()),'full50_shoulder_max':int(goals[:,1].max()),
                'shoulder_indices':{str(i):int(goals[i,1])for i in [0,9,19,29,39,49]}}
            print(label,result['scenarios'][label]['shoulder_indices'],'first20 max',goals[:20,1].max(),flush=True)
    (ROOT/'reports/act-current-start-probe.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
