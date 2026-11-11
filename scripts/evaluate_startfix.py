"""Compare old/new ACT on identical recorded observations, without hardware.

No claim of robot success is made from this teacher-forced comparison.
"""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from run_act_inference import denormalize_joints

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    reference = LeRobotDataset('local/so101_studio', root=ROOT/'datasets/lerobot_20261106T172457Z')
    calibration=json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
    raw_source=json.loads((ROOT/'datasets/studio_dataset_v2_20261106_170913/episode_0000.json').read_text())
    # Compare both policies on exactly the same image and joint observations.
    indices=[0,5,10,15,20,25,30,40,60,80,100,120,140,160,180]
    models={'old':ROOT/'outputs/act_20261106T172457Z/final','startfix':args.model.resolve()}
    report={'reference':str(reference.root),'models':{}}
    for label,path in models.items():
        policy=ACTPolicy.from_pretrained(str(path)).to('mps').eval()
        pre,post=make_pre_post_processors(policy.config,pretrained_path=path,
            preprocessor_overrides={'device_processor':{'device':'mps'}})
        values=[]
        with torch.inference_mode():
            for index in indices:
                row=reference[index]
                batch={k:row[k].unsqueeze(0) for k in ('observation.state','observation.images.external_front')}
                prediction=post(policy.predict_action_chunk(pre(batch)))[0].cpu().numpy()
                assert np.isfinite(prediction).all()
                raw=np.array([denormalize_joints(a,calibration) for a in prediction])
                entry={'index':index,'time_s':float(row['timestamp']),
                       'pred0':raw[0].tolist(),'pred9':raw[9].tolist(),'pred19':raw[19].tolist(),
                       'pred_last':raw[-1].tolist(),
                       'shoulder_max_first20':int(raw[:20,1].max()),
                       'first20_departure_step':next((i for i,v in enumerate(raw[:20,1]) if v>1100),None)}
                print(label,json.dumps(entry),flush=True)
                values.append(entry)
            # Exact standby position used in the trial, with recorded initial image.
            # This is a counterfactual robustness probe, NOT a real sensor observation.
            from run_act_inference import normalize_joints
            standby=[2081,1025,3063,2875,2093,2101]
            row=reference[0]
            batch={'observation.state':torch.from_numpy(normalize_joints(standby,calibration)).unsqueeze(0),
                   'observation.images.external_front':row['observation.images.external_front'].unsqueeze(0)}
            prediction=post(policy.predict_action_chunk(pre(batch)))[0].cpu().numpy()
            raw=np.array([denormalize_joints(a,calibration) for a in prediction])
            report['models'][label]={'path':str(path),'chunk_size':policy.config.chunk_size,
                'n_action_steps':policy.config.n_action_steps,'stages':values,
                'standby_counterfactual_first20':raw[:20].tolist()}
        del policy
        torch.mps.empty_cache()
    output=ROOT/'reports/startfix-comparison.json'
    output.write_text(json.dumps(report,indent=2))
    print('COMPARISON SAVED:',output,flush=True)


if __name__=='__main__':
    main()
