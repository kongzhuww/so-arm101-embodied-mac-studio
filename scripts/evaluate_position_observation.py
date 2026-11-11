"""Evaluate SAVED real position observations. No serial/camera or motion calls.
Template distances describe predicted trajectories, not recognition probabilities.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from studio_act import PolicyBackend
from studio_samples import resample_episode,decode_sample
from run_act_inference import normalize_joints,denormalize_joints
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('observation',type=Path)
    parser.add_argument('--model',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    obs=json.loads(args.observation.read_text());folder=Path(obs['folder'])
    model=args.model or ROOT/'outputs'/obs['model']/'final'
    backend=PolicyBackend(model,ROOT)
    manifest=json.loads((model.parent/'run-manifest.json').read_text())
    refs=[];poses=[];reference_images=[]
    for src in manifest['source']:
        src=Path(src);file=sorted(src.glob('episode_*.json'))[0];rows=resample_episode(json.loads(file.read_text()))
        refs.append(np.array([r['action']for r in rows[:50]],dtype=float))
        poses.append(json.loads((src/'metadata'/file.name).read_text())['version']['waypoints']['p2_grasp'])
        reference_images.append(decode_sample(rows[0]))
    frames=sorted(folder.glob(f'position{obs["position"]}-frame*.jpg'))
    images=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)for p in frames]+reference_images
    if backend.view is not None:
        images=[backend.view.transform_rgb(image,cache=False)for image in images]
    state=torch.from_numpy(normalize_joints(obs['positions'],backend.calibration))
    batch={'observation.state':torch.stack([state]*len(images)),
           'observation.images.external_front':torch.stack([torch.from_numpy(im.copy()).permute(2,0,1).float()/255 for im in images])}
    with torch.inference_mode():out=backend.post(backend.policy.predict_action_chunk(backend.pre(batch))).cpu().numpy()
    entries=[]
    for i,chunk in enumerate(out):
        goals=np.array([denormalize_joints(a,backend.calibration)for a in chunk]);joint_indices=[0,1,2,3]
        scores=[float(np.mean(abs(goals[:len(ref),joint_indices]-ref[:,joint_indices])))for ref in refs]
        pose_scores=[float(np.min(np.mean(abs(goals[:,joint_indices]-np.array(p)[joint_indices]),axis=1)))for p in poses]
        item={'image':str(frames[i])if i<len(frames)else f'recorded_reference_{i-len(frames)+1}',
              'trajectory_mae_to_versions':scores,'nearest_trajectory':int(np.argmin(scores))+1,
              'closest_predicted_pose_mae_to_grasps':pose_scores,'nearest_grasp_pose':int(np.argmin(pose_scores))+1,
              'first20_shoulder_max':int(goals[:20,1].max()),'last_goal':goals[-1].tolist(),'goals':goals.tolist()}
        entries.append(item);print(json.dumps({k:v for k,v in item.items()if k!='goals'}),flush=True)
    report={'observation':obs,'evaluated_model':str(model),'grasp_targets':poses,'results':entries,
            'caveat':'One static 5-second prediction, NOT a physical rollout or calibrated class probability. Fixed-pose current images and recorded-reference images are compared offline only.'}
    destination=args.output or folder/f'prediction-position{obs["position"]}.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2));print('SAVED',destination)

if __name__=='__main__':main()
