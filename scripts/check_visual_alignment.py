"""Offline camera-domain diagnostic. Never opens camera/serial or commands motion.
Fit uses a fixed cabinet box patch only, not cube labels or policy predictions.
Transforms are sensitivity probes, NOT calibrated 3-D extrinsics.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/position-check-20261108T175741Z'
OUTPUT=SOURCE/'visual-alignment'


def images():
    from studio_samples import decode_sample
    m=json.loads((ROOT/'outputs/act_20261108T151508Z_repair/run-manifest.json').read_text())
    old=[]
    for src in m['source']:
        row=json.loads(sorted(Path(src).glob('episode_*.json'))[0].read_text())[0]
        old.append(cv2.cvtColor(decode_sample(row),cv2.COLOR_RGB2BGR))
    return old


def fit():
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/'fit.json').exists():raise ValueError('Refuse to refit existing frozen parameters')
    old=images()[0];gray=cv2.cvtColor(old,cv2.COLOR_BGR2GRAY)
    # The stationary dark box on the back wall; no robot or red cube in patch.
    patch=gray[70:112,70:145]
    measurements=[]
    for pos in (1,2,3):
        for frame in range(3):
            image=cv2.imread(str(SOURCE/f'position{pos}-frame{frame}.jpg'))
            search=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[45:145,50:205]
            scores=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
            _,score,_,location=cv2.minMaxLoc(scores)
            measurements.append({'position':pos,'frame':frame,'correlation':score,
                'dx':location[0]+50-70,'dy':location[1]+45-70})
    first=measurements[0]
    assert first['correlation']>.8
    assert all(abs(x['dx']-first['dx'])<=2 and abs(x['dy']-first['dy'])<=2 and x['correlation']>.8 for x in measurements)
    matrix=[[1.,0.,float(first['dx'])],[0.,1.,float(first['dy'])]]
    result={'reference_patch_xywh':[70,70,75,42],'search_xywh':[50,45,155,100],
        'matrix_reference_to_current':matrix,'selection_rule':'Translation from FIRST current position1 background-box match. Frozen before policy evaluation; positions2/3 validate background only.',
        'measurements':measurements,'warning':'Image-plane approximation only, no real motor run authorized by this fit.'}
    (OUTPUT/'fit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def evaluate():
    import torch
    from studio_act import PolicyBackend
    from run_act_inference import normalize_joints,denormalize_joints
    from studio_samples import resample_episode
    frozen=json.loads((OUTPUT/'fit.json').read_text());matrix=np.array(frozen['matrix_reference_to_current'],np.float32)
    obs=json.loads((SOURCE/'observation.json').read_text());backend=PolicyBackend(ROOT/'outputs'/obs['model']/'final',ROOT)
    m=json.loads((ROOT/'outputs'/obs['model']/'run-manifest.json').read_text())
    references=[]
    for src in m['source']:
        rows=resample_episode(json.loads(sorted(Path(src).glob('episode_*.json'))[0].read_text()))
        references.append(np.array([r['action'] for r in rows[:50]]))
    results=[];sheet=np.zeros((3*240,3*320,3),np.uint8)
    for pos in (1,2,3):
        observation=json.loads((SOURCE/('observation.json'if pos==1 else f'observation-position{pos}.json')).read_text())
        for frame in range(3):
            bgr=cv2.imread(str(SOURCE/f'position{pos}-frame{frame}.jpg'))
            aligned=cv2.warpAffine(bgr,matrix,(320,240),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
            if frame==2:
                sheet[0:240,(pos-1)*320:pos*320]=images()[pos-1]
                sheet[240:480,(pos-1)*320:pos*320]=bgr
                sheet[480:720,(pos-1)*320:pos*320]=aligned
            for label,image in [('original',bgr),('background_aligned',aligned)]:
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                batch={'observation.state':torch.from_numpy(normalize_joints(observation['positions'],backend.calibration)).unsqueeze(0),
                       'observation.images.external_front':torch.from_numpy(rgb.copy()).permute(2,0,1).float().unsqueeze(0)/255}
                with torch.inference_mode():a=backend.post(backend.policy.predict_action_chunk(backend.pre(batch))).cpu().numpy()[0]
                goals=np.array([denormalize_joints(x,backend.calibration)for x in a])
                scores=[float(abs(goals[:len(r),:4]-r[:,:4]).mean())for r in references]
                item={'actual_position':pos,'frame':frame,'input':label,'trajectory_mae':scores,'nearest_trajectory':int(np.argmin(scores))+1,
                      'first20_shoulder_max':int(goals[:20,1].max()),'final_goal':goals[-1].tolist(),'goals':goals.tolist()}
                results.append(item);print(json.dumps({k:v for k,v in item.items()if k!='goals'}),flush=True)
    cv2.imwrite(str(OUTPUT/'recorded-original-aligned.png'),sheet)
    (OUTPUT/'evaluation.json').write_text(json.dumps({'frozen_fit':frozen,'results':results,
        'caveat':'Static observation prediction, not success rate. Current points were used for diagnostic tuning, not independent final tests.'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['fit','evaluate']);a=parser.parse_args()
    fit()if a.action=='fit'else evaluate()
