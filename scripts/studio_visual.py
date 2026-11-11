"""Fixed-background image normalization; no hardware and no action generation.
Only image translation is estimated from a stationary background patch. The
startup offset is cached for a run; never fit using cube labels/policy outputs.
"""
import base64,json
from pathlib import Path
import cv2
import numpy as np


class BackgroundView:
    def __init__(self,config):
        self.config=config;self.offset=None
        self.patch=cv2.imdecode(np.frombuffer(base64.b64decode(config['patch_png']),np.uint8),cv2.IMREAD_GRAYSCALE)
        if self.patch is None:raise ValueError('Invalid visual reference patch')

    @classmethod
    def from_rgb(cls,rgb):
        patch=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)[70:112,70:145]
        ok,png=cv2.imencode('.png',patch)
        if not ok:raise ValueError('Cannot encode visual reference')
        return cls({'version':1,'kind':'background_translation','patch_png':base64.b64encode(png).decode(),
            'reference_xy':[70,70],'search_xywh':[50,45,155,100],'minimum_correlation':.75,
            'frame_size':[320,240],'border':'replicate','fit':'startup-only, no object labels',
            'training_augmentation':{'translation_pixels':12,'gain':[.9,1.1],'bias':[-.03,.03]}})

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))
    def save(self,path):Path(path).write_text(json.dumps(self.config,indent=2))

    def estimate(self,rgb):
        if rgb.shape!=(240,320,3):raise ValueError('Expected RGB 320x240')
        x,y,w,h=self.config['search_xywh'];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        scores=cv2.matchTemplate(gray[y:y+h,x:x+w],self.patch,cv2.TM_CCOEFF_NORMED)
        _,score,_,where=cv2.minMaxLoc(scores)
        if score<self.config['minimum_correlation']:
            raise ValueError(f'固定背景未匹配（{score:.3f}），不使用不确定的视觉变换')
        rx,ry=self.config['reference_xy']
        return {'dx':where[0]+x-rx,'dy':where[1]+y-ry,'correlation':score}

    def transform_rgb(self,rgb,*,cache=True):
        offset=self.offset if cache else None
        if offset is None:
            offset=self.estimate(rgb)
            if cache:self.offset=offset
        matrix=np.array([[1,0,offset['dx']],[0,1,offset['dy']]],np.float32)
        return cv2.warpAffine(rgb,matrix,(320,240),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)


def translate_batch(images,offsets,*,augment=False):
    """Equivalent translation to CV warp; states/actions are never changed."""
    import torch
    import torch.nn.functional as F
    n,_,h,w=images.shape
    shifts=torch.tensor([[r['dx'],r['dy']]for r in offsets],device=images.device,dtype=images.dtype)
    if augment:shifts=shifts+torch.randint(-12,13,(n,2),device=images.device).to(images.dtype)
    theta=torch.zeros((n,2,3),dtype=images.dtype,device=images.device)
    theta[:,0,0]=theta[:,1,1]=1;theta[:,0,2]=2*shifts[:,0]/w;theta[:,1,2]=2*shifts[:,1]/h
    grid=F.affine_grid(theta,images.shape,align_corners=False)
    transformed=F.grid_sample(images,grid,mode='bilinear',padding_mode='border',align_corners=False)
    if augment:
        gain=.9+.2*torch.rand((n,1,1,1),device=images.device)
        bias=-.03+.06*torch.rand((n,1,1,1),device=images.device)
        transformed=(transformed*gain+bias).clamp(0,1)
    return transformed


class BatchView:
    def __init__(self,offsets):self.offsets=offsets
    def __call__(self,batch,*,augment=False):
        ids=batch['episode_index'].reshape(-1).tolist()
        copied=dict(batch)
        copied['observation.images.external_front']=translate_batch(batch['observation.images.external_front'],
            [self.offsets[int(i)]for i in ids],augment=augment)
        return copied
