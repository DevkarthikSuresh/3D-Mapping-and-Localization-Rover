"""Evaluate paired rendered images or camera poses with explicit metric scale."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def psnr(reference, estimate):
    if reference.shape != estimate.shape: raise ValueError('Image shape mismatch')
    mse=np.mean((reference.astype(float)-estimate.astype(float))**2)
    return float('inf') if mse==0 else float(10*np.log10(255**2/mse))


def pose_error(pred, truth, meters_per_unit=None):
    c=np.asarray(pred['camera_center_world']); g=np.asarray(truth['camera_center_world'])
    R=np.asarray(pred['R_world_to_camera']); G=np.asarray(truth['R_world_to_camera'])
    if c.shape!=(3,) or g.shape!=(3,) or R.shape!=(3,3) or G.shape!=(3,3): raise ValueError('Invalid pose dimensions')
    error=float(np.linalg.norm(c-g)); angle=float(np.degrees(np.arccos(np.clip((np.trace(R@G.T)-1)/2,-1,1))))
    result={'position_scene_units':error,'rotation_deg':angle}
    if meters_per_unit is not None:
        if meters_per_unit<=0: raise ValueError('Scale must be positive')
        result['position_m']=error*meters_per_unit
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    im=sub.add_parser('images'); im.add_argument('--reference',required=True); im.add_argument('--renders',required=True)
    po=sub.add_parser('pose'); po.add_argument('--prediction',required=True); po.add_argument('--truth',required=True); po.add_argument('--meters-per-unit',type=float)
    a=p.parse_args()
    if a.mode=='images':
        ref=Path(a.reference); est=Path(a.renders); names=sorted(x.name for x in ref.glob('*.png'))
        if not names or set(names)!={x.name for x in est.glob('*.png')}: raise ValueError('PNG sets must match exactly')
        scores={}
        for n in names:
            x=cv2.imread(str(ref/n)); y=cv2.imread(str(est/n))
            if x is None or y is None: raise ValueError(f'Cannot decode {n}')
            scores[n]=psnr(x,y)
        result={'count':len(scores),'mean_psnr_db':float(np.mean(list(scores.values()))),'per_image_db':scores}
    else:
        pred=json.loads(Path(a.prediction).read_text()); truth=json.loads(Path(a.truth).read_text())
        result=pose_error(pred.get('pose',pred),truth.get('pose',truth),a.meters_per_unit)
    def safe(v):
        if isinstance(v,dict): return {k:safe(x) for k,x in v.items()}
        return 'Infinity' if isinstance(v,float) and np.isinf(v) else v
    print(json.dumps(safe(result),indent=2,allow_nan=False))


if __name__=='__main__': main()
