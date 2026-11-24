"""ORB keyframe retrieval and PnP-RANSAC using COLMAP's sparse 2D–3D tracks."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def records(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.lstrip().startswith('#'):
                yield line.strip()


def load_tracks(model):
    model = Path(model)
    points = {}
    for line in records(model/'points3D.txt'):
        if line:
            p = line.split(); points[int(p[0])] = np.array(p[1:4], dtype=float)
    frames = []
    it = iter(records(model/'images.txt'))
    for line in it:
        if not line: continue
        p = line.split(maxsplit=9)
        if len(p) != 10: raise ValueError('Malformed COLMAP image record')
        track = next(it).split()
        if len(track)%3: raise ValueError('Malformed COLMAP observations')
        xy, ids = [], []
        for i in range(0,len(track),3):
            pid = int(track[i+2])
            if pid in points:
                xy.append([float(track[i]),float(track[i+1])]); ids.append(pid)
        frames.append((p[9], np.asarray(xy).reshape(-1,2), np.array(ids,dtype=np.int64)))
    return points, frames


def build_index(model, images, max_distance=3.0):
    points, frames = load_tracks(model)
    orb = cv2.ORB_create(nfeatures=4000)
    index = []
    # Match detected ORB keypoints spatially to observed COLMAP/SIFT landmarks.
    for name, xy, ids in frames:
        gray = cv2.imread(str(Path(images)/name), cv2.IMREAD_GRAYSCALE)
        if gray is None: raise FileNotFoundError(Path(images)/name)
        kp, desc = orb.detectAndCompute(gray, None)
        if desc is None or len(xy)==0: continue
        seen = set(); rows=[]; landmarks=[]
        for k, descriptor in zip(kp, desc):
            dist = np.sum((xy-np.asarray(k.pt))**2, axis=1); j = int(np.argmin(dist)); pid = int(ids[j])
            if dist[j] <= max_distance**2 and pid not in seen:
                seen.add(pid); rows.append(descriptor); landmarks.append(points[pid])
        if len(rows)>=6:
            index.append((name, np.asarray(rows, np.uint8), np.asarray(landmarks, np.float64)))
    if not index: raise RuntimeError('No usable keyframes; check image/track coordinates and scene texture')
    return index


def solve_pose(xyz, uv, K, distortion=None, min_inliers=6):
    xyz=np.ascontiguousarray(xyz,dtype=np.float64); uv=np.ascontiguousarray(uv,dtype=np.float64)
    K=np.asarray(K,dtype=np.float64)
    if len(xyz)<max(6,min_inliers) or len(uv)!=len(xyz): return None
    ok,r,t,inliers=cv2.solvePnPRansac(xyz,uv,K,distortion,iterationsCount=1000,
                                      reprojectionError=4.0,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<min_inliers: return None
    ii=inliers.ravel()
    r,t=cv2.solvePnPRefineLM(xyz[ii],uv[ii],K,distortion,r,t)
    R=cv2.Rodrigues(r)[0]; projected=cv2.projectPoints(xyz[ii],r,t,K,distortion)[0].reshape(-1,2)
    depth=(xyz[ii]@R.T+t.ravel())[:,2]
    if np.any(depth<=0): return None
    return {'R_world_to_camera':R.tolist(),'t_world_to_camera':t.ravel().tolist(),
            'camera_center_world':(-R.T@t).ravel().tolist(),'inliers':len(ii),
            'median_reprojection_px':float(np.median(np.linalg.norm(projected-uv[ii],axis=1)))}


def localize(index, query, K, distortion=None, top_k=5, ratio=.75):
    kp,desc=cv2.ORB_create(nfeatures=4000).detectAndCompute(query,None)
    if desc is None: return None
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING); candidates=[]
    for name, ref, xyz in index:
        pairs=matcher.knnMatch(desc,ref,k=2)
        matches=sorted((pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<ratio*pair[1].distance),key=lambda m:m.distance)
        used=set(); unique=[]
        for m in matches:
            if m.trainIdx not in used: used.add(m.trainIdx); unique.append(m)
        candidates.append((name,xyz,unique))
    best=None
    for name,xyz,matches in sorted(candidates,key=lambda c:len(c[2]),reverse=True)[:top_k]:
        if len(matches)<6: continue
        pose=solve_pose(xyz[[m.trainIdx for m in matches]], [kp[m.queryIdx].pt for m in matches], K,distortion)
        if pose:
            pose.update(keyframe=name, tentative_matches=len(matches))
            if best is None or (pose['inliers'],-pose['median_reprojection_px'])>(best['inliers'],-best['median_reprojection_px']): best=pose
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--images',required=True); p.add_argument('--query',required=True)
    p.add_argument('--camera',required=True,help='JSON with K, image_size [width,height], optional distortion')
    p.add_argument('--output'); a=p.parse_args()
    camera=json.loads(Path(a.camera).read_text()); query=cv2.imread(a.query,cv2.IMREAD_GRAYSCALE)
    if query is None: raise FileNotFoundError(a.query)
    if list(query.shape[::-1])!=camera['image_size']: raise ValueError('Query dimensions do not match calibration')
    K=np.asarray(camera['K'],float); distortion=np.asarray(camera.get('distortion',[]),float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0: raise ValueError('Invalid camera matrix')
    pose=localize(build_index(a.model,a.images),query,K,distortion if distortion.size else None)
    result=json.dumps({'success':pose is not None,'pose':pose,'units':'COLMAP scene units (not necessarily metres)'},indent=2)
    print(result)
    if a.output: Path(a.output).write_text(result+'\n')
    return 0 if pose else 2


if __name__=='__main__': raise SystemExit(main())
