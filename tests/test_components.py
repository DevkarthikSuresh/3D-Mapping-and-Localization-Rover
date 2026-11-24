import cv2
import numpy as np
import pytest
from compression.prune import prune, read_ply, write_ply
from localization.relocalize import solve_pose, load_tracks, build_index, localize
from benchmarks.evaluate import psnr, pose_error


def test_pruning_round_trip(tmp_path):
    v=np.zeros(10,dtype=[(s,'<f4') for s in ('x','y','z','opacity','f_dc_0','scale_0')]); v['opacity']=np.arange(10); v['x']=np.arange(10)*2
    write_ply(tmp_path/'original.ply',v); out=prune(read_ply(tmp_path/'original.ply'),.6)
    write_ply(tmp_path/'pruned.ply',out)
    np.testing.assert_array_equal(read_ply(tmp_path/'pruned.ply'),v[6:])
    with pytest.raises(ValueError): prune(v,1)


def test_pnp_with_outliers():
    rng=np.random.default_rng(42); xyz=rng.uniform(-1,1,(80,3)); xyz[:,2]+=5
    K=np.array([[600,0,320],[0,600,240],[0,0,1]],float); r=np.array([.1,-.1,.05]); t=np.array([.2,.1,.3])
    uv=cv2.projectPoints(xyz,r,t,K,None)[0].reshape(-1,2); uv[:10]+=80
    pred=solve_pose(xyz,uv,K); R=cv2.Rodrigues(r)[0]; truth={'R_world_to_camera':R,'camera_center_world':-R.T@t}
    assert pred['inliers']>=65
    e=pose_error(pred,truth); assert e['position_scene_units']<.001 and e['rotation_deg']<.01
    assert solve_pose(xyz[:3],uv[:3],K) is None


def test_orb_retrieval_and_pose(tmp_path):
    rng=np.random.default_rng(4); img=rng.integers(0,256,(480,640),dtype=np.uint8)
    kp,_=cv2.ORB_create(nfeatures=4000).detectAndCompute(img,None)
    xy=np.array([k.pt for k in kp[:300]]); z=rng.uniform(3,8,len(xy)); xyz=np.column_stack(((xy[:,0]-320)*z/600,(xy[:,1]-240)*z/600,z))
    (tmp_path/'points3D.txt').write_text('\n'.join(f'{i+1} {x} {y} {zz} 0 0 0 0' for i,(x,y,zz) in enumerate(xyz)))
    (tmp_path/'images.txt').write_text('1 1 0 0 0 0 0 0 1 frame.png\n'+' '.join(f'{x} {y} {i+1}' for i,(x,y) in enumerate(xy))+'\n2 1 0 0 0 0 0 0 1 empty.png\n\n')
    cv2.imwrite(str(tmp_path/'frame.png'),img); cv2.imwrite(str(tmp_path/'empty.png'),img)
    points,frames=load_tracks(tmp_path); assert len(points)==300 and len(frames)==2
    index=build_index(tmp_path,tmp_path,max_distance=.01)
    pose=localize(index,img,np.array([[600,0,320],[0,600,240],[0,0,1]],float)); assert pose is not None
    assert pose['inliers']>50 and np.linalg.norm(pose['camera_center_world'])<.01


def test_metrics():
    a=np.zeros((2,2,3),np.uint8); b=np.full_like(a,255)
    assert psnr(a,b)==0 and np.isinf(psnr(a,a))
    R=np.eye(3); p={'camera_center_world':[1,0,0],'R_world_to_camera':R}; g={'camera_center_world':[0,0,0],'R_world_to_camera':R}
    assert pose_error(p,g,.5)['position_m']==.5


def test_mapping_orchestration(tmp_path, monkeypatch):
    from mapping import reconstruct as mod
    calls=[]
    def fake(*args,**kwargs):
        calls.append(args)
        if args[1]=='mapper':
            model=tmp_path/'run'/'sparse'/'0'; model.mkdir(); (model/'images.bin').touch()
        if args[1]=='image_undistorter':
            sparse=tmp_path/'run'/'undistorted'/'sparse'; sparse.mkdir()
            for name in ('cameras.bin','images.bin','points3D.bin'): (sparse/name).touch()
    monkeypatch.setattr(mod,'run',fake)
    video=tmp_path/'video.mp4'; video.touch()
    result=mod.reconstruct(video,tmp_path/'run')
    assert (result/'sparse'/'0'/'images.bin').exists()
    assert len(calls)==6 and calls[-1][1]=='model_converter'
    with pytest.raises(ValueError): mod.reconstruct(video,tmp_path/'run')
