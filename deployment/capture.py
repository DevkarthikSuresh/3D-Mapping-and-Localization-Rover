"""Record a short RGB video from an OpenCV camera (no rover motor control)."""
import argparse
from pathlib import Path
import time
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',type=int,default=0)
    p.add_argument('--output',required=True); p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--fps',type=float,default=20); p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480)
    a=p.parse_args(); path=Path(a.output)
    if path.exists(): raise FileExistsError(path)
    if a.seconds<=0 or a.fps<=0: raise ValueError('Positive duration and fps required')
    cap=cv2.VideoCapture(a.device); writer=None
    try:
        if not cap.isOpened(): raise RuntimeError('Camera not available')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height); cap.set(cv2.CAP_PROP_FPS,a.fps)
        ok,frame=cap.read()
        if not ok: raise RuntimeError('No frames received')
        h,w=frame.shape[:2]; writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(w,h))
        if not writer.isOpened(): raise RuntimeError('Video writer failed')
        start=time.monotonic(); count=0
        while ok and time.monotonic()-start<a.seconds:
            writer.write(frame); count+=1
            delay=start+count/a.fps-time.monotonic()
            if delay>0: time.sleep(delay)
            ok,frame=cap.read()
        print(f'Wrote {count} frames to {path}')
    finally:
        cap.release()
        if writer is not None: writer.release()


if __name__=='__main__': main()
