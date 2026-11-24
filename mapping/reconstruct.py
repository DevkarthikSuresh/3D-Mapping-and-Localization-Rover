"""Video -> COLMAP -> undistorted dataset -> external 3DGS training/rendering."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(*args, cwd=None):
    args = [str(x) for x in args]
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})


def reconstruct(video, workspace, gs_repo=None, fps=3, iterations=7000, evaluate=False, colmap='colmap'):
    video, work = Path(video).resolve(), Path(workspace).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if work.exists() and any(work.iterdir()):
        raise ValueError('Use a new or empty workspace; existing data will not be overwritten.')
    if fps <= 0 or iterations <= 0:
        raise ValueError('fps and iterations must be positive')
    if gs_repo is not None:
        gs_repo = Path(gs_repo).resolve()
        if not all((gs_repo / x).is_file() for x in ('train.py', 'render.py')):
            raise ValueError('Pass the installed INRIA Gaussian Splatting checkout.')
    images, sparse, data = work/'images', work/'sparse', work/'undistorted'
    for p in (images, sparse, data):
        p.mkdir(parents=True, exist_ok=True)
    db = work/'database.db'
    run('ffmpeg', '-nostdin', '-i', video, '-vf', f'fps={fps}', '-q:v', '2', images/'frame_%04d.jpg')
    run(colmap, 'feature_extractor', '--database_path', db, '--image_path', images,
        '--ImageReader.single_camera', '1', '--SiftExtraction.use_gpu', '0')
    run(colmap, 'exhaustive_matcher', '--database_path', db, '--SiftMatching.use_gpu', '0')
    run(colmap, 'mapper', '--database_path', db, '--image_path', images, '--output_path', sparse)
    models = sorted(p for p in sparse.iterdir() if (p/'images.bin').exists())
    if len(models) != 1:
        raise RuntimeError(f'Expected one connected reconstruction, found {len(models)}; inspect sparse/ before proceeding.')
    run(colmap, 'image_undistorter', '--image_path', images, '--input_path', models[0],
        '--output_path', data, '--output_type', 'COLMAP')
    model = data/'sparse'/'0'
    model.mkdir(exist_ok=True)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        src = data/'sparse'/name
        if src.exists():
            src.rename(model/name)
        if not (model/name).is_file():
            raise FileNotFoundError(model/name)
    text = work/'colmap_text'
    text.mkdir()
    run(colmap, 'model_converter', '--input_path', model, '--output_path', text, '--output_type', 'TXT')
    if gs_repo is not None:
        flags = ['--eval'] if evaluate else []
        output = work/'gs_output'
        run(sys.executable, gs_repo/'train.py', '-s', data, '-m', output, '--iterations', iterations,
            '--save_iterations', iterations, '--test_iterations', iterations, *flags, cwd=gs_repo)
        run(sys.executable, gs_repo/'render.py', '-s', data, '-m', output, '--iteration', iterations, *flags, cwd=gs_repo)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True); p.add_argument('--workspace', required=True)
    p.add_argument('--gs-repo'); p.add_argument('--fps', type=float, default=3)
    p.add_argument('--iterations', type=int, default=7000); p.add_argument('--eval', action='store_true')
    p.add_argument('--colmap', default='colmap', help='COLMAP executable; flags follow COLMAP 3.7')
    a = p.parse_args()
    reconstruct(a.video, a.workspace, a.gs_repo, a.fps, a.iterations, a.eval, a.colmap)


if __name__ == '__main__':
    main()
