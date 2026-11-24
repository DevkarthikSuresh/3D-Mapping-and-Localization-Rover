"""Opacity-ranked pruning of INRIA vertex-only binary little-endian float32 PLYs."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_ply(path):
    with open(path, 'rb') as f:
        lines = []
        for _ in range(4096):
            raw = f.readline(4096)
            if not raw or not raw.endswith(b'\n'):
                raise ValueError('Invalid or truncated PLY header')
            line = raw.decode('ascii').strip()
            lines.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError('PLY header too long')
        if lines[:2] != ['ply', 'format binary_little_endian 1.0']:
            raise ValueError('Expected binary little-endian PLY')
        elems = [l.split() for l in lines if l.startswith('element ')]
        if len(elems) != 1 or elems[0][1] != 'vertex':
            raise ValueError('Only vertex-only Gaussian PLY files are supported')
        n = int(elems[0][2]); props = [l.split() for l in lines if l.startswith('property ')]
        if n < 1 or not props or any(len(p) != 3 or p[1] not in ('float','float32') for p in props):
            raise ValueError('Expected nonempty float32 vertex properties')
        dtype = np.dtype([(p[2], '<f4') for p in props])
        if not {'x','y','z','opacity'}.issubset(dtype.names):
            raise ValueError('Missing required Gaussian properties')
        if Path(path).stat().st_size - f.tell() != n*dtype.itemsize:
            raise ValueError('Unexpected vertex payload length')
        return np.fromfile(f, dtype=dtype, count=n)


def write_ply(path, vertices):
    names = vertices.dtype.names
    if not names or any(vertices.dtype[n] != np.dtype('<f4') for n in names):
        raise ValueError('Expected structured float32 vertices')
    with open(path, 'xb') as f:
        header = ['ply','format binary_little_endian 1.0',f'element vertex {len(vertices)}',
                  *[f'property float {n}' for n in names], 'end_header']
        f.write(('\n'.join(header)+'\n').encode('ascii')); vertices.tofile(f)


def prune(vertices, fraction):
    if not 0 <= fraction < 1 or len(vertices) == 0:
        raise ValueError('Prune fraction must be in [0,1), with nonempty input')
    scores = vertices['opacity'] # sigmoid is monotonic; rank raw logits directly
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite opacity')
    count = int(np.floor(len(vertices)*fraction))
    keep = np.ones(len(vertices), dtype=bool)
    keep[np.argsort(scores, kind='stable')[:count]] = False
    return vertices[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input'); p.add_argument('output'); p.add_argument('--fraction', type=float, default=.6)
    a = p.parse_args(); v = read_ply(a.input); out = prune(v, a.fraction); write_ply(a.output, out)
    before, after = Path(a.input).stat().st_size, Path(a.output).stat().st_size
    print(json.dumps({'input_gaussians':len(v),'output_gaussians':len(out),'input_bytes':before,
                      'output_bytes':after,'file_reduction_fraction':1-after/before}, indent=2))


if __name__ == '__main__': main()
