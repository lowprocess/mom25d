"""Quasi-static backend scaling and accuracy; no BLAS solve in these timings.

    python -m mom25d.core.build
    OPENBLAS_NUM_THREADS=1 python tools/benchmark_static_fill.py --dmax .5 --repeat 3

GPU explicitly: --backends numpy,cupy --workers 1 --device 0
Unavailable explicitly selected backends fail rather than changing the workload.
"""
import argparse
import json
import time

import numpy as np
from shapely.geometry import box

from mom25d.geo import mesh_polygons
from mom25d.static import static_moments


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dmax', type=float, default=.5, help='triangle mesh size (mm)')
    ap.add_argument('--backends', default='numpy,threaded,openmp')
    ap.add_argument('--workers', default='1,2')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--chunk', type=int, default=16384)
    ap.add_argument('--scalar', action='store_true')
    ap.add_argument('--device', type=int)
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error('--repeat must be positive')
    nodes, tris, _ = mesh_polygons(box(0, 0, .02, .01), args.dmax*1e-3, args.dmax*1e-3)
    z = np.zeros(len(tris))
    reference = static_moments(nodes, tris, z, backend='numpy', chunk=args.chunk, scalar_only=args.scalar)
    report = dict(triangles=len(tris), scalar_only=args.scalar, cases=[])
    for backend in args.backends.split(','):
        counts = [1] if backend in ('numpy', 'cupy') else [int(n) for n in args.workers.split(',')]
        for workers in counts:
            times, error = [], 0.
            for _ in range(args.repeat):
                start = time.perf_counter()
                result, info = static_moments(nodes, tris, z, backend=backend, workers=workers,
                                              device=args.device if backend == 'cupy' else None,
                                              chunk=args.chunk, scalar_only=args.scalar, return_info=True)
                times.append(time.perf_counter()-start)
                for a, b in zip(reference[1:], result[1:]):
                    error = max(error, float(np.linalg.norm(a-b)/np.linalg.norm(a)))
                del result
            case = dict(info, seconds_median=float(np.median(times)), seconds_samples=times,
                        max_relative_moment_error=error)
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
    if args.out:
        with open(args.out, 'w') as out:
            json.dump(report, out, indent=2)
            out.write('\n')


if __name__ == '__main__':
    main()
