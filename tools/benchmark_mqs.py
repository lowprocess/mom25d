"""Reproducible CPU MQS fill/field benchmark, with reference error metrics.

    OPENBLAS_NUM_THREADS=2 python tools/benchmark_mqs.py --dmax 0.8

Times are wall seconds; the all-analytic reference uses the same refined close
terms as the hybrid path. `--skip-reference` omits that quadratic comparison.
"""
import argparse
import json
import time

import numpy as np
from shapely.geometry import box, mapping

from mom25d.mqs import MQS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dmax', type=float, default=.8, help='mesh size in mm')
    parser.add_argument('--skip-reference', action='store_true')
    parser.add_argument('--fill-backend', default='auto')
    parser.add_argument('--fill-workers', type=int)
    args = parser.parse_args()
    board = dict(stack=dict(layers=[], copper=[dict(name='F', z=0, t=.035)]),
                 copper={'F': [dict(net='N', polygon=mapping(box(0, 0, 20, 10)))]}, vias=[])
    terms = {'a': [('F', box(0, 4, 1, 6))], 'b': [('F', box(19, 4, 20, 6))]}
    q = MQS(board, 'N', terminals=terms, dmax=args.dmax, log=lambda *_: None,
            fill_backend=args.fill_backend, fill_workers=args.fill_workers)
    q.source('I', 'a', 'b', part_h=0)
    stats = dict(triangles=len(q.tris), edges=q.ne)
    start = time.perf_counter()
    R, L = q._edge_matrices()
    stats['hybrid_fill_s'] = time.perf_counter() - start
    stats['fill'] = dict(q.fill_info['magnetic'])
    if not args.skip_reference:
        q.near = np.inf
        start = time.perf_counter()
        _, ref = q._edge_matrices()
        stats['analytic_fill_s'] = time.perf_counter() - start
        stats['inductance_relative_frobenius_error'] = float(np.linalg.norm(L-ref) / np.linalg.norm(ref))
        q.near = 3.0
    q._copper = R, L, *q._via_matrices()
    start = time.perf_counter()
    result = q.solve([1e3, 1e6, 1e7])
    stats['assembly_and_sweep_s'] = time.perf_counter() - start
    x, y, z = np.linspace(.002, .018, 41), np.linspace(.001, .009, 31), np.array([.0015, .003, .005])
    start = time.perf_counter()
    H = q.h_grid(result, [0, 1, 2], x, y, z)
    stats['fft_field_s'] = time.perf_counter() - start
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    start = time.perf_counter()
    direct = q.h_field(result, [0, 1, 2], points).reshape(H.shape)
    stats['direct_field_s'] = time.perf_counter() - start
    stats['field_relative_max_error'] = float(abs(H-direct).max() / abs(direct).max())
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
