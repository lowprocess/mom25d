"""PEEC preparation/factor reuse and capacitance-only timing on two plates.

    OPENBLAS_NUM_THREADS=2 python tools/benchmark_peec.py
"""
import json
import time

import numpy as np
from shapely.geometry import box, mapping

from mom25d.mqs import MQS


def main():
    board = dict(stack=dict(layers=[], copper=[dict(name='B', z=0, t=.035), dict(name='F', z=.5, t=.035)]),
                 copper={l: [dict(net=l, polygon=mapping(box(0, 0, 10, 10)))] for l in ('B', 'F')}, vias=[])
    terms = {l: [(l, box(0, 0, 1, 1))] for l in ('B', 'F')}
    q = MQS(board, {'B', 'F'}, terminals=terms, dmax=1.2, capacitance=True, log=lambda *_: None)
    stats = dict(triangles=len(q.tris), edges=q.ne)
    start = time.perf_counter()
    q.capacitance_matrix()
    stats['capacitance_only_s'] = time.perf_counter()-start
    stats['capacitance_only_built_magnetic_matrices'] = q._copper is not None
    q.source('I', 'F', 'B', part_h=0)
    start = time.perf_counter()
    a = q.solve([1e3, 1e6, 1e7])
    stats['first_solve_s'] = time.perf_counter()-start
    n = q._peec['factorizations']
    start = time.perf_counter()
    b = q.solve([1e3, 1e6, 1e7], amps={'I': 2j})
    stats['repeated_solve_s'] = time.perf_counter()-start
    stats['additional_factorizations'] = q._peec['factorizations']-n
    stats['factor_cache_bytes'] = q._peec['factor_bytes']
    stats['excitation_linearity_relative_error'] = float(abs(b.i-2j*a.i).max()/abs(a.i).max())
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
