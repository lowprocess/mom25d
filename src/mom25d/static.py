"""Reciprocal free-space 1/R moments, tiled across CPU/device fill backends."""
from math import isqrt
import time

import numpy as np

from .static_backends import FillEngine, bounded_map


def static_moments(nodes, tris, z, near=3.0, outer_levels=1, chunk=16384, scalar_only=False,
                   backend='auto', workers=None, device=None, return_info=False):
    """Return (local nodes, M0, Mrx, Mry, Mrr), in SI units.

    Pairs within `near` times the sum of their bounding radii use analytic
    source integration; close observation triangles subdivide `outer_levels`
    times. Far pairs use the three-point product rule. near=inf is the
    all-analytic reference, scalar_only returns just (local nodes, M0).

    backend='auto' uses OpenMP if built, otherwise threaded NumPy. 'numpy'
    selects the serial reference. Explicit unavailable backends raise.
    workers controls CPU threads; chunk bounds tile pair count. Independent
    pair results are assembled by one writer, with identical reciprocity for
    every backend. Storage remains O(Nt**2), scratch is O(chunk*workers).
    return_info=True returns (moments, execution metadata).
    """
    started = time.perf_counter()
    engine = FillEngine(nodes, tris, z, near, outer_levels, backend, workers, chunk, device)
    g = engine.geometry
    n, side = len(tris), isqrt(engine.chunk)
    matrices = [np.empty((n, n)) for _ in range(1 if scalar_only else 4)]

    def tiles():
        for p0 in range(0, n, side):
            for q0 in range(p0, n, side):
                yield p0, min(n, p0+side), q0, min(n, q0+side)

    def evaluate(tile):
        p0, p1, q0, q1 = tile
        p, q = np.meshgrid(np.arange(p0, p1, dtype=np.int32), np.arange(q0, q1, dtype=np.int32), indexing='ij')
        p, q = p.ravel(), q.ravel()
        if p0 == q0:
            upper = p <= q
            p, q = p[upper], q[upper]
        d = g.z[p]-g.z[q]
        kind = engine.kinds(p, q, d)
        return tile, p, q, engine.evaluate(p, q, d, kind, scalar_only), np.count_nonzero(kind >= 0)

    count, analytic = 0, 0
    for tile, p, q, m, close_count in bounded_map(evaluate, tiles(), engine.tile_workers):
        count += len(p); analytic += int(close_count)
        p0, p1, q0, q1 = tile
        if p0 != q0:
            # Contiguous block copies avoid repeated scattered index writes,
            # especially when compiled/device kernels make integration cheap.
            block = m.reshape(p1-p0, q1-q0, m.shape[1])
            matrices[0][p0:p1, q0:q1] = block[..., 0]
            matrices[0][q0:q1, p0:p1] = block[..., 0].T
            if not scalar_only:
                for matrix, forward, reverse in ((matrices[1], 1, 3), (matrices[2], 2, 4), (matrices[3], 5, 5)):
                    matrix[p0:p1, q0:q1] = block[..., forward]
                    matrix[q0:q1, p0:p1] = block[..., reverse].T
            continue
        matrices[0][p, q] = matrices[0][q, p] = m[:, 0]
        if not scalar_only:
            mx, my, rr = matrices[1:]
            mx[p, q], my[p, q] = m[:, 1], m[:, 2]
            mx[q, p], my[q, p] = m[:, 3], m[:, 4]
            rr[p, q] = rr[q, p] = m[:, 5]
            diagonal = p == q
            mx[p[diagonal], q[diagonal]] = (m[diagonal, 1]+m[diagonal, 3])/2
            my[p[diagonal], q[diagonal]] = (m[diagonal, 2]+m[diagonal, 4])/2
    result = (g.nodes, *matrices)
    info = dict(engine.info, pairs=count, analytic_pairs=analytic, scalar_only=bool(scalar_only),
                seconds=time.perf_counter()-started)
    return (result, info) if return_info else result
