"""Cropped demo transition with via ports (reciprocal): triangles against quad cells (--mom-quads), and a fine
reference. python tools/mom25d/quad_study.py"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_zone_study_meshes import make
from mom25d.run import renorm
F = [2e9, 5e9, 8e9]
FINE = dict(dmax=0.2e-3, dmax_layer={2: 0.5e-3}, ncross=6)
cases = [('REF tris (fine)', FINE), ('default tris', {}), ('default quads', dict(quads=True)),
         ('fine quads', dict(FINE, quads=True)), ('ncross 2 tris', dict(ncross=2)), ('ncross 2 quads', dict(ncross=2, quads=True))]
out, ref = {}, None
for name, kw in cases:
    s = make(port='via', **kw); zr = np.full(len(s.ports), 50.0)
    t = time.time()
    S = np.array([renorm(s.sparams(f), np.array([p.zc for p in s.ports]), zr) for f in F])
    t = (time.time() - t) / len(F)
    N = s.qT.shape[1] if getattr(s, 'qT', None) is not None else s.Ne + s.Nv
    if ref is None: ref = S
    tot = s.log.totals()
    print(f"{name:18s} N={N:6d} {t:5.1f} s/f (LU {tot.get('LU factor', 0)/3:4.1f}, proj {tot.get('quad projection', 0)/3:3.1f})"
          f"  |dS| vs ref per f {np.round(np.abs(S - ref).max((1, 2)), 4).tolist()}  recip {np.abs(S[:, 0, 1] - S[:, 1, 0]).max():.1e}", flush=True)
    out[name] = dict(N=int(N), t=t, S=[[v.real, v.imag] for v in S.ravel()])
json.dump(out, open('quad_study.json', 'w'))
