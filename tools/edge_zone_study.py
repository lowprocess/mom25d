"""Edge rings only near the signal copper (geo.mesh_polygons edge_zone) against the default mesh and a finer
reference, on the cropped demo transition: python tools/mom25d/edge_zone_study.py"""
import sys, time, json, numpy as np; sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from mom25d.run import renorm
from edge_zone_study_meshes import make, VARIANTS
from edge_zone_study_zone import zoned, ZONES
F = [2e9, 5e9, 8e9]
def solve(sim):
    zr = np.full(len(sim.ports), 50.0)
    t = time.time()
    S = np.array([renorm(sim.sparams(f), np.array([p.zc for p in sim.ports]), zr) for f in F])
    return S, (time.time() - t) / len(F)
cases = [('REF dmax 0.2 ncross 6', lambda: make(**VARIANTS['REF dmax 0.2 ncross 6']))]
cases += [(n, (lambda kw=VARIANTS[n]: make(**kw))) for n in ['default (triangle, ncross 4)', 'ncross 2', 'dmax 0.5']]
cases += [(n, (lambda b=ZONES[n][0]: zoned(b))) for n in ['zone 1.0 mm', 'zone 0.5 mm']]
cases += [('zone 0.3 mm', lambda: zoned(0.3))]
ref = None; out = {}
for n, mk in cases:
    sim = mk(); S, t = solve(sim); N = sim.Ne + sim.Nv
    if ref is None: ref = S
    d = np.abs(S - ref).max()
    s21 = 20 * np.log10(np.abs(S[:, 1, 0])); s21r = 20 * np.log10(np.abs(ref[:, 1, 0]))
    s11 = 20 * np.log10(np.abs(S[:, 0, 0]))
    print(f"{n:28s} N={N:6d} {t:6.1f} s/freq  max|dS|={d:.4f}  |dS21|={np.abs(s21 - s21r).max():.3f} dB  "
          f"S11 dB {np.round(s11, 1).tolist()}", flush=True)
    out[n] = dict(N=int(N), t=t, S=[[v.real, v.imag] for v in S.ravel()])
json.dump(out, open(sys.argv[0].rsplit('/', 1)[0] + '/accuracy.json', 'w'))
