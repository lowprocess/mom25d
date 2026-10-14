"""Signal zone (--mom-zone, Simulation(zone=)) against the default mesh and a fine reference (dmax 0.2 mm,
ncross 6) on example boards, with via ports:
    kicad2ems examples/rf_gcpw.toml --solvers lines --out out/b_rf_gcpw     # writes rf_gcpw_board.json
    python tools/mom25d/zone_study.py out rf_gcpw rf_diffvia
The references are cached as out/ref_<board>.npy."""
import sys, os, json, time, numpy as np
from mom25d.board import BoardModel
from mom25d.run import renorm
SP = sys.argv[1]; F = [2e9, 5e9, 8e9]
def run(D, dmax=0.3e-3, plane=0.9e-3, **kw):
    m = BoardModel(D, port='via', log=lambda *a: None)
    planes = {k: plane for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
    s = m.simulation(dmax=dmax, dmax_layer=planes, characterize=False, verbose=False, solver='auto', **kw)
    t = time.time(); S = np.array([renorm(s.sparams(f), np.array([p.zc for p in s.ports]), np.full(len(s.ports), 50.0)) for f in F])
    return s.Ne + s.Nv, (time.time() - t) / len(F), S
for b in sys.argv[2:]:
    D = json.load(open(f'{SP}/b_{b}/{b}_board.json'))
    rp = f'{SP}/ref_{b}.npy'
    if os.path.exists(rp):
        R = np.load(rp)
    else:
        n, t, R = run(D, 0.2e-3, 0.5e-3, ncross=6); np.save(rp, R); print(f"{b} reference N={n} {t:.1f} s/f", flush=True)
    for name, kw in [('default', {}), ('zone', dict(zone='auto', dmax_out=1.5e-3)), ('fine + zone', dict(dmax=0.2e-3, plane=0.5e-3, ncross=6, zone='auto', dmax_out=1.0e-3))]:
        n, t, S = run(D, **kw)
        print(f"{b:12s} {name:12s} N={n:6d} {t:5.1f} s/f  |dS| vs ref at 2/5/8 GHz {np.round(np.abs(S - R).max((1, 2)), 4).tolist()}", flush=True)
