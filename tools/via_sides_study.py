"""Signal vias as N-gons (--mom-signal-vias N, with and without the barrel N-gon), with the zone, against fine
references: python tools/mom25d/via_sides_study.py <dir> crop rf_diffvia (see via_study.py for <dir>)."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = sys.argv[1]
from mom25d.board import BoardModel
from mom25d.run import renorm
F = [2e9, 5e9, 8e9]
Z = dict(zone='auto', dmax_out=1.5e-3)
VAR = [('round', 'round', True)] + [(f'{n}-gon', n, True) for n in (16, 12, 8, 6, 4)] + \
      [('8-gon, no barrel', 8, False), ('4-gon, no barrel', 4, False)]
out = {}
for b in sys.argv[2:]:
    if b == 'crop':
        from edge_zone_study_meshes import make
        mk = lambda **kw: make(port='via', **kw)
        ref = json.load(open(f'{SP}/viastudy.json'))
        R = np.array([complex(*v) for v in ref['REF tris (fine)']['S']]).reshape(3, 2, 2)
    else:
        D = json.load(open(f'{SP}/b_{b}/{b}_board.json')); R = np.load(f'{SP}/ref_{b}.npy')

        def mk(**kw):
            m = BoardModel(D, port='via', log=lambda *a: None)
            planes = {k: 0.9e-3 for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
            return m.simulation(dmax=0.3e-3, dmax_layer=planes, characterize=False, verbose=False, solver='auto', **kw)
    for name, sv, vb in VAR:
        s = mk(signal_vias=sv, via_barrel=vb, **Z)
        t = time.time()
        S = np.array([renorm(s.sparams(f), np.array([p.zc for p in s.ports]), np.full(len(s.ports), 50.0)) for f in F])
        t = (time.time() - t) / len(F)
        e = np.abs(S - R).max((1, 2))
        out[f'{b}|{name}'] = dict(N=int(s.Ne + s.Nv), err=e.tolist())
        print(f"{b:10s} {name:18s} N={s.Ne + s.Nv:6d} {t:5.1f} s/f  |dS| vs ref at 2/5/8 GHz {np.round(e, 4).tolist()}", flush=True)
        json.dump(out, open(f'{SP}/nstudy.json', 'w'))
