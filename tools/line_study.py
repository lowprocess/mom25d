"""Line functions (--mom-lines, Simulation(line_basis=)) against fine references, with column lengths 1-3 x dmax:
    python tools/mom25d/line_study.py <dir> rf_coupler crop rf_gcpw rf_diffvia   (<dir> as in via_study.py)"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = sys.argv[1]
from mom25d.board import BoardModel
from mom25d.run import renorm
F = [2e9, 5e9, 8e9]
Z = dict(zone='auto', dmax_out=1.5e-3)
for b in sys.argv[2:]:
    if b == 'crop':
        from edge_zone_study_meshes import make
        base = Z
        mk = lambda **kw: make(port='via', **kw)
        ref = json.load(open(f'{SP}/viastudy.json')); R = np.array([complex(*v) for v in ref['REF tris (fine)']['S']]).reshape(3, 2, 2)
    else:
        D = json.load(open(f'{SP}/b_{b}/{b}_board.json')); R = np.load(f'{SP}/ref_{b}.npy')
        base = Z if b in ('rf_gcpw', 'rf_diffvia') else {}
        def mk(**kw):
            m = BoardModel(D, port='via', log=lambda *a: None)
            planes = {k: 0.9e-3 for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
            return m.simulation(dmax=0.3e-3, dmax_layer=planes, characterize=False, verbose=False, solver='auto', **kw)
    for name, kw in [('base', {}), ('lines even', dict(line_basis='even')), ('lines on', dict(line_basis='on')),
                     ('lines on, step 2', dict(line_basis='on', line_step=2.0)), ('lines on, step 3', dict(line_basis='on', line_step=3.0))]:
        s = mk(**base, **kw); t = time.time()
        S = np.array([renorm(s.sparams(f), np.array([p.zc for p in s.ports]), np.full(len(s.ports), 50.0)) for f in F])
        t = (time.time() - t) / 3
        N = s.qT.shape[1] if getattr(s, 'qT', None) is not None else s.Ne + s.Nv
        note = next((n[7:] for n in s.log.notes if n.startswith('lines:')), '')
        print(f"{b:10s} {name:17s} N={N:6d} {t:5.1f} s/f  |dS| vs ref 2/5/8 GHz {np.round(np.abs(S - R).max((1, 2)), 4).tolist()}  {note[:70]}", flush=True)
