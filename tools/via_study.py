"""Via meshing (--mom-signal-vias, --mom-stitching-vias) with and without the signal zone, against fine
references, on the cropped demo transition ('crop') and example boards (their <name>_board.json under
<dir>/b_<name>/, references cached by tools/mom25d/zone_study.py as <dir>/ref_<name>.npy):
    python tools/mom25d/via_study.py <dir> crop rf_gcpw rf_diffvia"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = sys.argv[1]
from mom25d.board import BoardModel
from mom25d.run import renorm
F = [2e9, 5e9, 8e9]
Z = dict(zone='auto', dmax_out=1.5e-3)
VAR = [('default', {}), ('zone', Z), ('zone, stitching square', dict(Z, stitching_vias='square')),
       ('zone, both square', dict(Z, signal_vias='square', stitching_vias='square')),
       ('no zone, both square', dict(signal_vias='square', stitching_vias='square'))]
def solve(s):
    t = time.time(); S = np.array([renorm(s.sparams(f), np.array([p.zc for p in s.ports]), np.full(len(s.ports), 50.0)) for f in F])
    return S, (time.time() - t) / len(F)
for b in sys.argv[2:]:
    if b == 'crop':
        from edge_zone_study_meshes import make
        mk = lambda **kw: make(port='via', **kw)
        ref = json.load(open(f'{SP}/viastudy.json')); R = np.array([complex(*v) for v in ref['REF tris (fine)']['S']]).reshape(3, 2, 2)
    else:
        D = json.load(open(f'{SP}/b_{b}/{b}_board.json')); R = np.load(f'{SP}/ref_{b}.npy')
        def mk(**kw):
            m = BoardModel(D, port='via', log=lambda *a: None)
            planes = {k: 0.9e-3 for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
            return m.simulation(dmax=0.3e-3, dmax_layer=planes, characterize=False, verbose=False, solver='auto', **kw)
    for name, kw in VAR:
        s = mk(**kw); S, t = solve(s)
        print(f"{b:10s} {name:24s} N={s.Ne + s.Nv:6d} {t:5.1f} s/f  |dS| vs ref at 2/5/8 GHz {np.round(np.abs(S - R).max((1, 2)), 4).tolist()}", flush=True)
