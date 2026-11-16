"""
FD mesher trial: Frontal-Delaunay and other alternatives to the mesher's interior (cdtmesh.mesh), compared on the
solvers' answers. Each variant has cdtmesh.mesh's signature and is swapped in for it; geo.mesh_polygons (the
outline, the edge rings, the constraints) is unchanged.

    python tools/fd_mesher_trial.py s  [--boards demo.json,launch.json,rf/coupler.json,thru.json] [--out DIR]
    python tools/fd_mesher_trial.py qs --pcbqs PATH/TO/pcbqs       (a pcbqs checkout: its tests' boards)
    python tools/fd_mesher_trial.py quality

  cdt          the mesher as it is: a hexagonal seed at the local size, circumcentre rounds to the area limits, ODT
               smoothing; mesh2d's constrained Delaunay triangulation
  refine_only  Delaunay refinement alone (Ruppert / Triangle's way): circumcentre rounds, no seed, no smoothing
  frontal      Frontal-Delaunay: rows of points advancing in from every segment, each row sqrt(3)/2 h in and shifted
               h/2 along it (equilateral cells off a straight front), kept where the nearest segment is the one it
               came from (the fronts meet); the rounds; Delaunay connectivity
  minmax       cdt's points, connected by edge flips that lower the largest angle (the max-angle condition) in
               place of Delaunay's largest smallest angle
  qhull        cdt's points, scipy's unconstrained Delaunay (quality only: it does not keep the segments)

Results (2026-11-16; s: dmax 0.4 mm against the cdt mesher at 0.2 mm, rms |dS| at 1, 3, 6 GHz, unknowns):

                    demo             demo, quads=all   launch           launch, quads=all
    cdt             0.0139 (3432)    0.0142 (2716)     0.0136 (4418)    0.0170 (3633)
    refine_only     0.0243 (3369)    0.0243 (2674)     0.0138 (4649)    0.0184 (3857)
    frontal         0.0106 (3462)    0.0088 (2792)     0.0110 (4544)    0.0214 (3795)
    minmax          0.0139 (3432)    0.0141 (2726)     0.0136 (4418)    0.0170 (3645)

  coupler and thru: all the same (traces: the edge rings' points, no interior). The reference is cdt's mesh, so a
  bias, if any, is cdt's way. Frontal with quads on launch is worse: its equilateral rows pair into 60/120 degree
  rhombi, which the 35 degree tolerance takes.
  qs (pcbqs): a via's spreading into a plate against a fine pad's, dmax 0.5 / 0.25 mm; the heated disc's largest
  error against the exact solution, 2 / 1 mm cells:
    cdt -2.87 / -1.73 %, 0.200 / 0.029 K; refine_only -2.56 / -1.33 %, 0.173 / 0.016 K;
    frontal -2.64 / -1.69 %, 0.137 / 0.027 K; minmax -2.87 / -1.76 %, 0.196 / 0.030 K
  quality: qhull loses 17-19 % of the segments on planes (the edge rings, the vias' n-gons): no ports, no via fans.
  minmax flips next to nothing: the Delaunay connectivity stays. frontal's prototype here takes 1-4 s a board (NumPy
  loops; the cdt seed well under 1 s).
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree, Delaunay


def variants(cm):
    """cm: the cdtmesh module (mom25d's or pcbqs's). {name: mesh function}."""
    orig = cm.mesh

    def cdt(*a, **k):
        return orig(*a, **k)

    def refine_only(V, segs, limit, inside, core, frozen=None, sizes=(1.0,), min_angle=0.0, max_steiner=0, smooth=0):
        seed = cm._hex_seed
        cm._hex_seed = lambda *a, **k: np.zeros((0, 2))
        try:
            return orig(V, segs, limit, inside, core, frozen, sizes, min_angle, max_steiner, smooth=0)
        finally:
            cm._hex_seed = seed

    def frontal(V, segs, limit, inside, core, frozen=None, sizes=(1.0,), min_angle=0.0, max_steiner=0, smooth=0):
        """Frontal-Delaunay: layers of points advancing in from the segments, each layer sqrt(3)/2 h in and shifted
        h/2 along (equilateral cells off a straight front), kept where the nearest segment is the one it came from
        (the fronts meet); then the area limits' rounds and the triangulation."""
        V = np.asarray(V, float)
        segs = np.asarray(segs, int).reshape(-1, 2)
        dseg = cm._seg_distance(V, segs)
        lim = lambda X: np.asarray(limit(X), float)
        A, B = V[segs[:, 0]], V[segs[:, 1]]
        mid = 0.5 * (A + B)
        lm = lim(mid)
        h = np.sqrt(2 * np.clip(np.where(lm > 0, lm, np.nan), 1e-300, None))
        h = np.where(np.isfinite(h), h, np.nanmax(h) if np.isfinite(h).any() else 1.0)
        L = np.linalg.norm(B - A, axis=1)
        T = (B - A) / np.maximum(L, 1e-300)[:, None]
        N = np.column_stack([-T[:, 1], T[:, 0]])
        span = (V.max(0) - V.min(0)).max()
        cand, prio, hc = [], [], []
        for side in (1.0, -1.0):
            for k in range(1, 400):
                t = k * np.sqrt(3) / 2 * h
                act = t < span
                if not act.any():
                    break
                for i in np.flatnonzero(act):
                    n = max(1, int(np.round(L[i] / h[i])))
                    s = (np.arange(n) + 0.5 * (k % 2)) / n
                    s = s[(s > 0) & (s < 1)] if k % 2 else s[s > 0]
                    P = A[i] + s[:, None] * (B[i] - A[i]) + side * t[i] * N[i]
                    cand.append(P); prio.append(np.full(len(P), -k)); hc.append(np.full(len(P), h[i]))
                if k > 3 and sum(map(len, cand)) > 3e6:
                    break
        P, pr, hh = np.vstack(cand), np.concatenate(prio), np.concatenate(hc)
        ok = inside(P)
        if frozen is not None:
            ok &= ~frozen(P)
        P, pr, hh = P[ok], pr[ok], hh[ok]
        d = dseg(P)
        lp = lim(P)
        hl = np.sqrt(2 * np.clip(lp, 1e-300, None))
        ok = (np.abs(d - (-pr) * np.sqrt(3) / 2 * hh) < 0.3 * hh) & (d > 0.45 * hl) & (lp > 0)
        P, pr, hl = P[ok], pr[ok], hl[ok]
        keep = cm._spaced(P, hl * (0.85 / cm.SEP), pr + 1e-3 * np.random.default_rng(0).random(len(P)))
        extra = P[keep]
        vt = cKDTree(V)
        extra = extra[vt.query(extra)[0] > 0.4 * np.sqrt(2 * np.clip(lim(extra), 1e-300, None))]
        n_seed = len(extra)
        extra, segs, rounds, over = cm._rounds(core, V, extra, segs, lim, inside, frozen, dseg, cm.TOL)
        v, t = cm.triangulate(core, np.vstack([V, extra]), segs, min_angle, max_steiner)
        return dict(vertices=v, triangles=t, info=dict(rounds=rounds, added=len(extra) - n_seed, seed=n_seed, over=over,
                                                       steiner=len(v) - len(V) - len(extra)))

    def minmax(V, segs, limit, inside, core, frozen=None, sizes=(1.0,), min_angle=0.0, max_steiner=0, smooth=cm.SMOOTH):
        T = orig(V, segs, limit, inside, core, frozen, sizes, min_angle, max_steiner, smooth=smooth)
        v, t = T['vertices'], T['triangles'].copy()
        seg_set = {tuple(sorted(s)) for s in np.asarray(segs).reshape(-1, 2)}
        T['triangles'] = flip_minmax(v, t, seg_set)
        return T

    def qhull(V, segs, limit, inside, core, frozen=None, sizes=(1.0,), min_angle=0.0, max_steiner=0, smooth=cm.SMOOTH):
        T = orig(V, segs, limit, inside, core, frozen, sizes, min_angle, max_steiner, smooth=smooth)
        v = T['vertices']
        t = Delaunay(v).simplices.astype(np.int32)
        t = t[inside(v[t].mean(1))]
        T['triangles'] = t
        E = {tuple(sorted(e)) for tri in t for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
        S = [tuple(sorted(s)) for s in np.asarray(segs).reshape(-1, 2)]
        T['info']['lost'] = (sum(s not in E for s in S), len(S))
        return T

    return dict(cdt=cdt, refine_only=refine_only, frontal=frontal, minmax=minmax, qhull=qhull)


def _maxang(P):
    e = [P[1] - P[0], P[2] - P[1], P[0] - P[2]]
    out = 0.0
    for i in range(3):
        a, b = -e[i - 1], e[i]
        c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        out = max(out, np.arccos(np.clip(c, -1, 1)))
    return out


def flip_minmax(v, t, seg_set, passes=20):
    """Edge flips that lower the larger of the two triangles' largest angles (the min-max-angle triangulation's local
    criterion); segments stay."""
    t = [list(x) for x in t]
    for _ in range(passes):
        emap = {}
        for i, tri in enumerate(t):
            for k in range(3):
                emap.setdefault(tuple(sorted((tri[(k + 1) % 3], tri[(k + 2) % 3]))), []).append((i, tri[k]))
        flips, used = 0, set()
        for e, side in emap.items():
            if len(side) != 2 or e in seg_set:
                continue
            (i, a), (j, d) = side
            if i in used or j in used:
                continue
            b, c = e
            P = v
            cr = lambda o, p, q: (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
            if not (cr(P[a], P[d], P[b]) * cr(P[a], P[d], P[c]) < 0):      # not convex
                continue
            old = max(_maxang(P[t[i]]), _maxang(P[t[j]]))
            n1, n2 = [a, b, d], [a, d, c]
            new = max(_maxang(P[n1]), _maxang(P[n2]))
            if new < old - 1e-9:
                if cr(P[n1[0]], P[n1[1]], P[n1[2]]) < 0: n1 = n1[::-1]
                if cr(P[n2[0]], P[n2[1]], P[n2[2]]) < 0: n2 = n2[::-1]
                t[i], t[j] = n1, n2
                used |= {i, j}
                flips += 1
        if not flips:
            break
    return np.array(t, np.int32)


def _rms(S, R):
    return float(np.sqrt(np.mean(np.abs(S - R) ** 2)))


def study_s(boards, out, F=(1e9, 3e9, 6e9), dmax=0.4e-3, dref=0.2e-3):
    from mom25d import cdtmesh
    from mom25d.api import build_board
    from mom25d.run import renorm
    V = variants(cdtmesh)
    here = os.path.join(os.path.dirname(__file__), '..', 'tests', 'boards')

    def run(board, d, quads):
        m, sim = build_board(os.path.join(here, board), F[0], F[-1], dmax=d, log=lambda *a: None, quads=quads)
        zr = np.array([m.port_z0[p.name] for p in sim.ports])
        S = np.array([renorm(sim.sparams(f), np.array([p.zc for p in sim.ports]), zr) for f in F])
        return S, sim
    os.makedirs(out, exist_ok=True)
    for board in boards:
        ref = os.path.join(out, 'ref_' + board.replace('/', '_') + '.npy')
        cdtmesh.mesh = V['cdt']
        if not os.path.exists(ref):
            np.save(ref, run(board, dref, False)[0])
        R = np.load(ref)
        for quads in (False, 'all'):
            for name in ('cdt', 'refine_only', 'frontal', 'minmax'):
                cdtmesh.mesh = V[name]
                t0 = time.time()
                S, sim = run(board, dmax, quads)
                qi = getattr(sim, 'quad_info', None)
                n = sim.Ne + sim.Nv
                print(f"{board:16s} quads={quads!s:5s} {name:12s} unknowns {n:5d} -> {qi['reduced'] if qi else n:5d}  "
                      f"rms|dS| {_rms(S, R):.4f}  max {np.abs(S - R).max():.4f}  {time.time() - t0:5.1f}s", flush=True)
        cdtmesh.mesh = V['cdt']


def study_qs(pcbqs):
    sys.path.insert(0, os.path.join(pcbqs, 'tests'))
    from shapely.geometry import box, Point
    from pcbqs._numerics import cdtmesh
    from pcbqs.pdn import dc
    from test_pdn import board
    from test_thermal import disc_problem, disc_exact, solve, quiet
    V = variants(cdtmesh)
    plate = {'F.Cu': [('P', box(0, 0, 10, 10))], 'B.Cu': [('P', box(0, 0, 10, 10))]}
    one = board([('F.Cu', 0.0, 0.035)], {'F.Cu': [('P', box(0, 0, 10, 10))]})
    pad = dc.Network(one, 'P', {'s': [('F.Cu', box(0, 0, 1, 10))], 'p': [('F.Cu', Point(5, 5).buffer(0.15, quad_segs=64))]},
                     dmax=0.0625, fine=['s', 'p'], log=lambda *a: None)
    spread = -pad.solve({'s': 0.0}, {'p': -1.0}).terminals['p']
    for name in ('cdt', 'refine_only', 'frontal', 'minmax'):
        cdtmesh.mesh = V[name]
        out = []
        for dmax in (0.5, 0.25):
            b = board([('B.Cu', 0.0, 0.035), ('F.Cu', 1.6, 0.035)], plate, vias=[(5, 5, 0.15, 0.0, 1.6, 'P')])
            nw = dc.Network(b, 'P', {'s': [('F.Cu', box(0, 0, 1, 10))], 'l': [('B.Cu', box(9, 0, 10, 10))]}, dmax=dmax,
                            plating=0.025, fine=['s', 'l'], log=lambda *a: None)
            via = -nw.solve({'s': 0.0}, {'l': -1.0}).terminals['l'] - 1 / dc._barrel_g(0.15, 1.6, 0.025, 5.8e7)
            out.append(f"via dmax {dmax}: {100 * (via / 2 / spread - 1):+.2f} %")
        for size in (2.0, 1.0):
            p = disc_problem()
            exact = disc_exact(p, 5.0, 1.0, 50.0)
            r = solve(p, 'mesh', size, log=quiet)
            rr = np.array([0.0, 2.5, 7.5, 10, 15, 20, 30])
            ang = np.arange(8) * np.pi / 4 + 0.1
            top = np.mean([r.sample('F.Cu', 60 + rr * np.cos(u), 60 + rr * np.sin(u)) for u in ang], axis=0)
            out.append(f"disc {size} mm: {np.abs(top - exact(rr)[0]).max():.3f} K")
        print(f"{name:12s} " + " | ".join(out), flush=True)
    cdtmesh.mesh = V['cdt']


def study_quality():
    import shapely
    from shapely.geometry import box, Point
    from mom25d import geo, cdtmesh
    V = variants(cdtmesh)
    rng = np.random.default_rng(3)
    plane = box(0, 0, 20e-3, 12e-3)
    for _ in range(12):
        plane = plane.difference(Point(rng.random() * 20e-3, rng.random() * 12e-3).buffer(0.5e-3, quad_segs=3))
    cases = {'trace': (box(0, 0, 20e-3, 0.4e-3).union(box(19.6e-3, 0, 20e-3, 8e-3)),
                       dict(lines=[((5e-3, 0), (5e-3, 0.4e-3))])),
             'plane+vias': (plane, dict(vias=[(x * 1e-3, 6e-3, 0.15e-3) for x in (3, 7, 11, 15)])),
             'plane coarsened': (plane, dict(coarsen=3.0))}
    for cname, (g, kw) in cases.items():
        for name, fn in V.items():
            info = {}

            def wrapped(*a, fn=fn, **k):
                T = fn(*a, **k)
                info.update(T['info'])
                return T
            cdtmesh.mesh = wrapped
            t0 = time.time()
            n, t, _ = geo.mesh_polygons(g, 0.3e-3, 0.4e-3, 4, **kw)
            P = n[t]
            e = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]], 1)
            c = np.stack([(-e[:, i - 1] * e[:, i]).sum(1) / (np.linalg.norm(e[:, i - 1], axis=1) *
                                                              np.linalg.norm(e[:, i], axis=1)) for i in range(3)], 1)
            a = np.degrees(np.arccos(np.clip(c, -1, 1)))
            lost = f"  segments lost {info['lost'][0]}/{info['lost'][1]}" if 'lost' in info else ''
            print(f"{cname:16s} {name:12s} triangles {len(t):6d}  min < 20: {np.mean(a.min(1) < 20):5.1%}  "
                  f"max > 120: {np.mean(a.max(1) > 120):5.1%}  {time.time() - t0:5.2f}s{lost}", flush=True)
    cdtmesh.mesh = V['cdt']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('study', choices=('s', 'qs', 'quality'))
    ap.add_argument('--boards', default='demo.json,launch.json,rf/coupler.json,thru.json')
    ap.add_argument('--out', default='fd_mesher_trial')
    ap.add_argument('--pcbqs', default=os.path.join(os.path.dirname(__file__), '..', '..', 'pcbqs'))
    a = ap.parse_args()
    if a.study == 's':
        study_s(a.boards.split(','), a.out)
    elif a.study == 'qs':
        study_qs(a.pcbqs)
    else:
        study_quality()


if __name__ == '__main__':
    main()
