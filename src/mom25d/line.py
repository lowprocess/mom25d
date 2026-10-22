"""
mom25d.line - line basis functions: the current on straight runs of trace as a few functions per column.

Along a straight run of a trace the current flows along the line, and across it has the shape the edges give it:
peaked at both edges, roughly as 1 / sqrt(distance to the edge). The triangle mesh resolves that shape with its
edge rings - several unknowns across the line at every step along it, plus the transverse ones in between.
Here the unknowns inside a run are replaced by, per column along it (a tent of length h), a few functions with a
fixed shape across the line:

    J(x, y) = t S_i(x) p(y)        p: 'maxwell'  1 / (pi sqrt(a^2 - y^2))   (the isolated strip, even)
                                      'uniform'  1 / (2 a)                  (with maxwell: the even shape free-er)
                                      'odd'      y / (a pi sqrt(a^2 - y^2)) (coupled lines: current pushed aside)

expressed in the RWG basis by interpolation (each RWG coefficient = the mean normal current through its edge),
so they are conforming RWG combinations: the matrix is filled on the triangles as before and projected
(quads.project), ports, cuts and dumps unchanged. The edges at the run's ends and everything outside the runs
stay unknowns of their own; the runs stop short of bends, tees and pads (by `margin`).

    runs = find_runs(geoms, h_of_layer, signal=..., w_max=..)
    T, info = line_subspace(rwg, n_total, tri_layer, runs, profiles=('maxwell', 'odd'), exclude_tris=..,
                            exclude_edges=..)
"""
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from scipy import sparse

PROFILES = {                       # weight w(theta) with  p(y) dy = w(theta) dtheta,  y = a sin(theta)
    'maxwell': lambda th: np.full_like(th, 1 / np.pi),
    'uniform': lambda th: 0.5 * np.cos(th),
    'odd': lambda th: np.sin(th) / np.pi,
}
_GX, _GW = np.polynomial.legendre.leggauss(8)


class Run:
    """A straight run of trace on layer k: centre line from c0 along unit t, half-width a, from x0 to x1 (m)."""
    def __init__(self, k, c0, t, a, x0, x1):
        self.k, self.c0, self.t, self.a, self.x0, self.x1 = k, np.asarray(c0, float), np.asarray(t, float), a, x0, x1
        self.n = np.array([-self.t[1], self.t[0]])

    def polygon(self, pad=0.0):
        P = [self.c0 + self.t * x + self.n * y for x, y in ((self.x0, -self.a - pad), (self.x1, -self.a - pad),
                                                          (self.x1, self.a + pad), (self.x0, self.a + pad))]
        return Polygon(P)

    def local(self, P):
        d = np.atleast_2d(P) - self.c0
        return d @ self.t, d @ self.n


def find_runs(geoms, h_of, signal=None, w_max=2.0e-3, min_len=3.0, margin=1.0, tol=1e-9):
    """Straight runs of trace: pairs of antiparallel outline segments facing each other across copper, at most
    w_max apart and overlapping by at least min_len x max(width, h); their overlap, less `margin` x max(width, h)
    at each end. geoms: {layer: (Multi)Polygon} (m); h_of: {layer: column length}; signal: only runs on it."""
    runs = []
    for k, g in geoms.items():
        h = h_of[k]
        for p in getattr(g, 'geoms', [g]):
            if p.is_empty:
                continue
            p = orient(p, 1.0)                                     # exterior ccw, holes cw: copper on the left
            segs = []
            for ring in [p.exterior] + list(p.interiors):
                c = np.asarray(ring.coords)[:, :2]
                for a_, b_ in zip(c[:-1], c[1:]):
                    L = np.linalg.norm(b_ - a_)
                    if L > tol:
                        d = (b_ - a_) / L
                        segs.append((a_, b_, d, np.array([-d[1], d[0]]), L))
            cand = []
            for i, (a1, b1, d1, n1, L1) in enumerate(segs):
                for j in range(i + 1, len(segs)):
                    a2, b2, d2, n2, L2 = segs[j]
                    if d1 @ d2 > -0.9999:
                        continue
                    w = (a2 - a1) @ n1
                    if not (tol < w <= w_max):
                        continue
                    s0, s1 = sorted(((b2 - a1) @ d1, (a2 - a1) @ d1))
                    lo, hi = max(0.0, s0), min(L1, s1)
                    e = max(w, h)
                    if hi - lo < min_len * e + 2 * margin * e:
                        continue
                    cand.append((hi - lo, k, a1 + n1 * w / 2, d1, w / 2, lo + margin * e, hi - margin * e))
            cand.sort(key=lambda c: -c[0])
            taken = []
            for _, k_, c0, t, a, x0, x1 in cand:
                r = Run(k_, c0, t, a, x0, x1)
                poly = r.polygon()
                if not p.buffer(tol).contains(poly.buffer(-tol)):
                    continue
                if signal is not None and not signal.buffer(tol).contains(poly.buffer(-tol)):
                    continue
                if any(poly.intersection(q).area > 1e-3 * poly.area for q in taken):
                    continue
                taken.append(poly)
                runs.append(r)
    return runs


def _edge_integral(run, P0, P1, lo, mid, hi, prof):
    """(1/pi-normalised) integral over the edge P0-P1 of S(x) p(y) dy for the tent S on (lo, mid, hi), in theta."""
    (xa, xb), (ya, yb) = run.local(np.array([P0, P1]))
    if abs(yb - ya) < 1e-12 * max(run.a, 1e-9):
        return 0.0
    if ya > yb:
        xa, xb, ya, yb = xb, xa, yb, ya
    a = run.a
    ys = [ya, yb]
    for xn in (lo, mid, hi):                                       # split where x crosses a kink of the tent
        if xn is not None and min(xa, xb) < xn < max(xa, xb) and xb != xa:
            ys.append(ya + (yb - ya) * (xn - xa) / (xb - xa))
    ys = np.clip(np.sort(ys), -a, a)
    tot = 0.0
    for y0, y1 in zip(ys[:-1], ys[1:]):
        t0, t1 = np.arcsin(y0 / a), np.arcsin(y1 / a)
        if t1 - t0 <= 0:
            continue
        th = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * _GX
        y = a * np.sin(th)
        x = xa + (xb - xa) * (y - ya) / (yb - ya)
        up = np.ones_like(x) if lo is None else (x - lo) / (mid - lo)       # an end tent stays 1 past the end
        down = np.ones_like(x) if hi is None else (hi - x) / (hi - mid)
        S = np.clip(np.minimum(up, down), 0, 1)
        tot += 0.5 * (t1 - t0) * np.sum(_GW * S * PROFILES[prof](th))
    return tot


def line_subspace(rwg, n_total, tri_layer, runs, h_of, profiles=('maxwell', 'odd'), exclude_tris=(),
                  exclude_edges=()):
    """T (n_total x n_reduced, CSC) and info (keep, elim, counts): the RWG unknowns inside the runs replaced by
    len(profiles) functions per column (quads.project's layout: the kept unknowns first, then the line functions,
    which live on the eliminated edges only)."""
    Ne = rwg.Ne
    bad_t = np.zeros(len(rwg.tris), bool)
    bad_t[np.asarray(list(exclude_tris), int)] = True
    bad_e = np.zeros(Ne, bool)
    bad_e[np.asarray(list(exclude_edges), int)] = True
    for e in np.nonzero(bad_e)[0]:                                 # port gaps, cuts, lumped edges stay unknowns
        bad_t[[rwg.tp[e], rwg.tm[e]]] = True
    owner = -np.ones(len(rwg.tris), int)
    cent = rwg.cent
    for i, r in enumerate(runs):
        sel = (tri_layer == r.k) & (owner < 0) & ~bad_t
        x, y = r.local(cent[sel])
        inside = (x > r.x0) & (x < r.x1) & (np.abs(y) < r.a)
        owner[np.nonzero(sel)[0][inside]] = i
    o1, o2 = owner[rwg.tp], owner[rwg.tm]
    interior = (o1 >= 0) & (o1 == o2) & ~bad_e
    elim = np.nonzero(interior)[0]
    erow = -np.ones(Ne, int)
    erow[elim] = np.arange(len(elim))
    keep = np.setdiff1d(np.arange(n_total), elim)
    rows, cols, vals = [], [], []
    ncol = len(keep)
    nfun = 0
    for i, r in enumerate(runs):
        es = elim[o1[elim] == i]
        if not len(es):
            continue
        M = max(2, int(np.ceil((r.x1 - r.x0) / h_of[r.k])))
        xs = np.linspace(r.x0, r.x1, M + 1)
        ex, _ = r.local(rwg.mid[es])
        tn = rwg.nrm[es] @ r.t                                      # t . n of each edge (RWG direction)
        hstep = xs[1] - xs[0]
        for j in range(0, M + 1):                                   # the end tents carry the current to the ends
            lo = xs[j - 1] if j > 0 else None
            hi = xs[j + 1] if j < M else None
            mid = xs[j]
            near = np.nonzero(((ex > mid - 1.5 * hstep) | (lo is None)) & ((ex < mid + 1.5 * hstep) | (hi is None)))[0]
            for prof in profiles:
                col = []
                for q in near:
                    e = es[q]
                    P0, P1 = rwg.ends[e]
                    I = _edge_integral(r, P0, P1, lo, mid, hi, prof)
                    if I == 0.0:
                        continue
                    _, (ya, yb) = r.local(np.array([P0, P1]))
                    c = tn[q] * I / abs(yb - ya)                     # mean normal current through the edge
                    if c != 0.0:
                        col.append((erow[e], c))
                if col:
                    for rr, c in col:
                        rows.append(rr); cols.append(ncol); vals.append(c)
                    ncol += 1
                    nfun += 1
    rows = np.asarray(rows, int)
    T = sparse.csc_matrix((np.concatenate([np.ones(len(keep)), vals]),
                           (np.concatenate([keep, elim[rows]]), np.concatenate([np.arange(len(keep)), cols]))),
                          shape=(n_total, ncol))
    info = dict(runs=len(runs), functions=nfun, elim=elim, keep=keep, reduced=ncol, full=n_total,
                edges_replaced=len(elim))
    return T, info
