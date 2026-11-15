"""
cdtmesh - the points of geo.mesh_polygons' triangulation: CDT (the compiled core's cdt_triangulate, core/cdt_mesh.hpp,
CDT vendored in core/CDT, MPL-2.0) triangulates; the points inside the copper are placed here. The same file in
mom25d (mom25d/cdtmesh.py) and pcbqs (pcbqs/_numerics/cdtmesh.py).

    T = mesh(V, segs, limit, inside, core, frozen=None, sizes=(1.0,), min_angle=0, max_steiner=0)

V (n, 2), segs (m, 2): geo's vertices and constraint segments, in its units (the layer's dmax). limit(C): the
largest area of a triangle whose centroid is at C (<= 0: no limit). inside(P): whether P is in the copper. core: the
package's compiled core. frozen(P): where no point is to be placed (the structured runs' cells; None: nowhere).

  seed     a hexagonal lattice at the local cell size (equilateral cells just under the limit): each region's own
           lattice where the cell size is that region's (sizes), and where it is graded (coarsening) the finest
           lattice thinned to the local size
  rounds   what the seed left over its limit: each round, the circumcentre of every such triangle (its centroid
           where the circumcentre falls outside the copper or near a segment), kept apart from each other and from
           the vertices (where neither has room: the triangle's longest edge split at its middle, a segment's halves
           segments), then the whole triangulated again (MAX_ROUNDS)
  smooth   ODT passes over the placed points (each to the area-weighted mean of its triangles' circumcentres; the
           outlines' points stay), then the limits again, to POST_TOL
  min_angle (degrees) at the end, no triangle with a smaller angle (CDT's Ruppert refinement, segments split where
           encroached, at most max_steiner points; pcbqs's DC and thermal meshes: 25)

Against Triangle, which this replaced (2026-11-15): on mom25d's Solver Matrix boards the same S within 0.005 on all
but a few, with 2-10 % fewer unknowns. Returns dict(vertices, triangles, info): V first, in its order (the triangles
outside the outline left out; geo also drops those in its holes), info dict(rounds, added, seed, over, steiner).
"""
import numpy as np
import shapely
from scipy.spatial import cKDTree

MAX_ROUNDS = 60
TOL = 1e-6           # a triangle over its limit by more than this (relative) is refined
SEP = 0.5            # a new point at least this many cells (h) from every other
EDGE_SEP = 0.4       # a circumcentre this close to a segment (cells): the triangle's centroid instead
SPLIT_EDGE = 0.25    # an edge's middle at least this many cells (h) from the vertices to split it there
SPLIT_SEG = 0.02     # a segment's (its middle may come close to copper across it: 0.06 h on rf_msvia's planes)
MIN_DIST = 1e-10     # cdt_triangulate: a point this close to a segment (dmax) is on it


def triangulate(core, V, segs, min_angle=0.0, max_steiner=0):
    """(vertices, triangles) of the constrained Delaunay triangulation of V with the segments segs, inside the
    outline (with min_angle refined to it)."""
    return core.cdt_triangulate(np.ascontiguousarray(V, np.float64),
                                np.ascontiguousarray(np.asarray(segs).reshape(-1, 2), np.int32), MIN_DIST,
                                float(min_angle), int(max_steiner))


def _seg_distance(V, segs):
    """A function: the distance (dmax) from points to the nearest segment."""
    if not len(segs):
        return lambda P: np.full(len(P), np.inf)
    tree = shapely.STRtree(shapely.linestrings(V[np.asarray(segs)]))

    def dist(P):
        if not len(P):
            return np.zeros(0)
        idx, d = tree.query_nearest(shapely.points(P), return_distance=True, all_matches=False)
        out = np.full(len(P), np.inf)
        out[idx[0]] = d
        return out
    return dist


def _spaced(P, h, prio):
    """The points of P (h: each one's cell size) to keep: a maximal set, the highest prio first, no two closer than
    SEP x the smaller h (Luby's rounds, vectorised)."""
    n = len(P)
    if n < 2:
        return np.ones(n, bool)
    pairs = cKDTree(P).query_pairs(SEP * float(h.max()), output_type='ndarray')
    if len(pairs):
        d = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
        pairs = pairs[d < SEP * np.minimum(h[pairs[:, 0]], h[pairs[:, 1]])]
    rank = np.empty(n)
    rank[np.lexsort((np.arange(n), prio))] = np.arange(n)            # distinct priorities
    alive, keep = np.ones(n, bool), np.zeros(n, bool)
    while alive.any():
        a, b = pairs[:, 0], pairs[:, 1]
        live = alive[a] & alive[b]
        nb = np.full(n, -1.0)
        np.maximum.at(nb, a[live], rank[b[live]])
        np.maximum.at(nb, b[live], rank[a[live]])
        win = alive & (rank > nb)
        keep |= win
        alive &= ~win
        lose = np.zeros(n, bool)
        lose[b[win[a]]] = True
        lose[a[win[b]]] = True
        alive &= ~lose
    return keep


def _circumcentres(P):
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    ab, ac = b - a, c - a
    d = 2 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    ab2, ac2 = (ab ** 2).sum(1), (ac ** 2).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ux = (ac[:, 1] * ab2 - ab[:, 1] * ac2) / d
        uy = (ab[:, 0] * ac2 - ac[:, 0] * ab2) / d
    return a + np.column_stack([ux, uy])


HEX = 0.93           # the hexagonal seed's side, of the side whose equilateral triangle is at the limit
HEX_NEAR = 0.15      # a point takes a region's own lattice where its cell size is within this of the region's
HEX_SEP = 0.9        # graded copper: seed points at least this x their side apart
MOVE = 0.3           # smoothing: a point moves at most this many cells (h) a pass
POST_TOL = 0.3       # after smoothing, a triangle over its limit by more than this is refined again
SMOOTH = 5           # smoothing passes


def _hex(lo, hi, s):
    """A hexagonal lattice of side s over the box lo..hi."""
    dy = s * np.sqrt(3) / 2
    nx, ny = int(np.ceil((hi[0] - lo[0]) / s)) + 2, int(np.ceil((hi[1] - lo[1]) / dy)) + 2
    if nx * ny > 2e7:
        raise RuntimeError(f"cdtmesh: a lattice of {nx * ny:,} points (side {s:.3g} dmax)")
    X, Y = np.meshgrid(np.arange(nx) * s, np.arange(ny) * dy)
    X = X + (np.arange(ny) % 2)[:, None] * (s / 2)
    c = 0.5 * (lo + hi) - 0.5 * np.array([(nx - 1) * s, (ny - 1) * dy])
    return np.column_stack([X.ravel(), Y.ravel()]) + c


def _hex_seed(V, limit, inside, frozen, dseg, sizes):
    """Seed points of equilateral cells: a hexagonal lattice of each region's own size where the cell size
    is that (sizes), and elsewhere (graded: coarsen's field) the finest lattice thinned to the local size."""
    lo, hi = V.min(0), V.max(0)
    side = lambda lim: HEX * np.sqrt(4 * np.clip(lim, 1e-300, None) / np.sqrt(3))
    vt = cKDTree(V)
    out = []

    def keep(P, sd):
        P = P[inside(P)]
        if frozen is not None and len(P):
            P = P[~frozen(P)]
        return P

    def clear(P, sd):
        ok = (dseg(P) >= 0.5 * sd) & (vt.query(P)[0] >= 0.6 * sd)
        if out:
            ok &= cKDTree(np.vstack(out)).query(P)[0] >= 0.8 * sd
        return ok
    probe = np.vstack([V, lo + (hi - lo) * np.random.default_rng(0).random((4096, 2))])
    probe = probe[inside(probe)]
    lp = np.asarray(limit(probe), float)
    lp = lp[lp > 0]
    if not len(lp):
        return np.zeros((0, 2))
    hmin = np.sqrt(2 * lp.min())
    near = lambda h, a: np.abs(h / a - 1) <= HEX_NEAR
    for a in sorted(set(sizes), reverse=True):                        # each region's own lattice
        sd = float(side(0.5 * a * a))
        P = keep(_hex(lo, hi, sd), sd)
        if not len(P):
            continue
        l_ = np.asarray(limit(P), float)
        on = l_ > 0
        on[on] = near(np.sqrt(2 * l_[on]), a)
        P = P[on]
        if len(P):
            P = P[clear(P, sd)]
        if len(P):
            out.append(P)
    sd0 = float(side(0.5 * hmin * hmin))                              # the graded rest: thinned
    P = keep(_hex(lo, hi, sd0), sd0)
    if len(P):
        l_ = np.asarray(limit(P), float)
        on = l_ > 0
        h = np.sqrt(2 * np.clip(l_, 1e-300, None))
        for a in sizes:
            on &= ~near(h, a)
        P, l_ = P[on], l_[on]
    if len(P):
        sd = side(l_)
        ok = clear(P, sd)
        P, sd = P[ok], sd[ok]
    if len(P):
        # _spaced keeps points SEP x h apart: h here the side scaled so that the distance is HEX_SEP sides
        k = _spaced(P, sd * (HEX_SEP / SEP), -sd)                     # the finest first
        out.append(P[k])
    return np.vstack(out) if out else np.zeros((0, 2))


def _smooth(core, V, extra, segs, limit, inside, frozen, dseg, passes):
    """ODT smoothing of the points placed inside the copper: each to the area-weighted mean of its triangles'
    circumcentres (the triangle's centroid where that lies outside), at most MOVE cells a pass, kept clear of the
    segments; the outlines' points and those on segments stay."""
    nV = len(V)
    movable = dseg(extra) > 1e-9 if len(extra) else np.zeros(0, bool)
    for _ in range(passes):
        if not movable.any():
            break
        v, t = triangulate(core, np.vstack([V, extra]), segs)
        P = v[t]
        C = P.mean(1)
        ins = inside(C)
        t, P, C = t[ins], P[ins], C[ins]
        A = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                         - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        cc = _circumcentres(P)
        bad = ~np.isfinite(cc).all(1)
        cc[bad] = C[bad]
        far = np.linalg.norm(cc - C, axis=1) > 2 * np.sqrt(A)          # an obtuse triangle's: far outside it
        cc[far] = C[far]
        out = ~inside(cc)
        cc[out] = C[out]
        n = nV + len(extra)
        num, den = np.zeros((n, 2)), np.zeros(n)
        for k in range(3):
            np.add.at(num, t[:, k], A[:, None] * cc)
            np.add.at(den, t[:, k], A)
        cur = extra
        new = np.where(den[nV:, None] > 0, num[nV:] / np.maximum(den[nV:, None], 1e-300), cur)
        h = np.sqrt(2 * np.clip(limit(cur), 1e-300, None))
        d = new - cur
        L = np.linalg.norm(d, axis=1)
        new = cur + d * np.minimum(1.0, MOVE * h / np.maximum(L, 1e-300))[:, None]
        ok = movable & inside(new) & (dseg(new) >= 0.35 * h)
        if frozen is not None:
            ok &= ~frozen(new)
        extra = np.where(ok[:, None], new, cur)
    return extra


def mesh(V, segs, limit, inside, core, frozen=None, sizes=(1.0,), min_angle=0.0, max_steiner=0, smooth=SMOOTH):
    V = np.asarray(V, float)
    segs = np.asarray(segs, int).reshape(-1, 2)
    dseg = _seg_distance(V, segs)
    limit0 = limit
    limit = lambda X: np.asarray(limit0(X), float)
    extra = _hex_seed(V, limit, inside, frozen, dseg, sizes)
    n_seed = len(extra)
    extra, segs, rounds, over = _rounds(core, V, extra, segs, limit, inside, frozen, dseg, TOL)
    if smooth:
        extra = _smooth(core, V, extra, segs, limit, inside, frozen, dseg, smooth)
        extra, segs, r2, over = _rounds(core, V, extra, segs, limit, inside, frozen, dseg, POST_TOL)
        rounds += r2
    n_in = len(V) + len(extra)
    v, t = triangulate(core, np.vstack([V, extra]), segs, min_angle, max_steiner)
    return dict(vertices=v, triangles=t,
                info=dict(rounds=rounds, added=len(extra) - n_seed, seed=n_seed, over=over, steiner=len(v) - n_in))


def _rounds(core, V, extra, segs, limit, inside, frozen, dseg, tol):
    """The rounds of the module docstring: (extra, segs, rounds, triangles left over their limit)."""
    v, t = triangulate(core, np.vstack([V, extra]), segs)
    over = rounds = 0
    for rounds in range(1, MAX_ROUNDS + 1):
        P = v[t]
        C = P.mean(1)
        A = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                   - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        ins = inside(C)
        lim = np.full(len(C), -1.0)
        lim[ins] = limit(C[ins])
        bad = ins & (lim > 0) & (A > lim * (1 + tol))
        over = int(bad.sum())
        if not over:
            break
        Pb, Cb, h = P[bad], C[bad], np.sqrt(2 * lim[bad])
        cc = _circumcentres(Pb)
        fine = np.isfinite(cc).all(1)
        cc[~fine] = Cb[~fine]
        ok = inside(cc) & (dseg(cc) >= EDGE_SEP * h)
        if frozen is not None:
            ok &= ~frozen(cc)
        cand = np.where(ok[:, None], cc, Cb)
        vtree = cKDTree(v)
        room = vtree.query(cand)[0] >= SEP * h
        tb = t[bad][~room]
        n_in = len(V) + len(extra)
        split = np.zeros((0, 2))
        if len(tb):
            ends = np.stack([tb, np.roll(tb, -1, 1)], 2)                   # (k, 3, 2): the three edges
            L = np.linalg.norm(v[ends[:, :, 0]] - v[ends[:, :, 1]], axis=2)
            e = ends[np.arange(len(tb)), L.argmax(1)]
            e = e[(e < n_in).all(1)]
            e = np.unique(np.sort(e, 1), axis=0)
            mid = 0.5 * (v[e[:, 0]] + v[e[:, 1]])
            hm = np.sqrt(2 * np.clip(limit(mid), 1e-300, None))
            skey = np.sort(segs, 1)
            kk = skey[:, 0].astype(np.int64) * n_in + skey[:, 1]
            is_seg = np.isin(e[:, 0].astype(np.int64) * n_in + e[:, 1], kk)
            fits = vtree.query(mid)[0] >= np.where(is_seg, SPLIT_SEG, SPLIT_EDGE) * hm
            e, mid, is_seg = e[fits], mid[fits], is_seg[fits]
            if len(e):
                ek = e[:, 0].astype(np.int64) * n_in + e[:, 1]
                ids = n_in + np.arange(len(e))
                if is_seg.any():
                    drop = np.isin(kk, ek[is_seg])
                    segs = np.vstack([segs[~drop], np.column_stack([e[is_seg, 0], ids[is_seg]]),
                                      np.column_stack([ids[is_seg], e[is_seg, 1]])])
                split = mid
        cand, h, prio = cand[room], h[room], A[bad][room]
        if not len(cand) and not len(split):
            break
        keep = _spaced(cand, h, prio) if len(cand) else np.zeros(0, bool)
        extra = np.vstack([extra, split, cand[keep]])
        v, t = triangulate(core, np.vstack([V, extra]), segs)
    return extra, segs, rounds, over
