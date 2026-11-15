"""
cdtmesh - the points of geo.mesh_polygons' triangulation: CDT (the compiled core's cdt_triangulate, core/cdt_mesh.hpp,
CDT vendored in core/CDT, MPL-2.0) triangulates; the points inside the copper are placed here. The same file in
mom25d (mom25d/cdtmesh.py) and pcbqs (pcbqs/_numerics/cdtmesh.py).

    T = mesh(V, segs, limit, inside, core, lattice=False, frozen=None, sizes=(1.0,), min_angle=0, max_steiner=0)

V (n, 2), segs (m, 2): geo's vertices and constraint segments, in its units (the layer's dmax). limit(C): the
largest area of a triangle whose centroid is at C (<= 0: no limit). inside(P): whether P is in the copper. core: the
package's compiled core. frozen(P): where no point is to be placed (the structured runs' cells; None: nowhere).

  rounds   each round, the circumcentre of every triangle over its limit (its centroid where the circumcentre falls
           outside the copper or near a segment), kept apart from each other and from the vertices (where neither
           has room: the triangle's longest edge split at its middle, a segment's halves segments), then the whole
           triangulated again; until no triangle is over (MAX_ROUNDS). Unlike Triangle's refinement, which leaves
           its triangles at about half their limit (and misses the limits of regions its flood fill does not reach),
           the triangles end near their limits: mom25d's cell sizes were set on Triangle's meshes (sim.py: planes)
  lattice  (experimental) first square lattices at the local cell size h = sqrt(2 limit) (each square two triangles
           just under the limit), on a ladder of sizes 2^k of each region's own (sizes), half a cell clear of the
           segments; then the rounds for what is left. Its squares pair into quad cells (mom25d's quads.py)
  min_angle (degrees) at the end, no triangle with a smaller angle (CDT's Ruppert refinement, segments split where
           encroached, at most max_steiner points; pcbqs's DC and thermal meshes: 25)

Returns dict(vertices, triangles, info): V first, in its order (the triangles outside the outline left out; geo
also drops those in its holes), info dict(rounds, added, lattice, over, steiner).
"""
import numpy as np
import shapely
from scipy.spatial import cKDTree

MAX_ROUNDS = 60
TOL = 1e-6           # a triangle over its limit by more than this (relative) is refined
SEP = 0.5            # a new point at least this many cells (h) from every other
EDGE_SEP = 0.4       # a circumcentre this close to a segment (cells): the triangle's centroid instead
LATTICE = 0.98       # the lattice's spacing, of the ladder's size (just under the limit)
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


def _ladder(h, sizes):
    """Each h's lattice spacing: the largest size x 2^k (k integer, any of sizes) not above it."""
    h = np.maximum(np.asarray(h, float), 1e-300)
    return np.max([a * 2.0 ** np.floor(np.log2(h / a) + 1e-9) for a in sizes], axis=0)


def _lattice(V, limit, inside, frozen, dseg, sizes=(1.0,)):
    """Points of square lattices at the local cell size h = sqrt(2 limit): spacing LATTICE x _ladder(h, sizes) (geo's
    units: the layer's dmax; sizes: the regions' own cell sizes, so each gets its own exactly), half a cell clear of
    the segments and the vertices, and of the coarser lattices' points."""
    lo, hi = V.min(0), V.max(0)
    probe = np.vstack([V, lo + (hi - lo) * np.random.default_rng(0).random((4096, 2))])
    probe = probe[inside(probe)]
    lim = limit(probe)
    lim = lim[lim > 0]
    if not len(lim):
        return np.zeros((0, 2))
    hs = np.sqrt(2 * lim)
    steps = sorted({float(a * 2.0 ** k) for a in sizes
                    for k in range(int(np.floor(np.log2(hs.min() / a))) - 1, int(np.floor(np.log2(hs.max() / a))) + 1)
                    if hs.min() / 2 <= a * 2.0 ** k <= hs.max() * (1 + 1e-9)}, reverse=True)
    vt = cKDTree(V)
    out = []
    for step in steps:
        s = LATTICE * step
        nx, ny = int(np.ceil((hi[0] - lo[0]) / s)) + 1, int(np.ceil((hi[1] - lo[1]) / s)) + 1
        if nx * ny > 2e7:
            raise RuntimeError(f"cdtmesh: a lattice of {nx * ny:,} points (spacing {s:.3g} dmax)")
        X, Y = np.meshgrid(lo[0] + s * (np.arange(nx) + 0.5), lo[1] + s * (np.arange(ny) + 0.5))
        P = np.column_stack([X.ravel(), Y.ravel()])
        P = P[inside(P)]
        if frozen is not None and len(P):
            P = P[~frozen(P)]
        if not len(P):
            continue
        lp = limit(P)
        on = lp > 0
        on[on] = np.abs(_ladder(np.sqrt(2 * lp[on]), sizes) - step) <= 1e-9 * step
        P = P[on]
        if not len(P):
            continue
        P = P[(dseg(P) >= 0.5 * s) & (vt.query(P)[0] >= 0.5 * s)]
        if out and len(P):                                   # clear of the coarser lattices' points
            P = P[cKDTree(np.vstack(out)).query(P)[0] >= 0.5 * s]
        if len(P):
            out.append(P)
    return np.vstack(out) if out else np.zeros((0, 2))


def _circumcentres(P):
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    ab, ac = b - a, c - a
    d = 2 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    ab2, ac2 = (ab ** 2).sum(1), (ac ** 2).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ux = (ac[:, 1] * ab2 - ab[:, 1] * ac2) / d
        uy = (ab[:, 0] * ac2 - ac[:, 0] * ab2) / d
    return a + np.column_stack([ux, uy])


def mesh(V, segs, limit, inside, core, lattice=False, frozen=None, sizes=(1.0,), min_angle=0.0, max_steiner=0):
    V = np.asarray(V, float)
    segs = np.asarray(segs, int).reshape(-1, 2)
    dseg = _seg_distance(V, segs)
    extra = _lattice(V, limit, inside, frozen, dseg, sizes) if lattice else np.zeros((0, 2))
    n_lat, rounds = len(extra), 0
    v, t = triangulate(core, np.vstack([V, extra]), segs)
    over = 0
    for rounds in range(1, MAX_ROUNDS + 1):
        P = v[t]
        C = P.mean(1)
        A = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                   - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        ins = inside(C)
        lim = np.full(len(C), -1.0)
        lim[ins] = limit(C[ins])
        bad = ins & (lim > 0) & (A > lim * (1 + TOL))
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
        # no room (a long thin triangle, its circumcentre and centroid near a vertex): its longest edge split at its
        # middle instead, as Triangle splits an encroached segment (a segment's halves become segments)
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
    steiner = 0
    if min_angle:
        n_in = len(V) + len(extra)
        v, t = triangulate(core, np.vstack([V, extra]), segs, min_angle, max_steiner)
        steiner = len(v) - n_in
    return dict(vertices=v, triangles=t,
                info=dict(rounds=rounds, added=len(extra) - n_lat, lattice=n_lat, over=over, steiner=steiner))
