"""
mom25d.geo - geometry helpers for the multilayer solver

  * LinePort      : wave port on a straight feed in any in-plane direction (KiCad traces at 45 deg etc.)
  * mesh_polygons : constrained-Delaunay mesh of arbitrary polygons (holes allowed) with
                    - graded edge refinement (inward offset rings, like the tensor mesher's cosine grading)
                    - port gap/cut lines inserted as constraint segments (so RWG edges lie on them)
                    - via points inserted as vertices, with a small ring of radius = via radius around
                      them so the via 'fan' (triangles sharing the via vertex) matches the via footprint
  * to_geometry   : rectangles / shapely geometry -> one cleaned shapely (Multi)Polygon
"""
import numpy as np
import shapely
from shapely.geometry import (Polygon, LineString, Point, box)
from shapely.ops import unary_union
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .base import graded


# ---- Ports in arbitrary directions ------------------------------------------
class LinePort:
    """
    Wave port on a straight feed line.
      layer     : metal interface index
      center    : (x, y) of the gap (delta-gap source) on the feed centre line
      direction : vector pointing INTO the device along the feed (any angle)
      width     : feed width (the gap and sample cuts span this, centred on `center`)
      ref_dist  : distance from the gap to the reference plane, along `direction`
      s0, step  : closest sample cut to the gap, and spacing between cuts
      s1        : farthest sample cut (default = ref_dist). Keep the cuts clear of discontinuities:
                  the reference plane may lie beyond s1 (the fitted waves are shifted to it).
      zc        : reference impedance
    """
    kind = 'wave'

    def __init__(self, layer, center, direction, width, ref_dist, s0=None, step=None, zc=50.0, name='',
                 s1=None):
        self.s1 = s1
        self.layer = layer
        d = np.asarray(direction, float)
        self.c, self.d = np.asarray(center, float), d / np.linalg.norm(d)
        self.t = np.array([-self.d[1], self.d[0]])
        self.w, self.ref, self.s0, self.step, self.zc, self.name = width, ref_dist, s0, step, zc, name

    def line(self, s):
        """End points of the transverse line at distance s from the gap (s=0: the gap)."""
        p = self.c + self.d * s
        return p - self.t * self.w / 2, p + self.t * self.w / 2

    @property
    def axis_aligned(self):
        return abs(self.d[0]) < 1e-12 or abs(self.d[1]) < 1e-12


# ---- Geometry ---------------------------------------------------------------
def _is_rect(s):
    return isinstance(s, (tuple, list)) and len(s) == 4 and np.isscalar(s[0])


def to_geometry(shapes):
    """List of rectangles (x0,y0,x1,y1) and/or shapely geometries -> cleaned (Multi)Polygon."""
    return unary_union([box(*s) if _is_rect(s) else s for s in shapes]).buffer(0)


def is_manhattan_rects(shapes):
    return all(_is_rect(s) for s in shapes)


def ngon(x, y, R, n, closed=False):
    """Vertices (n, 2) of the regular n-gon of circumradius R about (x, y), a flat side at the bottom (closed:
    the first vertex repeated at the end)."""
    a = 2 * np.pi * np.arange(n) / n + np.pi / n
    P = np.c_[x + R * np.cos(a), y + R * np.sin(a)]
    return np.vstack([P, P[:1]]) if closed else P


def _features(geom, lines, vias, dmax, turn_deg=25.0):
    """The points the coarsening grows from: vertices of the copper outline where it turns by more than turn_deg
    within 2 dmax (corners, and bends drawn as arcs), the ends and middles of the port / cut lines, via centres."""
    pts = []
    for ring in _rings(geom):
        P = np.asarray(ring.coords)[:-1, :2]
        n = len(P)
        if n < 3:
            continue
        d0, d1 = P - np.roll(P, 1, 0), np.roll(P, -1, 0) - P
        turn = np.abs(np.arctan2(d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0], (d0 * d1).sum(1)))
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(d1, axis=1))])
        L, s = s[-1], s[:-1]
        s3, t3 = np.concatenate([s - L, s, s + L]), np.concatenate([turn] * 3)    # the ring is closed
        c = np.concatenate([[0], np.cumsum(t3)])
        lo, hi = np.searchsorted(s3, s - 2 * dmax), np.searchsorted(s3, s + 2 * dmax, side='right')
        pts += list(P[c[hi] - c[lo] > np.radians(turn_deg)])
    for p0, p1 in lines:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        pts += [p0, p1, (p0 + p1) / 2]
    pts += [np.array(v[:2], float) for v in vias]
    return np.array(pts).reshape(-1, 2)


def _spaced(P, h, dmin):
    """Points along the polyline P (its vertices kept) at the local spacing h(points) gives (at least dmin)."""
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        if L == 0:
            continue
        t = np.linspace(0, 1, int(np.ceil(4 * L / dmin)) + 1)
        hv = h(a + t[:, None] * (b - a))
        n = np.concatenate([[0], np.cumsum(np.diff(t) * L * 2 / (hv[1:] + hv[:-1]))])    # cells so far
        k = max(1, int(np.ceil(n[-1] - 1e-6)))
        u = np.interp(np.arange(1, k + 1) * n[-1] / k, n, t)
        out.append(a + u[:, None] * (b - a))
    return np.vstack(out)


def _run_grid(geom, dmax, lines, vias, offsets, w_max, h=None, fill=0.98, within=None):
    """The tensor grid of mesh_polygons(structured=): (transverse and longitudinal constraint segments, the runs'
    polygons, the indices of the vias snapped to a cell of it). h(points): the local spacing (None: dmax), the
    stations at fill x it so that no outline or ring piece between two is split again. within: only the runs
    inside this area (the zone's fine region)."""
    from .line import find_runs
    h = h or (lambda P: np.full(len(P), dmax))
    segs, polys, snapped, on_run = [], [], set(), set()
    for r in find_runs({0: geom}, {0: dmax}, w_max=w_max, min_len=1.0, margin=0.0):
        a, cl = r.a, lambda x: r.c0 + r.t * np.asarray(x, float)[..., None]
        if within is not None and not within.contains(
                Polygon([cl(r.x0) - r.n * a, cl(r.x1) - r.n * a, cl(r.x1) + r.n * a, cl(r.x0) + r.n * a])):
            continue
        # fixed stations: the run's ends and the port / cut lines across it; gaps around vias
        fixed, cuts = [r.x0, r.x1], []
        for p0, p1 in lines:
            (xa, xb), (ya, yb) = r.local(np.array([p0, p1]))
            if abs(xa - xb) < 1e-3 * a and r.x0 < xa < r.x1 and min(ya, yb) <= 0 <= max(ya, yb):
                fixed.append(xa)
        # a square via (4 sides, its barrel square: sim.Simulation) on an axis-aligned run, clear of the edge rings'
        # lines, joins the grid: its sides become two stations and two longitudinal lines. One that may snap (a 5th
        # entry, True: stitching vias) takes a whole grid cell instead, centred on it, its own square dropped - no
        # thin column and row of cells across the run for each via (rf_gcpw, --mom-quads on: 3,391 unknowns with its
        # true square). Any other via: a gap in the grid
        rings = np.concatenate([[a], a - offsets]) if len(offsets) else np.array([a])
        clear = lambda yv, hs: all(abs(abs(y) - rings).min() > 0.15 * dmax for y in (yv - hs, yv + hs)) and \
            abs(yv) + hs < a
        vy, cells = [], []
        for i, v in enumerate(vias):
            xv, yv = r.local(np.array(v[:2], float))
            xv, yv = float(xv[0]), float(yv[0])
            rad = v[2] if len(v) > 2 else 0.0
            if not (abs(yv) < a + rad + 0.5 * dmax and r.x0 - rad - 0.5 * dmax < xv < r.x1 + rad + 0.5 * dmax):
                continue                                   # not on the run (a via beside it, on another trace)
            on_run.add(i)
            square, aligned = len(v) > 3 and v[3] == 4 and v[2] > 0, min(abs(r.t)) < 1e-9
            if square and aligned and len(v) > 4 and v[4]:
                hc = 0.5 * fill * float(h(np.array(v[:2], float)[None])[0])        # half a cell there
                sides = _cell_sides(yv, hc, rings)
                if sides is not None and r.x0 < xv - hc and xv + hc < r.x1 and \
                        not any(abs(xv - x) < 2 * hc and abs(yv - y) < 2 * hc for x, y in cells):
                    fixed += [xv - hc, xv + hc]
                    vy += sides
                    cells.append((xv, yv))
                    snapped.add(i)
                    continue
            hs = v[2] / np.sqrt(2) if square else 0.0                                  # the square's half side
            if hs and aligned and r.x0 < xv - hs and xv + hs < r.x1 and clear(yv, hs):
                fixed += [xv - hs, xv + hs]
                vy += [yv - hs, yv + hs]
                continue
            cuts.append((xv - rad - 2 * dmax, xv + rad + 2 * dmax))
        vy = np.unique(np.round(vy, 12))
        fixed = np.unique(np.clip(fixed + [x for c in cuts for x in c], r.x0, r.x1))
        inside = lambda x: not any(c0 - 1e-12 < x < c1 + 1e-12 for c0, c1 in cuts)
        xs = []
        for x0, x1 in zip(fixed[:-1], fixed[1:]):
            if x1 - x0 < 1e-9 or not inside(0.5 * (x0 + x1)):
                continue
            t = np.linspace(x0, x1, int(np.ceil(8 * (x1 - x0) / dmax)) + 1)
            P = np.stack([cl(t) + r.n * y for y in (-a, 0.0, a)])
            hv = fill * h(P.reshape(-1, 2)).reshape(3, -1).min(0)        # the finest across the run
            n = np.concatenate([[0], np.cumsum(np.diff(t) * 2 / (hv[1:] + hv[:-1]))])
            k = max(1, int(np.ceil(n[-1] - 1e-6)))
            xs.append(np.interp(np.arange(k + 1) * n[-1] / k, n, t))
            polys.append(Polygon([cl(x0) - r.n * a, cl(x1) - r.n * a, cl(x1) + r.n * a, cl(x0) + r.n * a]))
            # across the middle, inside the innermost edge ring: longitudinal lines at the spacing at the centre
            b = a - (offsets.max() if len(offsets) else 0.0)
            hc = fill * float(h(cl(0.5 * (x0 + x1))[None])[0])
            m = int(np.ceil(2 * b / hc)) if b > 0 else 0
            ys = -b + 2 * b * np.arange(1, m) / m
            ys = [y for y in ys if not len(vy) or abs(vy - y).min() > 0.5 * hc]      # not next to a via's sides
            for y in list(ys) + list(vy):
                segs.append((cl(x0) + r.n * y, cl(x1) + r.n * y))
        ext = 1e-3 * a
        segs += [(cl(x) - r.n * (a + ext), cl(x) + r.n * (a + ext)) for x in np.unique(np.concatenate(xs))] if xs else []
    return segs, polys, snapped, on_run


def _cell_sides(yv, hc, rings):
    """The longitudinal lines of the grid cell a via at yv (across the run) snaps to: yv -/+ hc, except where an edge
    ring's line (or the copper edge: rings, as |y|) lies within hc / 2 of one, which is then that side itself (no
    line: no sliver beside it). None if the via would not be inside its cell."""
    out = []
    for sgn in (-1.0, 1.0):
        y = yv + sgn * hc
        j = np.argmin(abs(abs(y) - rings))
        ring = rings[j] * (np.sign(y) or 1.0)
        if abs(abs(y) - rings[j]) < 0.5 * hc:
            if (ring - yv) * sgn < 0.25 * hc:                  # the via on or past that ring: not inside its cell
                return None
            continue
        if abs(y) >= rings[0]:                                 # past the copper edge
            return None
        out.append(y)
    return out


def _rings(g):
    polys = [p for p in ([g] if isinstance(g, Polygon) else getattr(g, 'geoms', []))
             if isinstance(p, Polygon) and not p.is_empty]
    return [LineString(r.coords) for p in polys for r in [p.exterior, *p.interiors]]


def gap_zones(geom, gmax, step, tol=1e-9):
    """The copper edges facing other copper across a gap narrower than gmax: [(zone, g)], each zone the copper
    within reach of the edges whose gap falls in one band (about 1.4x wide), g the narrowest gap of that band.
    The gap at an edge: a ray out of the copper along its normal, from points `step` apart, to the copper it hits,
    counted only where the edge it hits faces back (within 60 degrees) and is not the same outline within 20 gaps
    of the ray's start: the wedge where a track's round end meets a pad or strip at a tangent is no gap
    (rf_combline's bridged parts: gaps of 8 um, and rings that far in left its complex64 LU unable to precondition)."""
    from shapely.geometry.polygon import orient
    polys = [p for p in ([geom] if isinstance(geom, Polygon) else getattr(geom, 'geoms', [])) if isinstance(p, Polygon)]
    mids, normals, segs, rings, ring_of, along = [], [], [], [], [], []
    for p in polys:
        p = orient(p, 1.0)                                          # exterior ccw, holes cw: copper on the left
        for ring in [p.exterior, *p.interiors]:
            r = np.asarray(ring.coords)[:, :2]
            segs.append(np.stack([r[:-1], r[1:]], 1))
            c = np.asarray(shapely.segmentize(LineString(ring.coords), step).coords)[:, :2]
            d = np.diff(c, axis=0)
            ln = np.hypot(d[:, 0], d[:, 1])
            ok = ln > tol
            ring_of.append(np.full(ok.sum(), len(rings)))
            along.append((np.cumsum(ln) - 0.5 * ln)[ok])
            rings.append(LineString(ring.coords))
            mids.append(0.5 * (c[1:] + c[:-1])[ok])
            normals.append(np.c_[d[ok, 1], -d[ok, 0]] / ln[ok, None])   # to the right: out of the copper
    if not mids:
        return []
    M, N = np.vstack(mids), np.vstack(normals)
    eps = 1e-3 * gmax
    rays = shapely.linestrings(np.stack([M + eps * N, M + gmax * N], 1))
    hits = shapely.intersection(rays, geom)
    hit = ~shapely.is_empty(hits)
    gap = np.full(len(M), np.inf)
    pts = shapely.points(M[hit])
    gap[hit] = shapely.distance(pts, hits[hit])
    if hit.any():                                                   # the edge hit: its direction against the ray
        Sg = np.vstack(segs)
        Sg = Sg[np.hypot(*(Sg[:, 1] - Sg[:, 0]).T) > tol]
        tree = shapely.STRtree(shapely.linestrings(Sg))
        H = shapely.get_coordinates(shapely.get_point(shapely.shortest_line(pts, hits[hit]), 1))
        j = tree.query_nearest(shapely.points(H), all_matches=False)[1]
        dv = Sg[j, 1] - Sg[j, 0]
        dv /= np.hypot(dv[:, 0], dv[:, 1])[:, None]
        oblique = np.abs(np.einsum('ij,ij->i', dv, N[hit])) > 0.5
        R, A = np.concatenate(ring_of)[hit], np.concatenate(along)[hit]
        wedge = np.zeros(len(H), bool)
        for k in np.unique(R):                                      # the hit on the ray's own outline, close along it
            i = np.flatnonzero(R == k)
            ring = rings[k]
            on = shapely.distance(ring, shapely.points(H[i])) < 1e-6 * gmax + tol
            ds = np.abs(shapely.line_locate_point(ring, shapely.points(H[i])) - A[i])
            ds = np.minimum(ds, ring.length - ds)
            wedge[i] = on & (ds < 20 * gap[np.flatnonzero(hit)[i]])
        gap[np.flatnonzero(hit)[oblique | wedge]] = np.inf
    near = np.isfinite(gap) & (gap < gmax)
    if not near.any():
        return []
    band = np.round(2 * np.log2(gap[near])).astype(int)
    out = []
    for b in np.unique(band):
        sel = np.flatnonzero(near)[band == b]
        g = float(gap[sel].min())
        zone = shapely.union_all(shapely.buffer(shapely.points(M[sel]), max(0.6 * step, g)))
        out.append((zone, g))
    return out


# ---- Mesher -----------------------------------------------------------------
def _via_rings(vias, geom, linework, top, h, dmax, fine_u=None, dmax_fine=None):
    """mesh_polygons(via_rings=): the rings about the vias [(x, y, r, n, ngon)] (LineStrings). Each via's n-gon is
    repeated further out, its vertices opposite the last ring's, each ring 1 + 2 sin(pi / n) times the last's radius
    (cells about as deep as they are wide), while the ring's side is under `top` x the spacing there. A ring is kept
    only where it, with half its gap to the last, lies in the copper clear of the other lines and the other vias'
    rings; a via's rings stop at the first that is not."""
    P = np.array([v[:2] for v in vias], float)
    hv = np.full(len(P), float(dmax)) if h is None else np.asarray(h(P), float)
    if fine_u is not None and dmax_fine:
        hv = np.where(shapely.contains_xy(fine_u, P[:, 0], P[:, 1]), dmax_fine, hv)
    others = shapely.union_all([q for q in linework if not q.is_empty])
    radius = np.array([v[2] for v in vias], float)
    live = np.ones(len(vias), bool)
    out, kept = [], []                                                 # kept: the areas of the rings taken
    while live.any():
        idx = np.flatnonzero(live)
        G, g, R1 = [], [], []
        for i in idx:
            x, y, _, n, is_ngon = vias[i]
            R = radius[i] * (1 + 2 * np.sin(np.pi / n))
            if 2 * R * np.sin(np.pi / n) > top * hv[i]:
                G.append(None); g.append(0.0); R1.append(R)
                continue
            ring = ngon(x, y, R, n, closed=True) if is_ngon else np.asarray(Point(x, y).buffer(R, quad_segs=2).exterior.coords)
            G.append(LineString(ring)); g.append(R - radius[i]); R1.append(R)
        sel = np.array([q is not None for q in G])
        live[idx[~sel]] = False
        if not sel.any():
            break
        idx, g, R1 = idx[sel], np.array(g)[sel], np.array(R1)[sel]
        G = [q for q in G if q is not None]
        areas = shapely.buffer(np.array([Polygon(q.coords) for q in G], dtype=object), 0.5 * g)
        fits = shapely.within(areas, geom) & (shapely.distance(G, others) >= 0.5 * g)   # (its own n-gon: cleared)
        if kept:
            ktree = shapely.STRtree(np.array([a for _, a in kept], dtype=object))
            ka, kb = ktree.query(areas, predicate='intersects')
            owner = np.array([o for o, _ in kept])
            for i, j in zip(ka, kb):
                if owner[j] != idx[i]:
                    fits[i] = False
        f = np.flatnonzero(fits)
        tree = shapely.STRtree(areas[f])
        a, b = tree.query(areas[f], predicate='intersects')
        taken = np.zeros(len(f), bool)
        for k in range(len(f)):                                        # of two that touch, the first
            if not taken[b[(a == k) & (b != k)]].any():
                taken[k] = True
        ok = np.zeros(len(idx), bool)
        ok[f[taken]] = True
        for k in np.flatnonzero(ok):
            out.append(G[k])
            kept.append((idx[k], areas[k]))
            radius[idx[k]] = R1[k]
        live[idx[~ok]] = False
    return out


def _depth(pc, outline, tol):
    """The distance of the line pc from the outline if it is the same along it (an edge ring's piece, or the outline's
    own: 0), else None."""
    d = shapely.distance(outline, shapely.line_interpolate_point(pc, [0.0, 0.25, 0.5, 0.75, 1.0], normalized=True))
    return float(d[1:4].mean()) if np.ptp(d[1:4]) <= 1e-6 * max(d.max(), pc.length) + tol else None


def _aligned(pc, depth, stations, dmax, h=None, near=0.35, gap=1.3):
    """The points of an edge ring's piece pc (depth in from the outline) where the outline's points (stations) fall
    on it moved in along the outline's normal: those whose nearest point on pc is depth away (none round a corner
    of it). The piece's own vertices stay (its ends, its corners); a projection within near x the spacing of one of
    them, or of the last one kept, is dropped; a gap over gap x the spacing is filled at the spacing (h(points), or
    dmax) as spaced pieces are."""
    P = np.asarray(pc.coords)[:, :2]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_v = np.concatenate([[0.0], np.cumsum(seg)])                      # the vertices' arc length
    L = s_v[-1]
    hof = (lambda X: np.full(len(X), dmax)) if h is None else h
    S = np.zeros(0)
    if len(stations):
        lo, hi = P.min(0) - 1.1 * depth, P.max(0) + 1.1 * depth
        cand = stations[np.all((stations >= lo) & (stations <= hi), axis=1)]
        if len(cand):
            pts = shapely.points(cand)
            u = shapely.line_locate_point(pc, pts)
            Q = shapely.get_coordinates(shapely.line_interpolate_point(pc, u))
            dist = np.linalg.norm(Q - cand, axis=1)
            ok = (np.abs(dist - depth) <= 1e-3 * depth) & (u > 0) & (u < L)
            S = np.sort(u[ok])
    at = lambda s_: np.column_stack([np.interp(s_, s_v, P[:, 0]), np.interp(s_, s_v, P[:, 1])])
    hv = hof(at(S)) if len(S) else np.zeros(0)
    keep, last = [], -np.inf
    for s_, hs in zip(S, hv):                                         # clear of the vertices and of each other
        j = np.searchsorted(s_v, s_)
        dv = min(s_ - s_v[j - 1] if j > 0 else np.inf, s_v[j] - s_ if j < len(s_v) else np.inf)
        if dv >= near * hs and s_ - last >= near * hs:
            keep.append(s_)
            last = s_
    s_all = np.unique(np.concatenate([s_v, keep]))
    out = [s_all[:1]]
    for a, b in zip(s_all[:-1], s_all[1:]):                           # the gaps: at the spacing
        hm = float(hof(at(np.array([0.5 * (a + b)])))[0])
        k = int(np.ceil((b - a) / hm - 1e-9)) if b - a > gap * hm else 1
        out.append(a + (b - a) * np.arange(1, k + 1) / k)
    return at(np.concatenate(out))


def mesh_polygons(geom, dmax, w_ref, ncross=6, grade=1.0, lines=(), vias=(), amax=None,
                  max_offset_frac=0.45, tol=1e-9, fine=None, dmax_fine=None, edge_zone=None, split_chains=False,
                  coarsen=None, grow=0.5, structured=False, struct_w_max=20e-3, gap_rings=None, area_scale=1.0,
                  coarsen_fine=True, align=False, via_rings=0.0):
    """
    geom   : shapely (Multi)Polygon of one metal layer (triangulated in the compiled core, mesh2d: cdtmesh.py)
    dmax   : max edge length along conductors / in open regions
    w_ref  : reference strip width for the edge grading (narrowest trace on the layer)
    lines  : [(p0, p1), ...] constraint segments (port gap and cut lines), clipped to geom
    vias   : [(x, y, radius), ...] points to insert as vertices (+ an octagon of the via radius); (x, y, radius, n)
             an n-gon; (x, y, radius, 4, True) a square that may snap to a cell of the structured grid
    edge_zone : (experimental) shapely area where the current crowds at edges (the signal copper buffered by
             ~0.5 mm): the edge rings are kept only inside it. Demo transition (cropped): 4.9k unknowns instead of
             8.4k, S within 0.006 of the full rings (2.4x faster a solve); Simulation(zone=) passes it
    split_chains : each outline at the spacing of the region it runs through (zone=; the default spaces a whole
             chain by where its centroid falls, which a closed outline's centroid can get wrong)
    coarsen_fine : with coarsen, the fine regions coarsened too (True); False: they keep dmax_fine (sim.py: a
             meshed plane's, under the copper of the layers near it)
    coarsen : (experimental) cells grow away from the features (corners and bends of the copper outline, vias, port
             and cut lines; _features) up to coarsen x dmax: dmax + grow x the distance to the nearest one, along the
             outline and the edge rings (their spacing across the edge is kept) and over the area inside them
             (the middle of wide traces, planes). None: dmax everywhere
    structured : (experimental) straight runs of trace (line.find_runs, up to struct_w_max wide: traces and rectangular pads) as a tensor grid:
             transverse lines across the run at stations along it (the spacing of the field, or dmax), and
             longitudinal ones across its middle (inside the edge rings), so its cells are rectangles - one diagonal
             each, quad cells of 0 degrees. Port / cut lines on a run are stations; square vias join the grid (_run_grid),
             around other vias it is left as is
    gap_rings : fractions of the gap (e.g. (1/4, 1/8, 1/16)): rings that far in from the copper edges facing other
             copper across a gap g (gap_zones), where the width-graded rings are too coarse for the edge charge
             across it (g/4 under the first ring at w_ref's grading). Coupled lines: on the 10 dB coupler (0.14 mm
             gap, 1.28 mm strips) S31 -10.64 dB, against -10.24 to -10.29 from refined FDTD, until the edge cells
             came down to the gap's scale; isolated traces and planes keep their cost
    align  : the edge rings' points at the outline's, moved in along its normal (_aligned): the strips between the
             outline and the rings are rows of rectangles, two triangles each (quad cells pair them: quads.py), where
             each ring spaced on its own (align=False, the default) left them skewed, the stations out of step at
             every corner, port line and change of spacing. Where a ring has no outline point opposite (round a corner)
             it is spaced as before. Simulation turns it on with quads (align='auto'): on 4 boards against a mesh of
             half the size, 0.3-1 % fewer unknowns after the quads (launch: 58 % of the triangles paired for 53 %), S as
             before (0.0005); without quads 0.5-1 % more unknowns for nothing
    via_rings : rings about each via's n-gon, the n-gon repeated further out, each ring's vertices opposite the
             last's and its radius 1 + 2 sin(pi / n) times it (_via_rings), while a ring's side is under via_rings x
             the local spacing: the via in rings of n trapezoids (quad cells pair them: an n-gon annulus's angles are
             the same at any ratio) that grow out to the cells around, where the n-gon alone fanned out to them in
             slivers. Only where they clear the copper's edge, the other lines and the other vias' rings. 0 (the
             default): none. 0.8: on mom25d's demo board (a via fence) quads pair 53 % of the triangles for 43 %, but
             it costs 21 % more unknowns (16 % more after the quads) and S moved no closer to a finer mesh's; pcbqs's
             via spreading into a plate 2.1 % off a fine pad's for 2.8 % at dmax 0.5 mm, the same at 0.25
    Returns nodes (N,2), tris (M,3) counter-clockwise, and the list of via vertex ids.
    """
    if geom.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), int), []
    linework = _rings(geom)
    offsets = w_ref * graded(np.arange(1, ncross) / ncross, grade)      # inward rings: the tensor mesher's grading
    ring_d = list(offsets[offsets < 0.5 * w_ref - 1e-12])               # the rings' depths (align)
    for d in offsets[offsets < 0.5 * w_ref - 1e-12]:
        linework += [r if edge_zone is None else r.intersection(edge_zone)
                     for r in _rings(geom.buffer(-d, join_style='mitre', mitre_limit=2.0))]
    if gap_rings:
        first = offsets[0] if len(offsets) else 0.5 * w_ref
        for zone, g in gap_zones(geom, 4 * first, dmax if dmax_fine is None else min(dmax, dmax_fine), tol):
            if edge_zone is not None:
                zone = zone.intersection(edge_zone)
            for d in sorted({fr * g for fr in gap_rings}):
                if d < 0.75 * first:
                    ring_d.append(d)
                    linework += [r.intersection(zone)
                                 for r in _rings(geom.buffer(-d, join_style='mitre', mitre_limit=2.0))]
    linework += [LineString([p0, p1]).intersection(geom) for p0, p1 in lines]
    fine_parts = []
    if fine is not None and dmax_fine:
        fg = fine.intersection(geom).buffer(0)
        fine_parts = [q for q in getattr(fg, 'geoms', [fg]) if q.geom_type == 'Polygon' and q.area > (2 * dmax_fine) ** 2]
        linework += [r for q in fine_parts for r in _rings(q)]          # region boundaries as constraints
    fine_u = unary_union(fine_parts).buffer(tol) if fine_parts else None
    hfield = None
    if coarsen and coarsen > 1:
        F = _features(geom, lines, vias, min(dmax, dmax_fine or dmax))
        Fk = cKDTree(F) if len(F) else None
        dist = (lambda P: Fk.query(P)[0]) if Fk is not None else (lambda P: np.full(len(P), np.inf))
        hfield = lambda P, d_: np.minimum(coarsen * d_, d_ + grow * dist(P))
    run_polys, snapped, on_run = [], set(), set()
    if structured:
        hrun, within, dgrid = (lambda P: hfield(P, dmax)) if hfield is not None else None, None, dmax
        if edge_zone is not None and fine_u is not None:
            # zone=: only the runs inside its fine region (the traces), at its spacing. A run reaching out of it (a
            # plane under the zone, 8 mm wide on rf_stripline) took its grid's lines over the coarse cells:
            # rf_stripline 5,647 unknowns against 4,985 without the grid
            within, dgrid = fine_u.buffer(1e-6 * dmax), dmax_fine
            hrun = (lambda P: hfield(P, dmax_fine)) if hfield is not None else (lambda P: np.full(len(P), dmax_fine))
        st_lines, run_polys, snapped, on_run = _run_grid(geom, dgrid, lines, vias, offsets[offsets < 0.5 * w_ref - 1e-12],
                                                 struct_w_max, hrun, within=within)
        linework += [LineString([p0, p1]).intersection(geom) for p0, p1 in st_lines]
    vpts, via_out = [], []
    for i, via in enumerate(vias):
        x, y, r = via[:3]
        vpts.append((x, y))
        if len(via) > 4 and via[4] and i not in on_run:
            r = 0.0                                              # one that could snap, on no run: its vertex alone
        if r > 0 and i not in snapped:                           # (x, y, r, n): a regular n-gon, else an octagon
            ring = ngon(x, y, r, int(via[3]), closed=True) if len(via) > 3 else \
                Point(x, y).buffer(r, quad_segs=2).exterior.coords
            linework.append(LineString(ring).intersection(geom))
            if via_rings and i not in on_run:
                via_out.append((x, y, r, int(via[3]) if len(via) > 3 else 8, len(via) > 3))
    if via_out:
        linework += _via_rings(via_out, geom, linework, via_rings, (lambda P: hfield(P, dmax)) if hfield is not None
                               else None, dmax, fine_u, dmax_fine)
    runs_u = shapely.unary_union(run_polys).buffer(tol) if run_polys else None
    # snap-rounding noding: T-junctions (port lines ending on edges) must split the edge exactly
    noded = shapely.unary_union([g for g in linework if not g.is_empty], grid_size=tol)
    pieces = list(getattr(noded, 'geoms', [noded]))
    chains = []
    outline = shapely.union_all(linework[:len(_rings(geom))]) if align and ring_d else None
    deferred, stations = [], []                     # the rings' pieces (align: after the outline), the outline's points

    def spaced(q, d_):
        if hfield is None or (not coarsen_fine and fine_parts and d_ == dmax_fine):
            return np.asarray(shapely.segmentize(q, d_).coords)[:, :2]
        return _spaced(np.asarray(q.coords)[:, :2], lambda P: hfield(P, d_), d_)
    for pc in pieces:
        if pc.length < tol:
            continue
        if split_chains and fine_u is not None and not fine_u.contains(pc) and fine_u.intersects(pc):
            # a chain that runs in and out of the fine region (a pad or trace outline around the zone's edge):
            # each part at its own spacing (the centroid of a closed outline can lie anywhere)
            for part, d_ in ((pc.intersection(fine_u), dmax_fine), (pc.difference(fine_u), dmax)):
                for q in getattr(part, 'geoms', [part]):
                    if q.geom_type == 'LineString' and q.length >= tol:
                        chains.append(spaced(q, d_))
            continue
        inside = fine_u is not None and (fine_u.contains(pc) if split_chains else fine_u.contains(pc.centroid))
        dm = dmax_fine if inside else dmax
        depth = _depth(pc, outline, tol) if outline is not None else None
        if depth is not None and depth > 10 * tol and min(abs(depth - d) for d in ring_d) < 1e-6 * dmax + 10 * tol:
            deferred.append((pc, dm, depth))
            continue
        chains.append(spaced(pc, dm))
        if depth is not None and depth <= 10 * tol:
            stations.append(chains[-1])
    if deferred:
        st = np.vstack(stations) if stations else np.zeros((0, 2))
        for pc, dm, depth in deferred:
            h_ = None if hfield is None or (not coarsen_fine and fine_parts and dm == dmax_fine) else \
                (lambda P, d_=dm: hfield(P, d_))
            chains.append(_aligned(pc, depth, st, dm, h_))
    allp = np.vstack(chains + [np.array(vpts).reshape(-1, 2)]) if vpts else np.vstack(chains)
    # merge near-coincident vertices (noding round-off would otherwise create sliver triangles): each cluster of
    # vertices closer than `snap` becomes its lowest-numbered one
    n = len(allp)
    pairs = cKDTree(allp).query_pairs(max(10 * tol, 1e-4 * dmax), output_type='ndarray')
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), pairs.T), shape=(n, n)), directed=False)
    first = np.full(lab.max() + 1, n)
    np.minimum.at(first, lab, np.arange(n))
    uniq, inv = np.unique(first[lab], return_inverse=True)
    verts = allp[uniq]
    segs, off = set(), 0
    for ch in chains:
        ids, off = inv[off:off + len(ch)], off + len(ch)
        segs |= {(min(a, b), max(a, b)) for a, b in zip(ids[:-1], ids[1:]) if a != b}
    via_ids = list(inv[off:off + len(vpts)])
    segs = np.array(sorted(segs))
    # in units of dmax about the vertices' middle (the triangulation's numbers near 1)
    org = verts.mean(0)
    amax = (amax or 0.5 * dmax ** 2) / dmax ** 2
    T = _cdt((verts - org) / dmax, segs, geom, tol, dmax, org, amax, hfield, fine_parts, fine_u, dmax_fine, runs_u,
             area_scale, coarsen_fine)
    nodes, tris = T['vertices'] * dmax + org, T['triangles']
    cent = nodes[tris].mean(1)
    tris = tris[shapely.contains_xy(geom.buffer(tol), cent[:, 0], cent[:, 1])]
    # drop unused nodes, fix orientation
    used = np.unique(tris)
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    nodes, tris = nodes[used], remap[tris]
    via_ids = [int(remap[i]) if i < len(remap) and remap[i] >= 0 else -1 for i in via_ids]
    V = nodes[tris]
    ar = (V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0])
    tris[ar < 0] = tris[ar < 0][:, [0, 2, 1]]
    return nodes, tris[np.abs(ar) > 1e-10 * dmax ** 2], via_ids          # (a degenerate triangle: none)


def _core():
    """mom25d's compiled core (it triangulates: cdt_triangulate), built or not set aside (MOM25D_NO_CORE sets the
    kernels aside, not the mesher)."""
    from . import fast
    core = fast._built
    if core is None or not hasattr(core, 'cdt_triangulate'):
        raise RuntimeError("mom25d meshes with its compiled core, which is not built (or is older than its mesher): "
                           "python -m mom25d.core.build (needs a C++17 compiler and pybind11)")
    return core


def _cdt(V, segs, geom, tol, dmax, org, amax, hfield, fine_parts, fine_u, dmax_fine, runs_u, area_scale=1.0,
         coarsen_fine=True):
    """mesh_polygons' triangulation (cdtmesh: the core's mesh2d and the points inside), the area limits: amax, afine in the fine
    regions, with coarsen the field's 0.5 h^2 (none in the structured runs' cells; the fine regions kept at afine
    unless coarsen_fine); all of them x area_scale."""
    from .cdtmesh import mesh as cdt_mesh
    g_in = geom.buffer(tol)
    to_m = lambda c: np.asarray(c) * dmax + org
    afine = 0.5 * (dmax_fine / dmax) ** 2 if fine_parts else None

    def inside(c):
        P = to_m(c)
        return shapely.contains_xy(g_in, P[:, 0], P[:, 1])

    def limit(c):
        P = to_m(c)
        infine = shapely.contains_xy(fine_u, P[:, 0], P[:, 1]) if fine_parts else np.zeros(len(P), bool)
        if hfield is None:
            lim = np.where(infine, afine, amax) if fine_parts else np.full(len(P), amax)
        else:
            h = np.where(infine, hfield(P, dmax_fine or dmax) if coarsen_fine else dmax_fine, hfield(P, dmax))
            lim = 0.5 * (h / dmax) ** 2
            if runs_u is not None:
                lim[shapely.contains_xy(runs_u, P[:, 0], P[:, 1])] = -1.0
        return np.where(lim > 0, area_scale * lim, lim)
    frozen = None
    if runs_u is not None:
        frozen = lambda c: shapely.contains_xy(runs_u, *to_m(c).T)
    sizes = (1.0,) + ((dmax_fine / dmax,) if fine_parts else ())       # each region's cell size (dmax)
    return cdt_mesh(V, segs, limit, inside, _core(), frozen=frozen, sizes=sizes)
