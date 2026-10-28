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
import triangle as tr
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


# ---- Mesher -----------------------------------------------------------------
def mesh_polygons(geom, dmax, w_ref, ncross=6, grade=1.0, lines=(), vias=(), amax=None,
                  max_offset_frac=0.45, tol=1e-9, fine=None, dmax_fine=None, edge_zone=None, split_chains=False,
                  coarsen=None, grow=0.5, structured=False, struct_w_max=20e-3):
    """
    geom   : shapely (Multi)Polygon of one metal layer
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
    coarsen : (experimental) cells grow away from the features (corners and bends of the copper outline, vias, port
             and cut lines; _features) up to coarsen x dmax: dmax + grow x the distance to the nearest one, along the
             outline and the edge rings (their spacing across the edge is kept) and over the area inside them
             (the middle of wide traces, planes). None: dmax everywhere
    structured : (experimental) straight runs of trace (line.find_runs, up to struct_w_max wide: traces and rectangular pads) as a tensor grid:
             transverse lines across the run at stations along it (the spacing of the field, or dmax), and
             longitudinal ones across its middle (inside the edge rings), so its cells are rectangles - one diagonal
             each, quad cells of 0 degrees. Port / cut lines on a run are stations; square vias join the grid (_run_grid),
             around other vias it is left as is
    Returns nodes (N,2), tris (M,3) counter-clockwise, and the list of via vertex ids.
    """
    if geom.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), int), []
    linework = _rings(geom)
    offsets = w_ref * graded(np.arange(1, ncross) / ncross, grade)      # inward rings: the tensor mesher's grading
    for d in offsets[offsets < 0.5 * w_ref - 1e-12]:
        linework += [r if edge_zone is None else r.intersection(edge_zone)
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
    vpts = []
    for i, via in enumerate(vias):
        x, y, r = via[:3]
        vpts.append((x, y))
        if len(via) > 4 and via[4] and i not in on_run:
            r = 0.0                                              # one that could snap, on no run: its vertex alone
        if r > 0 and i not in snapped:                           # (x, y, r, n): a regular n-gon, else an octagon
            ring = ngon(x, y, r, int(via[3]), closed=True) if len(via) > 3 else \
                Point(x, y).buffer(r, quad_segs=2).exterior.coords
            linework.append(LineString(ring).intersection(geom))
    runs_u = shapely.unary_union(run_polys).buffer(tol) if run_polys else None
    # snap-rounding noding: T-junctions (port lines ending on edges) must split the edge exactly
    noded = shapely.unary_union([g for g in linework if not g.is_empty], grid_size=tol)
    pieces = list(getattr(noded, 'geoms', [noded]))
    chains = []

    def spaced(q, d_):
        if hfield is None:
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
        chains.append(spaced(pc, dm))
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
    # Triangle's switch parser cannot read exponents ('e' is a switch), so work in units of dmax
    org = verts.mean(0)
    A = dict(vertices=(verts - org) / dmax, segments=segs)
    amax = (amax or 0.5 * dmax ** 2) / dmax ** 2
    if hfield is not None:
        # coarse first (the largest cells the field allows), then each triangle refined to the field's area at its
        # centroid (fine regions: their own dmax). No quality switch (q), as below: it would split the thin
        # strips between the edge rings, now long, into fans of small triangles (rf_gcpw: twice the unknowns)
        afine = 0.5 * (dmax_fine / dmax) ** 2 if fine_parts else None
        if fine_parts:
            rest = geom.difference(unary_union(fine_parts)).buffer(-2 * tol)
            reg = [(q.representative_point(), afine) for q in fine_parts]
            reg += [(q.representative_point(), amax * coarsen ** 2) for q in getattr(rest, 'geoms', [rest])
                    if q.geom_type == 'Polygon' and not q.is_empty]
            A['regions'] = np.array([[(pt.x - org[0]) / dmax, (pt.y - org[1]) / dmax, i + 1, a]
                                     for i, (pt, a) in enumerate(reg)])
            T = tr.triangulate(A, "pQAa")
        else:
            T = tr.triangulate(A, f"pQa{amax * coarsen ** 2:.8f}")
        for _ in range(2):
            c = T['vertices'][T['triangles']].mean(1) * dmax + org
            infine = np.zeros(len(c), bool)
            if fine_parts:
                infine = np.isin(T['triangle_attributes'][:, 0], np.arange(1, len(fine_parts) + 1))
            h = np.where(infine, hfield(c, dmax_fine or dmax), hfield(c, dmax))
            amax_t = 0.5 * (h / dmax) ** 2
            if runs_u is not None:                     # the runs' grid cells as they are (no limit: -1)
                amax_t[shapely.contains_xy(runs_u, c[:, 0], c[:, 1])] = -1.0
            T = tr.triangulate(dict(T, triangle_max_area=amax_t), "rpQa" + ("A" if fine_parts else ""))
    elif fine_parts:
        # regional area limits: fine inside the projection of nearby metal, coarse elsewhere
        afine = 0.5 * (dmax_fine / dmax) ** 2
        rest = geom.difference(unary_union(fine_parts)).buffer(-2 * tol)
        reg = [(q.representative_point(), afine) for q in fine_parts]
        reg += [(q.representative_point(), amax) for q in getattr(rest, 'geoms', [rest])
                if q.geom_type == 'Polygon' and not q.is_empty]
        A['regions'] = np.array([[(pt.x - org[0]) / dmax, (pt.y - org[1]) / dmax, i + 1, a]
                                 for i, (pt, a) in enumerate(reg)])
        T = tr.triangulate(A, "pQa")
    else:
        T = tr.triangulate(A, f"pQa{amax:.8f}")
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
    return nodes, tris[np.abs(ar) > 1e-30], via_ids
