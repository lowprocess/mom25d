"""
mom25d_geo.py - geometry helpers for the multilayer solver

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
from shapely.geometry import (Polygon, MultiPolygon, LineString, MultiLineString, Point, box)
from shapely.ops import unary_union
import triangle as tr


# =============================================================================
# Ports in arbitrary directions
# =============================================================================
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
        self.c = np.asarray(center, float)
        d = np.asarray(direction, float)
        self.d = d / np.linalg.norm(d)
        self.t = np.array([-self.d[1], self.d[0]])
        self.w, self.ref, self.s0, self.step, self.zc, self.name = width, ref_dist, s0, step, zc, name

    # compatibility with the axis-aligned Port API used by the plotting code
    @property
    def dvec(self):
        return self.d

    def line(self, s):
        """End points of the transverse line at distance s from the gap (s=0: the gap)."""
        p = self.c + self.d * s
        return p - self.t * self.w / 2, p + self.t * self.w / 2

    @property
    def axis_aligned(self):
        return abs(self.d[0]) < 1e-12 or abs(self.d[1]) < 1e-12


def port_from_axis(p):
    """Convert a legacy mom25d.Port / MLPort (axis, pos, span, dir) into a LinePort."""
    tmid = 0.5 * (p.span[0] + p.span[1])
    w = p.span[1] - p.span[0]
    if p.axis == 'x':
        c, d = (p.pos, tmid), (p.dir, 0.0)
    else:
        c, d = (tmid, p.pos), (0.0, p.dir)
    lp = LinePort(p.layer, c, d, w, p.ref, s0=p.s0, step=p.step, zc=p.zc, name=p.name)
    return lp


# =============================================================================
# Geometry
# =============================================================================
def to_geometry(shapes):
    """List of rectangles (x0,y0,x1,y1) and/or shapely geometries -> cleaned (Multi)Polygon."""
    gs = []
    for s in shapes:
        if isinstance(s, (tuple, list)) and len(s) == 4 and np.isscalar(s[0]):
            gs.append(box(*s))
        else:
            gs.append(s)
    g = unary_union(gs).buffer(0)
    return g


def is_manhattan_rects(shapes):
    return all(isinstance(s, (tuple, list)) and len(s) == 4 and np.isscalar(s[0]) for s in shapes)


def _rings(g):
    polys = [g] if isinstance(g, Polygon) else list(getattr(g, 'geoms', []))
    out = []
    for p in polys:
        if p.is_empty or not isinstance(p, Polygon):
            continue
        out.append(LineString(p.exterior.coords))
        out += [LineString(r.coords) for r in p.interiors]
    return out


def edge_offsets(w_ref, ncross, grade=1.0):
    """Inward offsets mimicking the tensor mesher's graded cells across a strip of width w_ref."""
    k = np.arange(1, ncross)
    u = k / ncross
    u = (1 - grade) * u + grade * (1 - np.cos(np.pi * u)) / 2
    d = w_ref * u
    return [x for x in d if x < 0.5 * w_ref - 1e-12]


# =============================================================================
# Mesher
# =============================================================================
def mesh_polygons(geom, dmax, w_ref, ncross=6, grade=1.0, lines=(), vias=(), amax=None,
                  max_offset_frac=0.45, tol=1e-9, fine=None, dmax_fine=None):
    """
    geom   : shapely (Multi)Polygon of one metal layer
    dmax   : max edge length along conductors / in open regions
    w_ref  : reference strip width for the edge grading (narrowest trace on the layer)
    lines  : [(p0, p1), ...] constraint segments (port gap and cut lines), clipped to geom
    vias   : [(x, y, radius), ...] points to insert as vertices (+ an octagon of the via radius)
    Returns nodes (N,2), tris (M,3) counter-clockwise, and the list of via vertex ids.
    """
    if geom.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), int), []
    linework = _rings(geom)
    for d in edge_offsets(w_ref, ncross, grade):
        inner = geom.buffer(-d, join_style='mitre', mitre_limit=2.0)
        if not inner.is_empty:
            linework += _rings(inner)
    for p0, p1 in lines:
        seg = LineString([p0, p1]).intersection(geom)
        if not seg.is_empty:
            linework.append(seg)
    fine_parts = []
    if fine is not None and dmax_fine:
        fg = fine.intersection(geom).buffer(0)
        fine_parts = [q for q in getattr(fg, 'geoms', [fg]) if q.geom_type == 'Polygon' and q.area > (2 * dmax_fine) ** 2]
        for q in fine_parts:                                  # region boundaries as constraints
            linework += _rings(q)
    vpts = []
    for (x, y, r) in vias:
        vpts.append((x, y))
        if r > 0:
            ring = Point(x, y).buffer(r, quad_segs=2).exterior   # octagon
            linework.append(LineString(ring.coords).intersection(geom))
    # snap-rounding noding: T-junctions (port lines ending on edges) must split the edge exactly
    noded = shapely.unary_union([g for g in linework if not g.is_empty], grid_size=tol)
    pieces = list(getattr(noded, 'geoms', [noded]))
    chains = []
    fine_u = unary_union(fine_parts).buffer(tol) if fine_parts else None
    for pc in pieces:
        if pc.length < tol:
            continue
        dm = dmax_fine if (fine_u is not None and fine_u.contains(pc.centroid)) else dmax
        chains.append(np.asarray(shapely.segmentize(pc, dm).coords)[:, :2])
    allp = np.vstack(chains + [np.array(vpts).reshape(-1, 2)]) if vpts else np.vstack(chains)
    # merge near-coincident vertices (noding round-off would otherwise create sliver triangles)
    from scipy.spatial import cKDTree
    snap = max(10 * tol, 1e-4 * dmax)
    rep = np.arange(len(allp))
    for i, j in sorted(cKDTree(allp).query_pairs(snap)):
        ri, rj = rep[i], rep[j]
        while rep[ri] != ri: ri = rep[ri]
        while rep[rj] != rj: rj = rep[rj]
        if ri != rj:
            rep[max(ri, rj)] = min(ri, rj)
    for i in range(len(rep)):
        r_ = rep[i]
        while rep[r_] != r_: r_ = rep[r_]
        rep[i] = r_
    uniq, inv = np.unique(rep, return_inverse=True)
    verts = allp[uniq]
    segs, off = set(), 0
    for ch in chains:
        ids = inv[off:off + len(ch)]
        off += len(ch)
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                segs.add((min(a, b), max(a, b)))
    via_ids = list(inv[off:off + len(vpts)])
    segs = np.array(sorted(segs))
    # Triangle's switch parser cannot read exponents ('e' is a switch), so work in units of dmax
    org = verts.mean(0)
    A = dict(vertices=(verts - org) / dmax, segments=segs)
    amax = (amax or 0.5 * dmax ** 2) / dmax ** 2
    if fine_parts:
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
    keep = shapely.contains_xy(geom.buffer(tol), cent[:, 0], cent[:, 1])
    tris = tris[keep]
    # drop unused nodes, fix orientation
    used = np.unique(tris)
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    nodes, tris = nodes[used], remap[tris]
    via_ids = [int(remap[i]) if i < len(remap) and remap[i] >= 0 else -1 for i in via_ids]
    V = nodes[tris]
    ar = (V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0])
    flip = ar < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    good = np.abs(ar) > 1e-30
    return nodes, tris[good], via_ids
