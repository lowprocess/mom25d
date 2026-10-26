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


def _rings(g):
    polys = [p for p in ([g] if isinstance(g, Polygon) else getattr(g, 'geoms', []))
             if isinstance(p, Polygon) and not p.is_empty]
    return [LineString(r.coords) for p in polys for r in [p.exterior, *p.interiors]]


# ---- Mesher -----------------------------------------------------------------
def mesh_polygons(geom, dmax, w_ref, ncross=6, grade=1.0, lines=(), vias=(), amax=None,
                  max_offset_frac=0.45, tol=1e-9, fine=None, dmax_fine=None, edge_zone=None, split_chains=False):
    """
    geom   : shapely (Multi)Polygon of one metal layer
    dmax   : max edge length along conductors / in open regions
    w_ref  : reference strip width for the edge grading (narrowest trace on the layer)
    lines  : [(p0, p1), ...] constraint segments (port gap and cut lines), clipped to geom
    vias   : [(x, y, radius), ...] points to insert as vertices (+ an octagon of the via radius)
    edge_zone : (experimental) shapely area where the current crowds at edges (the signal copper buffered by
             ~0.5 mm): the edge rings are kept only inside it. Demo transition (cropped): 4.9k unknowns instead of
             8.4k, S within 0.006 of the full rings (2.4x faster a solve); Simulation(zone=) passes it
    split_chains : each outline at the spacing of the region it runs through (zone=; the default spaces a whole
             chain by where its centroid falls, which a closed outline's centroid can get wrong)
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
    vpts = []
    for via in vias:
        x, y, r = via[:3]
        vpts.append((x, y))
        if r > 0:                                                # (x, y, r, n): a regular n-gon, else an octagon
            ring = ngon(x, y, r, int(via[3]), closed=True) if len(via) > 3 else \
                Point(x, y).buffer(r, quad_segs=2).exterior.coords
            linework.append(LineString(ring).intersection(geom))
    # snap-rounding noding: T-junctions (port lines ending on edges) must split the edge exactly
    noded = shapely.unary_union([g for g in linework if not g.is_empty], grid_size=tol)
    pieces = list(getattr(noded, 'geoms', [noded]))
    chains = []
    fine_u = unary_union(fine_parts).buffer(tol) if fine_parts else None
    for pc in pieces:
        if pc.length < tol:
            continue
        if split_chains and fine_u is not None and not fine_u.contains(pc) and fine_u.intersects(pc):
            # a chain that runs in and out of the fine region (a pad or trace outline around the zone's edge):
            # each part at its own spacing (the centroid of a closed outline can lie anywhere)
            for part, d_ in ((pc.intersection(fine_u), dmax_fine), (pc.difference(fine_u), dmax)):
                for q in getattr(part, 'geoms', [part]):
                    if q.geom_type == 'LineString' and q.length >= tol:
                        chains.append(np.asarray(shapely.segmentize(q, d_).coords)[:, :2])
            continue
        inside = fine_u is not None and (fine_u.contains(pc) if split_chains else fine_u.contains(pc.centroid))
        dm = dmax_fine if inside else dmax
        chains.append(np.asarray(shapely.segmentize(pc, dm).coords)[:, :2])
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
