"""
mom25d.mesh_distmesh - the polygon mesher of geo.mesh_polygons with DistMesh node placement (Persson & Strang,
"A simple mesh generator in MATLAB", SIAM Review 46, 2004), same contract:

    nodes, tris, via_ids = mesh_polygons_distmesh(geom, dmax, w_ref, lines=..., vias=..., fine=..., dmax_fine=...)

The nodes relax like a truss: Delaunay bars push each other apart towards the target length h(p), nodes
leaving the metal are projected back with the signed distance function. Fixed nodes keep what must be in
the mesh: polygon corners, the boundary sampled at its target size, port cut lines, via points and
their octagons. DistMesh cannot enforce edges, so the final connectivity is a constrained Delaunay
triangulation of the relaxed nodes (Triangle, no added points): conformal like the other meshers.

  rings=False  isotropic: h grows from w_ref / ncross at the conductor edge to dmax (dmax_fine in `fine`)
  rings=True   the edge grading of geo.mesh_polygons (inward offset rings, nodes fixed every dmax along
               them), DistMesh relaxing the interior at dmax
"""
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import Delaunay, cKDTree


def _polys(g):
    return [p for p in ([g] if isinstance(g, Polygon) else getattr(g, 'geoms', [])) if isinstance(p, Polygon) and not p.is_empty]


def _sample(line, h):
    """Points along a LineString / ring every <= h (its vertices included)."""
    return np.asarray(shapely.segmentize(line, h).coords)[:, :2]


def mesh_polygons_distmesh(geom, dmax, w_ref, ncross=6, grade=1.0, lines=(), vias=(), amax=None, max_offset_frac=0.45,
                           tol=1e-9, fine=None, dmax_fine=None, rings=False, iters=80, grow=0.3, seed=0):
    import triangle as tr
    from .geo import edge_offsets
    if geom.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), int), []
    geom = geom.buffer(0)
    bnd = geom.boundary
    h_edge = max(w_ref / ncross, 1e-3 * dmax)
    offs = [d for d in edge_offsets(w_ref, ncross, grade) if d < max_offset_frac * w_ref] if rings else []
    core = geom.buffer(-offs[-1], join_style='mitre', mitre_limit=2.0) if offs else None
    fine_g = fine.intersection(geom).buffer(0) if (fine is not None and dmax_fine) else None

    def h(p):
        """Target edge length at points p (n, 2)."""
        far = np.full(len(p), dmax)
        if fine_g is not None and not fine_g.is_empty:
            far[shapely.contains_xy(fine_g, p[:, 0], p[:, 1])] = dmax_fine
        if rings:
            return far
        d = shapely.distance(bnd, shapely.points(p))
        return np.minimum(far, h_edge + grow * d)

    # ---- fixed nodes and constraint segments ----
    segs_pts = []                                                   # polylines that must be mesh edges
    hb = dmax if rings else h_edge
    for poly in _polys(geom):
        for ring in [poly.exterior] + list(poly.interiors):
            segs_pts.append(_sample(LineString(ring.coords), hb))
    for d in offs:
        for q in _polys(geom.buffer(-d, join_style='mitre', mitre_limit=2.0)):
            for ring in [q.exterior] + list(q.interiors):
                segs_pts.append(_sample(LineString(ring.coords), dmax))
    for p0, p1 in lines:
        seg = LineString([p0, p1]).intersection(geom)
        for q in getattr(seg, 'geoms', [seg]):
            if q.geom_type == 'LineString' and q.length > tol:
                segs_pts.append(_sample(q, min(hb, max(q.length / 4, tol * 10))))
    vpts = []
    for (x, y, r) in vias:
        vpts.append((x, y))
        if r > 0:
            ring = LineString(Point(x, y).buffer(r, quad_segs=2).exterior.coords).intersection(geom)
            for q in getattr(ring, 'geoms', [ring]):
                if q.geom_type == 'LineString' and q.length > tol:
                    segs_pts.append(np.asarray(q.coords)[:, :2])
    fixed = np.vstack(segs_pts + ([np.array(vpts)] if vpts else []))

    # ---- free nodes: rejection sampling by density 1 / h^2, then the truss relaxation ----
    region = core if (rings and core is not None and not core.is_empty) else geom
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = region.bounds
    hmin = h_edge if not rings else (dmax_fine or dmax)
    nx, ny = int((x1 - x0) / hmin) + 2, int((y1 - y0) / (hmin * np.sqrt(3) / 2)) + 2
    gx, gy = np.meshgrid(x0 + np.arange(nx) * hmin, y0 + np.arange(ny) * hmin * np.sqrt(3) / 2)
    gx[1::2] += hmin / 2
    p = np.column_stack([gx.ravel(), gy.ravel()])
    p = p[shapely.contains_xy(region.buffer(-0.5 * hmin), p[:, 0], p[:, 1])]
    if len(p):
        hp = h(p)
        p = p[rng.random(len(p)) < (hmin / hp) ** 2]
    if len(fixed) and len(p):                                       # not on top of the fixed nodes
        dd, _ = cKDTree(fixed).query(p)
        p = p[dd > 0.5 * h(p)]
    nf = len(fixed)
    reg_b = region.boundary
    for it in range(iters if len(p) else 0):
        P = np.vstack([fixed, p])
        T = Delaunay(P).simplices
        c = P[T].mean(1)
        T = T[shapely.contains_xy(region.buffer(1e-12), c[:, 0], c[:, 1])]
        bars = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]]), axis=1), axis=0)
        v = P[bars[:, 0]] - P[bars[:, 1]]
        L = np.hypot(v[:, 0], v[:, 1])
        hb_ = h(0.5 * (P[bars[:, 0]] + P[bars[:, 1]]))
        L0 = hb_ * 1.2 * np.sqrt((L ** 2).sum() / (hb_ ** 2).sum())
        Fb = np.maximum(L0 - L, 0)[:, None] * v / np.maximum(L, 1e-300)[:, None]
        F = np.zeros_like(P)
        np.add.at(F, bars[:, 0], Fb)
        np.add.at(F, bars[:, 1], -Fb)
        move = 0.2 * F[nf:]
        p = p + move
        out = ~shapely.contains_xy(region, p[:, 0], p[:, 1])        # project back onto the boundary
        if out.any():
            q = shapely.points(p[out])
            near = shapely.shortest_line(reg_b, q)
            p[out] = np.array([np.asarray(l.coords)[0] for l in near])
        if np.abs(move).max() < 1e-3 * hmin:
            break
    if len(p) and len(fixed):                                       # drop free nodes that crowd fixed ones
        dd, _ = cKDTree(fixed).query(p)
        p = p[dd > 0.3 * h(p)]
    allp = np.vstack([fixed, p]) if len(p) else fixed

    # ---- merge coincident nodes, constrained Delaunay of the nodes ----
    snap = max(10 * tol, 1e-4 * dmax)
    tree = cKDTree(allp)
    rep = np.arange(len(allp))
    for i, j in sorted(tree.query_pairs(snap)):
        a, b = rep[i], rep[j]
        while rep[a] != a:
            a = rep[a]
        while rep[b] != b:
            b = rep[b]
        if a != b:
            rep[max(a, b)] = min(a, b)
    for i in range(len(rep)):
        r = rep[i]
        while rep[r] != r:
            r = rep[r]
        rep[i] = r
    uniq, inv = np.unique(rep, return_inverse=True)
    verts = allp[uniq]
    segs, off = set(), 0
    for ch in segs_pts:
        ids = inv[off:off + len(ch)]
        off += len(ch)
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                segs.add((min(a, b), max(a, b)))
    via_ids = list(inv[off:off + len(vpts)])
    org = verts.mean(0)
    T = tr.triangulate(dict(vertices=(verts - org) / dmax, segments=np.array(sorted(segs))), 'pQ')
    nodes, tris = T['vertices'] * dmax + org, T['triangles']
    cent = nodes[tris].mean(1)
    tris = tris[shapely.contains_xy(geom.buffer(tol), cent[:, 0], cent[:, 1])]
    used = np.unique(tris)
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    nodes, tris = nodes[used], remap[tris]
    via_ids = [int(remap[i]) if i < len(remap) and remap[i] >= 0 else -1 for i in via_ids]
    V = nodes[tris]
    ar = (V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0])
    tris[ar < 0] = tris[ar < 0][:, [0, 2, 1]]
    return nodes, tris[np.abs(ar) > 1e-30], via_ids
