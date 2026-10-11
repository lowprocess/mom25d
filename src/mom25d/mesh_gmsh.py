"""
mom25d.mesh_gmsh - the polygon mesher of geo.mesh_polygons on gmsh (pip install gmsh), same contract:

    nodes, tris, via_ids = mesh_polygons_gmsh(geom, dmax, w_ref, lines=..., vias=..., fine=..., dmax_fine=...)

The geometry goes through gmsh's OpenCASCADE kernel: the metal polygons are fragmented with the port cut
lines, the via octagons (+ their centre points) and the fine regions, so all of them are conformal in the
mesh (RWG edges on the cuts, a via vertex with its fan). The edge refinement - where the surface current
is singular across the conductor edge and smooth along it - is one of

  mode 'layers'  (default) a boundary layer along the conductor edges: thin cells across the edge, long
                 along it, graded like geo.edge_offsets (first cell w_ref (1 - cos(pi / ncross)) / 2),
                 Frontal-Delaunay triangles inside with size dmax
  mode 'rings'   the same grading built geometrically, as geo.mesh_polygons does: inward offset rings of the
                 metal (geo.edge_offsets) embedded as curves, cut conformally with everything else by the
                 fragment; robust where a boundary layer is not (vias on pads inside the layer band). 'layers'
                 falls back to it when gmsh cannot recover the boundary layer
  mode 'graded'  isotropic: size growing from h_edge at the edge to dmax over w_ref (Distance + Threshold
                 fields): smoother but needs more unknowns for the same edge resolution

Away from the edges the size is dmax (dmax_fine inside `fine`), with gmsh's own smoothing of the
transition; corners and narrow necks get fans instead of the offset rings' slivers.
"""
import numpy as np
from shapely.geometry import Polygon, LineString, Point

SCALE = 1e3                 # gmsh works in mm here (its tolerances are absolute)


def _polys(g):
    return [p for p in ([g] if isinstance(g, Polygon) else getattr(g, 'geoms', [])) if isinstance(p, Polygon) and not p.is_empty]


def _ring_coords(ring, tol):
    c = np.asarray(ring.coords)[:-1]
    keep = [0]
    for i in range(1, len(c)):
        if np.hypot(*(c[i] - c[keep[-1]])) > tol:
            keep.append(i)
    c = c[keep]
    if len(c) > 2 and np.hypot(*(c[0] - c[-1])) <= tol:
        c = c[:-1]
    return c


def mesh_polygons_gmsh(geom, dmax, w_ref, ncross=6, grade=1.0, lines=(), vias=(), amax=None, max_offset_frac=0.45,
                       tol=1e-9, fine=None, dmax_fine=None, mode='layers', algorithm=6, verbose=False):
    kw = dict(lines=lines, vias=vias, max_offset_frac=max_offset_frac, tol=tol, fine=fine, dmax_fine=dmax_fine,
              algorithm=algorithm, verbose=verbose)
    if mode != 'layers':
        return _mesh(geom, dmax, w_ref, ncross, grade, mode=mode, **kw)
    try:
        return _mesh(geom, dmax, w_ref, ncross, grade, mode='layers', **kw)
    except Exception:                                               # e.g. 'Edge not recovered' at a via fan
        return _mesh(geom, dmax, w_ref, ncross, grade, mode='rings', **kw)


def _mesh(geom, dmax, w_ref, ncross, grade, lines, vias, max_offset_frac, tol, fine, dmax_fine, mode, algorithm,
          verbose):
    import gmsh
    from .geo import edge_offsets
    if geom.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), int), []
    S = SCALE
    ttol = max(tol, 1e-4 * dmax) * S
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize(interruptible=False)
    gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
    gmsh.model.add('mom25d')
    try:
        occ = gmsh.model.occ
        occ_tol = ttol / 10
        gmsh.option.setNumber('Geometry.Tolerance', occ_tol)
        gmsh.option.setNumber('Geometry.ToleranceBoolean', occ_tol)
        surfs = []
        for poly in _polys(geom):
            loops = []
            for ring in [poly.exterior] + list(poly.interiors):
                c = _ring_coords(ring, ttol / S) * S
                if len(c) < 3:
                    continue
                pts = [occ.addPoint(x, y, 0) for x, y in c]
                ls = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                loops.append(occ.addCurveLoop(ls))
            if loops:
                surfs.append(occ.addPlaneSurface(loops))
        tools, via_pts = [], []
        for p0, p1 in lines:                                       # port gap / cut lines, clipped to the metal
            seg = LineString([p0, p1]).intersection(geom)
            for q in getattr(seg, 'geoms', [seg]):
                if q.geom_type == 'LineString' and q.length > ttol / S:
                    c = np.asarray(q.coords) * S
                    a, b = occ.addPoint(*c[0], 0), occ.addPoint(*c[-1], 0)
                    tools.append((1, occ.addLine(a, b)))
        for (x, y, r) in vias:
            via_pts.append(occ.addPoint(x * S, y * S, 0))
            tools.append((0, via_pts[-1]))
            if r > 0:                                              # octagon of the via radius (the fan)
                ring = LineString(Point(x, y).buffer(r, quad_segs=2).exterior.coords).intersection(geom)
                for q in getattr(ring, 'geoms', [ring]):
                    if q.geom_type == 'LineString' and q.length > ttol / S:
                        c = np.asarray(q.coords) * S
                        p = [occ.addPoint(u, v, 0) for u, v in c]
                        tools += [(1, occ.addLine(p[i], p[i + 1])) for i in range(len(p) - 1)]
        if mode == 'rings':                                        # inward offset rings: the edge grading
            for dd in [d for d in edge_offsets(w_ref, ncross, grade) if d < max_offset_frac * w_ref]:
                inner = geom.buffer(-dd, join_style='mitre', mitre_limit=2.0)
                for q in _polys(inner):
                    for ring in [q.exterior] + list(q.interiors):
                        c = _ring_coords(ring, ttol / S) * S
                        if len(c) < 3:
                            continue
                        p = [occ.addPoint(x, y, 0) for x, y in c]
                        tools += [(1, occ.addLine(p[i], p[(i + 1) % len(p)])) for i in range(len(p))]
        fine_tags = []
        if fine is not None and dmax_fine:
            fg = fine.intersection(geom).buffer(0)
            for q in _polys(fg):
                if q.area > (2 * dmax_fine) ** 2:
                    c = _ring_coords(q.exterior, ttol / S) * S
                    pts = [occ.addPoint(x, y, 0) for x, y in c]
                    ls = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                    fine_tags.append(occ.addPlaneSurface([occ.addCurveLoop(ls)]))
            tools += [(2, t) for t in fine_tags]
        obj = [(2, s) for s in surfs]
        _, omap = occ.fragment(obj, tools)
        occ.synchronize()
        final = sorted({t for d, t in gmsh.model.getEntities(2)})
        fine_final = sorted({t for m in omap[len(obj) + len(tools) - len(fine_tags):] for d, t in m if d == 2}) \
            if fine_tags else []
        # conductor edges: the boundary of the union of the surfaces (not the fine-region interfaces)
        edges = sorted({abs(t) for d, t in gmsh.model.getBoundary([(2, s) for s in final], combined=True, oriented=False)
                        if d == 1})
        F = gmsh.model.mesh.field
        size_fields = []
        h_out = dmax * S
        if mode == 'graded':
            offs = edge_offsets(w_ref, ncross, grade)
            # isotropic cells pay the edge size along the edge too: w_ref / ncross, not the first offset ring
            h_edge = max(w_ref / ncross * S, 1e-3 * h_out)
            dist = F.add('Distance')
            F.setNumbers(dist, 'CurvesList', edges)
            F.setNumber(dist, 'Sampling', 200)
            th = F.add('Threshold')
            F.setNumber(th, 'InField', dist)
            F.setNumber(th, 'SizeMin', h_edge)
            F.setNumber(th, 'SizeMax', h_out)
            F.setNumber(th, 'DistMin', 0.0)
            F.setNumber(th, 'DistMax', max(w_ref * S, 3 * h_edge))
            size_fields.append(th)
        elif mode == 'layers':
            offs = [d for d in edge_offsets(w_ref, ncross, grade) if d < max_offset_frac * w_ref]
            if offs:
                bl = F.add('BoundaryLayer')
                F.setNumbers(bl, 'CurvesList', edges)
                F.setNumber(bl, 'Size', offs[0] * S)
                F.setNumber(bl, 'Ratio', (offs[1] - offs[0]) / offs[0] if len(offs) > 1 else 1.0)
                F.setNumber(bl, 'Thickness', offs[-1] * S * 1.0001)
                F.setNumber(bl, 'SizeFar', h_out)
                F.setNumber(bl, 'Quads', 0)
                corners = sorted({abs(t) for d, t in gmsh.model.getBoundary([(1, e) for e in edges], oriented=False)
                                  if d == 0})
                F.setNumbers(bl, 'FanPointsList', corners)
                F.setAsBoundaryLayer(bl)
        if fine_final:
            cf = F.add('Constant')
            F.setNumber(cf, 'VIn', dmax_fine * S)
            F.setNumber(cf, 'VOut', h_out)
            F.setNumbers(cf, 'SurfacesList', fine_final)
            size_fields.append(cf)
        cst = F.add('MathEval')
        F.setString(cst, 'F', f"{h_out}")
        size_fields.append(cst)
        mn = F.add('Min')
        F.setNumbers(mn, 'FieldsList', size_fields)
        F.setAsBackgroundMesh(mn)
        for k, v in (('Mesh.MeshSizeExtendFromBoundary', 0), ('Mesh.MeshSizeFromPoints', 0),
                     ('Mesh.MeshSizeFromCurvature', 0), ('Mesh.Algorithm', algorithm),
                     ('Mesh.MeshSizeMax', h_out)):
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        nodes = xyz.reshape(-1, 3)[:, :2] / S
        idx = {int(t): i for i, t in enumerate(tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tri = np.concatenate([np.asarray(en, int).reshape(-1, 3) for et, en in zip(etypes, enodes) if et == 2]) \
            if len(etypes) else np.zeros((0, 3), int)
        tris = np.vectorize(idx.get)(tri) if len(tri) else tri
        via_ids = []
        for p in via_pts:
            # the via point may have been merged into another entity by the fragment: find it by position
            vt, vx, _ = gmsh.model.mesh.getNodes(0, p) if (0, p) in gmsh.model.getEntities(0) else ([], [], None)
            via_ids.append(idx[int(vt[0])] if len(vt) else -1)
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()
    for i, v in enumerate(via_ids):                                 # by position where the tag was lost
        if v < 0:
            x, y, _ = vias[i]
            j = int(np.argmin(np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)))
            via_ids[i] = j if np.hypot(nodes[j, 0] - x, nodes[j, 1] - y) < 1e-3 * dmax else -1
    used = np.unique(tris)
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    nodes, tris = nodes[used], remap[tris]
    via_ids = [int(remap[i]) if i >= 0 and remap[i] >= 0 else -1 for i in via_ids]
    V = nodes[tris]
    ar = (V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0])
    tris[ar < 0] = tris[ar < 0][:, [0, 2, 1]]
    return nodes, tris[np.abs(ar) > 1e-30], via_ids


def quality(nodes, tris):
    """Per triangle: minimum angle (deg) and aspect ratio (longest edge / height on it)."""
    V = nodes[tris]
    e = np.stack([V[:, 1] - V[:, 0], V[:, 2] - V[:, 1], V[:, 0] - V[:, 2]], 1)
    L = np.linalg.norm(e, axis=2)
    area = 0.5 * np.abs(e[:, 0, 0] * e[:, 2, 1] - e[:, 0, 1] * e[:, 2, 0])
    ang = []
    for i in range(3):
        a, b = -e[:, i - 1], e[:, i]
        ang.append(np.degrees(np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1, 1))))
    lmax = L.max(1)
    return np.min(ang, axis=0), lmax / (2 * area / lmax)
