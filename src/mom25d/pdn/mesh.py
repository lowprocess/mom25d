"""
The copper of a net as triangle meshes, one per layer (mom25d's own mesher, geo.mesh_polygons), and the linear
(P1) finite-element matrices on them: what the DC drop (dc.py) and the plane-pair cavity (cavity.py) solve on.

    sheets = net_sheets(board, '+1V0', points=[(x, y, r), ...], dmax=1.0)   # {layer: Sheet}, board.json units (mm)
    K, M = stiffness(sh.nodes, sh.tris), mass(sh.nodes, sh.tris)

A point (a via, a pad) is a vertex of the mesh with an octagon of its radius around it (the via fan of
mesh_polygons); Sheet.at(i) is the vertex ids of point i on that sheet: its centre and its ring.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import shape
from shapely.ops import unary_union
from scipy.sparse import coo_matrix

from ..geo import mesh_polygons


@dataclass
class Sheet:
    layer: str
    z: float                    # mm
    t: float                    # copper thickness, mm
    geom: object                # shapely, mm
    nodes: np.ndarray           # (N, 2) mm
    tris: np.ndarray            # (M, 3)
    points: list                # [(x, y, r)] mm: the points asked for that lie on this copper
    rings: list                 # vertex ids of each of them (centre first), [] where it is not on this copper
    key: list                   # the index in the caller's list of each of `points`

    def at(self, i):
        """The vertex ids of the caller's point i on this sheet ([]: not on it)."""
        return self.rings[self.key.index(i)] if i in self.key else []


def copper_of(board, net, layer):
    """The copper of `net` on `layer` (board.json's dict), one shapely geometry in mm, or None."""
    gs = [shape(c['polygon']) for c in board['copper'].get(layer, []) if c.get('net') == net]
    g = unary_union(gs).buffer(0) if gs else None
    return None if g is None or g.is_empty else g


def stack_z(board):
    """{layer: (z, t)} of the copper layers, mm."""
    return {c['name']: (c['z'], c['t']) for c in board['stack']['copper']}


def mesh_sheet(geom, points=(), dmax=1.0, w_ref=None, coarsen=None):
    """Mesh one piece of copper (mm): (nodes mm, tris, [vertex ids of each point: centre + ring]) for the points
    (x, y, r) inside it, None where one is not. coarsen: mesh_polygons' (cells grow away from the features to coarsen x
    dmax)."""
    xy = np.asarray([(x, y) for x, y, _ in points], float).reshape(-1, 2)
    hit = shapely.contains_xy(geom.buffer(1e-6), xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, bool)
    inside = list(np.flatnonzero(hit))                     # the copper buffered once: not per point
    vias = [(points[i][0] * 1e-3, points[i][1] * 1e-3, points[i][2] * 1e-3) for i in inside]
    g = scale(geom, 1e-3, 1e-3, origin=(0, 0))
    nodes, tris, vid = mesh_polygons(g, dmax * 1e-3, (w_ref or dmax) * 1e-3, ncross=1, vias=vias, coarsen=coarsen)
    nodes = nodes * 1e3
    rings = [None] * len(points)
    for i, v in zip(inside, vid):
        x, y, r = points[i]
        d = np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)
        ids = np.flatnonzero(d <= r * 1.02 + 1e-9)
        c = int(np.argmin(d))
        rings[i] = [c] + [int(k) for k in ids if k != c]
    return nodes, tris, rings


def net_sheets(board, net, points=(), dmax=1.0, layers=None, coarsen=None):
    """{layer: Sheet} of the copper of `net` on each copper layer (or `layers`), every point (x, y, r) of `points`
    that lies on a layer's copper made a vertex with its ring."""
    zt = stack_z(board)
    out = {}
    for lay in (layers or zt):
        g = copper_of(board, net, lay)
        if g is None:
            continue
        nodes, tris, rings = mesh_sheet(g, points, dmax, coarsen=coarsen)
        key = [i for i, r in enumerate(rings) if r is not None]
        out[lay] = Sheet(lay, zt[lay][0], zt[lay][1], g, nodes, tris, [points[i] for i in key],
                         [rings[i] for i in key], key)
    return out


def _grads(nodes, tris):
    """Areas (M,) and the gradients of the three hat functions of each triangle (M, 3, 2)."""
    P = nodes[tris]                                         # (M, 3, 2)
    d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    # grad of hat i: perpendicular of the opposite edge / (2 area)
    e = np.stack([P[:, 2] - P[:, 1], P[:, 0] - P[:, 2], P[:, 1] - P[:, 0]], 1)
    g = np.stack([-e[..., 1], e[..., 0]], -1) / (2 * area[:, None, None])
    return np.abs(area), g * np.sign(area)[:, None, None]


def stiffness(nodes, tris, coef=1.0):
    """The P1 stiffness matrix: sum over triangles of coef * area * grad(phi_i) . grad(phi_j). coef: a scalar or
    one per triangle (a sheet conductance, S per square, makes it the sheet's conductance matrix)."""
    area, g = _grads(nodes, tris)
    k = np.einsum('mia,mja->mij', g, g) * (area * np.broadcast_to(coef, area.shape))[:, None, None]
    r = np.repeat(tris, 3, axis=1).ravel()
    c = np.tile(tris, (1, 3)).ravel()
    n = len(nodes)
    return coo_matrix((k.ravel(), (r, c)), shape=(n, n)).tocsr()


def mass(nodes, tris, lumped=True):
    """The P1 mass matrix (lumped: a third of each triangle's area on each of its vertices, diagonal)."""
    area, _ = _grads(nodes, tris)
    n = len(nodes)
    if lumped:
        return coo_matrix((np.repeat(area / 3, 3), (tris.ravel(), tris.ravel())), shape=(n, n)).tocsr()
    loc = (np.ones((3, 3)) + np.eye(3)) / 12
    k = loc[None] * area[:, None, None]
    r = np.repeat(tris, 3, axis=1).ravel()
    c = np.tile(tris, (1, 3)).ravel()
    return coo_matrix((k.ravel(), (r, c)), shape=(n, n)).tocsr()


def gradient(nodes, tris, v):
    """The gradient of a P1 field v (one value per node) on each triangle, (M, 2) per unit of the node units."""
    _, g = _grads(nodes, tris)
    return np.einsum('mia,mi->ma', g, v[tris])
