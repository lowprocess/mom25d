"""The mesher's triangulation: the compiled core's cdt_triangulate (CDT) and cdtmesh.py's points inside the copper."""
import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from mom25d import cdtmesh, fast, geo

pytestmark = pytest.mark.skipif(fast._built is None or not hasattr(fast._built, 'cdt_triangulate'),
                                reason="the compiled core without cdt_triangulate")


def _areas(nodes, tris):
    P = nodes[tris]
    return 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                  - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))


def _min_angles(nodes, tris):
    P = nodes[tris]
    e = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]], 1)
    cos = [(-e[:, i - 1] * e[:, i]).sum(1) / (np.linalg.norm(e[:, i - 1], axis=1) * np.linalg.norm(e[:, i], axis=1))
           for i in range(3)]
    return np.degrees(np.arccos(np.clip(np.max(cos, 0), -1, 1)))


def test_core_keeps_points_and_segments():
    core = fast._built
    V = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.25, 0.5], [0.75, 0.5], [0.5, 0.2]], float)
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]], np.int32)
    v, t = core.cdt_triangulate(V, E, 1e-12)
    assert np.allclose(v[:len(V)], V)                         # the input points keep their indices
    a = _areas(v, t)
    assert (a > 0).all() and abs(a.sum() - 1.0) < 1e-12        # counter-clockwise, the outline's inside
    edges = {tuple(sorted(e)) for tri in t for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    assert (4, 5) in edges                                    # the constraint is an edge
    v2, t2 = core.cdt_triangulate(V[:4], np.array([[0, 2], [1, 3], [0, 1], [1, 2], [2, 3], [3, 0]], np.int32), 1e-12)
    assert len(v2) == 5 and np.allclose(v2[4], [0.5, 0.5])    # crossing segments split where they cross
    v3, t3 = core.cdt_triangulate(np.array([[0, 0], [4, 0], [4, 1], [0, 1]], float), E[:4], 1e-12, 25.0)
    assert _min_angles(v3, t3).min() >= 25.0 - 1e-9 and abs(_areas(v3, t3).sum() - 4.0) < 1e-12


def test_no_slivers_outside_a_straight_outline():
    """Points a hair off a straight outline: the hull's slivers outside it are not kept (they once were, with their
    centroids on the edge: a zero-area triangle in pcbqs's wpthb PDN)."""
    x = np.linspace(0, 10, 41)
    top = np.column_stack([x, 1 + 1e-9 * np.sin(7 * x)])
    V = np.vstack([[[0, 0], [10, 0]], top[::-1]])
    n = len(V)
    E = np.array([(k, (k + 1) % n) for k in range(n)], np.int32)
    v, t = fast._built.cdt_triangulate(V, E, 1e-10)
    a = _areas(v, t)
    assert (np.abs(a) > 1e-6).all() and abs(a.sum() - 10.0) < 1e-6


@pytest.mark.parametrize('coarsen', [None, 2.0])
def test_mesh_polygons_meets_the_limits(coarsen):
    """A plane with a hole and a slot: the copper covered exactly, every triangle inside it and within its area
    limit (without coarsening: 0.5 dmax^2, to the smoothing's slack), the count near what the limit asks for."""
    g = box(0, 0, 12e-3, 8e-3).difference(Point(4e-3, 4e-3).buffer(1e-3)).difference(box(7e-3, 2e-3, 7.4e-3, 6e-3))
    dmax = 0.5e-3
    nodes, tris, _ = geo.mesh_polygons(g, dmax, 0.5e-3, 4, coarsen=coarsen)
    a = _areas(nodes, tris)
    assert (a > 0).all()
    assert abs(a.sum() - g.area) < 1e-7 * g.area                 # (the snapping of near vertices)
    c = nodes[tris].mean(1)
    assert shapely.contains_xy(g.buffer(1e-9), c[:, 0], c[:, 1]).all()
    if coarsen is None:
        assert a.max() <= 0.5 * dmax ** 2 * (1 + cdtmesh.POST_TOL + 1e-6)      # (the smoothing's slack)
        assert g.area / (0.5 * dmax ** 2) < len(tris) < 4 * g.area / (0.5 * dmax ** 2)   # (the edge rings: more)


def test_the_mesher_needs_the_core_not_the_kernels(monkeypatch):
    """MOM25D_NO_CORE (use_core('python')) sets the kernels aside; the mesher still triangulates with the core."""
    prev = fast.use_core('python')
    try:
        nodes, tris, _ = geo.mesh_polygons(box(0, 0, 4e-3, 2e-3), 0.5e-3, 0.5e-3, 2)
        assert len(tris) > 0
    finally:
        fast.set_core(prev)
    monkeypatch.setattr(fast, '_built', None)
    with pytest.raises(RuntimeError, match='compiled core'):
        geo.mesh_polygons(box(0, 0, 4e-3, 2e-3), 0.5e-3, 0.5e-3, 2)


def test_the_interior_is_equilateral_cells():
    """A wide rectangle: its middle is the hexagonal seed's equilateral triangles, and the smoothing keeps them."""
    W, H, dm = 40.0, 20.0, 1.0
    xs, ys = np.linspace(0, W, 41), np.linspace(0, H, 21)
    ring = np.vstack([np.column_stack([xs, 0 * xs]), np.column_stack([W + 0 * ys[1:], ys[1:]]),
                      np.column_stack([xs[::-1][1:], H + 0 * xs[1:]]), np.column_stack([0 * ys[::-1][1:-1], ys[::-1][1:-1]])])
    n = len(ring)
    segs = np.array([(k, (k + 1) % n) for k in range(n)])
    T = cdtmesh.mesh(ring, segs, lambda C: np.full(len(C), 0.5 * dm ** 2),
                     lambda P: (P[:, 0] > 0) & (P[:, 0] < W) & (P[:, 1] > 0) & (P[:, 1] < H), fast._built)
    v, t = T['vertices'], T['triangles']
    a = _areas(v, t)
    assert abs(a.sum() - W * H) < 1e-9 and T['info']['seed'] > 0.8 * (len(v) - n)
    assert (_min_angles(v, t) > 55).mean() > 0.6 and a.max() <= 0.5 * dm ** 2 * (1 + cdtmesh.POST_TOL)
