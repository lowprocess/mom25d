"""Hybrid 1/R integration compared with its analytic reference."""
import numpy as np
import pytest
from shapely.geometry import box

from mom25d.base import RWG, combine
from mom25d.geo import mesh_polygons
from mom25d.static import static_moments
from mom25d.sing import offplane_pairs


def test_hybrid_reciprocity_accuracy_and_translation():
    nodes, tris, _ = mesh_polygons(box(0, 0, .01, .004), .001, .001)
    z = np.zeros(len(tris))
    local, *m = static_moments(nodes, tris, z)
    local_ref, *ref = static_moments(nodes, tris, z, near=np.inf)
    def inductive(nd, moments):
        m0, mx, my, mrr = moments
        return combine(RWG(nd, tris), mrr, mx, my, m0)
    L, Lref = inductive(local, m), inductive(local_ref, ref)
    assert L == pytest.approx(L.T, abs=1e-20)
    assert np.linalg.norm(L - Lref) / np.linalg.norm(Lref) < 2e-4
    assert np.linalg.eigvalsh(L).min() > 0
    shifted, *shift_m = static_moments(nodes + [10, -20], tris, z)
    assert inductive(shifted, shift_m) == pytest.approx(L, rel=2e-7, abs=1e-18)
    _, scalar = static_moments(nodes, tris, z, scalar_only=True)
    assert scalar == pytest.approx(m[0], rel=1e-12)


def test_refined_self_integral_converges():
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    tris = np.array([[0, 1, 2]])
    vals = [offplane_pairs(nodes, tris, [0], [0], 0, outer_levels=k)[0][0] for k in (0, 1, 2)]
    assert abs(vals[1] - vals[2]) < abs(vals[0] - vals[2]) / 3
    assert vals[1] == pytest.approx(vals[2], rel=.01)


def test_vector_offsets_equal_scalar_offsets():
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    tris = np.array([[0, 1, 2]])
    vector = offplane_pairs(nodes, tris, [0, 0], [0, 0], [0, .1])
    for j, offset in enumerate([0, .1]):
        scalar = offplane_pairs(nodes, tris, [0], [0], offset)
        for v, s in zip(vector, scalar):
            assert v[j] == pytest.approx(s[0])
