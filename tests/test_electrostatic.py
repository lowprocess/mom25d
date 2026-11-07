"""Dielectric images against the independent spectral transmission-line solver."""
import numpy as np
import pytest

from mom25d.base import EPS0
from mom25d.electrostatic import DielectricStack
from mom25d.multilayer import Layer, LayerStack
from mom25d.mqs import MQS
from test_peec import plates


@pytest.mark.parametrize('o,s', [(0, 0), (0, 3), (1, 2), (2, 2), (3, 1)])
def test_images_match_static_limit_of_spectral_network(o, s):
    layers = [Layer(.0002, 4, .02), Layer(.0003, 2, .01), Layer(.0001, 6, .03)]
    reference = LayerStack(layers, bottom='open')
    images = DielectricStack(reference.z, [1]+[l.er_c for l in layers]+[1], tol=1e-10)
    k = np.geomspace(10, 1e4, 21)
    transform = sum(c*np.exp(-k*d) for d, c in images.images(s, o))/(2*EPS0*k)
    exact = reference.spectral(k.astype(complex), 2*np.pi*.001, o, s)[1]
    assert transform == pytest.approx(exact, rel=3e-8)


def layered_plates(loss=0.):
    p = plates(h=.1, whole=True, dmax=2)
    b = p.board
    b['stack']['layers'] = [dict(z0=0, z1=.05, er=2, tand=loss),
                            dict(z0=.05, z1=.1, er=6, tand=3*loss)]
    q = MQS(b, {'B', 'F'}, terminals=p.terms, dmax=2, capacitance=True,
            dielectric='stack', dielectric_tol=1e-9, log=lambda *_: None)
    q.source('I', 'F', 'B', part_h=0)
    return q


def test_layers_in_series_and_displacement_boundary_condition():
    q = layered_plates()
    r = q.solve([1e3])
    measured = -1/(2*np.pi*1e3*r.z['I'][0].imag)
    parallel = EPS0*.01**2/(.00005/2+.00005/6)
    assert measured == pytest.approx(parallel, rel=.05)
    E = q.e_field(r, 0, np.array([[.005, .005, .000025], [.005, .005, .000075]]))
    assert E[0, 2]*2 == pytest.approx(E[1, 2]*6, rel=.02)
    assert r.charge.sum() == pytest.approx(0, abs=1e-17)
    assert q._potential == pytest.approx(q._potential.T)
    assert np.linalg.eigvalsh(q._potential.real).min() > 0


def test_dielectric_loss_is_passive_and_matches_series_loss():
    q = layered_plates(loss=.01)
    r = q.solve([1e3, 1e6])
    z = r.z['I']
    assert (z.real > 0).all()
    assert z.real/-z.imag == pytest.approx(.015, rel=.1)
    C, _, _ = q.capacitance_matrix()
    assert np.linalg.eigvalsh(-C.imag).min() > 0


def test_uniform_stack_agrees_with_homogeneous_model():
    p = plates(h=.5, whole=True, dmax=2, er=4)
    b = p.board
    b['stack']['layers'] = [dict(z0=0, z1=.5, er=4)]
    b['stack'].update(bottom_er=4, top_er=4)
    q = MQS(b, {'B', 'F'}, terminals=p.terms, dmax=2, capacitance=True,
            dielectric='stack', log=lambda *_: None)
    q.source('I', 'F', 'B', part_h=0)
    a, b = p.solve([1e3]), q.solve([1e3])
    assert b.z['I'] == pytest.approx(a.z['I'], rel=1e-10)
    pts = np.array([[.005, .005, .00025], [.016, .01, .003]])
    assert q.e_field(b, 0, pts) == pytest.approx(p.e_field(a, 0, pts), rel=1e-10, abs=1e-4)


def test_image_sum_quadrature_matches_all_analytic_reference():
    q = layered_plates()
    C, _, _ = q.capacitance_matrix()
    q.near = np.inf
    q._potential = None
    reference, _, _ = q.capacitance_matrix()
    assert np.linalg.norm(C-reference)/np.linalg.norm(reference) < 2e-4
