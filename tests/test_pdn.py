"""The DC drop and the plane-pair PDN impedance against closed forms."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from mom25d.pdn import dc, cavity
from mom25d.base import EPS0


def board(layers, copper, vias=()):
    """A board.json dict: copper {layer: [(net, shapely)]}, layers [(name, z, t)], one slab between."""
    zs = [z for _, z, _ in layers]
    return dict(stack=dict(layers=[dict(name='d', z0=min(zs), z1=max(zs) or 1.0, er=4.0, tand=0.0, kind='diel')],
                           copper=[dict(name=n, z=z, t=t) for n, z, t in layers]),
                copper={lay: [dict(net=n, polygon=mapping(g)) for n, g in items] for lay, items in copper.items()},
                vias=[dict(x=x, y=y, r=r, z0=z0, z1=z1, net=n) for x, y, r, z0, z1, n in vias])


def test_strip_resistance():
    """A 60 x 4 mm strip of 35 um copper, a pad at each end: R = L / (sigma t W) between them, the pads' discs
    and the spreading from them aside (~2% for these)."""
    b = board([('F.Cu', 0.0, 0.035)], {'F.Cu': [('P', box(0, 0, 60, 4))]})
    r = dc.solve(b, 'P', dict(x=3, y=2, r=1.8, layer='F.Cu', V=1.0), [dict(name='L', x=57, y=2, r=1.8, layer='F.Cu', I=1.0)],
                 dmax=0.5, log=lambda *a: None)
    R = 1.0 - r.loads['L']['V']
    R0 = 54e-3 / (5.8e7 * 35e-6 * 4e-3)                     # between the pad centres
    assert R == pytest.approx(R0, rel=0.06)
    assert r.loss == pytest.approx(R * 1.0 ** 2, rel=0.02)   # the copper's loss is the drop times the current


def test_vias_join_layers_and_carry_the_current():
    """Two layers, the source on one and the load on the other, joined by two vias: Kirchhoff holds, and the
    vias carry all of the load's current."""
    cu = {'F.Cu': [('P', box(0, 0, 20, 10))], 'B.Cu': [('P', box(0, 0, 20, 10))]}
    b = board([('B.Cu', 0.0, 0.035), ('F.Cu', 1.6, 0.035)], cu,
              vias=[(10, 3, 0.15, 0.0, 1.6, 'P'), (10, 7, 0.15, 0.0, 1.6, 'P')])
    r = dc.solve(b, 'P', dict(x=2, y=5, r=1, layer='F.Cu', V=1.0), [dict(name='L', x=18, y=5, r=1, layer='B.Cu', I=2.0)],
                 dmax=0.5, log=lambda *a: None)
    assert sum(abs(v['segments'][0][2]) for v in r.vias) == pytest.approx(2.0, rel=1e-6)
    assert r.loss == pytest.approx(2.0 * (1.0 - r.loads['L']['V']), rel=1e-6)
    # one barrel: rho dz / area, 1.6 mm of 25 um plating on a 0.3 mm hole ~ 1.3 mOhm
    g = dc._barrel_g(0.15, 1.6, 0.025, 5.8e7)
    assert 1 / g == pytest.approx(1.6e-3 / (5.8e7 * np.pi * (0.15e-3 ** 2 - 0.125e-3 ** 2)))


def test_ground_return_puts_the_current_back():
    b = board([('F.Cu', 0.0, 0.035)], {'F.Cu': [('G', box(0, 0, 30, 10))]})
    r = dc.solve(b, 'G', dict(x=2, y=5, r=1, layer='F.Cu', V=1.0), [dict(name='L', x=28, y=5, r=1, layer='F.Cu', I=1.0)],
                 ground=True, dmax=0.5, log=lambda *a: None)
    assert r.loads['L']['V'] > 0                            # the load's ground rises above the source's 0 V


@pytest.fixture(scope='module')
def plane():
    a, bb, d = 50.0, 30.0, 0.2
    ports = [dict(name='A', x=10.0, y=8.0, r=0.5), dict(name='B', x=38.0, y=21.0, r=0.5)]
    pp = cavity.PlanePair(geom=box(0, 0, a, bb), d=d, er=4.4, tand=0.0, ports=ports, dmax=0.8, log=lambda *x: None)
    return pp, (a, bb, d)


def test_plane_pair_is_its_capacitance_at_low_frequency(plane):
    pp, (a, b, d) = plane
    C = EPS0 * 4.4 * a * b * 1e-6 / (d * 1e-3)
    assert pp.C == pytest.approx(C, rel=1e-6)
    z = pp.z([1e6], bare=True)[0]
    assert abs(z[0, 0]) == pytest.approx(1 / (2 * np.pi * 1e6 * C), rel=0.02)


def test_plane_pair_matches_the_cavity_double_sum(plane):
    """Below, between and above the first modes (f10 = c / (2 a sqrt(er)) = 1.43 GHz), the FEM against the
    cavity formula at ports of the same area (a square of the disc's area). The formula's ports carry a uniform
    current, the FEM's are equipotential discs: a self term of a uniform-current port has mu0 d / (8 pi) more
    inductance (10 pH here), constant over frequency, which the self impedance is compared with."""
    pp, (a, b, d) = plane
    f = np.array([0.3e9, 1.0e9, 2.0e9, 3.3e9])
    w = np.sqrt(np.pi) * 0.5
    ref = cavity.rect_cavity_z(a, b, d, 4.4, 0.0, [(10.0, 8.0, w), (38.0, 21.0, w)], f, modes=120)
    z = pp.z(f, bare=True)
    uniform = 1j * 2 * np.pi * f * 4e-7 * np.pi * d * 1e-3 / (8 * np.pi)
    for i in range(len(f)):
        assert abs(z[i, 1, 0]) == pytest.approx(abs(ref[i, 1, 0]), rel=0.05), f[i]
        # 3.3 GHz, between the (2,0) and (2,1) modes where |Z| moves fastest: 6% on this 0.8 mm mesh, 0.5% on 0.4 mm
        assert abs(z[i, 0, 0] + uniform[i]) == pytest.approx(abs(ref[i, 0, 0]), rel=0.08), f[i]


def test_decap_pulls_the_impedance_down_and_resonates_with_its_esl():
    pp = cavity.PlanePair(geom=box(0, 0, 20, 20), d=0.1, er=4.0, ports=[
        dict(name='U', x=10, y=10, r=0.3), dict(name='C', x=12, y=10, r=0.3, C=1e-6, ESR=5e-3, ESL=0.5e-9)],
        dmax=0.6, log=lambda *x: None)
    fsr = 1 / (2 * np.pi * np.sqrt(1e-6 * 0.5e-9))                        # 7.1 MHz
    f = np.logspace(5, 9, 161)
    z = np.abs(pp.z(f, observe=['U'])[:, 0, 0])
    z0 = np.abs(pp.z(f, bare=True, observe=['U'])[:, 0, 0])
    assert (z <= z0 * 1.001).all() or z[f < 1e8].max() < z0[f < 1e8].min()
    k = np.argmin(z[f < 1e8])
    assert f[k] == pytest.approx(fsr, rel=0.15)                         # its series resonance, near the ESR
    assert z[k] == pytest.approx(5e-3, rel=0.6)
