"""Free-space electrostatics, displacement current, and low-frequency PEEC."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from mom25d.base import EPS0
from mom25d.mqs import MQS


def plates(h=0.5, size=10, dmax=2, whole=False, er=1):
    copper = {lay: [dict(net=lay, polygon=mapping(box(0, 0, size, size)))] for lay in ('B', 'F')}
    b = dict(stack=dict(layers=[], copper=[dict(name='B', z=0, t=.035), dict(name='F', z=h, t=.035)]),
             copper=copper, vias=[])
    pad = box(0, 0, size, size) if whole else box(0, 0, 1, 1)
    q = MQS(b, {'B', 'F'}, terminals={l: [(l, pad)] for l in copper}, dmax=dmax,
            capacitance=True, er=er, log=lambda *_: None)
    q.source('I', 'F', 'B', part_h=0)
    return q


def test_disconnected_plates_carry_displacement_current():
    q = plates()
    r = q.solve([1e2, 1e5, 1e7])
    C, labels, terminals = q.capacitance_matrix()
    # Equal/opposite conductor charges, floating pair; infinity is no terminal.
    drive = np.zeros(len(C)); drive[terminals['B']] = 1; drive[terminals['F']] = -1
    expected = 1 / (drive @ np.linalg.solve(C, drive))
    measured = -1 / (2 * np.pi * r.f[:2] * r.z['I'][:2].imag)
    assert measured == pytest.approx(expected, rel=2e-5)
    assert np.all(r.z['I'].real >= -1e-6)
    # Finite plates have positive fringe capacitance, above eps A / h.
    parallel = EPS0 * .01**2 / .0005
    assert parallel < expected < 1.5 * parallel
    assert r.charge.sum(1) == pytest.approx(np.zeros(3), abs=1e-17)
    ends = q.br + [(q.term_node[p['a']], q.term_node[p['b']]) for p in q.parts]
    for fi, freq in enumerate(r.f):
        balance = 2j * np.pi * freq * r.charge[fi]
        for current, (a, b) in zip(r.i[fi], ends):
            balance[a] += current; balance[b] -= current
        assert abs(balance).max() < 1e-9
        assert r.potential[fi] == pytest.approx(q._potential @ r.charge[fi])


def test_uniform_dielectric_and_parallel_plate_electric_field():
    q = plates(h=.1, whole=True, er=4)
    r = q.solve([1e3, 1e6])
    pts = np.array([[.005, .005, .00005], [.005, .005, .02]])
    E = q.e_field(r, [0, 1], pts)
    rho = r.charge[0, q.term_node['F']] / .01**2
    assert E[0, 2, 0] == pytest.approx(-rho / (4 * EPS0), rel=.01)
    assert abs(E[0, :2]).max() < 1e-10 * abs(E[0, 2]).max()
    assert np.abs(E[1]).max() < .01 * np.abs(E[0]).max()
    assert q.e_field(r, 0, pts) == pytest.approx(E[:, :, 0])
    air = plates(h=.1, whole=True)
    ra = air.solve([1e3])
    assert r.z['I'][0] == pytest.approx(ra.z['I'][0] / 4, rel=1e-10)


def test_electric_field_off_axis_matches_coulomb_quadrature():
    from mom25d.base import _BAR6, _W6
    q = plates(whole=True)
    r = q.solve([1e3])
    points = np.array([[.025, .016, .02], [.025, .016, 0.]])
    obs = np.einsum('qk,tkc->tqc', _BAR6, q.nodes[q.tris])
    xyz = np.concatenate([obs, np.broadcast_to(q.z_of[:, None, None], obs.shape[:-1] + (1,))], axis=-1)
    area = np.bincount(q.node_of, weights=q.rwg.area)
    density = r.charge[0, q.node_of] / area[q.node_of]
    delta = points[:, None, None] - xyz[None]
    kernel = delta / np.linalg.norm(delta, axis=-1)[..., None]**3
    expected = np.einsum('ptqc,t,q,t->pc', kernel, q.rwg.area, _W6, density) / (4*np.pi*EPS0)
    assert q.e_field(r, 0, points) == pytest.approx(expected, rel=2e-6)


def test_peec_handles_passive_capacitor_between_islands():
    q = plates(whole=True)
    q.element('decap', 'F', 'B', C=1e-9, ESR=.01, ESL=.5e-9)
    r = q.solve([1e2, 1e6])
    C, _, _ = q.capacitance_matrix()
    e = np.array([1., -1.])
    cgeom = 1 / (e @ np.linalg.solve(C, e))
    w = 2 * np.pi * r.f
    zpart = .01 + 1j*w*.5e-9 + 1/(1j*w*1e-9)
    expected = 1 / (1/zpart + 1j*w*cgeom)
    assert r.z['I'] == pytest.approx(expected, rel=1e-7)


def test_peec_approaches_mqs_for_a_closed_copper_path():
    board = dict(stack=dict(layers=[], copper=[dict(name='F', z=0, t=.035)]),
                 copper={'F': [dict(net='N', polygon=mapping(box(0, 0, 5, 2)))]}, vias=[])
    terms = {'a': [('F', box(0, 0, 1, 2))], 'b': [('F', box(4, 0, 5, 2))]}
    zs = []
    for cap in (False, True):
        q = MQS(board, 'N', terminals=terms, dmax=.7, capacitance=cap, log=lambda *_: None)
        q.source('I', 'a', 'b', part_h=0)
        result = q.solve([1e-3, 1e2, 1e5])
        zs.append(result.z['I'])
        if cap:
            assert result.potential == pytest.approx(result.charge @ q._potential.T, rel=1e-8, abs=1e-12)
    assert zs[1] == pytest.approx(zs[0], rel=1e-5)


@pytest.mark.parametrize('er', [0, -1, np.inf, np.nan])
def test_invalid_permittivity(er):
    with pytest.raises(ValueError, match='er'):
        plates(er=er)


def test_capacitance_only_skips_magnetic_fill(monkeypatch):
    q = plates(whole=True)
    def fail():
        raise AssertionError('magnetic assembly is unnecessary')
    monkeypatch.setattr(q, '_edge_matrices', fail)
    C, _, _ = q.capacitance_matrix()
    assert C.shape == (2, 2)
    assert q._copper is None and q._mats is None


def test_preparation_factor_reuse_and_part_invalidation():
    q = plates(dmax=3)
    a = q.solve([1e3, 1e6])
    prepared = q._peec
    n = prepared['factorizations']
    b = q.solve([1e3, 1e6], amps={'I': 2j})
    assert q._peec is prepared and prepared['factorizations'] == n
    assert b.i == pytest.approx(2j*a.i)
    assert b.charge == pytest.approx(2j*a.charge)
    assert b.z['I'] == pytest.approx(a.z['I'])
    assert prepared['factor_bytes'] <= q.factor_cache_bytes
    q.element('C', 'F', 'B', C=1e-9)
    assert q._peec is None
    q.solve([1e3])
    assert q._peec is not prepared


def test_closed_multiple_sources_do_not_create_common_mode_charge():
    board = dict(stack=dict(layers=[], copper=[dict(name='F', z=0, t=.035)]),
                 copper={'F': [dict(net='N', polygon=mapping(box(0, 0, 6, 2)))]}, vias=[])
    terms = {n: [('F', box(x, 0, x+1, 2))] for n, x in [('a', 0), ('b', 5), ('c', 2)]}
    q = MQS(board, 'N', terminals=terms, dmax=.7, capacitance=True, log=lambda *_: None)
    q.source('I', 'a', 'b', part_h=0)
    q.source('J', 'c', 'b', part_h=0)
    a = q.solve([1e-3], amps={'I': 1, 'J': 0})
    b = q.solve([1e-3], amps={'I': 0, 'J': 1})
    combined = q.solve([1e-3], amps={'I': .1+.2j, 'J': .2+.3j})
    expected = (.1+.2j)*a.charge + (.2+.3j)*b.charge
    assert abs(combined.charge-expected).max() < 1e-10*abs(expected).max()
    assert abs(combined.charge.sum()) < 1e-12*abs(combined.charge).max()
