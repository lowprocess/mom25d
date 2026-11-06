"""The rail tree, the DC drop over it with pads as terminals and series parts, decaps through their copper, and the
PDN over several plane pairs, on a small synthetic board."""
import numpy as np
import pytest
from shapely.geometry import box, mapping, Point

from mom25d.base import MU0, EPS0
from mom25d.pdn.rail import Rail, parse_value
from mom25d.pdn.dc import solve_rail
from mom25d.pdn.stack import RailPDN

W, H = 40.0, 30.0
# F.Cu 0.6 / 0.1 / In1 GND 0.5 / 0.1 / In2 P plane 0.4 / 0.1 / In3 GND 0.3 / 0.3 / B.Cu 0
STACK = dict(layers=[dict(name='d4', z0=0.0, z1=0.3, er=4.0, tand=0.0, kind='diel'),
                     dict(name='d3', z0=0.3, z1=0.4, er=4.0, tand=0.0, kind='diel'),
                     dict(name='d2', z0=0.4, z1=0.5, er=4.0, tand=0.0, kind='diel'),
                     dict(name='d1', z0=0.5, z1=0.6, er=4.0, tand=0.0, kind='diel')],
             copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='In3.Cu', z=0.3, t=0.035),
                     dict(name='In2.Cu', z=0.4, t=0.035), dict(name='In1.Cu', z=0.5, t=0.035),
                     dict(name='F.Cu', z=0.6, t=0.035)], thickness=0.6)


def pad(name, net, x, y, w=1.0, h=1.0, layer='F.Cu'):
    return dict(name=name, net=net, xy=[x, y], layers=[layer], geom=mapping(box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)))


def fp(ref, value, pads, lib='C_0402_1005Metric'):
    return dict(ref=ref, value=value, props={'Value': value}, lib=lib, side='F', dnp=False, exclude_from_sim=False,
                pads=pads, x=0, y=0, rot=0)


def board(trace_decap=True, via_in_pad=False):
    """A 40 x 30 mm board: a buck (U3 driving L1 from SW) to VOUT, a 10 mOhm sense R1 to the plane net P (In2,
    between the In1 and In3 grounds), a bead FB1 to the filtered PF (F.Cu pour, its own C3 and load U2), the feedback
    R2 (10 k) to FBK, decaps C1 (on vias) and C2 (5 mm of 0.5 mm trace to its via), the load U1."""
    cu = {l['name']: [] for l in STACK['copper']}
    plane = box(1, 1, W - 1, H - 1)
    cu['In1.Cu'].append(dict(net='GND', polygon=mapping(plane)))
    cu['In3.Cu'].append(dict(net='GND', polygon=mapping(plane)))
    cu['In2.Cu'].append(dict(net='P', polygon=mapping(plane)))
    F = []                                              # F.Cu: (net, geometry)
    F += [('SW', box(3, 13, 6, 17)), ('VOUT', box(7, 13, 12, 17)), ('P', box(13, 13, 17, 17)),
          ('GND', box(3, 6, 8, 11)), ('FBK', box(7, 20, 9, 21))]
    F += [('P', box(30, 13, 34, 17)), ('GND', box(30, 8, 34, 12))]          # U1's pads
    F += [('P', box(22, 4, 23, 5)), ('P', box(22.25, 5, 22.75, 6.4)), ('GND', box(22, 2, 23, 3)),
          ('GND', box(22.25, 1.1, 22.75, 2))]                                # C1 and its short necks
    F += [('P', box(25, 24, 26, 25)), ('P', box(25.25, 19.5, 25.75, 24.5)), ('GND', box(25, 26, 26, 27))]   # C2 + trace
    F += [('P', box(17, 22, 18, 23)), ('PF', box(19, 21, 24.8, 23)), ('GND', box(19, 26.5, 21, 28))]
    for n, gm in F:
        cu['F.Cu'].append(dict(net=n, polygon=mapping(gm)))
    vias = []
    def via(x, y, net, z0=0.0, z1=0.6, r=0.15):
        vias.append(dict(x=x, y=y, r=r, z0=z0, z1=z1, net=net))
    for x in (14, 15, 16):
        via(x, 14, 'P')                                 # the sense resistor's plane side down to In2
    for x in (31, 33):
        via(x, 14, 'P'); via(x, 10, 'GND')              # U1
    via(22.5, 6.0, 'P'); via(22.5, 1.5, 'GND')        # C1 (a short neck to each via)
    via(25.5, 19.75, 'P'); via(25.5, 26.5, 'GND')     # C2: its via at the end of the trace (ground: in its pad)
    via(17.5, 22.5, 'P')                              # the bead's plane side
    via(20, 27.2, 'GND')
    for x in (4, 6):
        via(x, 8, 'GND')                                # the regulator's ground
    fps = [fp('U3', 'BUCK', [pad('1', 'SW', 4.5, 15), pad('2', 'GND', 5.5, 8.5, 3, 3)]),
           fp('L1', '2.2uH', [pad('1', 'SW', 5.5, 15), pad('2', 'VOUT', 8.0, 15)], lib='L_1210'),
           fp('R1', '0R01', [pad('1', 'VOUT', 11.0, 15), pad('2', 'P', 14.0, 15)], lib='R_0805_2012Metric'),
           fp('R2', '10k', [pad('1', 'VOUT', 8.0, 16.5, 0.5, 0.5), pad('2', 'FBK', 8.0, 20.5, 0.5, 0.5)]),
           fp('FB1', 'BLM18 600R@100MHz', [pad('1', 'P', 17.5, 22.5), pad('2', 'PF', 19.5, 22)], lib='L_0603_1608Metric'),
           fp('C3', '1u', [pad('1', 'PF', 20.0, 22.5), pad('2', 'GND', 20.0, 27.2)]),
           fp('U2', 'LOAD2', [pad('1', 'PF', 24.0, 22), pad('2', 'GND', 20.5, 27.5)]),
           fp('U1', 'LOAD', [pad('1', 'P', 32, 15, 3, 3), pad('2', 'GND', 32, 10, 3, 3)]),
           fp('C1', '100n', [pad('1', 'P', 22.5, 4.5), pad('2', 'GND', 22.5, 2.5)]),
           fp('C2', '10u', [pad('1', 'P', 25.5, 24.5), pad('2', 'GND', 25.5, 26.5)], lib='C_0805_2012Metric')]
    if via_in_pad:
        via(32, 15, 'P')                                # a via in the middle of U1's supply pad
    return dict(stack=STACK, outline=mapping(box(0, 0, W, H)), copper=cu, vias=vias,
                prims=dict(footprints=fps))


def test_values():
    assert parse_value('0R001') == pytest.approx(1e-3) and parse_value('4k7') == 4700
    assert parse_value('1u8') == pytest.approx(1.8e-6) and parse_value('100n') == pytest.approx(1e-7)


def test_rail_tree_follows_sense_resistors_and_beads_not_dividers():
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    assert r.nets == {'P', 'VOUT', 'PF'}
    kinds = {s.part.ref: s.kind for s in r.series}
    assert kinds == {'R1': 'R', 'FB1': 'bead'}
    assert [p.ref for p in r.source_parts] == ['L1'] and {q.net for q in r.source_pads} == {'VOUT'}
    assert {q.part for q in r.ground_source_pads} == {'U3'}
    assert {p.ref for p in r.decaps} == {'C1', 'C2', 'C3'}
    # a 0.5 ohm sense resistor is in the tree at the default 1 ohm, out of it at 0.1
    b = board()
    next(f for f in b['prims']['footprints'] if f['ref'] == 'R1')['props']['Value'] = '0R5'
    assert 'VOUT' in Rail(b, 'P', source='L1').nets
    with pytest.raises(ValueError):                     # VOUT is then not on the tree: L1 feeds nothing
        Rail(b, 'P', max_ohms=0.1)


def test_dc_over_the_tree_drops_the_sense_resistor():
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    res = solve_rail(r, 3.3, dmax=0.8, log=lambda *a: None)
    s = res['series']
    assert s['R1']['I'] == pytest.approx(2.5, rel=1e-4)              # all of the loads' current
    assert s['R1']['drop'] == pytest.approx(2.5 * 0.01, rel=1e-4)    # through its 10 mOhm
    assert s['FB1']['I'] == pytest.approx(0.5, rel=1e-4)
    u1 = res['loads']['U1']
    assert 0.025 < u1['power_drop'] < 0.05                           # the sense R and some copper
    assert u1['ground_rise'] > 0


def test_via_in_pad_is_part_of_the_pad():
    r = Rail(board(via_in_pad=True), 'P', loads={'U1': 2.0, 'U2': 0.5})
    res = solve_rail(r, 3.3, dmax=0.8, log=lambda *a: None)
    v = [q for q in res['power'].vias if (q['x'], q['y']) == (32, 15)][0]
    assert v['I'] > 0.1                                              # it carries a share of U1's current


def test_decap_path_through_its_trace():
    """C2's supply pad is 4.25 mm of 0.5 mm trace from its via: 8.5 squares less the via's ring, 0.1 mm over In1."""
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    pdn = RailPDN(r, dmax=1.0, log=lambda *a: None)
    c2 = pdn.parts['C2']['rail']
    sq = c2.L_path / (MU0 * c2.h * 1e-3)
    assert 6.5 < sq < 9.5 and c2.h == pytest.approx(0.1)            # In1 is 0.1 mm under F.Cu
    assert pdn.parts['C1']['rail'].L_path < c2.L_path / 2
    assert pdn.parts['C3']['rail'].node == 'PF'                      # the filtered net: a lumped node


def test_two_plane_pairs_in_parallel():
    """In2 between the In1 and In3 grounds: two cavities of 0.1 mm, joined; at low frequency the planes alone are
    their two capacitances in parallel."""
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    pdn = RailPDN(r, dmax=1.0, log=lambda *a: None)
    assert {(c.rail_layer, c.ground_layer) for c in pdn.cavities} == {('In2.Cu', 'In1.Cu'), ('In2.Cu', 'In3.Cu')}
    C = sum(EPS0 * c.er * c.geom.area * 1e-6 / (c.d * 1e-3) for c in pdn.cavities)
    f = 1e6
    z = pdn.z([f], observe=['U1'], bare=True)['U1'][0]
    assert abs(z) == pytest.approx(1 / (2 * np.pi * f * C), rel=0.03)


def test_rail_pdn_with_its_parts():
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    pdn = RailPDN(r, dmax=1.0, vrm=dict(R=2e-3, L=20e-9), log=lambda *a: None)
    f = np.logspace(3, 9, 61)
    z = pdn.z(f)
    # at 1 kHz the regulator, its sense resistor: R_vrm + R1 (+ copper)
    assert abs(z['U1'][0]) == pytest.approx(2e-3 + 0.01, rel=0.15)
    # U2, behind the bead: at 1 kHz the bead's DCR in series too; at 100 MHz its own decap C3 (0402: 0.35 nH ESL and
    # a lumped node's nominal 0.5 nH), the bead's 600 ohm cutting it off from the plane
    assert abs(z['U2'][0]) == pytest.approx(2e-3 + 0.01 + 0.05, rel=0.15)
    k = np.argmin(abs(f - 1e8))
    assert abs(z['U2'][k]) == pytest.approx(2 * np.pi * f[k] * 0.85e-9, rel=0.1)
    z0 = pdn.z(f[k:k + 1], observe=['U1'], bare=True)['U1'][0]
    assert abs(z['U1'][k]) < 0.5 * abs(z0)                           # C1, C2 (~0.7 nH together) on the planes
