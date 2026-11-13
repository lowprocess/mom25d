"""mom25d.board: a kicad2ems board (the content of <name>_board.json) as a mom25d model - the ground, stack
and interfaces, vias over the interfaces they join, edge walls as via rows, parts as series elements, ports
(single and differential) - and mom25d.api.solve_board, the one call kicad2ems makes. The boards are in
tests/boards: kicad2ems's example boards as it writes them (examples/make_demo_board.py there)."""
import json
import os

import numpy as np
import pytest
from mom25d.board import BoardModel, WALL_GAP, wall_points

quiet = lambda *a: None
BOARDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boards')


def board(name):
    """A board file: demo (the transition; nets RF, GND; ground B.Cu), diff (D_P, D_N; B.Cu), launch (RF, GND;
    edge walls xmin, ymin), rlc (RF1, RF2, GND), lowpass, thru (a 3 mm line on 1.6 mm FR4)."""
    with open(os.path.join(BOARDS, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def test_stack_vias_and_ports():
    """The transition demo: ground B.Cu, three interfaces, the blind signal via from In2.Cu to F.Cu, a port
    on each end (F.Cu and In2.Cu)."""
    m = BoardModel(board('demo'), log=quiet)
    assert m.ground == 'B.Cu' and m.iface == {'In2.Cu': 1, 'In1.Cu': 2, 'F.Cu': 3}
    assert [l.h for l in m.stack.layers] == pytest.approx([0.2e-3] * 3)
    ps = m.ports(dmax=0.3e-3)
    assert [p.name for p in ps] == ['P1', 'P2'] and [p.layer for p in ps] == [3, 1]
    assert ps[0].w == pytest.approx(0.35e-3, rel=1e-6) and ps[1].w == pytest.approx(0.15e-3, rel=1e-6)
    assert len(m.vias) == 23
    signal = [v for v in m.vias if v.name.startswith('RF')]
    assert len(signal) == 1 and (signal[0].k_bot, signal[0].k_top) == (1, 3)
    assert m.port_z0 == {'P1': 50.0, 'P2': 50.0}


def test_ground_found_and_differential_ports():
    """Without a ground setting the lowest solid layer is the ground; P<n>+ / P<n>- become one mixed-mode
    port with pairs=True."""
    from mom25d.sim import DiffPort
    data = board('diff')
    m = BoardModel(data, pairs=True, log=quiet)
    ps = m.ports(dmax=0.3e-3)
    assert [type(p) for p in ps] == [DiffPort, DiffPort] and [p.name for p in ps] == ['P1', 'P2']
    assert ps[0].pos.w == pytest.approx(0.25e-3, rel=1e-6)
    assert abs((ps[0].pos.c - ps[0].neg.c) @ ps[0].pos.t) == pytest.approx(0.45e-3, rel=1e-3)
    # a pair's wave port needs its cuts to span enough of a wavelength at the lowest frequency, as a single port's
    # does: with port auto it is refused below that (kicad2ems then solves the legs as lumped ports), with wave not
    assert len(m.ports(dmax=0.3e-3, fmin=1e9)) == 2
    with pytest.raises(ValueError, match=r'P1: straight coupled feed too short for a differential wave port down to '
                                         r'0\.05 GHz \(\d+\.\d+ mm, need \d+\.\d+ mm\)'):
        m.ports(dmax=0.3e-3, fmin=5e7)
    assert len(BoardModel(data, pairs=True, port='wave', log=quiet).ports(dmax=0.3e-3, fmin=5e7)) == 2
    single = BoardModel(data, log=quiet)
    assert [sp['name'] for sp in single.port_specs] == ['P1+', 'P1-', 'P2+', 'P2-']
    assert [type(p).__name__ for p in single.ports(dmax=0.3e-3, fmin=5e7)] == ['ViaPort'] * 4     # lumped legs
    data['ground'] = None
    assert BoardModel(data, log=quiet).ground == 'B.Cu'                    # the solid plane --ground made


def test_walls_parts_and_errors():
    launch = board('launch')
    m = BoardModel(launch, log=quiet)
    walls = [v for v in m.vias if v.name.startswith('wall')]
    length = sum(w['t1'] - w['t0'] for w in launch['walls'])
    assert launch['walls'] and len(walls) >= length / 0.5 and not any(v.ring for v in walls)
    assert any('edge walls' in n for n in m.notes)
    xy = np.array([(v.x, v.y) for v in m.vias])                  # walls meeting at a corner share no via
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)) + np.eye(len(xy))
    assert d.min() > 0.9 * WALL_GAP * 1e-3
    rlc = board('rlc')
    m = BoardModel(rlc, log=quiet)
    elements = rlc['parts']                                       # the series elements (shorts are left out)
    assert len(m.lumped) == len(elements) and [e.name for e in m.lumped] == [p['ref'] for p in elements]
    c1 = next(p for p in elements if p['ref'] == 'C1')
    z = next(e for e in m.lumped if e.name == 'C1').Z(1e9)
    assert z == pytest.approx(1 / (2j * np.pi * 1e9 * c1['C']) + (c1['R'] or 0) + 2j * np.pi * 1e9 * (c1['L'] or 0), rel=1e-9)
    data = board('rlc')
    data['version'] = 99
    with pytest.raises(ValueError, match='newer'):
        BoardModel(data)
    data = board('rlc')
    data['ports'][0]['kind'] = 'gap'
    with pytest.raises(ValueError, match='no planar equivalent'):
        BoardModel(data, log=quiet)


def test_solve_board_is_the_steps_it_wraps():
    """api.solve_board: the same S-parameters as building the model and sweeping by hand."""
    pytest.importorskip('triangle')
    from mom25d.api import solve_board
    from mom25d.run import solve_band
    data = board('lowpass')
    f = np.linspace(1e9, 3e9, 5)
    calls = []
    res = solve_board(data, f, dmax=1.6e-3, max_solves=5, log=quiet, progress=lambda *a: calls.append(a))
    m = BoardModel(data, log=quiet)
    planes = {k: 2.4e-3 for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
    sim = m.simulation(dmax=1.6e-3, dmax_layer=planes, fmin=1e9, fmax=3e9, solver='auto', verbose=False, wave_down_to=1e9)
    fs, S, info = solve_band(sim, 1e9, 3e9, points=5, zref=m.port_z0, max_solves=5, afs_tol=5e-3, log=quiet)
    assert np.allclose(res.S, S) and res.names == info['names'] == ['P1', 'P2']
    assert res.solves == len(calls) and calls[-1][:2] == (res.solves, 5)
    # reciprocal to 1e-3: the core's direct solver (near pairs corrected per ordered pair) leaves Z symmetric to
    # about 1e-4 in S, NumPy's dense one (without the core) exactly
    assert np.abs(res.S).max() <= 1 + 1e-6 and np.abs(res.S - res.S.transpose(0, 2, 1)).max() < 1e-3


def test_wave_ports_with_the_characterised_gamma():
    """A 3 mm 50 ohm line on 1.6 mm FR4 characterises to ~50 ohm (a 16 mm test line gave 1.6 / 24 / 33 ohm), and
    a thru keeps wave ports down to frequencies where its feed spans too little to fit gamma: the characterised
    gamma takes over, and the line stays matched and lossless."""
    from mom25d.run import renorm, _zc
    m = BoardModel(board('thru'), port='wave', log=quiet)
    sim = m.simulation(dmax=0.4e-3, fmin=0.2e9, fmax=6e9, verbose=False, wave_down_to=0.2e9)
    assert all(abs(p.zc - 51) < 2 for p in sim.ports)
    for f in (0.3e9, 1e9):
        S = renorm(sim.sparams(f), _zc(sim, f), [50, 50])
        assert all(p.known_gamma for p in sim.ports)
        assert abs(S[0, 0]) < 0.04 and abs(abs(S[1, 0]) - 1) < 0.01


def test_a_pair_s_modes_take_their_gamma_from_the_characterisation():
    """A pair's differential and common modes are fitted with the gamma of the pair's characterisation, as a single
    line's wave is: fitted on the feed's own cuts where they span a quarter of a radian it made S wrong at the low
    end of the band (rf_diffvia at 1 GHz: |Sdd11| 0.19 for 0.01, |Sdd21| 1.21 one way and 0.79 the other, the sweep
    54 solves long). The ports' scales are then taken out (sim.balance): S is reciprocal."""
    from mom25d.sim import ModePort
    m = BoardModel(board('diff'), pairs=True, log=quiet)
    sim = m.simulation(dmax=0.3e-3, fmin=1e9, fmax=10e9, verbose=False, wave_down_to=1e9)
    assert [type(p) for p in sim.ports] == [ModePort] * 4 and [p.name for p in sim.ports] == ['P1d', 'P1c', 'P2d', 'P2c']
    k0 = 2 * np.pi * 1e9 / 299792458.0
    for p in sim.ports:
        g = p.gamma_fn(1e9)
        assert 1.3 * k0 < g.imag < 2.2 * k0 and abs(g.real) < 0.05 * g.imag     # between air and the dielectric
    S = sim.sparams(1e9)
    assert all(p.known_gamma for p in sim.ports)
    assert np.abs(S - S.T).max() < 1e-12 and np.linalg.svd(S, compute_uv=False).max() < 1.0
    assert abs(S[0, 0]) < 0.03 and 0.98 < abs(S[2, 0]) < 0.997 and 0.98 < abs(S[3, 1]) < 0.997
    d2 = sim.last['scale'] ** 2
    assert np.abs(d2 - 1).max() < 0.03 and sim.unbalance == pytest.approx(d2.max() / d2.min() - 1)


def test_a_plane_only_vias_run_through_as_a_pec_plane_with_holes():
    """planes='holes' (experimental): the demo transition's inner plane, whose one hole is the signal via's, is a
    PEC plane inside the stack, not meshed, and the hole a few unknowns (mom25d.frill): 1,500 unknowns for
    3,900, S within 0.02 of the meshed plane's (0.09 before the hole had its first azimuthal order: the traces
    over it saw the plane closed). A plane with any other hole stays meshed."""
    import copy
    from shapely.geometry import shape
    from mom25d.board import plane_holes
    from mom25d.api import solve_board
    data = board('demo')
    gn, outline = set(data['ground_nets']), shape(data['outline'])
    geom, holes = plane_holes(data, 'In1.Cu', gn, outline)
    assert len(holes) == 1 and holes[0][0] == 0 and holes[0][1] == pytest.approx(0.6, abs=0.01) and holes[0][2] is None
    assert plane_holes(data, 'F.Cu', gn, outline) is None and plane_holes(data, 'B.Cu', gn, outline) == (shape(data['copper']['B.Cu'][0]['polygon']).buffer(0), [])
    bad = copy.deepcopy(data)                                   # a second via in the hole
    bad['vias'].append(dict(bad['vias'][0], x=bad['vias'][0]['x'] + 0.3))
    assert plane_holes(bad, 'In1.Cu', gn, outline) is None
    bad = copy.deepcopy(data)                                   # a slot, not a via's hole
    x, y = data['vias'][0]['x'], data['vias'][0]['y']
    bad['copper']['In1.Cu'][0]['polygon']['coordinates'][1] = [[x - 3, y - 0.3], [x + 3, y - 0.3], [x + 3, y + 0.3], [x - 3, y + 0.3], [x - 3, y - 0.3]]
    assert plane_holes(bad, 'In1.Cu', gn, outline) is None
    bad = copy.deepcopy(data)                                   # the via ends on the plane's layer
    bad['vias'][0]['z0'] = 0.4
    assert plane_holes(bad, 'In1.Cu', gn, outline) is None
    assert BoardModel(bad, planes='holes', log=quiet).stack.pec == ()

    m = BoardModel(data, planes='holes', log=quiet)
    assert m.stack.pec == (2,) and sorted(m.shapes) == [1, 3] and len(m.frills) == 1
    fr = m.frills[0]
    assert (fr.k, fr.via.k_bot, fr.via.k_top) == (2, 1, 3) and fr.b == pytest.approx(0.6e-3, abs=1e-5)
    assert any('In1.Cu: a PEC plane inside the stack' in n for n in m.notes) and 'a PEC plane In1.Cu' in m.describe()
    assert BoardModel(data, log=quiet).stack.pec == () and not BoardModel(data, log=quiet).frills
    f = np.array([2e9, 6e9])
    res = {p: solve_board(data, f, planes=p, port='via', log=quiet, max_solves=4) for p in ('pec', 'holes')}
    assert res['holes'].unknowns < 0.45 * res['pec'].unknowns
    S, ref = res['holes'].S, res['pec'].S
    assert np.abs(S - ref).max() < 0.03 and np.abs(np.abs(S[:, 1, 0]) - np.abs(ref[:, 1, 0])).max() < 0.005
    assert np.abs(S - S.transpose(0, 2, 1)).max() < 1e-6 and np.linalg.svd(S, compute_uv=False).max() < 1.0


def test_wall_points_skip_a_via_already_there():
    """Two walls meeting at a corner give one via there, not two (a duplicated via makes the via block
    singular), and a wall via on top of a board via is left out."""
    walls = [('x', 0.0, 0.0, 2.0, 0, 1), ('y', 0.0, 0.0, 2.0, 0, 1)]          # x = 0 and y = 0, meeting at (0, 0)
    pts = wall_points(walls, 0.5)
    assert len(pts) == 9 and sum((x, y) == (0.0, 0.0) for _, x, y in pts) == 1
    assert len(wall_points(walls, 0.5, taken=[(0.0, 1.02)])) == 8


def stripline():
    """The demo transition made a stripline: In1.Cu and F.Cu solid ground planes (the signal via, P1 and F.Cu's
    copper gone), the trace on In2.Cu between B.Cu and In1.Cu, its stitching vias' lone pads beside it."""
    data = board('demo')
    full = dict(polygon=data['outline'], net='GND')
    data['copper']['In1.Cu'] = [dict(full)]
    data['copper']['F.Cu'] = [dict(full)]
    data['vias'] = [v for v in data['vias'] if v['net'] in data['ground_nets']]
    data['ports'] = [p for p in data['ports'] if p['layer'] == 'In2.Cu']
    return data


def test_solid_planes_as_pec_and_lone_pads_dropped():
    """planes='pec': the solid plane over the trace is a PEC cover (the copper above it left out), the ground
    stays B.Cu (the trace's layer is not solid), the stitching vias run from the ground to the cover.
    stitching_pads='drop': the pads beside the trace, and nothing else, are left out; the vias are kept."""
    from mom25d.board import pec_planes, lone_pads
    data = stripline()
    assert pec_planes(data, 'B.Cu') == ('B.Cu', 'In1.Cu')
    assert pec_planes(board('demo'), 'B.Cu') == ('B.Cu', None)         # In1.Cu has the signal via's antipad
    m = BoardModel(data, planes='pec', stitching_pads='keep', log=quiet)
    assert m.cover == 'In1.Cu' and m.stack.top == 'pec' and m.stack.N == 2 and m.iface == {'In2.Cu': 1}
    assert any('above the cover (F.Cu)' in n for n in m.notes)
    assert {(v.k_bot, v.k_top) for v in m.vias} == {(0, 2)} and [sp['kref'] for sp in m.port_specs] == [0]
    pads = lone_pads(data, ['In2.Cu'])['In2.Cu']
    nets = [p['net'] for p in data['copper']['In2.Cu']]
    assert len(pads) == nets.count('GND') and all(nets[i] == 'GND' for i in pads)
    d = BoardModel(data, planes='pec', stitching_pads='drop', log=quiet)
    assert len(d.vias) == len(m.vias) and d.shapes[1][0].area < m.shapes[1][0].area
    meshed = BoardModel(data, planes='mesh', stitching_pads='keep', log=quiet)
    assert meshed.cover is None and len(meshed.iface) == 3
    assert BoardModel(data, log=quiet).cover == 'In1.Cu' and len(BoardModel(data, log=quiet).vias) == len(m.vias)   # the defaults


def test_a_raised_ground():
    """planes='pec' on a stripline whose lower plane is not the board's ground (rf_stripline: B.Cu, In2.Cu
    solid, the trace on In1.Cu): the ground moves up to In2.Cu and B.Cu is left out."""
    from mom25d.board import pec_planes
    data = stripline()
    data['copper']['In2.Cu'], data['copper']['In1.Cu'] = data['copper']['In1.Cu'], data['copper']['In2.Cu']
    for p in data['ports']:
        p['layer'], p['ref_layer'] = 'In1.Cu', 'In2.Cu'
    assert pec_planes(data, 'B.Cu') == ('In2.Cu', 'F.Cu')
    m = BoardModel(data, planes='pec', log=quiet)
    assert m.ground == 'In2.Cu' and m.cover == 'F.Cu' and m.iface == {'In1.Cu': 1} and m.stack.N == 2
    assert [sp['kref'] for sp in m.port_specs] == [0] and {(v.k_bot, v.k_top) for v in m.vias} == {(0, 2)}


def test_a_pad_a_part_lands_on_is_not_lone():
    """stitching_pads='drop' keeps a ground pad that a part terminates on: a shunt part's pad with its via to the
    ground (demo_bandpass: dropped, it cut the part off the ground, |dS| 1.98). The part's s and e are its gap's
    corners, so its pads touch that box without containing either."""
    from shapely.geometry import shape
    from mom25d.board import lone_pads
    data = stripline()
    pads = lone_pads(data, ['In2.Cu'])['In2.Cu']
    p = shape(data['copper']['In2.Cu'][pads[0]]['polygon'])
    x0, y0, x1, y1 = p.bounds
    data['parts'] = [dict(ref='C1', R=0.1, L=0.0, C=1e-12, layer='In2.Cu', z=0.0, axis=0, s=[x1, y0, 0.0],
                          e=[x1 + 0.5, y1, 0.0])]                    # the gap beside the pad's right edge
    kept = lone_pads(data, ['In2.Cu'])['In2.Cu']
    assert pads[0] not in kept and len(kept) == len(pads) - 1


def test_a_ground_via_under_a_parts_pad_is_meshed_as_a_signal_via():
    """A shunt part's ground via is not a stitching via: squared outside the zone, its pad lost the part
    (demo_bandpass with --mom-zone on: |dS| 1.98)."""
    data = stripline()
    v = next(v for v in data['vias'] if v['net'] in data['ground_nets'])
    data['parts'] = [dict(ref='C1', R=0.1, L=0.0, C=1e-12, layer='In2.Cu', z=0.0, axis=0, s=[v['x'] + 0.3, v['y'] - 0.2, 0.0],
                          e=[v['x'] + 0.7, v['y'] + 0.2, 0.0])]
    from shapely.geometry import box, Point
    from mom25d.board import PART_VIA
    gap = box(v['x'] + 0.3, v['y'] - 0.2, v['x'] + 0.7, v['y'] + 0.2)
    near = [gap.distance(Point(w['x'], w['y'])) < PART_VIA for w in data['vias']]
    kinds = [w.kind for w in BoardModel(data, log=quiet).vias]
    assert near[data['vias'].index(v)] and kinds == ['signal' if n else 'stitching' for n in near]


def test_metal_finish_per_layer():
    """A surface finish on some interfaces: Zs there times 1 + share (F - 1), elsewhere bare (sim.Metal)."""
    import numpy as np
    from mom25d.sim import Metal
    enig = [(4.1e7, 0.08e-6), (1.45e6, 5e-6)]
    m = Metal(5.8e7, 35e-6, finish=enig, on={2: 0.4})
    f = 3e9
    F = m.face_factor(f)
    assert 4 < F.real < 6
    assert m.zs(f, 2) == pytest.approx(m.zs(f) * (1 + 0.4 * (F - 1)))
    assert m.zs(f, 1) == m.zs(f)
    zs = m.zs_edges(f, np.array([1, 2, 2, 1]))
    assert zs[0] == m.zs(f) and zs[1] == m.zs(f, 2)
    assert np.ndim(Metal(5.8e7, 35e-6).zs_edges(f, np.array([1, 2]))) == 0     # no finish: one value


def test_ground_apertures():
    """ground_apertures: a slot cut into the ground under the trace is found, with its area; a via's antipad
    (In1.Cu's, the signal via through it) and a pour pulled back from the board edge are not."""
    from shapely.geometry import box, mapping, shape
    from mom25d.board import ground_apertures
    data = board('demo')
    assert ground_apertures(data, 'B.Cu') == [] and ground_apertures(data, 'In1.Cu') == []
    gnd = data['copper']['B.Cu'][0]
    gnd['polygon'] = mapping(shape(gnd['polygon']).buffer(-0.3, join_style='mitre'))     # the pour's pullback
    assert ground_apertures(data, 'B.Cu') == []
    gnd['polygon'] = mapping(shape(gnd['polygon']).difference(box(5.0, -5.8, 5.4, -4.2)))  # a slot across the trace
    (slot, area), = ground_apertures(data, 'B.Cu')
    assert area == pytest.approx(0.4 * 1.6, rel=1e-3) and slot.centroid.x == pytest.approx(5.2)


def test_a_feed_with_another_trace_beside_it_takes_a_lumped_port():
    """port auto: a wave port fits one line's two waves, and what a neighbour couples in along the feed is not
    one of them (kicad2ems's rf_crosstalk, two lines 3 mm apart on 0.8 mm: NEXT 20 dB low with wave ports). A
    lone line keeps its wave ports, and port='wave' is the user's word."""
    xtalk = board(os.path.join('rf', 'xtalk'))                      # (kicad2ems's rf_crosstalk: make_rf_examples.py)
    m = BoardModel(xtalk, log=quiet)
    ps = m.ports(dmax=0.3e-3)
    assert [type(p).__name__ for p in ps] == ['ViaPort'] * 4
    assert sum('another trace runs beside its feed' in n for n in m.notes) == 4
    m = BoardModel(xtalk, port='wave', log=quiet)
    assert [type(p).__name__ for p in m.ports(dmax=0.3e-3)] == ['LinePort'] * 4
    m = BoardModel(board('thru'), log=quiet)
    assert [type(p).__name__ for p in m.ports(dmax=0.3e-3)] == ['LinePort'] * 2 and not any('beside' in n for n in m.notes)
