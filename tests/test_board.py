"""mom25d.board: a kicad2ems board (the content of <name>_board.json) as a mom25d model - the ground, stack
and interfaces, vias over the interfaces they join, edge walls as via rows, parts as series elements, ports
(single and differential) - and mom25d.api.solve_board, the one call kicad2ems makes."""
import numpy as np
import pytest
from kicad2ems.board import Board
from kicad2ems.importers.kicad import KiCadBoard
from mom25d.board import BoardModel

quiet = lambda *a: None


def board(base, **kw):
    return Board(KiCadBoard(base + '.kicad_pcb'), pos_file=base + '-top.pos', log=quiet, **kw).to_dict()


def test_stack_vias_and_ports(demo_files):
    """The transition demo: ground B.Cu, three interfaces, the blind signal via from In2.Cu to F.Cu, a port
    on each end (F.Cu and In2.Cu)."""
    demo, _ = demo_files
    m = BoardModel(board(demo, nets=['RF', 'GND'], ground='B.Cu'), log=quiet)
    assert m.ground == 'B.Cu' and m.iface == {'In2.Cu': 1, 'In1.Cu': 2, 'F.Cu': 3}
    assert [l.h for l in m.stack.layers] == pytest.approx([0.2e-3] * 3)
    ps = m.ports(dmax=0.3e-3)
    assert [p.name for p in ps] == ['P1', 'P2'] and [p.layer for p in ps] == [3, 1]
    assert ps[0].w == pytest.approx(0.35e-3, rel=1e-6) and ps[1].w == pytest.approx(0.15e-3, rel=1e-6)
    assert len(m.vias) == 23
    signal = [v for v in m.vias if v.name.startswith('RF')]
    assert len(signal) == 1 and (signal[0].k_bot, signal[0].k_top) == (1, 3)
    assert m.port_z0 == {'P1': 50.0, 'P2': 50.0}


def test_ground_found_and_differential_ports(demo_files):
    """Without a ground setting the lowest solid layer is the ground; P<n>+ / P<n>- become one mixed-mode
    port with pairs=True."""
    from mom25d.sim import DiffPort
    _, diff = demo_files
    data = board(diff, nets=['D_P', 'D_N'], ground='B.Cu')
    m = BoardModel(data, pairs=True, log=quiet)
    ps = m.ports(dmax=0.3e-3)
    assert [type(p) for p in ps] == [DiffPort, DiffPort] and [p.name for p in ps] == ['P1', 'P2']
    assert ps[0].pos.w == pytest.approx(0.25e-3, rel=1e-6)
    assert abs((ps[0].pos.c - ps[0].neg.c) @ ps[0].pos.t) == pytest.approx(0.45e-3, rel=1e-3)
    single = BoardModel(data, log=quiet)
    assert [sp['name'] for sp in single.port_specs] == ['P1+', 'P1-', 'P2+', 'P2-']
    data['ground'] = None
    assert BoardModel(data, log=quiet).ground == 'B.Cu'                    # the solid plane --ground made


def test_walls_parts_and_errors(boards):
    launch = Board(KiCadBoard(boards['launch'] + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=boards['launch'] + '-top.pos',
                   edge_walls=['xmin', 'ymin'], log=quiet)
    m = BoardModel(launch.to_dict(), log=quiet)
    walls = [v for v in m.vias if v.name.startswith('wall')]
    length = sum(w[3] - w[2] for w in launch.walls)
    assert launch.walls and len(walls) >= length / 0.5 and not any(v.ring for v in walls)
    assert any('edge walls' in n for n in m.notes)
    rlc = Board(KiCadBoard(boards['rlc'] + '.kicad_pcb'), nets=['RF1', 'RF2', 'GND'], pos_file=boards['rlc'] + '-top.pos',
                log=quiet)
    m = BoardModel(rlc.to_dict(), log=quiet)
    elements = [p for p in rlc.parts if not p.short]
    assert len(m.lumped) == len(elements) and [e.name for e in m.lumped] == [p.ref for p in elements]
    c1 = next(p for p in rlc.parts if p.ref == 'C1')
    z = next(e for e in m.lumped if e.name == 'C1').Z(1e9)
    assert z == pytest.approx(1 / (2j * np.pi * 1e9 * c1.C) + (c1.R or 0) + 2j * np.pi * 1e9 * (c1.L or 0), rel=1e-9)
    data = rlc.to_dict()
    data['version'] = 99
    with pytest.raises(ValueError, match='newer'):
        BoardModel(data)
    data = rlc.to_dict()
    data['ports'][0]['kind'] = 'gap'
    with pytest.raises(ValueError, match='no planar equivalent'):
        BoardModel(data, log=quiet)


def test_solve_board_is_the_steps_it_wraps(boards):
    """api.solve_board: the same S-parameters as building the model and sweeping by hand."""
    pytest.importorskip('triangle')
    from mom25d.api import solve_board
    from mom25d.run import solve_band
    data = board(boards['lowpass'])
    f = np.linspace(1e9, 3e9, 5)
    calls = []
    res = solve_board(data, f, dmax=0.8e-3, max_solves=5, log=quiet, progress=lambda *a: calls.append(a))
    m = BoardModel(data, log=quiet)
    planes = {k: 2.4e-3 for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
    sim = m.simulation(dmax=0.8e-3, dmax_layer=planes, fmin=1e9, fmax=3e9, solver='auto', verbose=False, wave_down_to=1e9)
    fs, S, info = solve_band(sim, 1e9, 3e9, points=5, zref=m.port_z0, max_solves=5, afs_tol=5e-3, log=quiet)
    assert np.allclose(res.S, S) and res.names == info['names'] == ['P1', 'P2']
    assert res.solves == len(calls) and calls[-1][:2] == (res.solves, 5)
    assert np.abs(res.S).max() <= 1 + 1e-6 and np.abs(res.S - res.S.transpose(0, 2, 1)).max() < 1e-6


def test_wave_ports_with_the_characterised_gamma(tmp_path):
    """A 3 mm 50 ohm line on 1.6 mm FR4 characterises to ~50 ohm (a 16 mm test line gave 1.6 / 24 / 33 ohm), and
    a thru keeps wave ports down to frequencies where its feed spans too little to fit gamma: the characterised
    gamma takes over, and the line stays matched and lossless."""
    import os, subprocess, sys
    from mom25d.run import renorm, _zc
    gen = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'make_demo_board.py')
    subprocess.check_call([sys.executable, gen, str(tmp_path / 'thru'), 'thru'], stdout=subprocess.DEVNULL)
    b = Board(KiCadBoard(str(tmp_path / 'thru.kicad_pcb')), pos_file=str(tmp_path / 'thru-top.pos'), log=lambda *a: None)
    m = BoardModel(b.to_dict(), port='wave', log=lambda *a: None)
    sim = m.simulation(dmax=0.4e-3, fmin=0.2e9, fmax=6e9, verbose=False, wave_down_to=0.2e9)
    assert all(abs(p.zc - 51) < 2 for p in sim.ports)
    for f in (0.3e9, 1e9):
        S = renorm(sim.sparams(f), _zc(sim, f), [50, 50])
        assert all(p.known_gamma for p in sim.ports) == (f < 0.5e9)      # 1 GHz: span enough to fit gamma
        assert abs(S[0, 0]) < 0.04 and abs(abs(S[1, 0]) - 1) < 0.01
