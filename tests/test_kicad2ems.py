"""mom25d as a solver: it gets the board itself (no CSX file, no openEMS), --mom / --solvers mom end to end,
and --sigma as conductor loss."""
import json, os
import numpy as np
import pytest

from k2e_fixtures import ROOT                                       # kicad2ems's checkout (conftest.py)

FILTER = os.path.join(ROOT, 'examples', 'rf_interdig.toml')


def test_import():
    import mom25d
    from mom25d.board import BoardModel                             # noqa: F401
    assert hasattr(mom25d, '__version__')


def test_preview_solves_no_full_wave_solver(tmp_path, run_cli):
    """--preview builds the model and the meshes and solves only the line model: mom25d named (or added with
    --mom) is not run."""
    out = tmp_path / 'out'
    code, so, se = run_cli(FILTER, '--preview', '--solvers', 'mom25d', '--mom', '--points', '41', '--out', str(out))
    assert code == 0, so + se
    assert not (out / 'rf_interdig_mom25d.s2p').exists() and list(out.glob('rf_interdig_*.s2p'))      # the line model's


def test_mom_run(tmp_path, run_cli, no_csxcad):
    """--solvers mom25d end to end with no openEMS (MoM reads the board), and the line model the config's
    fast = "auto" picks compared against it."""
    pytest.importorskip('triangle')
    out = tmp_path / 'out'
    code, so, se = run_cli(FILTER, '--solvers', 'mom25d', '--mom-dmax', '1.6', '--mom-solves', '8', '--points', '41',
                           '--out', str(out))
    assert code == 0, so + se
    sm = json.load(open(out / 'rf_interdig_run.json', encoding='utf-8'))
    line = next(n for n in sm['solved'] if n != 'mom25d')          # qucs or scikit, whichever is installed
    for f in ('rf_interdig_mom25d.s2p', 'rf_interdig_mom25d.png', f'rf_interdig_{line}.s2p', 'rf_interdig_report.html', 'rf_interdig_board.json'):
        assert (out / f).exists(), f
    mo = sm['solved']['mom25d']
    assert mo['solves'] <= 8 and mo['unknowns'] > 100 and {p['kind'] for p in mo['ports']} == {'lumped'}
    assert mo['table'][0]['param'] == 'S(P1,P1)' and sm['results']['reference'] == 'mom25d'
    assert sm['solved'][line]['reference'] == 'mom25d' and sm['solved'][line]['compare']
    samples = mo['samples_ghz']                                     # the adaptive sweep's frequencies
    assert len(samples) == mo['solves'] and samples == sorted(samples)
    assert 'mom25d' in so


def test_sigma_reaches_mom(demo_files):
    """--sigma is conductor loss in openEMS's conducting sheets; MoM gets the same, with the copper thickness
    of the ports' layer."""
    from mom25d.kicad2ems.mom import _metal
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu',
              log=lambda *a: None)
    assert _metal(m, None) == (None, None)
    metal, t = _metal(m, 5.8e7)
    assert metal.sigma == 5.8e7 and t == pytest.approx(m.cu_t[m.ports[0].layer]) and metal.t == pytest.approx(t * 1e-3)
    zs = metal.zs(np.array([1e9]))
    assert zs.real[0] > 0                                           # a lossy surface impedance
    rough, _ = _metal(m, 5.8e7, 1.0)                                # --rough 1 um: Hammerstad on the resistance
    f = np.array([1e6, 5e9, 1e12])
    k = rough.zs(f).real / metal.zs(f).real
    delta = 1 / np.sqrt(np.pi * f * 4e-7 * np.pi * 5.8e7)
    np.testing.assert_allclose(k, 1 + 2 / np.pi * np.arctan(1.4 * (1e-6 / delta) ** 2))
    assert k[0] == pytest.approx(1, abs=0.01) and k[2] == pytest.approx(2, abs=0.01)
    np.testing.assert_allclose(rough.zs(f).imag, metal.zs(f).imag)


def test_current_dumps(boards, run_cli, tmp_path, monkeypatch):
    """--mom-dump: mom25d's surface currents at the frequencies asked, a |J| map each, with --dump-animate one
    period of Re(J e^j phase), and the dumps where the GUI's Fields tab finds them. The sweep solves at the dump
    frequencies among its samples and the dumps reuse those currents: no solve of their own."""
    pytest.importorskip('triangle')
    from kicad2ems.fields import frames as F
    from mom25d.sim import Simulation
    save = F.save_fd_outputs
    monkeypatch.setattr(F, 'save_fd_outputs', lambda *a, **kw: save(*a, n_phase=4, **kw))     # 4 frames, not 24
    calls, currents = [], Simulation.currents
    monkeypatch.setattr(Simulation, 'currents', lambda self, f, E: (calls.append(f), currents(self, f, E))[1])
    base, out = boards['lowpass'], tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--solvers', 'mom', '--mom-dmax', '1.6',
                            '--mom-solves', '6', '--points', '21', '--mom-dump', '2,5', '--dump-animate', 'gif',
                            '--mom-near', '1', '--out', out)
    assert code == 0, so + err
    for f in (2, 5):
        for ext in ('png', 'gif'):
            assert (out / f"lowpass_mom25d_P1_dump_J_F_Cu_fd_{f}GHz.{ext}").exists()
    E, Hn, src = F.find_dumps(str(out))                          # sorted by name: E@top+1mm, H@top+1mm, J@F.Cu
    assert src.solver == 'mom25d' and src.freqs == [2e9, 5e9] and 'A/m' in src.unit
    assert E.field == 'E' and 'V/m' in E.unit and np.abs(E.load(0)[1]).max() > 0
    assert Hn.field == 'H' and 'A/m' in Hn.unit and np.abs(Hn.load(0)[1]).max() > 0     # --mom-near: H as well
    assert (out / 'lowpass_mom25d_P1_dump_E_topp1mm_fd_5GHz.gif').exists()
    mesh, J = src.load(0)
    assert np.abs(J).max() > 0 and np.all(J[..., 2] == 0)                # in-plane currents only
    sm = json.loads((out / 'lowpass_run.json').read_text(encoding='utf-8'))
    assert any(d['label'].startswith('mom25d dump_J_F_Cu_fd') for d in sm['dumps'])
    mom = sm['solved']['mom25d']
    assert {2.0, 5.0} <= set(mom['samples_ghz']) and len(calls) == mom['solves'], (calls, mom['solves'])


def test_antenna_mode(run_cli, tmp_path):
    """mom25d in antenna mode on the patch example: the resonance, efficiency below 1, and openEMS's antenna
    outputs (JSON, plots, the sphere the GUI reads)."""
    pytest.importorskip('triangle')
    import subprocess, sys
    ex = os.path.join(os.path.dirname(FILTER), 'make_antennas.py')
    subprocess.check_call([sys.executable, ex, 'patch', str(tmp_path / 'patch')], stdout=subprocess.DEVNULL)
    out = tmp_path / 'out'
    code, so, err = run_cli(tmp_path / 'patch.kicad_pcb', '--mode', 'antenna', '--fmin', '2.3', '--fmax', '2.6',
                            '--points', '61', '--solvers', 'mom', '--mom-dmax', '2', '--mom-solves', '12', '--out', out)
    assert code == 0, so + err
    res = json.loads((out / 'patch_mom25d_antenna.json').read_text(encoding='utf-8'))
    assert 2.38e9 < res['match']['f_res'] < 2.52e9 and res['solver'] == 'mom25d'
    row = min(res['rows'], key=lambda r: abs(r['f'] - res['f_pattern']))
    assert 0.1 < row['eff'] < 1 and 5 < row['D_dbi'] < 9
    assert set(res['cuts']) == {'xz (phi = 0)', 'yz (phi = 90)', 'xy (theta = 90)'}
    for f in ('patch_mom25d_antenna.png', 'patch_mom25d_pattern.png', 'patch_mom25d_sphere.npz'):
        assert (out / f).exists(), f


def test_phased_array(run_cli, tmp_path):
    """The 2 x 2 array with a feed per row placed as a 1 x 2 grid (--grid), driven with P2 20 deg behind P1
    (--drive) and steered by a Y step of 40 deg on top (--steer 0,40) with a taper (--taper: two rows, so uniform):
    P2 60 deg behind P1, the beam tilted in the yz plane, the run's base drive, grid, taper and steps recorded for
    the GUI's Reset (the grid in its short form, 1x2), the per-port
    fields kept, and recombining them (phased.combine) gives the run's result back and broadside for the drive
    in phase."""
    pytest.importorskip('triangle')
    import subprocess, sys
    from kicad2ems import phased
    ex = os.path.join(os.path.dirname(FILTER), 'make_antennas.py')
    subprocess.check_call([sys.executable, ex, 'array2x2_phased', str(tmp_path / 'arr')], stdout=subprocess.DEVNULL)
    out = tmp_path / 'out'
    code, so, err = run_cli(tmp_path / 'arr.kicad_pcb', '--mode', 'antenna', '--drive', 'P1@0,P2@20', '--grid',
                            'P1:0,0 P2:0,1', '--steer', '0,40', '--taper', 'taylor:30', '--fmin', '2.3', '--fmax',
                            '2.6', '--points', '61', '--solvers', 'mom', '--mom-dmax', '3', '--mom-solves', '12',
                            '--out', out)
    assert code == 0, so + err
    res = json.loads((out / 'arr_mom25d_antenna.json').read_text(encoding='utf-8'))
    assert res['drive'] == ['P1@0', 'P2@60'] and res['drive_base'] == ['P1@0', 'P2@20']
    assert res['grid'] == '1x2' and res['taper'] == 'taylor:30' and res['steer'] == [0.0, 40.0]
    assert 5 <= res['peak']['theta'] <= 25 and res['peak']['phi'] == 90
    assert res['port'] == 'P1@0, P2@60' and set(res['active_db']) == {'P1', 'P2'}
    data = phased.load(out / res['array_file'])
    assert data['ports'] == ['P1', 'P2'] and data['Eth'].shape[:2] == (len(data['freq']), 2)
    again = phased.combine(data, data['drive'])
    assert np.allclose([r['G_dbi'] for r in again['rows']], [r['G_dbi'] for r in res['rows']], atol=0.01)
    assert phased.combine(data, [1, 1])['peak']['theta'] <= 5
    code, so, err = run_cli(tmp_path / 'arr.kicad_pcb', '--mode', 'antenna', '--drive', 'P1,P2', '--steer', '0,60',
                            '--solvers', 'mom', '--preview', '--out', tmp_path / 'bad')
    assert code != 0 and 'needs grid' in so + err


def test_stop_ends_mom25d_after_its_current_solve(run_cli, tmp_path, monkeypatch):
    """The GUI's Stop (a STOP file, kicad2ems.stop): mom25d ends after the frequency it is solving, with no
    result; the line model (the --fast reference) still gives its. ABORT alone had left mom25d running."""
    pytest.importorskip('triangle')
    from kicad2ems import pipeline
    from kicad2ems.runs import stop_runs
    out = tmp_path / 'out'
    seen = []
    status = pipeline.Job.status

    def stop_at_second(self, run, st):
        status(self, run, st)
        if run == 'mom25d':
            seen.append(st['solves'])
            if st['solves'] == 2:
                stop_runs(str(out))
    monkeypatch.setattr(pipeline.Job, 'status', stop_at_second)
    code, so, se = run_cli(FILTER, '--solvers', 'mom25d', '--fast', 'auto', '--mom-dmax', '2.4', '--points', '41',
                           '--no-report', '--out', out)
    assert code == 1 and seen == [1, 2], so + se
    assert not (out / 'rf_interdig.xml').exists()                        # openEMS importable: still no CSX file
    assert 'stopped: mom25d (MoM) ended early, no result' in so + se
    assert not (out / 'rf_interdig_mom25d.s2p').exists() and list(out.glob('rf_interdig_*.s2p'))


def test_short_differential_feed_falls_back_to_single_ended_legs(monkeypatch):
    """--mom-port auto: a pair whose coupled feed is too short for a differential wave port (ports on pads) is
    solved with its legs as single-ended ports and the modes taken from them, instead of failing."""
    import types
    import mom25d.api
    from kicad2ems.board import Board
    from kicad2ems.importers.kicad import KiCadBoard
    from mom25d.kicad2ems.mom import solve
    ex = os.path.join(ROOT, 'examples')
    m = Board(KiCadBoard(os.path.join(ex, 'rf_diffvia.kicad_pcb')), ground='B.Cu', log=lambda *a: None)
    names = [p.name for p in m.ports]
    calls = []

    def fake(board, freq, pairs=False, **kw):
        calls.append(pairs)
        if pairs:
            raise ValueError("P1: straight coupled feed too short for a differential wave port (0.15 mm, need 2.80 mm)")
        n = len(names)
        model = types.SimpleNamespace(describe=lambda: 'model')
        return types.SimpleNamespace(S=np.zeros((len(freq), n, n), complex), names=names, model=model,
                                     sim=types.SimpleNamespace(log=types.SimpleNamespace(notes=[])), unknowns=1,
                                     solver='fake', ports=[dict(name=x, kind='via') for x in names], seconds=0.0,
                                     solves=1, triangles=1, notes=[], solved=list(freq), afs=False, afs_tol=None)

    monkeypatch.setattr(mom25d.api, 'solve_board', fake)
    log = []
    S, out_names, zref, info = solve(m, np.array([1e9, 2e9]), port='auto', log=log.append)
    assert calls == [True, False] and out_names == ['P1d', 'P1c', 'P2d', 'P2c'] and S.shape == (2, 4, 4)
    assert any('single-ended' in line for line in log)


def test_mixed_connect_cascades_lines():
    """mom25d.kicad2ems.mixed.connect: two line pieces joined by a third through their inner ports equal one
    line of the total length (mismatched Zc, so reflections are in play)."""
    from mom25d.kicad2ems.mixed import connect, line_S
    g, zc, z0 = 0.5 + 180j, 37.0, 50.0
    A, B, C = line_S(g, zc, 3e-3, z0), line_S(g, zc, 5e-3, z0), line_S(g, zc, 7e-3, z0)
    S = np.zeros((4, 4), complex)
    S[:2, :2], S[2:, 2:] = A, B
    out = connect(S, [(1, 2)], [C])
    assert np.allclose(out, line_S(g, zc, 15e-3, z0), atol=1e-12)


def _two_traces(gap, length=30.0, w=0.38, third=None, w2=None):
    """thru.json's stack with two straight traces side by side (mm), a port on each end of each: P1-P2 one trace,
    P3-P4 the other; third: a third trace that far (centre to centre) beyond the second."""
    data = json.load(open(os.path.join(os.path.dirname(__file__), 'boards', 'thru.json'), encoding='utf-8'))
    x0, x1, yc = 2.0, 2.0 + length, -5.0
    box = lambda a, b, c, d: dict(type='Polygon', coordinates=[[[a, b], [c, b], [c, d], [a, d], [a, b]]])
    data['outline'] = box(0.0, -10.0, x1 + 2.0, 0.0)
    ys, ws = (yc + (gap + w) / 2, yc - (gap + w) / 2), (w, w2 or w)
    tr = lambda y, net, wd: dict(net=net, polygon=box(x0, y - wd / 2, x1, y + wd / 2))
    data['copper'] = {'B.Cu': [dict(net='GND', polygon=box(0.0, -10.0, x1 + 2.0, 0.0))],
                      'F.Cu': [tr(ys[0], 'A', ws[0]), tr(ys[1], 'B', ws[1])] + ([tr(ys[1] - third, 'C', w)] if third else [])}
    p0 = data['ports'][0]
    data['ports'] = [dict(p0, name=f"P{2 * i + 1 + e}", c=[(x0 + 0.1, x1 - 0.1)[e], ys[i]], w=ws[i], sgn=(1, -1)[e])
                     for i in range(2) for e in range(2)]
    return data


def test_mixed_plans_a_pair_s_stretch_as_one_cut():
    """mixed.plan_pair_cuts: two traces of one width side by side are cut together where both run straight, less
    the feed each face's mixed-mode port needs; not where a third trace runs beside them, not two of different
    widths, and not two far enough apart to be lone lines (each then a cut of its own)."""
    pytest.importorskip('triangle')
    from mom25d.board import BoardModel
    from mom25d.kicad2ems import mixed
    quiet = lambda *a, **k: None

    def plan(data, **kw):
        model = BoardModel(data, port='via', log=quiet)
        ctx = mixed._context(model, data, 0.3e-3)
        pairs, taken = mixed.plan_pair_cuts(model, data, 0.3e-3, 2e9, ctx=ctx, **kw)
        return pairs, mixed.plan_cuts(model, data, 0.3e-3, 2e9, ctx=ctx, skip=taken, log=quiet, **kw)
    pairs, lone = plan(_two_traces(0.3))
    assert len(pairs) == 1 and lone == []
    ca, cb, need = pairs[0]
    assert (ca.x0, ca.x1) == (cb.x0, cb.x1) and np.allclose(ca.t, cb.t) and ca.x1 - ca.x0 > 10e-3
    assert abs(abs((cb.c0 - ca.c0) @ ca.n) - 0.68e-3) < 1e-9
    lone_need = mixed._need(BoardModel(_two_traces(0.3), port='via', log=quiet), 1, 0.38e-3, 0.3e-3, 2e9)
    assert lone_need + 3e-3 < need < 12e-3                           # (a pair's faces: more feed than a lone line's)
    assert plan(_two_traces(0.3, third=0.9)) == ([], [])            # three side by side: three modes, none cut
    assert plan(_two_traces(0.3, w2=0.3)) == ([], [])               # not a symmetric pair
    pairs, lone = plan(_two_traces(3.0))                            # 8 widths apart: two lone lines
    assert pairs == [] and len(lone) == 2
    pairs, lone = plan(_two_traces(0.3), keep_out=0.2e-3)           # (a keep-out below the gap: lone, by request)
    assert pairs == [] and len(lone) == 2


def test_mixed_cuts_a_coupled_pair_as_two_modal_lines(monkeypatch):
    """Two traces 0.3 mm apart over 30 mm: mixed cuts most of their length as one coupled line (each face a
    mixed-mode wave port, the stretch two lines: its differential and its common mode) and agrees with mom25d on
    every term (0.007 in |S|), the near- and far-end coupling among them, with under two thirds of the unknowns
    (a 30 mm pair: the faces' feeds are half of it)."""
    pytest.importorskip('triangle')
    pytest.importorskip('mom25d._core')
    from types import SimpleNamespace as NS
    import kicad2ems.api.solver as ks
    from mom25d.kicad2ems import mixed
    from mom25d.api import solve_board
    quiet = lambda *a, **k: None
    data = _two_traces(0.3)
    freq = np.linspace(3e9, 9e9, 13)
    ref = solve_board(data, freq, log=quiet, max_solves=24)
    monkeypatch.setattr(ks, 'mixed_mode', lambda S, m: (S, [p.name for p in m.ports], [50.0] * len(m.ports)))
    monkeypatch.setattr(mixed, '_metal', lambda *a, **k: (None, None))
    board = NS(to_dict=lambda: data, ports=[NS(name=p['name']) for p in data['ports']])
    log, hold = [], {}
    S, names, zref, info = mixed.solve_mixed(board, freq, log=log.append, hold=hold, max_solves=24)
    assert info['cuts'] == 1 and info['pairs'] == 1 and info['cut_mm'] > 25 and info['unknowns'] < 0.65 * ref.unknowns
    assert any('a pair w=0.380 mm, 0.300 mm apart' in line for line in log)
    assert [j['mode'] for j in hold['joins']] == ['d', 'c'] and len(hold['stretches']) == 2
    assert [share for _, share in hold['stretches'][1]['of']] == [-1.0, 0.5]        # I- = Ic / 2 - Id
    assert names == ['P1', 'P2', 'P3', 'P4'] and np.abs(S - ref.S).max() < 0.015
    far = np.abs(ref.S[:, 3, 0])                                    # the far-end coupling: the modes' two speeds
    assert far.max() > 0.3 and np.abs(np.abs(S[:, 3, 0]) - far).max() < 0.01
    assert np.abs(S - S.transpose(0, 2, 1)).max() < 1e-6
    currents, extra = mixed.dump_fields(hold)                       # the joined solution's currents: a column a port
    assert currents(6e9).shape == (hold['sim'].Ne + hold['sim'].Nv, len(hold['sim'].ports))


def test_whole_stack_ports_below_the_ground(run_cli, tmp_path):
    """--mom-stack whole on rf_msvia (F.Cu microstrip through a via to B.Cu, two inner planes): the B.Cu port that
    no ground plane allows, lumped ports to the nearest plane, and S21 as gfdtd gives it (-0.38 dB at 5 GHz);
    with the board's own ground (the default) that port cannot be."""
    pytest.importorskip('triangle')
    import subprocess, sys
    from mom25d.board import BoardModel
    ex = os.path.join(os.path.dirname(FILTER), 'make_rf_examples.py')
    subprocess.check_call([sys.executable, ex, 'msvia', str(tmp_path / 'rf_msvia')], stdout=subprocess.DEVNULL)
    cfg = os.path.join(os.path.dirname(FILTER), 'rf_msvia.toml')
    toml = (tmp_path / 'msvia.toml')
    toml.write_text(open(cfg, encoding='utf-8').read().replace('rf_msvia.kicad_pcb', (tmp_path / 'rf_msvia.kicad_pcb').as_posix()))
    out = tmp_path / 'out'
    code, so, err = run_cli(toml, '--solvers', 'mom25d', '--mom-stack', 'whole', '--mom-dmax', '0.8', '--out', out,
                            '--no-report')
    assert code == 0, so + err
    assert 'whole stack: no infinite ground' in so and 'P2 (lumped)' in so
    from kicad2ems.touchstone import read_touchstone
    f, S = read_touchstone(str(out / 'rf_msvia_mom25d.s2p'))
    i = int(np.argmin(np.abs(f - 5e9)))
    assert abs(20 * np.log10(abs(S[i, 1, 0])) + 0.38) < 0.15
    board = json.loads((out / 'rf_msvia_board.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='B.Cu is not a metal interface above the ground In2.Cu'):
        BoardModel(board, port='via', log=lambda *a: None)
    m = BoardModel(board, ground='none', port='via', log=lambda *a: None)
    assert m.whole and m.stack.bottom == 'open' and [(s['k'], s['kref']) for s in m.port_specs] == [(3, 2), (0, 1)]


def test_whole_stack_antenna_radiates_behind_its_ground(run_cli, tmp_path):
    """The patch over its own 80 mm ground meshed (--mom-stack whole): a back lobe (gfdtd, the finite board: -13
    dBi), which the infinite ground cannot have, and gain near gfdtd's 3.1 dBi (the infinite ground's 1.9). No
    ground named is the board's own ground, not the whole stack (ground=None once read as 'none')."""
    pytest.importorskip('triangle')
    import subprocess, sys
    ex = os.path.join(os.path.dirname(FILTER), 'make_antennas.py')
    subprocess.check_call([sys.executable, ex, 'patch', str(tmp_path / 'patch')], stdout=subprocess.DEVNULL)
    out = tmp_path / 'out'
    code, so, err = run_cli(tmp_path / 'patch.kicad_pcb', '--mode', 'antenna', '--fmin', '2.3', '--fmax', '2.6',
                            '--points', '61', '--solvers', 'mom', '--mom-dmax', '2', '--mom-solves', '12',
                            '--mom-stack', 'whole', '--out', out, '--no-report')
    assert code == 0, so + err
    res = json.loads((out / 'patch_mom25d_antenna.json').read_text(encoding='utf-8'))
    c = res['cuts']['xz (phi = 0)']
    a, g = np.array(c['angle']), np.array(c['gain_dbi'])
    assert -25 < g[np.abs(np.abs(a) - 180) < 1.5].max() < -10                     # behind the board
    row = min(res['rows'], key=lambda r: abs(r['f'] - res['f_pattern']))
    assert 2.4 < row['G_dbi'] < 3.6 and 'whole stack' in res['model']
    from mom25d.board import BoardModel
    board = json.loads((out / 'patch_board.json').read_text(encoding='utf-8'))
    assert not BoardModel(board, log=lambda *a: None).whole and BoardModel(board, ground='none', log=lambda *a: None).whole


def test_factorisations_and_reused_solves_from_the_stage_records():
    from types import SimpleNamespace
    from mom25d.kicad2ems.mom import factorisations, reused
    log = SimpleNamespace(records=[('LU factor', 1.0, 'N=100, complex64, kept'), ('LU factor (GPU)', 1.0, ''),
                                   ('refinement (GMRES, reused factorisation)', 0.5, ''),
                                   ('refinement (GMRES, reused factorisation)', 0.5, ''), ('LU factor', 1.0, 'N=100')],
                          notes=['the LU of 1 GHz did not precondition 3 GHz in 20 iterations (residual 1e-2) - a new one'])
    assert factorisations(log) == (1, 2)
    assert reused(log) == 1                                         # one of the two attempts converged
    assert reused(SimpleNamespace()) == 0


def test_mixed_models_the_board_with_mom25d_s_options(monkeypatch):
    """--mom-planes, --mom-stitching-pads and --diel reach the mixed solver's BoardModel, --mom-coarsen and
    --mom-quad-tol its mesh: it had taken mom25d's defaults, so --solvers mom,mixed compared two models."""
    from types import SimpleNamespace as NS
    import mom25d.board as B
    from mom25d.kicad2ems import mixed
    from mom25d.kicad2ems.mom import _model_kw
    seen = {}

    class Stop(Exception):
        pass

    def model(data, **kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(B, 'BoardModel', model)
    monkeypatch.setattr(mixed, '_metal', lambda *a, **k: (None, None))
    s = NS(mom_planes='holes', mom_stitching_pads='keep', diel='wideband', diel_fref=2.0)
    board = NS(to_dict=lambda: {}, ports=[])
    with pytest.raises(Stop):
        mixed.solve_mixed(board, np.array([1e9, 2e9]), model_kw=_model_kw(s), coarsen=4, quad_tol=20.0)
    assert seen['planes'] == 'holes' and seen['stitching_pads'] == 'keep' and seen['wideband'] == 2e9
