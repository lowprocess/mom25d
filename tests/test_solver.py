"""
Stage-3 checks (fast): vias, off-plane integrals, mesher robustness, lumped elements, AAA,
KiCad / Gerber import, fast operator vs dense.
"""
import os
import sys
import subprocess

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'mom25d')

from mom25d.base import C0, MU0, EPS0                                          # noqa: E402
from mom25d.multilayer import Layer, LayerStack, line_char_ml                     # noqa: E402
from mom25d.via import ViaKernels                                         # noqa: E402
from mom25d.sing import offplane_pairs                                    # noqa: E402
from mom25d.geo import mesh_polygons, LinePort                            # noqa: E402
from mom25d.sim import Simulation, SeriesLumped, Via, ViaPort, rlc        # noqa: E402
from mom25d.afs import aaa                                                # noqa: E402


def test_via_kernels_exact_in_air():
    """Via-via reaction (self and mutual) and the via->strip correction D against images."""
    h, a, f = 0.5e-3, 0.15e-3, 10e9
    w = 2 * np.pi * f; k = w / C0
    vk = ViaKernels(f, LayerStack([Layer(h, 1.0)]), [(0, 1)], [1], 20e-3)
    assert np.max(np.abs(vk.D[(1, (0, 1))](np.array([1e-4, 1e-3])))) < 1e-10      # D == 0 in air
    for d in (a, 1e-3):
        tot = vk.R((0, 1), (0, 1), d)[0] + vk.homog((0, 1), (0, 1), d)[0]
        Rp = np.hypot(d, 2 * h)
        phi = (np.exp(-1j * k * d) / d - np.exp(-1j * k * Rp) / Rp) / (1j * w * 4 * np.pi * EPS0)
        g = lambda z, zp: np.exp(-1j * k * np.hypot(d, z - zp)) / np.hypot(d, z - zp)
        I = sum((1j ** p) * quad(lambda z: quad(lambda zp: (g(z, zp).real, g(z, zp).imag)[p], -h, h,
                                                points=[z], limit=200)[0], 0, h, limit=200)[0] for p in (0, 1))
        ref = 1j * w * MU0 / (4 * np.pi) * I + phi
        assert abs(tot + phi - ref) / abs(ref) < 1e-4


def test_offplane_reduces_to_inplane_and_converges():
    from mom25d.base import make_mesh, RWG, singular_moments
    nodes, tris = make_mesh([(0, 0, 0.3e-3, 0.06e-3)], 0.15e-3, 0.03e-3)
    M0, _, _ = singular_moments(RWG(nodes, tris))
    m0, *_ = offplane_pairs(nodes, tris, [0, 0], [0, 3], 0.0)
    assert np.allclose(m0, [M0[0, 0], M0[0, 3]], rtol=1e-12)
    a, *_ = offplane_pairs(nodes, tris, [0], [3], 1e-3)          # far: ~ A_p A_q / R
    V = nodes[tris]
    ar = 0.5 * np.abs((V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0]))
    R = np.sqrt(np.sum((V[0].mean(0) - V[3].mean(0)) ** 2) + 1e-6)
    assert abs(a[0] / (ar[0] * ar[3] / R) - 1) < 0.02


def test_mesher_t_junction_port_lines():
    """Port cut lines ending on a polygon edge must split the edge exactly (no sliver cascade)."""
    from shapely.geometry import box
    w = 0.15e-3
    g = box(14e-3, -5.075e-3, 22.075e-3, -4.925e-3)
    lp = LinePort(1, (21.85e-3, -5e-3), (-1, 0), w, 0.0)
    lines = [lp.line(s) for s in np.arange(0, 6e-3, 0.3e-3)]
    nodes, tris, _ = mesh_polygons(g, 0.3e-3, w, 4, lines=lines)
    near = (np.abs(nodes[:, 0] - 17.95e-3) < 0.02e-3) & (np.abs(nodes[:, 1] + 5.075e-3) < 1e-6)
    assert near.sum() <= 2


def test_series_resistor_matches_circuit_theory():
    w, h = 0.35e-3, 0.2e-3
    st = LayerStack([Layer(h, 4.3)])
    g, zc = line_char_ml(st, 1, w, [5e9]); z = zc[0].real
    L = 12e-3
    p1 = LinePort(1, (-L / 2 + 0.5e-3, 0), (1, 0), w, 4e-3, s0=1.5e-3, step=0.7e-3, zc=z)
    p2 = LinePort(1, (L / 2 - 0.5e-3, 0), (-1, 0), w, 4e-3, s0=1.5e-3, step=0.7e-3, zc=z)
    sim = Simulation(st, {1: [(-L / 2, -w / 2, L / 2, w / 2)]}, [p1, p2],
                     lumped=[SeriesLumped(1, (0, 0), (1, 0), w, 50.0)], dmax=0.25e-3, w_ref=w, verbose=False)
    S = sim.sparams(5e9)
    assert abs(abs(S[1, 0]) - 2 * z / (2 * z + 50)) < 0.01


def test_via_ports_on_a_line_are_reciprocal_and_lossless():
    from shapely.geometry import box, Point
    w, h, L = 0.35e-3, 0.2e-3, 10e-3
    st = LayerStack([Layer(h, 4.3)])
    v1, v2 = Via(-L / 2, 0, 0, 1, 0.1e-3), Via(L / 2, 0, 0, 1, 0.1e-3)
    geo = [box(-L / 2, -w / 2, L / 2, w / 2), Point(-L / 2, 0).buffer(0.25e-3), Point(L / 2, 0).buffer(0.25e-3)]
    sim = Simulation(st, {1: geo}, [ViaPort(v1, 54.0), ViaPort(v2, 54.0)], dmax=0.3e-3, w_ref=w, verbose=False)
    S = sim.sparams(2e9)
    assert abs(S[0, 1] - S[1, 0]) < 1e-8
    assert abs(np.sum(np.abs(S[:, 0]) ** 2) - 1) < 2e-3


def test_aaa_interpolates_resonances():
    def F(f):
        g = f / 1e9
        return np.array([sum(1 / (1j * (g - p) + 0.01) for p in (2.4, 3.6))])
    fs = np.linspace(1e9, 6e9, 25)
    r = aaa(fs / 1e9, np.array([F(f) for f in fs]))
    fg = np.linspace(1e9, 6e9, 301)
    assert np.max(np.abs(r(fg / 1e9)[:, 0] - np.array([F(f)[0] for f in fg]))) < 1e-6


@pytest.fixture(scope='module')
def demo_board(tmp_path_factory):
    d = tmp_path_factory.mktemp('board')
    base = str(d / 'demo')
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'make_demo_board.py'), base])
    return base


def test_kicad_import(demo_board):
    from mom25d.kicad import KiCadBoard
    b = KiCadBoard(demo_board + '.kicad_pcb')
    m = b.build(nets=['RF', 'GND'], ground='B.Cu', pos_file=demo_board + '-top.pos', include_mask=False)
    assert m.iface == {'In2.Cu': 1, 'In1.Cu': 2, 'F.Cu': 3}
    assert [l.h for l in m.stack.layers] == pytest.approx([0.2e-3] * 3)
    assert sorted(m.port_fps) == [1, 2]
    ps = m.ports(dmax=0.3e-3)
    assert [p.layer for p in ps] == [3, 1]
    assert ps[0].w == pytest.approx(0.35e-3, rel=1e-6) and ps[1].w == pytest.approx(0.15e-3, rel=1e-6)
    assert len(m.vias) == 23
    signal = [v for v in m.vias if v.name.startswith('RF')][0]
    assert (signal.k_bot, signal.k_top) == (1, 3)


def test_gerber_roundtrip_matches_native(demo_board, tmp_path):
    from shapely.ops import unary_union
    from mom25d.kicad import KiCadBoard
    from mom25d.gerber import GerberBoard
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'export_demo_gerbers.py'),
                           demo_board + '.kicad_pcb', str(tmp_path)])
    k = KiCadBoard(demo_board + '.kicad_pcb')
    g = GerberBoard(str(tmp_path / 'demo-job.gbrjob'))
    for lay in ('In1.Cu', 'In2.Cu', 'B.Cu'):
        a = unary_union([x for l, n, x in k.items if l == lay])
        b = unary_union([x for l, n, x in g.items if l == lay])
        assert a.symmetric_difference(b).area < 0.05                    # mm^2
    assert len(g.vias) == len(k.vias)


def test_fast_operators_match_dense():
    """Direct (centroid far field + exact near correction) and ACA/GMRES against the dense reference
    on a two-port line between via ports."""
    from shapely.geometry import box, Point
    w, h, L = 0.35e-3, 0.2e-3, 10e-3
    st = LayerStack([Layer(h, 4.3, 0.02)])
    v1, v2 = Via(-L / 2, 0, 0, 1, 0.1e-3), Via(L / 2, 0, 0, 1, 0.1e-3)
    geo = [box(-L / 2, -w / 2, L / 2, w / 2), Point(-L / 2, 0).buffer(0.25e-3), Point(L / 2, 0).buffer(0.25e-3)]
    out = {}
    for solver in ('dense', 'direct', 'aca'):
        sim = Simulation(st, {1: geo}, [ViaPort(v1, 54.0), ViaPort(v2, 54.0)], dmax=0.3e-3, w_ref=w,
                         solver=solver, verbose=False)
        out[solver] = sim.sparams(4e9)
    assert np.max(np.abs(out['direct'] - out['dense'])) < 0.01
    assert np.max(np.abs(out['aca'] - out['direct'])) < 1e-3



def test_core_matches_numpy_and_solvers_agree():
    """Compiled core (if built) vs NumPy fallback, and direct/mixed/BLR agreement."""
    core = pytest.importorskip('mom25d._core')
    from shapely.geometry import box, Point
    st = LayerStack([Layer(0.2e-3, 4.3, 0.02), Layer(0.2e-3, 4.3, 0.02)])
    w, L = 0.35e-3, 8e-3
    v1 = Via(L / 2, 0, 0, 2, 0.1e-3)
    geo2 = [box(-L / 2, -w / 2, L / 2, w / 2), Point(L / 2, 0).buffer(0.25e-3)]
    geo1 = [box(-L / 2, -2e-3, L / 2 + 1e-3, 2e-3).difference(Point(L / 2, 0).buffer(0.5e-3))]
    mk = lambda s, p='double': Simulation(st, {1: geo1, 2: geo2}, [ViaPort(Via(-L / 2, 0, 1, 2, 0.1e-3)), ViaPort(v1)],
                                          dmax=0.4e-3, dmax_layer={1: 0.6e-3}, w_ref={2: w, 1: 1e-3}, solver=s,
                                          precision=p, verbose=False)
    sim = mk('direct')
    fo = sim.fast
    fo.setup(4e9, mode='aca')
    rows, cols = np.arange(0, sim.Ne, 7), np.arange(0, sim.Ne, 5)
    Bc = fo.entry(rows, cols)
    fo.core = None
    Bp = fo.entry(rows, cols)
    assert np.abs(Bc - Bp).max() < 1e-12 * np.abs(Bp).max()
    S = {k: mk(*k).sparams(4e9) for k in (('direct', 'double'), ('direct', 'mixed'), ('blr', 'double'))}
    ref = S[('direct', 'double')]
    for k, v in S.items():
        assert np.abs(v - ref).max() < 1e-6, k


def test_coupled_stripline_even_odd_vs_cohn():
    from mom25d.sim import line_char_pair, coupled_stripline_cohn
    b, er, w, s = 0.4e-3, 4.3, 0.15e-3, 0.15e-3
    st = LayerStack([Layer(b / 2, er), Layer(b / 2, er)], top='pec')
    ge, ze, go, zo = line_char_pair(st, 1, w, s, [5e9])
    Ze, Zo = coupled_stripline_cohn(w, s, b, er)
    assert abs(ze[0].real / Ze - 1) < 0.025 and abs(zo[0].real / Zo - 1) < 0.025
    k0 = 2 * np.pi * 5e9 / C0
    assert abs((ge[0].imag / k0) ** 2 / er - 1) < 0.003 and abs((go[0].imag / k0) ** 2 / er - 1) < 0.003


def test_differential_through_line():
    from mom25d.sim import DiffPort
    st = LayerStack([Layer(0.2e-3, 4.3)])
    w, s, L = 0.25e-3, 0.2e-3, 16e-3
    yc = (w + s) / 2
    shapes = {1: [(-L / 2, yc - w / 2, L / 2, yc + w / 2), (-L / 2, -yc - w / 2, L / 2, -yc + w / 2)]}

    def dp(x, d, name):
        kw = dict(s0=0.6e-3, s1=6e-3, step=0.6e-3)
        return DiffPort(LinePort(1, (x, yc), (d, 0), w, -0.3e-3, **kw), LinePort(1, (x, -yc), (d, 0), w, -0.3e-3, **kw),
                        112.4, 36.5, name)
    sim = Simulation(st, shapes, [dp(-L / 2 + 0.3e-3, 1, 'P1'), dp(L / 2 - 0.3e-3, -1, 'P2')], dmax=0.3e-3,
                     w_ref=w, ncross=4, verbose=False)
    S = sim.sparams(5e9)
    assert [p.name for p in sim.ports] == ['P1d', 'P1c', 'P2d', 'P2c']
    assert abs(abs(S[2, 0]) - 1) < 2e-3 and abs(S[0, 0]) < 0.01          # matched, lossless, differential
    assert abs(S[3, 0]) < 1e-3 and abs(S[0, 1]) < 1e-3                    # no mode conversion (symmetric)
    assert np.abs(S - S.T).max() < 1e-8
    assert sim.last['resid'].max() < 1e-2


def test_kicad_differential_ports(tmp_path):
    base = str(tmp_path / 'diff')
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'make_demo_board.py'), base, 'diff'])
    from mom25d.kicad import KiCadBoard
    from mom25d.sim import DiffPort
    m = KiCadBoard(base + '.kicad_pcb').build(nets=['D_P', 'D_N'], ground='B.Cu', pos_file=base + '-top.pos',
                                              include_mask=False)
    ps = m.ports(dmax=0.3e-3)
    assert [type(p) for p in ps] == [DiffPort, DiffPort]
    assert ps[0].pos.w == pytest.approx(0.25e-3, rel=1e-6)
    assert abs((ps[0].pos.c - ps[0].neg.c) @ ps[0].pos.t) == pytest.approx(0.45e-3, rel=1e-3)
