"""
Stage-3 checks (fast): vias, off-plane integrals, mesher robustness, lumped elements, AAA,
fast operator vs dense.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from mom25d.base import C0, MU0, EPS0
from mom25d.multilayer import Layer, LayerStack, line_char_ml
from mom25d.via import ViaKernels
from mom25d.sing import offplane_pairs
from mom25d.geo import mesh_polygons, LinePort
from mom25d.sim import Simulation, SeriesLumped, Via, ViaPort
from mom25d.afs import aaa


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


def test_via_via_table_matches_the_direct_transform():
    """The tabulated via-via remainder (boards with many via pairs) against the transform at each distance."""
    st = LayerStack([Layer(0.2e-3, 4.3), Layer(1.0e-3, 4.3), Layer(0.2e-3, 4.3)])
    vk = ViaKernels(5e9, st, [(0, 3), (1, 3)], [1, 2, 3], 40e-3)
    d = np.concatenate([[0.15e-3], np.geomspace(0.3e-3, 38e-3, 60)])
    for sm, sn in (((0, 3), (0, 3)), ((0, 3), (1, 3))):
        exact, tab = vk.R(sm, sn, d), vk.R_table(sm, sn)(d)
        assert np.abs(tab - exact).max() < 1e-4 * np.abs(exact).max()


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


def test_afs_does_not_stop_on_a_model_that_ignores_new_samples(monkeypatch):
    """Regression (d5787ca): a fit that a new sample does not move changes by nothing between iterations while
    it is still wrong (seen on a 10 dB coupler: stopped at 9 solves, -14 dB coupling instead of -11 dB). The
    sweep must also see the model predict each new sample before it stops."""
    import mom25d.afs as afs
    real, first = afs.aaa, []
    def stale(z, F, **kw):                                          # fitted to the first samples, ever after
        first.append(first[0] if first else (z, F))
        return real(*first[-1], **kw)
    monkeypatch.setattr(afs, 'aaa', stale)
    truth = lambda f: np.array([[0.3 + 0.2 * np.sin(f / 4e8)]], complex)
    seen = []
    _, _, fs, _, hist = afs.adaptive_sweep(lambda f: seen.append(f) or truth(f), 1e9, 6e9, n0=5, tol=2e-3,
                                           max_points=12, ngrid=401, log=lambda *a: None)
    assert len(fs) == 12                                            # ran to the cap: never converged
    assert all(change < 2e-3 for _, change in hist[2:])             # although the model stopped changing


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
    pytest.importorskip('mom25d._core')
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


def test_far_field_of_a_dipole_over_ground():
    """mom25d.farfield on a small x-directed current at height h above the ground in air against the image
    solution: |r E| = eta k p / (4 pi) * 2 sin(k h cos theta) * (cos theta cos phi, -sin phi), shape and size."""
    from types import SimpleNamespace as NS
    from mom25d.farfield import far_field
    f = 3e9
    k0 = 2 * np.pi * f / C0
    for h in (0.01, 0.2):
        H = h * C0 / f
        sim = NS(st=LayerStack([Layer(H, 1.0)]), rwg=NS(cent=np.array([[0.0, 0.0, H]]), area=np.array([1e-6])),
                 tri_idx={1: np.array([0])}, current_density=lambda I: (np.array([1.0 + 0j]), np.array([0j])))
        th, ph = np.array([0.0, 30, 60, 85]), np.array([0.0, 45, 90])
        Et, Ep = far_field(sim, None, f, th, ph)
        T, P = np.meshgrid(np.radians(th), np.radians(ph))
        amp = np.sqrt(MU0 / EPS0) * k0 * 1e-6 / (4 * np.pi) * 2 * np.sin(k0 * H * np.cos(T))
        assert np.allclose(np.abs(Et), amp * np.abs(np.cos(T) * np.cos(P)), rtol=1e-6, atol=1e-12)
        assert np.allclose(np.abs(Ep), amp * np.abs(np.sin(P)), rtol=1e-6, atol=1e-12)
