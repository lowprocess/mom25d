"""Slot planes (mom25d.slots): an opening's kernels against the frill holes' (the same field, superposed from point
elements), and the openings in a solve."""
import numpy as np
import pytest

from mom25d.multilayer import Layer, LayerStack
from mom25d.sim import Via
from mom25d.frill import Frill, FrillKernels, _nodes, DIPOLE_SHAPES
from mom25d.slots import SlotKernels, Slot

F = 3e9


def _stack():
    """A plane at interface 1 between two FR-4 layers, air above and below: metal on 0 and 2."""
    return LayerStack([Layer(0.8e-3, 4.4, 0.02), Layer(0.5e-3, 3.6, 0.01)], top='open', bottom='open', pec=[1])


def _annulus(a, b, nphi=192):
    """Points (x, y), weights (dA) and the polar (rho, phi, t) of the annulus a < rho < b, rho = b sin t (frill's
    nodes: the rim's 1 / sqrt in the Jacobian)."""
    t0, t, wq = _nodes(a, b)
    ph = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    T, PH = np.meshgrid(t, ph, indexing='ij')
    rho = b * np.sin(T)
    W = (wq[:, None] * b * np.cos(T) * rho) * (2 * np.pi / nphi)
    return rho * np.cos(PH), rho * np.sin(PH), W, rho, PH, T


def _superpose(kern, key, xo, yo, X, Y, W, Ex, Ey):
    """Psi (TM potential) and E (TE field) at (xo, yo) from the field (Ex, Ey) dA over the points X, Y."""
    dx, dy = xo - X, yo - Y
    rho = np.hypot(dx, dy)
    cs, sn = dx / rho, dy / rho
    psi = ((Ex * cs + Ey * sn) * kern.P[key](rho) * W).sum()
    u = kern.U[key](rho)
    c2, s2 = cs * cs - sn * sn, 2 * sn * cs
    u0, u2 = u[..., 0], u[..., 1]
    ex = (((u0 + u2 * c2) * Ex + u2 * s2 * Ey) * W).sum()
    ey = ((u2 * s2 * Ex + (u0 - u2 * c2) * Ey) * W).sum()
    return psi, ex, ey


@pytest.mark.parametrize('k', [0, 2])
def test_axisymmetric_field_is_the_frills(k):
    """A frill's coaxial field shape, superposed from the opening's point elements: the same potential on either
    side of the plane, and no TE field."""
    st = _stack()
    a, b = 0.2e-3, 0.5e-3
    via = Via(0.0, 0.0, 0, 2, radius=a)
    fr = Frill(via, 1, b, a=a, modes=1, dipoles=0)
    fk = FrillKernels(F, st, [fr], [0, 2], 0.05)
    sk = SlotKernels(F, st, [Slot(1, None)], [0, 2], 0.05)
    X, Y, W, rho, PH, T = _annulus(a, b)
    g = 1.0 / (rho * np.cos(T)) / np.arccosh(b / a)          # profile m = 0: 1 / (rho sqrt(1 - rho^2 / b^2))
    Ex, Ey = g * np.cos(PH), g * np.sin(PH)
    s = fk.shape(fr, 0)
    for xo in (0.3e-3, 0.9e-3, 2.5e-3, 6e-3):
        psi, ex, ey = _superpose(sk, (1, k), xo, 0.0, X, Y, W, Ex, Ey)
        ref = fk.psi[(s, k)](xo)
        assert abs(psi - ref) < 2e-3 * abs(ref), (xo, psi, ref)
        assert abs(ex) + abs(ey) < 1e-4 * abs(ref) / xo         # (axisymmetric: TM only; E ~ Psi / rho)


def test_dipole_field_is_the_frills():
    """A frill's first-order shape (cos), superposed: its potential (cos phi) and its TE field as frill's."""
    st = _stack()
    a, b, k = 0.15e-3, 0.45e-3, 2
    via = Via(0.0, 0.0, 0, 2, radius=a)
    fr = Frill(via, 1, b, a=a, modes=1, dipoles=2)
    fk = FrillKernels(F, st, [fr], [0, 2], 0.05)
    sk = SlotKernels(F, st, [Slot(1, None)], [0, 2], 0.05)
    X, Y, W, rho, PH, T = _annulus(a, b)
    for j in range(2):
        R, Fn = DIPOLE_SHAPES[j]
        c, q = np.cos(T), a / rho
        Rv = R(c, q) / b if R else 0 * rho
        Fv = Fn(c, q) / b if Fn else 0 * rho
        # E = rho^ R cos(phi) + phi^ F sin(phi)
        Er, Ep = Rv * np.cos(PH), Fv * np.sin(PH)
        Ex, Ey = Er * np.cos(PH) - Ep * np.sin(PH), Er * np.sin(PH) + Ep * np.cos(PH)
        s = fk.shape(fr, ('c', j))
        for xo, yo in ((0.8e-3, 0.3e-3), (2.0e-3, -1.1e-3), (0.2e-3, 0.25e-3)):
            psi, ex, ey = _superpose(sk, (1, k), xo, yo, X, Y, W, Ex, Ey)
            d = np.hypot(xo, yo)
            cs, sn = xo / d, yo / d
            ref = fk.psi[(s, k)](d) * cs
            U = fk.U[(s, k)](d)
            c2, s2 = cs * cs - sn * sn, 2 * sn * cs
            rx, ry = U[0] + U[1] * c2, U[1] * s2
            assert abs(psi - ref) < 3e-3 * abs(ref), (j, xo, psi, ref)
            scale = max(abs(rx), abs(ry))
            assert abs(ex - rx) < 3e-3 * scale and abs(ey - ry) < 3e-3 * scale, (j, xo, (ex, ey), (rx, ry))


def test_remainder_is_the_frills():
    """A chain's remainder from the coaxial shape, superposed: frill's."""
    st = _stack()
    a, b = 0.2e-3, 0.5e-3
    via = Via(0.0, 0.0, 0, 2, radius=a)
    fr = Frill(via, 1, b, a=a, modes=1, dipoles=0)
    fk = FrillKernels(F, st, [fr], [0, 2], 0.05)
    sk = SlotKernels(F, st, [Slot(1, None)], [0, 2], 0.05)
    X, Y, W, rho, PH, T = _annulus(a, b)
    g = 1.0 / (rho * np.cos(T)) / np.arccosh(b / a)
    Ex, Ey = g * np.cos(PH), g * np.sin(PH)
    s = fk.shape(fr, 0)
    for span in ((1, 2), (0, 1)):
        R = sk.rem(1, span)
        for xo in (0.9e-3, 3e-3):
            dx, dy = xo - X, -Y
            d = np.hypot(dx, dy)
            got = ((Ex * dx / d + Ey * dy / d) * R(d) * W).sum()
            ref = fk.rem(s, span, np.array([xo]))[0]
            assert abs(got - ref) < 3e-3 * abs(ref), (span, xo, got, ref)


@pytest.mark.parametrize('er', [1.0, 4.4])
def test_babinet(er):
    """An opening's own matrix in an infinite PEC plane in a homogeneous medium is (4 / eta^2) x the impedance
    matrix of a strip of its shape there, on the same mesh (Babinet): the dual of mom25d's strips, exactly."""
    from shapely.geometry import box
    from mom25d.sim import Simulation
    from mom25d.slots import SlotTable, self_block
    from mom25d.base import MU0, EPS0, quad_pts, _BAR3, _BAR6, _W3, _W6
    mm = 1e-3
    shape = box(0, 0, 8 * mm, 1.0 * mm)
    st = lambda pec: LayerStack([Layer(1 * mm, er, 0.0)], top='open', top_er=er, bottom='open', bottom_er=er, pec=pec)
    sim = Simulation(st(()), {1: [shape]}, dmax=0.5 * mm, solver='dense', verbose=False)
    Z = sim.impedance(F)[:sim.Ne, :sim.Ne]
    s = Slot(0, shape)
    r = s.rwg = sim.rwg                                  # the strip's own mesh
    s.P3, s.W3 = quad_pts(r, _BAR3), r.area[:, None] * _W3[None, :]
    s.P6, s.W6 = quad_pts(r, _BAR6), r.area[:, None] * _W6[None, :]
    V = r.nodes[r.tris]
    s.size = np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2).max(1)
    Y = self_block([s], SlotTable(F, st((0,)), 0, sim.rho_max), 2 * np.pi * F, sim.near_fac)
    dual = 4 * EPS0 * er / MU0 * Z
    assert np.linalg.norm(Y - dual) < 1e-4 * np.linalg.norm(dual)


def test_slot_under_a_line():
    """A microstrip over a narrow slot across it in its ground (air below): the slot plane against the ground
    meshed (whole stack, 0.3 mm cells, a 24 mm ground: its edges barely matter here), within 0.03 in S, and the
    line alone (no opening) as the solid ground's."""
    from shapely.geometry import box
    from mom25d.sim import Simulation, ViaPort
    mm = 1e-3
    L, W = 16 * mm, 1.5 * mm
    strip = box(0, -W / 2, L, W / 2)
    slot = box(L / 2 - 0.25 * mm, -5 * mm, L / 2 + 0.25 * mm, 5 * mm)
    ports = lambda: [ViaPort(Via(0.25 * mm, 0, 0, 1, radius=0.15 * mm), name='P1'),
                     ViaPort(Via(L - 0.25 * mm, 0, 0, 1, radius=0.15 * mm), name='P2')]
    st = LayerStack([Layer(0.8 * mm, 4.4, 0.0)], top='open', bottom='open', pec=[0])
    sim = Simulation(st, {1: [strip]}, ports=ports(), dmax=0.3 * mm, solver='auto', verbose=False,
                     slots=[Slot(0, slot)])
    got = np.array([sim.sparams(f) for f in (3e9, 6e9, 8e9)])
    # the whole stack with the 24 x 24 mm ground meshed at 0.3 mm (30,435 unknowns), 2026-11-15
    ref = {3e9: (-13.57, -0.22, -122.6), 6e9: (-6.62, -1.32, 106.4), 8e9: (-2.10, -5.11, 9.6)}
    for s, (f, (s11, s21, ph)) in zip(got, ref.items()):
        r21 = 10 ** (s21 / 20) * np.exp(1j * np.radians(ph))
        assert abs(abs(s[0, 0]) - 10 ** (s11 / 20)) < 0.03 and abs(s[1, 0] - r21) < 0.04, (f, s)


def test_frill_dipoles_couple_to_vias_ending_on_the_plane():
    """Two via ports between the ground and an inner plane, either side of a 0.6 mm hole round a floating via (no
    traces: only the vias' coupling to the hole's first-order shapes and the hole's own admittance count). The hole
    lowers Z21; with the plane's potential at the vias' feet (frill.py, 2026-11-15) the hole model gives the
    plane meshed at 0.15 mm's Z21 to 3 % (0.246 / 0.683 / 2.091 ohm at 2 / 5 / 10 GHz; no hole 0.289 / 0.804 /
    2.491). Without it the dipoles did nothing here (0.287 / 0.792 / 2.354)."""
    from shapely.geometry import Point
    from mom25d.sim import Simulation, ViaPort
    from mom25d.frill import Frill
    mm = 1e-3
    H, ER, TD, A, B, D = 0.2 * mm, 4.3, 0.02, 0.125 * mm, 0.6 * mm, 1.3 * mm
    layers = [Layer(H, ER, TD) for _ in range(3)]
    pads = [Point(0, 0).buffer(0.2 * mm)]
    sig = Via(0, 0, 1, 3, A, name='sig')
    ports = [ViaPort(Via(-D, 0, 0, 2, 0.125 * mm), name='P1'), ViaPort(Via(D, 0, 0, 2, 0.125 * mm), name='P2')]
    gv = [Via(x * mm, s * 2.0 * mm, 0, 2, 0.125 * mm) for x in np.arange(-4.8, 4.81, 1.2) for s in (1, -1)]
    gv += [Via(sx * 4.8 * mm, y * mm, 0, 2, 0.125 * mm) for sx in (1, -1) for y in (-1, 0, 1)]
    sim = Simulation(LayerStack(layers, pec=(2,)), {1: pads, 3: pads}, ports, vias=[sig] + gv,
                     frills=[Frill(sig, 2, B, dipoles=2)], dmax=0.3 * mm, solver='direct', precision='double',
                     gpu='off', verbose=False)
    for f, ref in ((2e9, 0.246), (5e9, 0.683), (10e9, 2.091)):
        S = sim.sparams(f)
        Z = 50 * (np.eye(2) + S) @ np.linalg.inv(np.eye(2) - S)
        assert abs(Z[1, 0].imag - ref) < 0.03 * ref, (f, Z[1, 0])
