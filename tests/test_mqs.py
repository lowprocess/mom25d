"""The magneto-quasistatic solver (mom25d.mqs) and the near field (mom25d.farfield.near_fields, mom25d.biot) against
closed forms: a trace over a plane (its return current under it, its inductance, its field by image theory), a
via's partial inductance, the spectral H against Biot-Savart."""
import numpy as np
import pytest
from shapely.geometry import box, mapping, Point

from mom25d.base import MU0
from mom25d.mqs import MQS, filament_mutual

H_TR, W_TR = 0.5, 0.5                     # mm: the trace's height over the plane, its width


def trace_board(dmax_plane=1.0):
    """A 30 x 20 mm plane (B.Cu) and a 0.5 mm trace 0.5 mm over it (F.Cu) from x 3 to 27 at y 10, one net: a via at
    the far end joins them; at the near end a pad on F.Cu with its via down to the plane, beside the trace's pad."""
    cu = {'B.Cu': [dict(net='N', polygon=mapping(box(0, 0, 30, 20)))],
          'F.Cu': [dict(net='N', polygon=mapping(box(3, 10 - W_TR / 2, 27.5, 10 + W_TR / 2))),
                   dict(net='N', polygon=mapping(box(2.0, 9.5, 3.0, 10.5))),          # the trace's start pad
                   dict(net='N', polygon=mapping(box(2.0, 7.6, 3.0, 8.6)))]}          # the return's pad
    vias = [dict(x=27.2, y=10, r=0.15, z0=0, z1=H_TR, net='N'), dict(x=2.5, y=8.1, r=0.15, z0=0, z1=H_TR, net='N')]
    stack = dict(layers=[dict(name='d', z0=0, z1=H_TR, er=1.0, tand=0, kind='diel')],
                 copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=H_TR, t=0.035)], thickness=H_TR)
    return dict(stack=stack, copper=cu, vias=vias, outline=mapping(box(0, 0, 30, 20)))


@pytest.fixture(scope='module')
def solved():
    b = trace_board()
    q = MQS(b, 'N', terminals={'s': [('F.Cu', box(2.0, 9.6, 2.6, 10.4))], 'g': [('F.Cu', box(2.0, 7.7, 2.6, 8.5))]},
            dmax=0.6, log=lambda *a: None)
    q.source('I', 's', 'g', part_h=0.0)
    return q, q.solve([1e3, 1e7])


def plane_profile(q, r, fi, x0=15.0):
    """The plane's Jx (A/m) across y at x0 (mm), from its triangles' centroids near x0."""
    cen, J = q.current_density(r, fi)['B.Cu']
    sel = np.abs(cen[:, 0] * 1e3 - x0) < 1.0
    return cen[sel, 1] * 1e3, J[sel, 0]


def test_return_current_comes_under_the_trace_at_high_frequency(solved):
    q, r = solved
    frac = []
    for fi in (0, 1):
        y, jx = plane_profile(q, r, fi)
        order = np.argsort(y)
        y, jx = y[order], np.abs(jx[order])
        tot = np.trapezoid(jx, y)
        near = np.trapezoid(np.where(np.abs(y - 10) < H_TR, jx, 0), y)
        frac.append(near / tot)
    # 10 MHz: inductive, the image's 1 / (1 + (y / h)^2): half within +-h; 1 kHz: resistive, spread over the plane
    assert frac[1] == pytest.approx(0.5, abs=0.08)
    assert frac[0] < 0.3


def test_loop_inductance_and_resistance(solved):
    q, r = solved
    z = r.z['I']
    L = z[1].imag / (2 * np.pi * 1e7)
    # Wheeler, a thin strip over a plane (w / h <= 1): mu0 / 2 pi ln(8 h / w + w / 4 h) per length; 24.5 mm of it,
    # its two vias and the pads' turn besides
    Lw = MU0 / (2 * np.pi) * np.log(8 * H_TR / W_TR + W_TR / (4 * H_TR)) * 24.5e-3
    assert Lw < L < 1.25 * Lw
    # at 1 kHz the loop is its resistance: 24.5 mm of 0.5 mm, 35 um trace (and the plane, the vias)
    Rt = 24.5e-3 / (5.8e7 * 35e-6 * 0.5e-3)
    assert Rt < z[0].real < 1.4 * Rt


def test_field_above_the_trace_by_image_theory(solved):
    q, r = solved
    zp = H_TR + 2.0                               # mm above the plane
    pts = np.array([[15e-3, 10e-3, zp * 1e-3]])
    H = q.h_field(r, 1, pts)[0]
    Himg = 1 / (2 * np.pi) * (1 / ((zp - H_TR) * 1e-3) - 1 / ((zp + H_TR) * 1e-3))   # a long wire and its image
    assert abs(H[1]) == pytest.approx(Himg, rel=0.06)
    assert abs(H[0]) < 0.05 * abs(H[1]) and abs(H[2]) < 0.05 * abs(H[1])


def test_via_partial_inductance_is_rosas():
    h, a = 1.6e-3, 0.15e-3
    L = filament_mutual(0, h, 0, h, a)
    rosa = MU0 / (2 * np.pi) * (h * np.log((h + np.sqrt(h * h + a * a)) / a) - np.sqrt(h * h + a * a) + a)
    assert L == pytest.approx(rosa, rel=1e-9)
    assert filament_mutual(0, h, 0, h, 1e-3) < L


def test_open_boundary_fft_grid_matches_biot_savart(solved):
    q, r = solved
    # Observation window covers only a small part of the copper: deposition
    # must still retain the rest of the trace and its return plane.
    x, y = np.linspace(.012, .018, 25), np.linspace(.007, .013, 25)
    heights = [.0025, -.002]
    H = q.h_grid(r, [0, 1], x, y, heights)
    X, Y, Z = np.meshgrid(x[::6], y[::6], heights, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    ref = q.h_field(r, [0, 1], pts).reshape(5, 5, 2, 3, 2)
    assert abs(H[::6, ::6] - ref).max() < .01 * abs(ref).max()
    assert q.h_grid(r, 1, x, y, heights) == pytest.approx(H[..., 1])
    refined, info = q.h_grid(r, [0, 1], x, y, heights, rtol=.01, return_info=True)
    assert info['converged'] and info['refine'] >= 2
    assert abs(refined[::6, ::6]-ref).max() < abs(H[::6, ::6]-ref).max()
    with pytest.raises(RuntimeError, match='did not converge'):
        q.h_grid(r, 1, x, y, heights, rtol=.01, max_refine=1)


def test_spectral_near_field_h_is_biot_savart():
    """mom25d.farfield.near_fields' H (magnetostatic, with the ground's image) against mom25d.biot on the same
    currents: a strip's RWG currents, random."""
    from types import SimpleNamespace
    from mom25d.geo import mesh_polygons
    from mom25d.base import RWG
    from mom25d.multilayer import LayerStack, Layer
    from mom25d.farfield import near_fields, current_deposit
    from mom25d.biot import rwg_currents, h_sheets
    nodes, tris, _ = mesh_polygons(box(0, -1e-3, 10e-3, 1e-3), 0.4e-3, 0.4e-3, ncross=2)
    rwg = RWG(nodes, tris)
    I = np.random.default_rng(0).normal(size=rwg.Ne) + 0j
    st = LayerStack([Layer(0.5e-3, 4.0)])
    sim = SimpleNamespace(rwg=rwg, tri_idx={1: np.arange(len(tris))}, st=SimpleNamespace(at=lambda f: st, z=st.z))
    sim.st = st
    gx, gy = np.arange(-8, 18, 0.1) * 1e-3, np.arange(-12, 12, 0.1) * 1e-3
    Jx, Jy = current_deposit(sim, I, gx, gy, 1)
    _, H = near_fields(sim, {1: (Jx, Jy)}, gx, gy, 1e8, [1.5e-3])
    a, b = rwg_currents(rwg, I)
    pts = np.array([[x, y, 2.0e-3] for x in (2e-3, 5e-3, 8e-3) for y in (-1e-3, 0.0, 1.5e-3)])
    Hb = h_sheets([(nodes, tris, a, b, 0.5e-3), (nodes, tris, -a, -b, -0.5e-3)], pts)
    for p, hb in zip(pts, Hb):
        i, j = np.argmin(abs(gx - p[0])), np.argmin(abs(gy - p[1]))
        assert np.abs(H[i, j, 0] - hb).max() < 0.03 * np.abs(hb).max() + 1e-3 * np.abs(Hb).max()
