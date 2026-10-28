"""
mom25d.farfield - the far field of the surface currents of a solve: patterns, directivity, gain, efficiency.

    ff = far_field(sim, I, f, theta, phi)        # r E_theta, r E_phi (V) on a theta x phi grid (degrees)
    U, prad = radiated(Eth, Eph, theta, phi)       # radiation intensity and radiated power

Every metal interface radiates into the half-space above the stack (the top medium, top_er): its current
J(x, y), Fourier transformed at k_rho = k_top sin(theta), gives the tangential E on the top interface through
the stack's TM / TE transmission-line voltages (LayerStack.voltage; Michalski: E_u = -V^e J_u, E_v = -V^h J_v,
u along k_rho), and that E the far field (exact for the half-space above a plane, by stationary phase):

    r E_theta = j k_top / (2 pi) E_u,   r E_phi = j k_top cos(theta) / (2 pi) E_v      (e^{-j k r} dropped)

Vertical currents (the via chains: one unknown each, a +z current through its layers) radiate through the TM
network too: a unit current through layer j gives E_t = grad_t S on interface o, S~ = sum_j [V(z_j | z_o) -
V(z_{j-1} | z_o)] / kz_j^2 (mom25d.via), so E_u = -j k_rho S~ e^{j k.rho_via} and nothing in E_v.

Over the PEC ground nothing radiates below it (theta > 90 deg: zero). A whole stack (LayerStack bottom='open':
no infinite ground, the planes meshed) radiates below as well: mirrored in z, the lower half-space's far field
comes from the tangential E on interface 0, k_bot the half-space's wavenumber (cos(theta) < 0):

    r E_theta = -j k_bot / (2 pi) E_u,   r E_phi = -j k_bot cos(theta) / (2 pi) E_v

What the substrate guides away (surface waves) and its loss are not radiated, so efficiency = P_rad / P_accepted
counts both as loss.
"""
import numpy as np

from .base import C0, MU0, EPS0

ETA0 = np.sqrt(MU0 / EPS0)


def _surface_E(st, kr, w, k, Jx, Jy, c, s):
    """E_u, E_v on the top interface from the current spectrum (Jx, Jy) on interface k (u along k_rho, at angle
    cos c, sin s from x)."""
    return (-st.voltage(kr, w, st.N, k, 'e') * (Jx * c + Jy * s), -st.voltage(kr, w, st.N, k, 'h') * (-Jx * s + Jy * c))


def _via_Eu(st, kr, w, o, layers):
    """E_u on interface o (spectral) from a unit +z current through `layers` (TM only): -j k_rho S~ (module doc)."""
    k2 = (w / C0) ** 2
    S = 0.0
    for j in layers:
        S = S + (st.voltage(kr, w, j, o, 'e') - st.voltage(kr, w, j - 1, o, 'e')) / (k2 * st.eps(j) - kr ** 2)
    return -1j * kr * S


def _tangential(sim, I, w, o, kr, cphi, sphi):
    """E_u, E_v on interface o at the plane-wave directions (kr, cos / sin phi) from all the currents: the surface
    currents of every layer, and the via chains' (sim.chains, unknowns after the Ne edges, when the sim has them)."""
    st = sim.st
    kx, ky = kr * cphi, kr * sphi
    jx, jy = sim.current_density(I)                                      # per triangle (A/m)
    cen, area = sim.rwg.cent[:, :2], sim.rwg.area
    Eu, Ev = np.zeros(kr.shape, complex), np.zeros(kr.shape, complex)
    for k, idx in sim.tri_idx.items():
        if not len(idx):
            continue
        ph_ = np.exp(1j * (np.outer(kx, cen[idx, 0]) + np.outer(ky, cen[idx, 1])))   # e^{j k.rho}
        eu, ev = _surface_E_at(st, kr, w, o, k, ph_ @ (jx[idx] * area[idx]), ph_ @ (jy[idx] * area[idx]), cphi, sphi)
        Eu, Ev = Eu + eu, Ev + ev
    chains, ne = getattr(sim, 'chains', None) or [], getattr(sim, 'Ne', None)
    if chains and ne is not None:
        Iv = np.asarray(I)[ne:ne + len(chains)]
        spans = {}
        for c, (v, a, b) in enumerate(chains):                          # one kernel per span, summed over its vias
            spans.setdefault((a, b), []).append((v.x, v.y, Iv[c]))
        for (a, b), vs in spans.items():
            xy, cur = np.array([(x, y) for x, y, _ in vs]), np.array([i for _, _, i in vs])
            Eu = Eu + _via_Eu(st, kr, w, o, range(a + 1, b + 1)) * (np.exp(1j * (np.outer(kx, xy[:, 0]) +
                                                                                    np.outer(ky, xy[:, 1]))) @ cur)
    return Eu, Ev


def _surface_E_at(st, kr, w, o, k, Jx, Jy, c, s):
    """E_u, E_v on interface o from the current spectrum (Jx, Jy) on interface k."""
    return (-st.voltage(kr, w, o, k, 'e') * (Jx * c + Jy * s), -st.voltage(kr, w, o, k, 'h') * (-Jx * s + Jy * c))


def far_field(sim, I, f, theta, phi):
    """r E_theta, r E_phi (V, complex, shape (len(phi), len(theta))) of the currents I (one excitation column)
    at f (Hz), the surface currents and the vias'; theta from the board normal (0..180 deg: above the board to 90,
    below it beyond; zero below the PEC ground), phi from +x (degrees)."""
    st = sim.st
    th, ph = np.radians(np.asarray(theta, float)), np.radians(np.asarray(phi, float))
    TH, PH = np.meshgrid(th, ph)                                         # (nphi, ntheta)
    if st.top != 'open' and (TH < np.pi / 2).any():
        raise ValueError("a PEC cover on the stack: nothing radiates above it")
    w = 2 * np.pi * f
    Et, Ep = np.zeros(TH.size, complex), np.zeros(TH.size, complex)
    t, cphi, sphi = TH.ravel(), np.cos(PH).ravel(), np.sin(PH).ravel()
    for up, sel in ((True, t <= np.pi / 2), (False, t > np.pi / 2)):
        if not sel.any() or (not up and st.bottom == 'pec'):
            continue                                                     # below the ground: nothing
        er = st.top_er if up else st.bottom_er
        k = w / C0 * np.sqrt(complex(er)).real
        kr = k * np.sin(t[sel]) + 1e-9 * k                               # k_rho = 0 at broadside: just off it
        Eu, Ev = _tangential(sim, I, w, st.N if up else 0, kr, cphi[sel], sphi[sel])
        c = (1 if up else -1) * 1j * k / (2 * np.pi)
        Et[sel], Ep[sel] = c * Eu, c * np.cos(t[sel]) * Ev
    return Et.reshape(TH.shape), Ep.reshape(TH.shape)


def radiated(Eth, Eph, theta, phi, eta=ETA0):
    """Radiation intensity U (W/sr, (nphi, ntheta)) and its integral over the grid (the upper hemisphere when
    theta spans 0..90 and phi 0..360 deg)."""
    U = (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / (2 * eta)
    th, ph = np.radians(theta), np.radians(phi)
    return U, float(np.trapezoid(np.trapezoid(U * np.sin(th)[None, :], th, axis=1), ph))


def port_power(sim, I, port=0):
    """Power accepted at a port driven by mom25d's 1 V source (excitation column `port`): Re(V I*) / 2."""
    return 0.5 * abs(np.real(np.conj(np.sum(sim.port_current(I, port)))))


def near_field(sim, layers_J, x, y, f, height, pad=2):
    """E (V/m, (nx, ny, 3) complex) on the plane `height` (m) above the top of the stack, from the surface
    currents of each layer sampled on the grid x, y (m, uniform): layers_J = {interface: (Jx, Jy)}, each
    (nx, ny) (A/m). The plane-wave spectrum of the currents (FFT) through the stack's TM / TE line voltages
    to the top surface (as far_field), then up by e^{-j kz height}; E_z from the divergence. The FFT is
    periodic: the currents are zero-padded to `pad` x the grid (a port's strong field otherwise reappeared on
    the far side of a 10 mm margin) and the result cropped back."""
    st, w = sim.st, 2 * np.pi * f
    k_top = w / C0 * np.sqrt(complex(st.top_er)).real
    nx0, ny0 = len(x), len(y)
    nx, ny = pad * nx0, pad * ny0
    dx, dy = x[1] - x[0], y[1] - y[0]
    KX, KY = np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, dx), 2 * np.pi * np.fft.fftfreq(ny, dy), indexing='ij')
    kr = np.hypot(KX, KY) + 1e-9 * k_top
    cu, su = KX / kr, KY / kr
    Ex, Ey = np.zeros((nx, ny), complex), np.zeros((nx, ny), complex)
    for k, (Jx, Jy) in layers_J.items():
        Eu, Ev = _surface_E(st, kr, w, k, np.fft.fft2(Jx, s=(nx, ny)) * dx * dy, np.fft.fft2(Jy, s=(nx, ny)) * dx * dy,
                            cu, su)
        Ex += Eu * cu - Ev * su
        Ey += Eu * su + Ev * cu
    kz = -1j * np.sqrt(kr ** 2 - k_top ** 2 + 0j)                     # e^{-j kz z}: decaying when evanescent
    up = np.exp(-1j * kz * height)
    Ez = (KX * Ex + KY * Ey) / kz                                    # div E = 0 for e^{+j(kx x + ky y) - j kz z}
    return np.stack([np.fft.ifft2(c * up)[:nx0, :ny0] / (dx * dy) for c in (Ex, Ey, Ez)], axis=-1)


def current_on_grid(sim, I, x, y, k):
    """The surface current J (A/m) of interface k on the grid x, y (m): the RWG basis functions evaluated at each
    point (linear in each triangle: J = a r + b, continuous normal component across edges), 0 off the copper.
    Returns Jx, Jy ((nx, ny) complex) and the triangle of each point (-1 off the copper)."""
    from matplotlib.tri import Triangulation
    r = sim.rwg
    I = np.asarray(I)[:r.Ne]
    a, b = np.zeros(len(r.tris), complex), np.zeros((len(r.tris), 2), complex)
    for t, alpha, v in ((r.tp, r.l / (2 * r.area[r.tp]), r.vp), (r.tm, -r.l / (2 * r.area[r.tm]), r.vm)):
        np.add.at(a, t, I * alpha)
        np.add.at(b, t, -(I * alpha)[:, None] * v)
    idx = sim.tri_idx[k]
    used, local = np.unique(r.tris[idx], return_inverse=True)        # the layer's nodes, its triangles in them
    X, Y = np.meshgrid(x, y, indexing='ij')
    hit = Triangulation(r.nodes[used, 0], r.nodes[used, 1], local.reshape(-1, 3)).get_trifinder()(X, Y)
    t = idx[np.clip(hit, 0, None)]
    on = hit >= 0
    Jx, Jy = np.where(on, a[t] * X + b[t, 0], 0), np.where(on, a[t] * Y + b[t, 1], 0)
    return Jx, Jy, hit

