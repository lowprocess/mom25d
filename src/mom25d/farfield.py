"""
mom25d.farfield - the far field of the surface currents of a solve: patterns, directivity, gain, efficiency.

    ff = far_field(sim, I, f, theta, phi)        # r E_theta, r E_phi (V) on a theta x phi grid (degrees)
    p = antenna_powers(sim, I, f, port=0)          # accepted and radiated power, efficiency, directivity

Every metal interface radiates into the half-space above the stack (the top medium, top_er): its current
J(x, y), Fourier transformed at k_rho = k_top sin(theta), gives the tangential E on the top interface through
the stack's TM / TE transmission-line voltages (Michalski: E_u = -V^e J_u, E_v = -V^h J_v, u along k_rho),
and that E the far field (exact for the half-space above a plane, by stationary phase):

    r E_theta = j k_top / (2 pi) E_u,   r E_phi = j k_top cos(theta) / (2 pi) E_v      (e^{-j k r} dropped)

The ground is infinite: nothing radiates below it (theta > 90 deg), and what the substrate guides away
(surface waves) and its loss are not radiated, so efficiency = P_rad / P_accepted counts both as loss.
Vertical currents (vias) are left out of the far field: a probe or shorting via of a patch radiates
little next to the patch's edges. A board whose radiation depends on its finite ground (an inverted-F, a
monopole at the board edge) is not this model's antenna.
"""
import numpy as np

from .base import C0, MU0, EPS0

ETA0 = np.sqrt(MU0 / EPS0)


def _tl_voltage(stack, kr, w, o, s, mode):
    """Voltage at interface o for a unit shunt current source at interface s, in the TM ('e') or TE ('h')
    transmission-line network of the stack (LayerStack.spectral's network)."""
    Y0, tn, sc, Yd, Yu = stack._network(kr, w, mode)
    v = 1.0 / (Yu[s] + Yd[s])
    if o > s:
        for k in range(s + 1, o + 1):
            v = v * sc[k] / (1 + 1j * (Yu[k] / Y0[k]) * tn[k])
    elif o < s:
        for k in range(s, o, -1):
            v = v * sc[k] / (1 + 1j * (Yd[k - 1] / Y0[k]) * tn[k])
    return v


def far_field(sim, I, f, theta, phi):
    """r E_theta, r E_phi (V, complex, shape (len(phi), len(theta))) of the currents I (one excitation column)
    at f (Hz); theta from the board normal (0..90 deg: above the ground), phi from +x (degrees)."""
    st = sim.st
    if st.top != 'open':
        raise ValueError("a PEC cover on the stack: nothing radiates")
    w = 2 * np.pi * f
    k_top = w / C0 * np.sqrt(complex(st.top_er)).real
    th, ph = np.radians(np.asarray(theta, float)), np.radians(np.asarray(phi, float))
    TH, PH = np.meshgrid(th, ph)                                         # (nphi, ntheta)
    kr = k_top * np.sin(TH).ravel() + 1e-9 * k_top                      # k_rho = 0 at broadside: just off it
    kx, ky = kr * np.cos(PH).ravel(), kr * np.sin(PH).ravel()
    jx, jy = sim.current_density(I)                                      # per triangle (A/m)
    r = sim.rwg
    cen, area = r.cent[:, :2], r.area
    Eu = np.zeros(kr.shape, complex)
    Ev = np.zeros(kr.shape, complex)
    cphi, sphi = np.cos(PH).ravel(), np.sin(PH).ravel()
    for k, idx in sim.tri_idx.items():
        if not len(idx):
            continue
        ph_ = np.exp(1j * (np.outer(kx, cen[idx, 0]) + np.outer(ky, cen[idx, 1])))   # e^{j k.rho}
        Jx, Jy = ph_ @ (jx[idx] * area[idx]), ph_ @ (jy[idx] * area[idx])
        Ju, Jv = Jx * cphi + Jy * sphi, -Jx * sphi + Jy * cphi
        Eu += -_tl_voltage(st, kr, w, st.N, k, 'e') * Ju
        Ev += -_tl_voltage(st, kr, w, st.N, k, 'h') * Jv
    c = 1j * k_top / (2 * np.pi)
    Eth = (c * Eu).reshape(TH.shape)
    Eph = (c * np.cos(TH).ravel() * Ev).reshape(TH.shape)
    return Eth, Eph


def radiated(Eth, Eph, theta, phi, eta=ETA0):
    """Radiation intensity U (W/sr, (nphi, ntheta)) and its integral over the grid (the upper hemisphere when
    theta spans 0..90 and phi 0..360 deg)."""
    U = (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / (2 * eta)
    th, ph = np.radians(theta), np.radians(phi)
    return U, float(np.trapezoid(np.trapezoid(U * np.sin(th)[None, :], th, axis=1), ph))


def port_power(sim, I, port=0):
    """Power accepted at a port driven by mom25d's 1 V source (excitation column `port`): Re(V I*) / 2."""
    return 0.5 * abs(np.real(np.conj(np.sum(sim.port_current(I, port)))))


def antenna_powers(sim, I, f, port=0, step=2.0):
    """Accepted and radiated power, efficiency (radiated / accepted: substrate loss and surface waves are loss)
    and directivity of the currents I (driven at `port`), from the far field over the upper hemisphere."""
    th, ph = np.arange(0, 90 + step / 2, step), np.arange(0, 360 + step / 2, step)
    Eth, Eph = far_field(sim, I, f, th, ph)
    U, prad = radiated(Eth, Eph, th, ph)
    pacc = port_power(sim, I, port)
    return dict(f=f, Pacc=pacc, Prad=prad, eff=prad / pacc if pacc > 0 else np.nan,
                D=4 * np.pi * U.max() / prad if prad > 0 else np.nan, theta=th, phi=ph, U=U)
