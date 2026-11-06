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
    currents of every layer, and the via chains' (sim.chains, unknowns after the Ne edges, when the sim has them).
    I: one excitation (N) or several (N x k: E_u, E_v (ndir, k)), the phases e^{j k.rho} made once for all."""
    st = sim.st.at(w / (2 * np.pi))
    one = I is None or np.ndim(I) < 2
    kx, ky = kr * cphi, kr * sphi
    jx, jy = sim.current_density(I)                                      # per triangle (A/m), (ntri[, k])
    jx, jy = np.asarray(jx).reshape(len(sim.rwg.area), -1), np.asarray(jy).reshape(len(sim.rwg.area), -1)
    cen, area = sim.rwg.cent[:, :2], sim.rwg.area
    nk = jx.shape[1]
    Eu, Ev = np.zeros((len(kr), nk), complex), np.zeros((len(kr), nk), complex)
    for k, idx in sim.tri_idx.items():
        if not len(idx):
            continue
        ph_ = np.exp(1j * (np.outer(kx, cen[idx, 0]) + np.outer(ky, cen[idx, 1])))   # e^{j k.rho}
        J = ph_ @ np.concatenate([jx[idx] * area[idx][:, None], jy[idx] * area[idx][:, None]], 1)   # one product
        eu, ev = _surface_E_at(st, kr[:, None], w, o, k, J[:, :nk], J[:, nk:], cphi[:, None], sphi[:, None])
        Eu, Ev = Eu + eu, Ev + ev
    chains, ne = getattr(sim, 'chains', None) or [], getattr(sim, 'Ne', None)
    if chains and ne is not None:
        Iv = np.asarray(I).reshape(np.shape(I)[0], -1)[ne:ne + len(chains)]
        spans = {}
        for c, (v, a, b) in enumerate(chains):                          # one kernel per span, summed over its vias
            spans.setdefault((a, b), []).append((v.x, v.y, c))
        for (a, b), vs in spans.items():
            xy, rows = np.array([(x, y) for x, y, _ in vs]), [c for _, _, c in vs]
            Eu = Eu + _via_Eu(st, kr, w, o, range(a + 1, b + 1))[:, None] * (
                np.exp(1j * (np.outer(kx, xy[:, 0]) + np.outer(ky, xy[:, 1]))) @ Iv[rows])
    return (Eu[:, 0], Ev[:, 0]) if one else (Eu, Ev)


def _surface_E_at(st, kr, w, o, k, Jx, Jy, c, s):
    """E_u, E_v on interface o from the current spectrum (Jx, Jy) on interface k."""
    return (-st.voltage(kr, w, o, k, 'e') * (Jx * c + Jy * s), -st.voltage(kr, w, o, k, 'h') * (-Jx * s + Jy * c))


def far_field(sim, I, f, theta, phi):
    """r E_theta, r E_phi (V, complex, shape (len(phi), len(theta))) of the currents I (one excitation column)
    at f (Hz), the surface currents and the vias'; theta from the board normal (0..180 deg: above the board to 90,
    below it beyond; zero below the PEC ground), phi from +x (degrees). I: N x k for k excitations at once:
    (k, nphi, ntheta) each, the phases shared (an array's ports: one pass, not one each)."""
    st = sim.st.at(f)
    one = I is None or np.ndim(I) < 2
    th, ph = np.radians(np.asarray(theta, float)), np.radians(np.asarray(phi, float))
    TH, PH = np.meshgrid(th, ph)                                         # (nphi, ntheta)
    if st.top != 'open' and (TH < np.pi / 2).any():
        raise ValueError("a PEC cover on the stack: nothing radiates above it")
    w = 2 * np.pi * f
    Et = Ep = None
    t, cphi, sphi = TH.ravel(), np.cos(PH).ravel(), np.sin(PH).ravel()
    for up, sel in ((True, t <= np.pi / 2), (False, t > np.pi / 2)):
        if not sel.any() or (not up and st.bottom == 'pec'):
            continue                                                     # below the ground: nothing
        er = st.top_er if up else st.bottom_er
        k = w / C0 * np.sqrt(complex(er)).real
        kr = k * np.sin(t[sel]) + 1e-9 * k                               # k_rho = 0 at broadside: just off it
        Eu, Ev = _tangential(sim, I, w, st.N if up else 0, kr, cphi[sel], sphi[sel])
        Eu, Ev = np.reshape(Eu, (len(kr), -1)), np.reshape(Ev, (len(kr), -1))
        if Et is None:
            Et, Ep = np.zeros((TH.size, Eu.shape[1]), complex), np.zeros((TH.size, Eu.shape[1]), complex)
        c = (1 if up else -1) * 1j * k / (2 * np.pi)
        Et[sel], Ep[sel] = c * Eu, c * np.cos(t[sel])[:, None] * Ev
    if Et is None:                                                       # nothing radiates on this grid
        nk = 1 if one else np.shape(I)[1]
        Et, Ep = np.zeros((TH.size, nk), complex), np.zeros((TH.size, nk), complex)
    nk = Et.shape[1]
    Et, Ep = Et.T.reshape((nk,) + TH.shape), Ep.T.reshape((nk,) + TH.shape)
    return (Et[0], Ep[0]) if one else (Et, Ep)


def radiated(Eth, Eph, theta, phi, eta=ETA0):
    """Radiation intensity U (W/sr, (nphi, ntheta)) and its integral over the grid (the upper hemisphere when
    theta spans 0..90 and phi 0..360 deg)."""
    U = (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / (2 * eta)
    th, ph = np.radians(theta), np.radians(phi)
    return U, float(np.trapezoid(np.trapezoid(U * np.sin(th)[None, :], th, axis=1), ph))


def near_field(sim, layers_J, x, y, f, height, pad=2):
    """E (V/m, (nx, ny, 3) complex) on the plane `height` (m) above the top of the stack (near_fields' E at one
    height)."""
    return near_fields(sim, layers_J, x, y, f, [height], pad)[0][:, :, 0]


def near_fields(sim, layers_J, x, y, f, heights, pad=2):
    """E (V/m) and H (A/m), each (nx, ny, nh, 3) complex, on the planes `heights` (m) above the top of the stack,
    from the surface currents of each layer sampled on the grid x, y (m, uniform): layers_J = {interface: (Jx,
    Jy)}, each (nx, ny) (A/m; current_deposit). The FFT is periodic: the currents are zero-padded to `pad` x the
    grid (a port's strong field otherwise reappeared on the far side of a 10 mm margin) and the result cropped
    back. One spectrum serves every height: a volume costs an inverse FFT per height and component.

    E: the plane-wave spectrum of the currents through the stack's TM / TE line voltages to the top surface (as
    far_field), up by e^{-j kz h}; E_z from the divergence.
    H: magnetostatic, each layer's current at its height and, over the PEC ground, its image below it:
    A = mu0 J~ (e^{-k (z - z_k)} - e^{-k (z + z_k)}) / 2k, H = curl A / mu0. Exact for the near field (h well under
    lambda / 2 pi; the dielectric does not change H there). H from E by Faraday's law through the stack's spectral
    voltages was 15% low at 3 mm over rf_gcpw at 0.1 to 5 GHz (Hy; Hz agreed), against Biot-Savart of the same
    currents and their images (mom25d.biot), which this matches."""
    (Ek, kz, KX, KY, dxdy, w, mu), _ = _near_spectrum(sim, layers_J, x, y, f, pad)
    st = sim.st.at(f)
    nx0, ny0 = len(x), len(y)
    nx, ny = KX.shape
    kr = np.hypot(KX, KY)
    kr_ = np.where(kr > 0, kr, 1.0)
    ztop = float(st.z[-1])
    pec = st.bottom == 'pec'
    spec = {k: (np.fft.fft2(Jx, s=(nx, ny)) * dxdy, np.fft.fft2(Jy, s=(nx, ny)) * dxdy) for k, (Jx, Jy) in layers_J.items()}
    outE = np.zeros((nx0, ny0, len(heights), 3), complex)
    outH = np.zeros_like(outE)
    for i, h in enumerate(heights):
        up = np.exp(-1j * kz * h)
        for c, comp in enumerate(Ek):
            outE[:, :, i, c] = np.fft.ifft2(comp * up)[:nx0, :ny0] / dxdy
        z = ztop + h
        Hx, Hy, Hz = 0, 0, 0
        for k, (jx, jy) in spec.items():
            zk = float(st.z[k])
            F = np.exp(-kr * (z - zk)) - (np.exp(-kr * (z + zk)) if pec else 0.0)
            Hx = Hx + jy * F / 2
            Hy = Hy - jx * F / 2
            Hz = Hz + 1j * F * (KX * jy - KY * jx) / (2 * kr_)
        for c, comp in enumerate((Hx, Hy, Hz)):
            outH[:, :, i, c] = np.fft.ifft2(comp)[:nx0, :ny0] / dxdy
    return outE, outH


def _near_spectrum(sim, layers_J, x, y, f, pad):
    """The E spectrum at the top surface (Ex, Ey, Ez), kz, the wavenumber grids, dx dy, w and mu0."""
    st, w = sim.st.at(f), 2 * np.pi * f
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
    Ez = (KX * Ex + KY * Ey) / kz                                    # div E = 0 for e^{+j(kx x + ky y) - j kz z}
    return (((Ex, Ey, Ez), kz, KX, KY, dx * dy, w, MU0), None)


JITTER = 0.0137                 # sample points this fraction of a cell off the grid: never on a copper edge


def current_on_grid_avg(sim, I, x, y, k, sub=2):
    """current_on_grid averaged over sub x sub points in each cell of the uniform grid x, y: the current a cell
    carries, not its value at one point. A point on a trace's edge (where the current crowds) and its mirror in the
    gap beside it had a field leaning to one side of the trace; worse, copper edges on round coordinates put a grid
    row exactly on them, where the triangle search counts the edge on one side only (rf_gcpw at 0.05 mm: the edge
    current on one edge of the trace and not the other, its 1 mm near field 0.15 mm off centre). The points are
    also kept JITTER of a cell off the grid's lines. Returns Jx, Jy."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    sub = max(1, int(sub))
    off = (np.arange(sub) + 0.5) / sub - 0.5 + JITTER
    Jx, Jy = 0, 0
    for a in off:
        for b in off:
            jx, jy, _ = current_on_grid(sim, I, x + a * dx, y + b * dy * 1.0007, k)
            Jx, Jy = Jx + jx, Jy + jy
    return Jx / sub ** 2, Jy / sub ** 2


def current_deposit(sim, I, x, y, k, per_cell=2.0):
    """The surface current of interface k as a density on the uniform grid x, y (m), for the near field's FFT:
    each triangle's current integrated by a 6-point rule (triangles larger than `per_cell` cells split into four
    until they are not), each point's current moment spread onto its four grid points (cloud in cell). The total
    current and its centre are kept on any grid; sampling the current at the grid points instead was fragile (it
    crowds at a trace's edges: a row of points on one edge and none on the other put rf_gcpw's 1 mm near field
    0.15 mm off the trace's centre, and at another step halved it). Returns Jx, Jy ((nx, ny) complex, A/m)."""
    from .base import _BAR6, _W6
    from .biot import rwg_currents, _split4
    r = sim.rwg
    a, b = rwg_currents(r, np.asarray(I)[:r.Ne])
    idx = sim.tri_idx[k]
    V, aa, bb = r.nodes[r.tris[idx]][:, :, :2], a[idx], b[idx]
    dx, dy = x[1] - x[0], y[1] - y[0]
    for _ in range(6):
        size = np.sqrt(np.abs(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])) / 2)
        big = size > per_cell * min(dx, dy)
        if not big.any():
            break
        V = np.concatenate([V[~big], _split4(V[big])])
        aa = np.concatenate([aa[~big], np.tile(aa[big], 4)])
        bb = np.concatenate([bb[~big], np.tile(bb[big], (4, 1))])
    area = 0.5 * np.abs(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]))
    Q = np.einsum('qk,tkc->tqc', _BAR6, V)                              # (M, 6, 2)
    mom = (aa[:, None, None] * Q + bb[:, None, :]) * (area[:, None] * _W6[None])[..., None]   # A m per point
    fx, fy = (Q[..., 0].ravel() - x[0]) / dx, (Q[..., 1].ravel() - y[0]) / dy
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - i0, fy - j0
    nx, ny = len(x), len(y)
    out = [np.zeros((nx, ny), complex), np.zeros((nx, ny), complex)]
    for di, dj, wgt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        ii, jj = i0 + di, j0 + dj
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        for c in range(2):
            np.add.at(out[c], (ii[ok], jj[ok]), (mom[..., c].ravel() * wgt)[ok])
    return out[0] / (dx * dy), out[1] / (dx * dy)


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

