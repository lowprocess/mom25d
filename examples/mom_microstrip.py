"""
Minimal 2D electrostatic Method of Moments: microstrip characteristic impedance.

Method (pulse basis + point matching):
  * Ground plane at y=0 handled with image charges (no mesh needed).
  * Strip (finite thickness) meshed into segments; unknown = total surface charge.
  * Dielectric interface (y=h, outside strip footprint) replaced by bound
    charge in free space; unknown bound charge enforces D-continuity.
  * Solve dense Z*sigma = V, get capacitance/length, then Z0 = 1/(c*sqrt(C*C0)).
"""
import numpy as np

EPS0 = 8.8541878128e-12
C0_LIGHT = 299792458.0
GX, GW = np.polynomial.legendre.leggauss(12)   # Gauss-Legendre nodes/weights


def graded(p0, p1, n):
    """n segments from p0 to p1, cosine-clustered at both ends (corner singularities)."""
    s = (1 - np.cos(np.pi * np.linspace(0, 1, n + 1))) / 2
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    pts = p0 + s[:, None] * (p1 - p0)
    return pts[:-1], pts[1:]


def one_sided(p0, p1, n, k=4.0):
    """Geometric-ish grading, fine at p0, coarse at p1."""
    s = (np.exp(k * np.linspace(0, 1, n + 1)) - 1) / (np.exp(k) - 1)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    pts = p0 + s[:, None] * (p1 - p0)
    return pts[:-1], pts[1:]


def build_mesh(w, t, h, L, n_face=30, n_if=80):
    a, b = (-w / 2, h), (w / 2, h)
    c, d = (w / 2, h + t), (-w / 2, h + t)
    faces = [(a, b, True), (b, c, False), (c, d, False), (d, a, False)]
    cp0, cp1, bottom = [], [], []
    for p, q, is_bot in faces:
        n = n_face if abs(p[0] - q[0]) > 0 else max(6, n_face // 3)
        s0, s1 = graded(p, q, n)
        cp0.append(s0); cp1.append(s1); bottom += [is_bot] * n
    cond = (np.vstack(cp0), np.vstack(cp1), np.array(bottom))
    # dielectric interface on both sides of strip, refined near the strip
    r0, r1 = one_sided((w / 2, h), (L, h), n_if)
    l0, l1 = one_sided((-w / 2, h), (-L, h), n_if)
    iface = (np.vstack([l0, r0]), np.vstack([l1, r1]))
    return cond, iface


def geom(p0, p1):
    mid, vec = (p0 + p1) / 2, p1 - p0
    return mid, np.linalg.norm(vec, axis=1)


def quad_points(p0, p1):
    mid, vec = (p0 + p1) / 2, p1 - p0
    Q = mid[:, None, :] + 0.5 * GX[None, :, None] * vec[:, None, :]   # (M,G,2)
    wl = 0.5 * np.linalg.norm(vec, axis=1)[:, None] * GW[None, :]     # (M,G)
    return Q, wl


def potential_matrix(obs, s0, s1, self_diag=False):
    """phi at obs from unit (sigma/eps0) on each source segment, with ground image."""
    Q, wl = quad_points(s0, s1)
    Qi = Q * np.array([1, -1])
    R = obs[:, None, None, :] - Q[None]
    Ri = obs[:, None, None, :] - Qi[None]
    with np.errstate(divide="ignore"):
        A = (wl[None] * np.log(np.linalg.norm(R, axis=-1))).sum(-1)
    B = (wl[None] * np.log(np.linalg.norm(Ri, axis=-1))).sum(-1)
    if self_diag:  # analytic integral of ln|s| over a segment centred on obs
        _, ln = geom(s0, s1)
        A[np.arange(len(ln)), np.arange(len(ln))] = ln * (np.log(ln / 2) - 1)
    return -(A - B) / (2 * np.pi)


def normal_field_matrix(obs, s0, s1):
    """E_y (normal, +y) at obs from unit sigma/eps0 sources + ground images."""
    Q, wl = quad_points(s0, s1)
    R = obs[:, None, None, :] - Q[None]
    Qi = Q * np.array([1, -1])
    Ri = obs[:, None, None, :] - Qi[None]
    r2 = (R ** 2).sum(-1)
    ri2 = (Ri ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.where(r2 > 1e-30, R[..., 1] / r2, 0.0)
    image = Ri[..., 1] / ri2
    return (wl[None] * (direct - image)).sum(-1) / (2 * np.pi)


def capacitance(w, t, h, er, L=None, n_face=30, n_if=80):
    """Capacitance per metre."""
    L = L or 25 * h
    (c0, c1, bot), (i0, i1) = build_mesh(w, t, h, L, n_face, n_if)
    cm, clen = geom(c0, c1)
    im, ilen = geom(i0, i1)
    Nc, Ni = len(cm), len(im)

    Pcc = potential_matrix(cm, c0, c1, self_diag=True)
    Pci = potential_matrix(cm, i0, i1)
    Eic = normal_field_matrix(im, c0, c1)
    Eii = normal_field_matrix(im, i0, i1)

    k = 2 * (er - 1) / (er + 1)
    Z = np.zeros((Nc + Ni, Nc + Ni))
    Z[:Nc, :Nc], Z[:Nc, Nc:] = Pcc, Pci
    Z[Nc:, :Nc] = -k * Eic
    Z[Nc:, Nc:] = np.eye(Ni) - k * Eii
    rhs = np.zeros(Nc + Ni)
    rhs[:Nc] = 1.0                       # strip at 1 V, ground at 0 V

    sol = np.linalg.solve(Z, rhs)
    sc = sol[:Nc]
    # total charge -> free charge: faces touching the substrate carry er * total
    q_free = (sc * clen * np.where(bot, er, 1.0)).sum()
    return q_free * EPS0


def hammerstad_z0(w, h, er):
    u = w / h
    eeff = (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 / u)
    z = 60 / np.sqrt(eeff) * np.log(8 / u + u / 4) if u <= 1 else \
        120 * np.pi / (np.sqrt(eeff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
    return z, eeff


def microstrip(w, t, h, er):
    C = capacitance(w, t, h, er)
    C0 = capacitance(w, t, h, 1.0)       # air-filled reference
    z0 = 1 / (C0_LIGHT * np.sqrt(C * C0))
    return z0, C / C0, C


if __name__ == "__main__":
    w, t, h, er = 0.35e-3, 35e-6, 0.2e-3, 4.3     # metres
    z0, eeff, C = microstrip(w, t, h, er)
    zh, eh = hammerstad_z0(w, h, er)
    print(f"MoM        : Z0 = {z0:6.2f} ohm, eps_eff = {eeff:.3f}, C = {C*1e12:.2f} pF/m")
    print(f"Hammerstad : Z0 = {zh:6.2f} ohm, eps_eff = {eh:.3f}  (zero thickness)")
    print("\nZ0 vs width (h=0.2mm, er=4.3, t=35um):")
    for wm in (0.15, 0.25, 0.35, 0.5, 0.8):
        z, e, _ = microstrip(wm * 1e-3, t, h, er)
        print(f"  w={wm:4.2f} mm  Z0={z:6.2f}  eps_eff={e:.3f}  Hammerstad={hammerstad_z0(wm*1e-3,h,er)[0]:6.2f}")
