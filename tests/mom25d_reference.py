"""
References for mom25d's tests, from its first (single-layer) solver: the Green's function of a grounded slab by
Sommerfeld integrals (an independent check of multilayer.GreenTableML) and a structured triangulation of
rectangles (a known mesh for the moment integrals).
"""
import numpy as np
from scipy.special import jv
from scipy.interpolate import CubicSpline

from mom25d.base import MU0, C0, EPS0


class Stackup:
    """Grounded dielectric slab: thickness h [m], relative permittivity er, loss tangent tand."""
    def __init__(self, h, er, tand=0.0):
        self.h, self.er, self.tand = h, er, tand

    @property
    def er_c(self):
        return self.er * (1 - 1j * self.tand)


def sommerfeld_dynamic(k0, h, er_c, rho, K, nq=8):
    """
    Numerical part of the Sommerfeld integrals:
        G_dyn(rho) = 1/(2 pi) * Int [ G~(kr) - G~_qs(kr) ] J0(kr rho) kr dkr
    along a contour pushed into the upper-half kr plane (stays clear of the surface-wave
    poles and the k0 branch point).  Returns (G_A_dyn, G_q_dyn) on the rho grid.
    """
    w = k0 * C0
    kd = k0 * np.sqrt(er_c)
    t1 = 1.5 * kd.real
    alpha = min(0.5 * k0, 4.0 / rho.max())
    wid1 = min(0.3 * k0, 0.8 * alpha)
    wid2 = 2 * np.pi / rho.max()
    b1 = np.linspace(0, t1, int(np.ceil(t1 / wid1)) + 1)
    b2 = np.append(np.arange(t1, K, wid2), K)
    br = np.concatenate([b1, b2[1:]])
    xg, wg = np.polynomial.legendre.leggauss(nq)
    a, b = br[:-1, None], br[1:, None]
    t = (0.5 * (b - a) * xg[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * wg[None, :]).ravel()
    s = np.where(t < t1, alpha * np.sin(np.pi * t / t1), 0.0)
    ds = np.where(t < t1, alpha * np.pi / t1 * np.cos(np.pi * t / t1), 0.0)
    kr = t + 1j * s
    dk = wt * (1 + 1j * ds)

    kz0 = -1j * np.sqrt(kr ** 2 - k0 ** 2)          # Im(kz) <= 0 sheet
    kz1 = -1j * np.sqrt(kr ** 2 - kd ** 2)
    cot = 1.0 / np.tan(kz1 * h)
    GA = MU0 / (1j * kz0 + kz1 * cot)                                       # TE
    Ve = 1j * kz0 * kz1 / (w * EPS0 * (er_c * kz0 * cot + 1j * kz1))        # TM
    Gq = 1j * w * (Ve - 1j * w * GA) / kr ** 2
    GA_qs = MU0 * (1 - np.exp(-2 * kr * h)) / (2 * kr)                      # kr -> inf limits
    Gq_qs = 1.0 / (EPS0 * kr * (1 + er_c / np.tanh(kr * h)))

    J = jv(0, rho[:, None] * kr[None, :])
    dA = J @ ((GA - GA_qs) * kr * dk) / (2 * np.pi)
    dQ = J @ ((Gq - Gq_qs) * kr * dk) / (2 * np.pi)
    return dA, dQ


class GreenTable:
    """
    G_A(rho) = cA / rho + A_rest(rho),   G_q(rho) = cq / rho + Q_rest(rho)
    A_rest / Q_rest are smooth (no singularity) and tabulated on a fine uniform grid.
    """
    def __init__(self, f, st, rho_max, nrho=260, nfine=8192, K=None, nimg=None):
        w = 2 * np.pi * f
        k0 = w / C0
        er_c, h = st.er_c, st.h
        K = K or (1e5 + 40 * k0 * np.sqrt(st.er))
        u = np.linspace(0, 1, nrho)
        rho = rho_max * u ** 2
        dA, dQ = sommerfeld_dynamic(k0, h, er_c, rho, K)
        fine = np.linspace(0, rho_max, nfine)
        Kc = (er_c - 1) / (er_c + 1)
        if nimg is None:
            nimg = int(min(400, max(10, np.ceil(np.log(1e-10) / np.log(abs(Kc) + 1e-12)))))
        n = np.arange(1, nimg + 1)
        img = ((-Kc) ** (n - 1))[None, :] / np.sqrt(fine[:, None] ** 2 + (2 * n * h)[None, :] ** 2)
        self.cA = MU0 / (4 * np.pi)
        self.cq = 1.0 / (2 * np.pi * EPS0 * (1 + er_c))
        self.rho = fine
        self.A_rest = CubicSpline(rho, dA)(fine) - self.cA / np.sqrt(fine ** 2 + 4 * h ** 2)
        self.Q_rest = CubicSpline(rho, dQ)(fine) - self.cq * (1 + Kc) * img.sum(1)

    def A(self, r):
        return np.interp(r, self.rho, self.A_rest.real) + 1j * np.interp(r, self.rho, self.A_rest.imag)

    def Q(self, r):
        return np.interp(r, self.rho, self.Q_rest.real) + 1j * np.interp(r, self.rho, self.Q_rest.imag)


def _axis_lines(vals, dmax):
    v = np.unique(np.round(np.asarray(vals, float), 10))
    out = [v[0]]
    for a, b in zip(v[:-1], v[1:]):
        n = max(1, int(np.ceil((b - a) / dmax - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


def make_mesh(rects, dx, dy, xlines=(), ylines=()):
    xs = [r[0] for r in rects] + [r[2] for r in rects] + list(xlines)
    ys = [r[1] for r in rects] + [r[3] for r in rects] + list(ylines)
    X, Y = _axis_lines(xs, dx), _axis_lines(ys, dy)
    cx, cy = 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])
    inside = np.zeros((len(cx), len(cy)), bool)
    for x0, y0, x1, y1 in rects:
        inside |= ((cx[:, None] > x0) & (cx[:, None] < x1) & (cy[None, :] > y0) & (cy[None, :] < y1))
    ids, pts, tris = {}, [], []

    def nid(i, j):
        if (i, j) not in ids:
            ids[(i, j)] = len(pts)
            pts.append((X[i], Y[j]))
        return ids[(i, j)]

    for i, j in zip(*np.nonzero(inside)):
        n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        if (i + j) % 2 == 0:
            tris += [(n0, n1, n2), (n0, n2, n3)]
        else:
            tris += [(n0, n1, n3), (n1, n2, n3)]
    return np.array(pts), np.array(tris)
