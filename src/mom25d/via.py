"""
mom25d.via - layered-media kernels for vertical (via) currents

Via model
---------
A via is a thin wire (radius a) on the axis (xv, yv). Its current is constant within each
'chain' - the run of dielectric layers between two interfaces where the via touches metal
(pad, plane, ground, PEC cover). Charge therefore only appears at chain ends, and at a pad
it is spread uniformly over the 'fan' of triangles around the via vertex, so it can cancel
the pad's own charge exactly (current continuity at the junction).

Spectral kernels (TM transmission-line network, unit shunt-current source at interface s:
voltages V(z_i | z_s) at all interfaces; kz_j, eps_j per layer). Derived from Maxwell with
the exp(+jwt), exp(-j k.rho) conventions used throughout:

  via segment in layer j (unit current, +z)  ->  E_t on interface k   :  E_t = grad_t S
        S~_{k,j}(kr) = [V(z_j|z_k) - V(z_{j-1}|z_k)] / kz_j^2
  reaction between via segments in layers i (obs) and j (src):
        P~_ij = kr^2/(kz_i^2 kz_j^2) [dV_i(z_j) - dV_i(z_{j-1})] + delta_ij j w mu0 h_j/(kr^2 - k_j^2)
        dV_i(z_s) = V(z_i|z_s) - V(z_{i-1}|z_s)    (zero if z_s is a PEC interface)
  (the delta(z-z') self term cancels the constant part of the second term exactly).

For numerics the point end charges are taken out spectrally:
  D~ = S~ - (1/jw) sum_ends div_e G~_q(k, k_e)        (smooth; identically zero in a homogeneous stack)
  R~ = P~ - (1/jw) sum_ends sum_ends' div div' G~_q - homogeneous K0 part
and the end charges are put back as fan-distributed charges through the ordinary G_q machinery.
"""
import numpy as np
from scipy.special import jv, kv
from scipy.interpolate import CubicSpline

from .base import MU0, C0, EPS0


# =============================================================================
# Sommerfeld contour (shared with the horizontal kernels)
# =============================================================================
def sommerfeld_contour(f, stack, rho_max, K=None, nq=8, hmin=None):
    w = 2 * np.pi * f
    k0 = w / C0
    kmax = k0 * np.sqrt(stack.max_er())
    hmin = hmin or stack.hmin
    K = K or (max(1e5, 30.0 / hmin) + 40 * kmax)
    t1 = 1.5 * kmax
    alpha = min(0.5 * k0, 4.0 / rho_max)
    wid1, wid2 = min(0.3 * k0, 0.8 * alpha), 2 * np.pi / rho_max
    b1 = np.linspace(0, t1, int(np.ceil(t1 / wid1)) + 1)
    b2 = np.append(np.arange(t1, K, wid2), K)
    br = np.concatenate([b1, b2[1:]])
    xg, wg = np.polynomial.legendre.leggauss(nq)
    a, b = br[:-1, None], br[1:, None]
    t = (0.5 * (b - a) * xg[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * wg[None, :]).ravel()
    s = np.where(t < t1, alpha * np.sin(np.pi * t / t1), 0.0)
    ds = np.where(t < t1, alpha * np.pi / t1 * np.cos(np.pi * t / t1), 0.0)
    return t + 1j * s, wt * (1 + 1j * ds), K


def j0_matrix(rho, kr):
    """J0(rho x kr): complex Bessel only on the lifted part of the contour, fast real J0 elsewhere."""
    from scipy.special import j0
    rho = np.atleast_1d(rho)
    real = np.abs(kr.imag) < 1e-300
    J = np.empty((len(rho), len(kr)), complex)
    J[:, real] = j0(rho[:, None] * kr.real[None, real])
    if (~real).any():
        J[:, ~real] = jv(0, rho[:, None] * kr[None, ~real])
    return J


def hankel(spec, kr, dk, rho):
    """(1/2pi) int spec(kr) J0(kr rho) kr dkr  for an array of rho; spec may be (npts,) or (nspec, npts)."""
    J = j0_matrix(rho, kr)
    spec = np.asarray(spec)
    if spec.ndim == 1:
        return (J @ (spec * kr * dk)) / (2 * np.pi)
    return (J @ (spec * (kr * dk)[None, :]).T).T / (2 * np.pi)


class Table1D:
    """Spline-resampled radial kernel on a uniform grid with direct-index linear interpolation."""
    def __init__(self, rho_coarse, vals, rho_max, nfine=8192):
        self.rho = np.linspace(0, rho_max, nfine)
        self.v = CubicSpline(rho_coarse, vals)(self.rho)
        self._dr = self.rho[1]

    def __call__(self, r):
        u = np.minimum(np.asarray(r) / self._dr, len(self.rho) - 1.000001)
        i = u.astype(np.int64)
        t = u - i
        return self.v[i] * (1 - t) + self.v[i + 1] * t


# =============================================================================
# TM network voltages for all source interfaces
# =============================================================================
class TMNetwork:
    """V[s][i](kr): TM voltage at interface i (0..N) for a unit shunt current source at interface s."""
    def __init__(self, stack, kr, w):
        self.st, self.kr, self.w = stack, kr, w
        N = stack.N
        k0 = w / C0
        Y0, tn, sc, Yd, Yu = stack._network(kr, w, 'e')
        self.kz2 = [None] + [k0 ** 2 * stack.eps(j) - kr ** 2 for j in range(1, N + 1)]
        self.k2 = [None] + [k0 ** 2 * stack.eps(j) for j in range(1, N + 1)]
        self.V = {}
        srcs = range(1, N + 1) if stack.top == 'open' else range(1, N)
        for s in srcs:
            v = [None] * (N + 1)
            v[s] = 1.0 / (Yu[s] + Yd[s])
            for k in range(s + 1, N + 1):
                if stack.top == 'pec' and k == N:
                    v[k] = np.zeros_like(kr)
                else:
                    v[k] = v[k - 1] * sc[k] / (1 + 1j * (Yu[k] / Y0[k]) * tn[k])
            for k in range(s, 0, -1):
                v[k - 1] = np.zeros_like(kr) if k - 1 == 0 else v[k] * sc[k] / (1 + 1j * (Yd[k - 1] / Y0[k]) * tn[k])
            self.V[s] = v

    def is_pec(self, k):
        return k == 0 or (self.st.top == 'pec' and k == self.st.N)

    def dV(self, i, s):
        """V(z_i|z_s) - V(z_{i-1}|z_s); zero for a PEC source interface."""
        if self.is_pec(s):
            return 0.0
        return self.V[s][i] - self.V[s][i - 1]

    def S(self, k, layers):
        """E_t potential on interface k from a unit upward current through `layers`."""
        out = 0.0
        for j in layers:
            out = out + self.dV(j, k) / self.kz2[j]
        return out

    def P(self, li, lj, with_homog=True):
        """Reaction between chains through layers li (obs) and lj (src)."""
        out = 0.0
        kr2 = self.kr ** 2
        for i in li:
            for j in lj:
                out = out + kr2 / (self.kz2[i] * self.kz2[j]) * (self.dV(i, j) - self.dV(i, j - 1))
                if with_homog and i == j:
                    out = out + 1j * self.w * MU0 * self.st.layers[j - 1].h / (kr2 - self.k2[j])
        return out


def chain_ends(k_bot, k_top):
    """[(interface, divergence)] of a +z unit-current chain (current leaves bottom, arrives top)."""
    return [(k_bot, +1.0), (k_top, -1.0)]


class ViaKernels:
    """
    Per-frequency via kernels for a set of chain spans [(k_bot, k_top), ...] and metal interfaces.
      D[(k, span)] : Table1D of the smooth via->interface kernel D(rho)
      R(span_m, span_n, rho) : smooth via-via reaction remainder at distances rho
      homog(span_m, span_n, rho) : analytic K0 part
    """
    def __init__(self, f, stack, spans, interfaces, rho_max, nrho=200):
        self.f, self.st = f, stack
        w = 2 * np.pi * f
        self.w = w
        kr, dk, K = sommerfeld_contour(f, stack, rho_max)
        self.kr, self.dk = kr, dk
        net = TMNetwork(stack, kr, w)
        self.net = net
        _, _ = None, None
        # G_q spectral between interfaces (both non-PEC), from the stack network
        self._gq = {}

        def gq(a, b):
            if net.is_pec(a) or net.is_pec(b):
                return 0.0
            key = (min(a, b), max(a, b))
            if key not in self._gq:
                self._gq[key] = stack.spectral(kr, w, a, b)[1]
            return self._gq[key]
        self.gq = gq
        rho = rho_max * np.linspace(0, 1, nrho) ** 2
        self.D = {}
        keys, specs = [], []
        for span in sorted(set(spans)):
            layers = range(span[0] + 1, span[1] + 1)
            for k in interfaces:
                spec = net.S(k, layers)
                for ke, div in chain_ends(*span):
                    spec = spec - div * gq(k, ke) / (1j * w)
                keys.append((k, span)); specs.append(spec * np.ones_like(kr))
        if keys:
            vals = hankel(np.array(specs), kr, dk, rho)          # one Bessel matrix for all tables
            for key, v in zip(keys, vals):
                self.D[key] = Table1D(rho, v, rho_max)

    def R(self, sm, sn, rho):
        lm, ln = range(sm[0] + 1, sm[1] + 1), range(sn[0] + 1, sn[1] + 1)
        spec = self.net.P(lm, ln)
        for ka, da in chain_ends(*sm):
            for kb, db in chain_ends(*sn):
                spec = spec - da * db * self.gq(ka, kb) / (1j * self.w)
        # remove the homogeneous same-layer term (added back analytically by homog())
        kr2 = self.kr ** 2
        for j in set(lm) & set(ln):
            spec = spec - 1j * self.w * MU0 * self.st.layers[j - 1].h / (kr2 - self.net.k2[j])
        return hankel(spec, self.kr, self.dk, rho)

    def homog(self, sm, sn, rho):
        lm, ln = range(sm[0] + 1, sm[1] + 1), range(sn[0] + 1, sn[1] + 1)
        rho = np.atleast_1d(rho)
        out = np.zeros(len(rho), complex)
        for j in set(lm) & set(ln):
            kj = np.sqrt(self.net.k2[j] + 0j)
            if kj.imag > 0:
                kj = -kj
            out += 1j * self.w * MU0 * self.st.layers[j - 1].h / (2 * np.pi) * kv(0, 1j * kj * rho)
        return out
