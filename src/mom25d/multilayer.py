"""
mom25d.multilayer - multilayer stackups for the 2.5D planar MoM solver (sim.Simulation):

  * Arbitrary stack of dielectric layers above a PEC ground, open (air) or PEC top cover
    (microstrip, embedded microstrip, stripline, offset/broadside stripline ...); or the whole stack
    with a half-space below too (bottom='open'): no infinite ground, metal on interface 0 as well
    (a board's every copper layer, its planes meshed, radiation above and below)
  * Layered-media Green's functions from a transmission-line network in the spectral domain,
    for source and observation on the same or different interfaces (MPIE "formulation C",
    exact for horizontal currents):
        G_A  = V_h / (j w)                 G_q = j w (V_e - V_h) / kr^2
    same interface: the 1/rho singular part (coefficients cA, cq) is subtracted spectrally and
    integrated analytically; the remainder is tabulated per frequency.
  * Non-uniform tensor mesh: n cells across narrow strips, dmax along them
  * The run log (timings and notes).

Time convention exp(+j w t).
"""
import time
from contextlib import contextmanager

import numpy as np
from scipy.special import ellipk

from .base import MU0, C0, EPS0, graded
from .via import sommerfeld_contour, hankel, Table1D


# ---- Run log ----------------------------------------------------------------
class RunLog:
    """Collects stage timings and notes; printed live when verbose."""
    def __init__(self, verbose=True):
        self.verbose, self.records, self.notes = verbose, [], []

    @contextmanager
    def timer(self, stage, detail=''):
        t = time.perf_counter()
        yield
        dt = time.perf_counter() - t
        self.records.append((stage, dt, detail))
        if self.verbose:
            print(f"  [{dt:7.2f} s] {stage}" + (f"  ({detail})" if detail else ''))

    def note(self, msg):
        self.notes.append(msg)
        if self.verbose:
            print("  " + msg)

    def totals(self):
        d = {}
        for s, t, _ in self.records:
            d[s] = d.get(s, 0.0) + t
        return d


# ---- Stackup and spectral-domain transmission-line network ------------------
class Layer:
    def __init__(self, h, er, tand=0.0, name=None):
        self.h, self.er, self.tand, self.name = h, er, tand, name

    @property
    def er_c(self):
        return self.er * (1 - 1j * self.tand)


class LayerStack:
    """
    Dielectric layers listed bottom -> top, interface 0 at z = 0 their bottom surface.
    Interface k (k = 1..N) is the top surface of layer k.
    top='open': half-space of permittivity top_er above layer N (metal allowed on 1..N)
    top='pec' : PEC cover on interface N (metal allowed on 1..N-1)  -> stripline-type stacks
    bottom='pec' : a PEC ground at z = 0 (interface 0 is the ground; the default)
    bottom='open': a half-space of permittivity bottom_er below layer 1, metal allowed on interface 0 too
    """
    def __init__(self, layers, top='open', top_er=1.0, bottom='pec', bottom_er=1.0):
        assert top in ('open', 'pec') and bottom in ('open', 'pec')
        self.layers, self.top, self.top_er = list(layers), top, top_er
        self.bottom, self.bottom_er = bottom, bottom_er
        self.N = len(self.layers)
        self.z = np.concatenate([[0.0], np.cumsum([l.h for l in self.layers])])
        for i, l in enumerate(self.layers):
            if l.name is None:
                l.name = f"D{i + 1}"

    @property
    def metal_interfaces(self):
        return list(range(0 if self.bottom == 'open' else 1, self.N + 1 if self.top == 'open' else self.N))

    def is_pec(self, k):
        """Interface k is a PEC plane of the stack itself (the ground, a cover), not meshed metal."""
        return (k == 0 and self.bottom == 'pec') or (k == self.N and self.top == 'pec')

    @property
    def hmin(self):
        return min(l.h for l in self.layers)

    def eps(self, j):
        """Permittivity of layer j (1..N); j = 0 the half-space below, j = N + 1 the one above."""
        if 1 <= j <= self.N:
            return self.layers[j - 1].er_c
        return complex(self.bottom_er if j <= 0 else self.top_er)

    def eps_sum(self, k):
        """permittivity below + above interface k (sets the 1/rho singular coefficient)."""
        return self.eps(k) + self.eps(k + 1)

    def nearest_discontinuity(self, k):
        """Distance from interface k to the closest interface with a permittivity step, or a ground."""
        down = next((self.z[k] - self.z[j] for j in range(k - 1, -1, -1)
                     if (j == 0 and self.bottom == 'pec') or self.eps(j) != self.eps(j + 1)), np.inf)
        up = next((self.z[j] - self.z[k] for j in range(k + 1, self.N + 1)
                   if (self.top == 'pec' and j == self.N) or self.eps(j) != self.eps(j + 1)), np.inf)
        return min(down, up)

    def max_er(self):
        return max([abs(l.er_c) for l in self.layers] + ([abs(self.top_er)] if self.top == 'open' else []) +
                   ([abs(self.bottom_er)] if self.bottom == 'open' else []))

    def _network(self, kr, w, mode):
        k0, N = w / C0, self.N

        def y0(e):                                  # line admittance and kz of a medium
            kz = -1j * np.sqrt(kr ** 2 - k0 ** 2 * e)
            return kz / (w * MU0) if mode == 'h' else w * EPS0 * e / kz, kz
        Y0, tn, sc = [None] * (N + 1), [None] * (N + 1), [None] * (N + 1)
        for j in range(1, N + 1):
            Y0[j], kz = y0(self.eps(j))
            th = kz * self.layers[j - 1].h
            E = np.exp(-2j * th)                    # tan and sec of th, overflow-free for Im(th) <= 0
            tn[j], sc[j] = -1j * (1 - E) / (1 + E), 2 * np.exp(-1j * th) / (1 + E)
        Yd = [None] * (N + 1)                       # looking down from interface k
        if self.bottom == 'pec':
            Yd[1], kfirst = -1j * Y0[1] / tn[1], 2  # shorted by the ground plane
        else:
            Yd[0], kfirst = y0(complex(self.bottom_er))[0], 1     # the half-space below
        for k in range(kfirst, N + 1):
            Yd[k] = Y0[k] * (Yd[k - 1] + 1j * Y0[k] * tn[k]) / (Y0[k] + 1j * Yd[k - 1] * tn[k])
        Yu = [None] * (N + 1)                       # looking up from interface k
        if self.top == 'open':
            Yu[N], kstart = y0(complex(self.top_er))[0], N
        else:
            Yu[N - 1], kstart = -1j * Y0[N] / tn[N], N - 1      # shorted by the top cover
        for k in range(kstart, 0 if self.bottom == 'open' else 1, -1):
            Yu[k - 1] = Y0[k] * (Yu[k] + 1j * Y0[k] * tn[k]) / (Y0[k] + 1j * Yu[k] * tn[k])
        return Y0, tn, sc, Yd, Yu

    def voltage(self, kr, w, o, s, mode):
        """Voltage at interface o for a unit shunt current source at interface s, in the TM ('e') or TE ('h')
        transmission-line network of the stack. Zero at a PEC interface (either end)."""
        if self.is_pec(o) or self.is_pec(s):
            return np.zeros_like(np.asarray(kr, complex))
        Y0, tn, sc, Yd, Yu = self._network(kr, w, mode)
        v = 1.0 / (Yu[s] + Yd[s])
        for k in range(s + 1, o + 1):               # up through layer k, load Yu[k]
            v = v * sc[k] / (1 + 1j * (Yu[k] / Y0[k]) * tn[k])
        for k in range(s, o, -1):                   # down through layer k, load Yd[k-1]
            v = v * sc[k] / (1 + 1j * (Yd[k - 1] / Y0[k]) * tn[k])
        return v

    def spectral(self, kr, w, o, s):
        """Spectral G_A(kr), G_q(kr) for observation on interface o, source on interface s."""
        Vh, Ve = self.voltage(kr, w, o, s, 'h'), self.voltage(kr, w, o, s, 'e')
        return Vh / (1j * w), 1j * w * (Ve - Vh) / kr ** 2


class GreenTableML:
    """
    Tabulated smooth parts of the layered-media kernels between interfaces o and s:
        G_A = cA/rho + A_rest(rho),  G_q = cq/rho + Q_rest(rho)   (cA = cq = 0 if o != s)
    Sommerfeld integrals along a contour lifted into the upper-half kr plane (clear of
    surface-wave / parallel-plate poles and the branch point).
    """
    def __init__(self, f, stack, o, s, rho_max, nrho=260, nfine=8192, K=None, nq=8, cross_sing=False,
                 image_sing=None):
        w = 2 * np.pi * f
        self.o, self.s, self.f = o, s, f
        same, self.d = o == s, abs(stack.z[o] - stack.z[s])
        self.cA, self.cq = (MU0 / (4 * np.pi), 1.0 / (2 * np.pi * EPS0 * stack.eps_sum(o))) if same else (0.0, 0.0)
        if cross_sing and not same:
            # near-singular 1/sqrt(rho^2+d^2) part between close interfaces: coefficients from the
            # large-kr limit of the network (reflections from other interfaces have died out)
            K1 = min(25.0 / stack.hmin, 300.0 / self.d)
            ga, gq = stack.spectral(np.array([K1 + 0j]), w, o, s)
            fac = K1 * np.exp(K1 * self.d) / (2 * np.pi)
            self.cA, self.cq = complex(ga[0] * fac), complex(gq[0] * fac)
        kr, dk, K = sommerfeld_contour(f, stack, rho_max, K, nq)
        GA, Gq = stack.spectral(kr, w, o, s)
        self.dimg, self.cAi, self.cqi = 0.0, 0.0, 0.0
        if same and image_sing:
            # nearest image (close dielectric step or ground): extract c/sqrt(rho^2 + dimg^2) too.
            # Coefficients from the static network (no dynamic remainder to pollute the limit).
            self.dimg = 2 * stack.nearest_discontinuity(o)
            K1 = 18.0 / self.dimg        # image term ~e^-18: above round-off, next images negligible
            ga, gq = stack.spectral(np.array([K1 + 0j]), 2 * np.pi, o, s)
            fac = K1 * np.exp(K1 * self.dimg) / (2 * np.pi)
            self.cAi = complex((ga[0] - MU0 / (2 * K1)) * fac).real
            self.cqi = complex((gq[0] - 1.0 / (EPS0 * stack.eps_sum(o) * K1)) * fac)
        for d, cA, cq in self.extracted():          # c / sqrt(rho^2 + d^2) <-> 2 pi c e^(-kr d) / kr
            e = 2 * np.pi * np.exp(-kr * d) / kr
            GA, Gq = GA - cA * e, Gq - cq * e
        rho = rho_max * np.linspace(0, 1, nrho) ** 2
        self._tab = Table1D(rho, hankel(np.stack([GA, Gq]), kr, dk, rho).T, rho_max, nfine)     # (nfine, 2)
        self.rho, self.A_rest, self.Q_rest = self._tab.rho, self._tab.v[:, 0], self._tab.v[:, 1]
        self.npts, self.K = len(kr), K

    def extracted(self):
        """[(d, cA, cq)]: the parts taken out of the tables, cA / sqrt(rho^2 + d^2) and cq / sqrt(rho^2 + d^2):
        the 1/rho singularity (same interface) and its nearest image, or the near-singular cross-layer part."""
        if self.o != self.s:
            return [(self.d, self.cA, self.cq)] if self.cA != 0 else []
        return [(0.0, self.cA, self.cq)] + ([(self.dimg, self.cAi, self.cqi)] if self.dimg else [])

    def total(self, rho):
        """Full G_A, G_q at distances rho (the extracted parts added back; 0 for them at rho = d = 0)."""
        GA, GQ = self.AQ(rho)
        for d, cA, cq in self.extracted():
            R = np.hypot(rho, d)
            inv = np.divide(1.0, R, out=np.zeros(np.shape(R)), where=R > 0)
            GA, GQ = GA + cA * inv, GQ + cq * inv
        return GA, GQ

    def AQ(self, r):
        v = self._tab(r)
        return v[..., 0], v[..., 1]

    def A(self, r):
        return self._tab(r)[..., 0]

    def Q(self, r):
        return self._tab(r)[..., 1]


# ---- Mesh: non-uniform tensor grid (fine across narrow strips, coarse along them) ----
def _axis_lines(edge_vals, port_vals, dmax, ncross, narrow, grade=0.0):
    e = np.unique(np.round(np.asarray(edge_vals, float), 12))
    pv = np.unique(np.round(np.asarray(port_vals, float), 12))
    out = [e[0]]
    for a, b in zip(e[:-1], e[1:]):
        inner = [p for p in pv if a + 1e-9 < p < b - 1e-9]
        if b - a <= narrow + 1e-12 and not inner:
            out.extend(a + (b - a) * graded(np.linspace(0, 1, ncross + 1)[1:], grade))   # toward both strip edges
            continue
        bps = [a] + inner + [b]
        for c, d in zip(bps[:-1], bps[1:]):
            n = max(1, int(np.ceil((d - c) / dmax - 1e-9)))
            out.extend(np.linspace(c, d, n + 1)[1:])
    return np.array(out)


def make_mesh_ml(rects, dmax, ncross, narrow, xlines=(), ylines=(), grade=1.0):
    X, Y = (_axis_lines([r[a] for r in rects] + [r[a + 2] for r in rects], ls, dmax, ncross, narrow, grade)
            for a, ls in ((0, xlines), (1, ylines)))
    cx, cy = 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])
    inside = np.zeros((len(cx), len(cy)), bool)
    for x0, y0, x1, y1 in rects:
        inside |= ((cx[:, None] > x0) & (cx[:, None] < x1) & (cy[None, :] > y0) & (cy[None, :] < y1))
    ids, tris = {}, []                                  # grid node -> node number, in order of first use
    for i, j in zip(*np.nonzero(inside)):
        n0, n1, n2, n3 = (ids.setdefault(ij, len(ids)) for ij in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
        tris += [(n0, n1, n2), (n0, n2, n3)] if (i + j) % 2 == 0 else [(n0, n1, n3), (n1, n2, n3)]
    return np.array([(X[i], Y[j]) for i, j in ids]), np.array(tris)


# ---- Utilities --------------------------------------------------------------
def stripline_z0_cohn(w, b, er):
    """Exact Z0 of a zero-thickness strip centred between ground planes spaced b."""
    k, kp = 1 / np.cosh(np.pi * w / (2 * b)), np.tanh(np.pi * w / (2 * b))
    return 30 * np.pi / np.sqrt(er) * ellipk(k ** 2) / ellipk(kp ** 2)
