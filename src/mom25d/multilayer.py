"""
mom25d.multilayer - multilayer stackups for the 2.5D planar MoM solver (sim.Simulation):

  * Arbitrary stack of dielectric layers above a PEC ground, open (air) or PEC top cover
    (microstrip, embedded microstrip, stripline, offset/broadside stripline ...)
  * Layered-media Green's functions from a transmission-line network in the spectral domain,
    for source and observation on the same or different interfaces (MPIE "formulation C",
    exact for horizontal currents):
        G_A  = V_h / (j w)                 G_q = j w (V_e - V_h) / kr^2
    same interface: the 1/rho singular part (coefficients cA, cq) is subtracted spectrally and
    integrated analytically; the remainder is tabulated per frequency.
  * Non-uniform tensor mesh: n cells across narrow strips, dmax along them
  * Layer-aware ports and the run log (timings and notes).

Time convention exp(+j w t).
"""
import time
from contextlib import contextmanager

import numpy as np
from scipy.special import ellipk
from scipy.interpolate import CubicSpline

from .base import MU0, C0, EPS0, Port


# =============================================================================
# Run log
# =============================================================================
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


# =============================================================================
# Stackup and spectral-domain transmission-line network
# =============================================================================
class Layer:
    def __init__(self, h, er, tand=0.0, name=None):
        self.h, self.er, self.tand, self.name = h, er, tand, name

    @property
    def er_c(self):
        return self.er * (1 - 1j * self.tand)


def _trig(th):
    """Overflow-free tan and sec for Im(th) <= 0."""
    E = np.exp(-2j * th)
    return -1j * (1 - E) / (1 + E), 2 * np.exp(-1j * th) / (1 + E)


class LayerStack:
    """
    Dielectric layers listed bottom -> top above a PEC ground at z = 0.
    Interface k (k = 1..N) is the top surface of layer k.
    top='open': half-space of permittivity top_er above layer N (metal allowed on 1..N)
    top='pec' : PEC cover on interface N (metal allowed on 1..N-1)  -> stripline-type stacks
    """
    def __init__(self, layers, top='open', top_er=1.0):
        assert top in ('open', 'pec')
        self.layers, self.top, self.top_er = list(layers), top, top_er
        self.N = len(self.layers)
        self.z = np.concatenate([[0.0], np.cumsum([l.h for l in self.layers])])
        for i, l in enumerate(self.layers):
            if l.name is None:
                l.name = f"D{i + 1}"

    @property
    def metal_interfaces(self):
        return list(range(1, self.N + 1 if self.top == 'open' else self.N))

    @property
    def hmin(self):
        return min(l.h for l in self.layers)

    def eps(self, j):
        return self.layers[j - 1].er_c if 1 <= j <= self.N else complex(self.top_er)

    def eps_sum(self, k):
        """permittivity below + above interface k (sets the 1/rho singular coefficient)."""
        return self.eps(k) + self.eps(k + 1)

    def nearest_discontinuity(self, k):
        """Distance from interface k to the closest interface with a permittivity step, or a ground."""
        d = np.inf
        for j in range(k - 1, -1, -1):                       # down
            if j == 0 or self.eps(j) != self.eps(j + 1):
                d = min(d, self.z[k] - self.z[j]); break
        for j in range(k + 1, self.N + 1):                   # up
            if (self.top == 'pec' and j == self.N) or self.eps(j) != self.eps(j + 1):
                d = min(d, self.z[j] - self.z[k]); break
        return d

    def max_er(self):
        e = [abs(l.er_c) for l in self.layers]
        if self.top == 'open':
            e.append(abs(self.top_er))
        return max(e)

    def _network(self, kr, w, mode):
        k0, N = w / C0, self.N
        Y0, tn, sc = [None] * (N + 1), [None] * (N + 1), [None] * (N + 1)
        for j in range(1, N + 1):
            e = self.eps(j)
            kz = -1j * np.sqrt(kr ** 2 - k0 ** 2 * e)
            Y0[j] = kz / (w * MU0) if mode == 'h' else w * EPS0 * e / kz
            tn[j], sc[j] = _trig(kz * self.layers[j - 1].h)
        Yd = [None] * (N + 1)                       # looking down from interface k
        Yd[1] = -1j * Y0[1] / tn[1]                 # shorted by the ground plane
        for k in range(2, N + 1):
            Yd[k] = Y0[k] * (Yd[k - 1] + 1j * Y0[k] * tn[k]) / (Y0[k] + 1j * Yd[k - 1] * tn[k])
        Yu = [None] * (N + 1)                       # looking up from interface k
        if self.top == 'open':
            e = complex(self.top_er)
            kz = -1j * np.sqrt(kr ** 2 - k0 ** 2 * e)
            Yu[N] = kz / (w * MU0) if mode == 'h' else w * EPS0 * e / kz
            kstart = N
        else:
            Yu[N - 1] = -1j * Y0[N] / tn[N]          # shorted by the top cover
            kstart = N - 1
        for k in range(kstart, 1, -1):
            Yu[k - 1] = Y0[k] * (Yu[k] + 1j * Y0[k] * tn[k]) / (Y0[k] + 1j * Yu[k] * tn[k])
        return Y0, tn, sc, Yd, Yu

    def spectral(self, kr, w, o, s):
        """Spectral G_A(kr), G_q(kr) for observation on interface o, source on interface s."""
        V = []
        for mode in ('h', 'e'):
            Y0, tn, sc, Yd, Yu = self._network(kr, w, mode)
            v = 1.0 / (Yu[s] + Yd[s])                # unit shunt current source
            if o > s:
                for k in range(s + 1, o + 1):        # up through layer k, load Yu[k]
                    v = v * sc[k] / (1 + 1j * (Yu[k] / Y0[k]) * tn[k])
            elif o < s:
                for k in range(s, o, -1):            # down through layer k, load Yd[k-1]
                    v = v * sc[k] / (1 + 1j * (Yd[k - 1] / Y0[k]) * tn[k])
            V.append(v)
        Vh, Ve = V
        return Vh / (1j * w), 1j * w * (Ve - Vh) / kr ** 2

    def describe(self):
        lines = []
        top = f"air (er={self.top_er})" if self.top == 'open' else "PEC cover"
        lines.append(f"top: {top}")
        for j in range(self.N, 0, -1):
            l = self.layers[j - 1]
            lines.append(f"  {l.name}: h={l.h * 1e3:.3f} mm  er={l.er}  tand={l.tand}")
        lines.append("ground (PEC)")
        return lines


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
        k0 = w / C0
        self.o, self.s, self.f = o, s, f
        same = o == s
        self.d = abs(stack.z[o] - stack.z[s])
        self.cA = MU0 / (4 * np.pi) if same else 0.0
        self.cq = 1.0 / (2 * np.pi * EPS0 * stack.eps_sum(o)) if same else 0.0
        if cross_sing and not same:
            # near-singular 1/sqrt(rho^2+d^2) part between close interfaces: coefficients from the
            # large-kr limit of the network (reflections from other interfaces have died out)
            K1 = min(25.0 / stack.hmin, 300.0 / self.d)
            ga, gq = stack.spectral(np.array([K1 + 0j]), w, o, s)
            fac = K1 * np.exp(K1 * self.d) / (2 * np.pi)
            self.cA, self.cq = complex(ga[0] * fac), complex(gq[0] * fac)
        kmax = k0 * np.sqrt(stack.max_er())
        K = K or (max(1e5, 30.0 / stack.hmin) + 40 * kmax)
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
        sdef = np.where(t < t1, alpha * np.sin(np.pi * t / t1), 0.0)
        ds = np.where(t < t1, alpha * np.pi / t1 * np.cos(np.pi * t / t1), 0.0)
        kr = t + 1j * sdef
        dk = wt * (1 + 1j * ds)
        GA, Gq = stack.spectral(kr, w, o, s)
        self.dimg, self.cAi, self.cqi = 0.0, 0.0, 0.0
        if same:
            GA = GA - MU0 / (2 * kr)
            Gq = Gq - 1.0 / (EPS0 * stack.eps_sum(o) * kr)
            if image_sing:
                # nearest image (close dielectric step or ground): extract c/sqrt(rho^2 + dimg^2) too.
                # Coefficients from the static network (no dynamic remainder to pollute the limit).
                self.dimg = 2 * stack.nearest_discontinuity(o)
                K1 = 18.0 / self.dimg        # image term ~e^-18: above round-off, next images negligible
                ws = 2 * np.pi * 1.0
                ga, gq = stack.spectral(np.array([K1 + 0j]), ws, o, s)
                fac = K1 * np.exp(K1 * self.dimg) / (2 * np.pi)
                self.cAi = complex((ga[0] - MU0 / (2 * K1)) * fac).real
                self.cqi = complex((gq[0] - 1.0 / (EPS0 * stack.eps_sum(o) * K1)) * fac)
                e = 2 * np.pi * np.exp(-kr * self.dimg) / kr
                GA, Gq = GA - self.cAi * e, Gq - self.cqi * e
        elif self.cA != 0:
            e = 2 * np.pi * np.exp(-kr * self.d) / kr
            GA, Gq = GA - self.cA * e, Gq - self.cq * e
        rho = rho_max * np.linspace(0, 1, nrho) ** 2
        from .via import j0_sum
        dA, dQ = j0_sum(rho, kr, np.stack([GA * kr * dk, Gq * kr * dk])) / (2 * np.pi)
        self.rho = np.linspace(0, rho_max, nfine)
        self.A_rest = CubicSpline(rho, dA)(self.rho)
        self.Q_rest = CubicSpline(rho, dQ)(self.rho)
        self.npts, self.K = len(kr), K

        self._dr = self.rho[1] - self.rho[0]
        self._AQ = np.stack([self.A_rest, self.Q_rest], -1)       # (nfine, 2) complex

    def _lerp(self, r):
        """Linear interpolation on the uniform grid by direct indexing (no search)."""
        u = np.minimum(np.asarray(r) / self._dr, len(self.rho) - 1.000001)
        i = u.astype(np.int64)
        t = (u - i)[..., None]
        return self._AQ[i] * (1 - t) + self._AQ[i + 1] * t

    def AQ(self, r):
        v = self._lerp(r)
        return v[..., 0], v[..., 1]

    def A(self, r):
        return self._lerp(r)[..., 0]

    def Q(self, r):
        return self._lerp(r)[..., 1]

    def total(self, r):
        """Full kernels (singular part included) - for plotting/debug."""
        r = np.asarray(r, float)
        R = np.sqrt(r ** 2 + self.d ** 2) if self.o != self.s else r
        A, Q = self.cA / R + self.A(r), self.cq / R + self.Q(r)
        if self.dimg:
            Ri = np.sqrt(r ** 2 + self.dimg ** 2)
            A, Q = A + self.cAi / Ri, Q + self.cqi / Ri
        return A, Q


# =============================================================================
# Mesh: non-uniform tensor grid (fine across narrow strips, coarse along them)
# =============================================================================
def _axis_lines(edge_vals, port_vals, dmax, ncross, narrow, grade=0.0):
    e = np.unique(np.round(np.asarray(edge_vals, float), 12))
    pv = np.unique(np.round(np.asarray(port_vals, float), 12)) if len(port_vals) else np.array([])
    out = [e[0]]
    for a, b in zip(e[:-1], e[1:]):
        inner = [p for p in pv if a + 1e-9 < p < b - 1e-9]
        if b - a <= narrow + 1e-12 and not inner:
            u = np.linspace(0, 1, ncross + 1)[1:]            # graded toward both strip edges
            u = (1 - grade) * u + grade * (1 - np.cos(np.pi * u)) / 2
            out.extend(a + (b - a) * u)
            continue
        bps = [a] + inner + [b]
        for c, d in zip(bps[:-1], bps[1:]):
            n = max(1, int(np.ceil((d - c) / dmax - 1e-9)))
            out.extend(np.linspace(c, d, n + 1)[1:])
    return np.array(out)


def make_mesh_ml(rects, dmax, ncross, narrow, xlines=(), ylines=(), grade=1.0):
    X = _axis_lines([r[0] for r in rects] + [r[2] for r in rects], xlines, dmax, ncross, narrow, grade)
    Y = _axis_lines([r[1] for r in rects] + [r[3] for r in rects], ylines, dmax, ncross, narrow, grade)
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


# =============================================================================
# Ports
# =============================================================================
class MLPort(Port):
    """Port (see mom25d.Port) on metal interface `layer`."""
    def __init__(self, layer, axis, pos, span, direction, ref_dist, **kw):
        super().__init__(axis, pos, span, direction, ref_dist, **kw)
        self.layer = layer


# =============================================================================
# Utilities
# =============================================================================
def stripline_z0_cohn(w, b, er):
    """Exact Z0 of a zero-thickness strip centred between ground planes spaced b."""
    k = 1 / np.cosh(np.pi * w / (2 * b))
    kp = np.tanh(np.pi * w / (2 * b))
    return 30 * np.pi / np.sqrt(er) * ellipk(k ** 2) / ellipk(kp ** 2)
