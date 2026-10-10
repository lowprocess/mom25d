"""
mom25d_ml.py - Stage 2 (part 1) of the 2.5D planar MoM solver: multilayer stackups.

  * Arbitrary stack of dielectric layers above a PEC ground, open (air) or PEC top cover
    (microstrip, embedded microstrip, stripline, offset/broadside stripline ...)
  * Metal (zero-thickness sheets) on any number of dielectric interfaces
  * Layered-media Green's functions from a transmission-line network in the spectral domain,
    for source and observation on the same or different interfaces (MPIE "formulation C",
    exact for horizontal currents):
        G_A  = V_h / (j w)                 G_q = j w (V_e - V_h) / kr^2
    same interface: the 1/rho singular part (coefficients cA, cq) is subtracted spectrally and
    integrated analytically; the remainder is tabulated per frequency.
  * Non-uniform tensor mesh: n cells across narrow strips, dmax along them
  * Layer-aware ports, run log (timings), and diagnostics (wave-fit residual, reciprocity,
    passivity) for the visualisation module (mom25d_viz.py).

Not yet: vias / vertical currents (stage 2, part 2), conductor thickness, lumped components.
Time convention exp(+j w t).
"""
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from scipy.special import jv, ellipk
from scipy.interpolate import CubicSpline

from mom25d import (MU0, C0, EPS0, RWG, qform, combine, singular_moments, quad_pts,
                    _BAR3, _W3, _BAR6, _W6, Port, PlanarMoM, renormalize)


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
        from mom25d_via import j0_matrix
        J = j0_matrix(rho, kr)
        dA = J @ (GA * kr * dk) / (2 * np.pi)
        dQ = J @ (Gq * kr * dk) / (2 * np.pi)
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
# Integrals between triangle sets (smooth kernels)
# =============================================================================
def smooth_block(Pa, wa, Pb, wb, gt, nb=24):
    """
    Moments of the tabulated smooth kernels between triangle sets a (observation) and b (source).
    Returns A0, Axo, Ayo, Axs, Ays, Arr, Q0 with  Axo[p,q] = int int x_p G,  Axs[p,q] = int int x_q G.
    """
    na, nt = len(Pa), len(Pb)
    out = [np.zeros((na, nt), complex) for _ in range(7)]
    for p0 in range(0, na, nb):
        sl = slice(p0, min(na, p0 + nb))
        pa = Pa[sl][:, None, :, None, :]
        pb = Pb[None, :, None, :, :]
        rho = np.sqrt(((pa - pb) ** 2).sum(-1))
        W = wa[sl][:, None, :, None] * wb[None, :, None, :]
        GA, GQ = gt.AQ(rho)
        GA, GQ = GA * W, GQ * W
        out[0][sl] = GA.sum((2, 3))
        out[1][sl] = (GA * pa[..., 0]).sum((2, 3))
        out[2][sl] = (GA * pa[..., 1]).sum((2, 3))
        out[3][sl] = (GA * pb[..., 0]).sum((2, 3))
        out[4][sl] = (GA * pb[..., 1]).sum((2, 3))
        out[5][sl] = (GA * (pa * pb).sum(-1)).sum((2, 3))
        out[6][sl] = GQ.sum((2, 3))
    return out


# =============================================================================
# Solver
# =============================================================================
class MLPort(Port):
    """Port (see mom25d.Port) on metal interface `layer`."""
    def __init__(self, layer, axis, pos, span, direction, ref_dist, **kw):
        super().__init__(axis, pos, span, direction, ref_dist, **kw)
        self.layer = layer


class MultiLayerMoM:
    """
    shapes : {interface index: [ (x0, y0, x1, y1), ... ]}   union of rectangles per metal layer
    ports  : list of MLPort
    dmax   : max cell size along conductors; ncross: cells across strips narrower than `narrow`
    grade  : 0 = uniform cells across narrow strips, 1 = cosine-graded toward the edges (default)
    near_fac: near/far threshold for the smooth-kernel quadrature (in cell sizes); larger = slower, safer
    """
    def __init__(self, stack, shapes, ports, dmax, ncross=6, narrow=None, grade=1.0, near_fac=5.0,
                 verbose=True, title=''):
        self.st, self.shapes, self.ports, self.title = stack, shapes, ports, title
        self.log = RunLog(verbose)
        self.dmax, self.ncross, self.grade = dmax, ncross, grade
        narrow = narrow or 1.5 * dmax
        for k in shapes:
            assert k in stack.metal_interfaces, f"interface {k} cannot carry metal in this stack"
        self.layers = sorted(shapes)
        self.samples, lines = [], {k: ([], []) for k in self.layers}
        for p in ports:
            step = p.step or dmax
            s0 = p.s0 if p.s0 is not None else 2 * step
            n = int(np.floor((p.ref - s0) / step + 1e-9)) + 1
            s = p.ref - step * np.arange(n)[::-1]
            self.samples.append(s)
            lines[p.layer][0 if p.axis == 'x' else 1].extend(p.pos + p.dir * np.append(s, 0.0))

        with self.log.timer('mesh'):
            nodes, tris, tl, off = [], [], [], 0
            for k in self.layers:
                nd, tr = make_mesh_ml(shapes[k], dmax, ncross, narrow, *lines[k], grade=grade)
                nodes.append(nd); tris.append(tr + off); tl.append(np.full(len(tr), k)); off += len(nd)
            self.rwg = RWG(np.vstack(nodes), np.vstack(tris))
            self.tri_layer = np.concatenate(tl)
            self.edge_layer = self.tri_layer[self.rwg.tp]
        r = self.rwg
        self.tri_idx = {k: np.nonzero(self.tri_layer == k)[0] for k in self.layers}
        Nt = len(r.tris)
        self.log.note(f"mesh: {Nt} triangles, {r.Ne} unknowns  ("
                      + ", ".join(f"M{k}: {len(self.tri_idx[k])} tri" for k in self.layers) + ")")

        with self.log.timer('singular integrals (static, once)'):
            M0, Mrr = np.zeros((Nt, Nt)), np.zeros((Nt, Nt))
            Mr = np.zeros((Nt, Nt, 2)); Q0 = np.zeros((Nt, Nt), complex)
            for k in self.layers:
                idx = self.tri_idx[k]
                m0, mr, mrr = singular_moments(SimpleNamespace(nodes=r.nodes, tris=r.tris[idx], area=r.area[idx]))
                ix = np.ix_(idx, idx)
                M0[ix] = 0.5 * (m0 + m0.T); Mrr[ix] = 0.5 * (mrr + mrr.T); Mr[ix] = mr
                Q0[ix] = M0[ix] / (2 * np.pi * EPS0 * stack.eps_sum(k))
            SA = MU0 / (4 * np.pi) * combine(r, Mrr, Mr[..., 0], Mr[..., 1], M0)
            self.SA = 0.5 * (SA + SA.T)
            SQ = qform(r.D, Q0, r.D)
            self.SQ = 0.5 * (SQ + SQ.T)
            del M0, Mrr, Mr, Q0
        V = r.nodes[r.tris]
        self.tri_size = np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2).max(1)
        self.near_fac = near_fac
        self.P3, self.W3 = quad_pts(r, _BAR3), r.area[:, None] * _W3[None, :]
        self.P6, self.W6 = quad_pts(r, _BAR6), r.area[:, None] * _W6[None, :]
        ext = r.nodes.max(0) - r.nodes.min(0)
        self.rho_max = 1.05 * np.hypot(*ext) + 1e-4
        self._edge_sets()
        self.tables, self.last = {}, None

    # ---- ports ----------------------------------------------------------------
    def _line_edges(self, p, c):
        r, tol = self.rwg, 1e-9
        k = 0 if p.axis == 'x' else 1
        t = 1 - k
        on = (np.abs(r.ends[:, 0, k] - c) < tol) & (np.abs(r.ends[:, 1, k] - c) < tol)
        on &= (r.mid[:, t] > p.span[0] - tol) & (r.mid[:, t] < p.span[1] + tol)
        on &= self.edge_layer == p.layer
        idx = np.nonzero(on)[0]
        return idx, np.sign(r.nrm[idx] @ p.dvec)

    def _edge_sets(self):
        self.gap, self.cuts = [], []
        for p, s in zip(self.ports, self.samples):
            self.gap.append(self._line_edges(p, p.pos))
            self.cuts.append([self._line_edges(p, p.pos + p.dir * x) for x in s])
            assert len(self.gap[-1][0]) > 0, f"port {p.name}: no mesh edges on gap line"
            assert all(len(c[0]) for c in self.cuts[-1]), f"port {p.name}: sample cut off the conductor"

    def excitation(self, which=None):
        r = self.rwg
        ids = list(range(len(self.ports))) if which is None else list(which)
        V = np.zeros((r.Ne, len(ids)), complex)
        for col, i in enumerate(ids):
            idx, sig = self.gap[i]
            V[idx, col] = sig * r.l[idx]
        return V

    def cut_current(self, I, i):
        l = self.rwg.l
        return np.array([((sig * l[idx])[:, None] * I[idx]).sum(0) for idx, sig in self.cuts[i]])

    # ---- matrix fill ----------------------------------------------------------
    def impedance(self, f):
        w = 2 * np.pi * f
        r = self.rwg
        Nt = len(r.tris)
        A0, Axo, Ayo, Arr, Q0 = (np.zeros((Nt, Nt), complex) for _ in range(5))
        self.tables = {}
        for i, a in enumerate(self.layers):
            for b in self.layers[i:]:
                with self.log.timer('Green tables', f"M{a}-M{b} @ {f / 1e9:.2f} GHz"):
                    gt = GreenTableML(f, self.st, a, b, self.rho_max)
                self.tables[(a, b)] = gt
                ia, ib = self.tri_idx[a], self.tri_idx[b]
                with self.log.timer('smooth-kernel fill'):
                    a0, axo, ayo, axs, ays, arr, q0 = self._block(ia, ib, a, b, gt)
                ix = np.ix_(ia, ib)
                A0[ix], Axo[ix], Ayo[ix], Arr[ix], Q0[ix] = a0, axo, ayo, arr, q0
                if a != b:
                    jx = np.ix_(ib, ia)
                    A0[jx], Axo[jx], Ayo[jx], Arr[jx], Q0[jx] = a0.T, axs.T, ays.T, arr.T, q0.T
        with self.log.timer('assemble'):
            sym = lambda M: 0.5 * (M + M.T)
            ZA = self.SA + combine(r, sym(Arr), Axo, Ayo, sym(A0))
            ZQ = self.SQ + qform(r.D, sym(Q0), r.D)
            Z = 1j * w * ZA + ZQ / (1j * w)
        return 0.5 * (Z + Z.T)

    def _block(self, ia, ib, a, b, gt):
        """
        Smooth-kernel moments between triangle sets ia (on M{a}) and ib (on M{b}).
        Far pairs: 1-point (centroid) rule.  Near pairs - where the kernel varies on the scale of
        a cell, i.e. sqrt(rho^2 + Ls^2) < near_fac * cell size with Ls = |dz| (cross-layer) or the
        distance to the nearest ground/dielectric boundary (same layer) - are redone with the
        3-point (same layer) or 6-point (cross-layer) rule.
        """
        r = self.rwg
        ca, cb, aa, ab = r.cent[ia], r.cent[ib], r.area[ia], r.area[ib]
        rho = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
        GA, GQ = gt.AQ(rho)
        w = aa[:, None] * ab[None, :]
        A0 = GA * w
        out = [A0, A0 * ca[:, None, 0], A0 * ca[:, None, 1], A0 * cb[None, :, 0], A0 * cb[None, :, 1],
               A0 * (ca @ cb.T), GQ * w]
        st = self.st
        Ls = abs(st.z[a] - st.z[b]) if a != b else min(st.z[a], (st.z[-1] - st.z[a]) if st.top == 'pec'
                                                         else st.layers[a - 1].h) * 2
        size = np.maximum(self.tri_size[ia][:, None], self.tri_size[ib][None, :])
        p, q = np.nonzero(rho ** 2 + Ls ** 2 < (self.near_fac * size) ** 2)
        self._near_count = getattr(self, '_near_count', 0) + len(p)
        P, W = (self.P3, self.W3) if a == b else (self.P6, self.W6)
        for c0 in range(0, len(p), 20000):
            pp, qq = ia[p[c0:c0 + 20000]], ib[q[c0:c0 + 20000]]
            pa, pb = P[pp][:, :, None, :], P[qq][:, None, :, :]
            Wt = W[pp][:, :, None] * W[qq][:, None, :]
            ga, gq = gt.AQ(np.sqrt(((pa - pb) ** 2).sum(-1)))
            ga, gq = ga * Wt, gq * Wt
            vals = [ga.sum((1, 2)), (ga * pa[..., 0]).sum((1, 2)), (ga * pa[..., 1]).sum((1, 2)),
                    (ga * pb[..., 0]).sum((1, 2)), (ga * pb[..., 1]).sum((1, 2)),
                    (ga * (pa * pb).sum(-1)).sum((1, 2)), gq.sum((1, 2))]
            for M, v in zip(out, vals):
                M[p[c0:c0 + 20000], q[c0:c0 + 20000]] = v
        return out

    # ---- solve & extract ------------------------------------------------------
    def sparams(self, f, keep=False):
        Z = self.impedance(f)
        with self.log.timer('LU solve', f"N={Z.shape[0]}"):
            I = np.linalg.solve(Z, self.excitation())
        n = len(self.ports)
        Am, Bm = np.zeros((n, n), complex), np.zeros((n, n), complex)
        gam, resid = [], []
        for i, p in enumerate(self.ports):
            s = self.samples[i]
            Ik = self.cut_current(I, i)
            g = PlanarMoM.fit_gamma(Ik, abs(s[1] - s[0]))
            a, b = PlanarMoM.fit_waves(Ik, s - p.ref, g)
            M = np.stack([np.exp(-g * (s - p.ref)), np.exp(g * (s - p.ref))], 1)
            resid.append(np.max(np.linalg.norm(Ik - M @ np.vstack([a, b]), axis=0)
                                / np.linalg.norm(Ik, axis=0)))
            Am[i], Bm[i] = a, b
            gam.append(g)
        zc = np.array([p.zc for p in self.ports], float)
        S = -np.diag(np.sqrt(zc)) @ Bm @ np.linalg.inv(Am) @ np.diag(1 / np.sqrt(zc))
        self.last = dict(f=f, I=I if keep else None, gamma=np.array(gam), resid=np.array(resid))
        return S

    def sweep(self, freqs, zc=None, currents_at=None):
        """zc: None (use port.zc) or callable f -> list of reference impedances."""
        out = SimpleNamespace(f=np.asarray(freqs), S=[], gamma=[], resid=[], recip=[], passiv=[],
                              I=None, fI=None)
        fI = None if currents_at is None else freqs[np.argmin(abs(np.asarray(freqs) - currents_at))]
        for f in freqs:
            if zc is not None:
                for p, z in zip(self.ports, zc(f)):
                    p.zc = z
            S = self.sparams(f, keep=(f == fI))
            out.S.append(S); out.gamma.append(self.last['gamma']); out.resid.append(self.last['resid'])
            out.recip.append(np.max(np.abs(S - S.T)))
            out.passiv.append(np.max(np.linalg.svd(S, compute_uv=False)))
            if f == fI:
                out.I, out.fI = self.last['I'], f
            if self.log.verbose:
                print(f"    {f / 1e9:6.2f} GHz  fit residual {out.resid[-1].max():.1e}  "
                      f"max sv(S) {out.passiv[-1]:.4f}")
        for k in ('S', 'gamma', 'resid', 'recip', 'passiv'):
            setattr(out, k, np.array(getattr(out, k)))
        return out

    def current_density(self, I):
        """Surface current density (Jx, Jy) [A/m] at triangle centroids."""
        r = self.rwg
        a = r.Aal @ I
        return r.cent[:, 0] * a - r.Bx @ I, r.cent[:, 1] * a - r.By @ I


# =============================================================================
# Utilities
# =============================================================================
def line_char_ml(stack, layer, w, freqs, length=20e-3, dmax=0.5e-3, ncross=6, grade=1.0, verbose=False):
    """Zc and gamma of a uniform strip of width w on interface `layer` (centre-fed long line)."""
    L2, margin = length / 2, 2e-3
    pR = MLPort(layer, 'x', 0.0, (-w / 2, w / 2), +1, L2 - margin, s0=1e-3, name='R')
    pL = MLPort(layer, 'x', 0.0, (-w / 2, w / 2), -1, L2 - margin, s0=1e-3, name='L')
    sim = MultiLayerMoM(stack, {layer: [(-L2, -w / 2, L2, w / 2)]}, [pR, pL], dmax, ncross,
                        narrow=max(w, 1.5 * dmax) * 1.01, grade=grade, verbose=verbose)
    gam, zc = [], []
    for f in freqs:
        I = np.linalg.solve(sim.impedance(f), sim.excitation([0]))
        IR, IL = sim.cut_current(I, 0), sim.cut_current(I, 1)
        g = PlanarMoM.fit_gamma(np.hstack([IR, IL]), dmax)
        A, B = PlanarMoM.fit_waves(IR, sim.samples[0], g)
        Cp, Dp = PlanarMoM.fit_waves(IL, sim.samples[1], g)
        gam.append(g); zc.append(1.0 / ((A[0] - B[0]) - (Cp[0] - Dp[0])))
    return np.array(gam), np.array(zc)


def stripline_z0_cohn(w, b, er):
    """Exact Z0 of a zero-thickness strip centred between ground planes spaced b."""
    k = 1 / np.cosh(np.pi * w / (2 * b))
    kp = np.tanh(np.pi * w / (2 * b))
    return 30 * np.pi / np.sqrt(er) * ellipk(k ** 2) / ellipk(kp ** 2)
