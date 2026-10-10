"""
mom25d.py - Stage 1 of a 2.5D planar Method-of-Moments solver (open source, pure NumPy/SciPy)

Structure : metal sheets on top of a grounded dielectric slab (microstrip-type, single layer)
Equation  : mixed-potential integral equation (MPIE), Galerkin, RWG basis on triangles
Kernels   : layered-media Green's functions via Sommerfeld integrals
            G = G_quasi_static(analytic 1/rho + images) + G_dynamic(numerical, deformed contour)
            the 1/rho part is integrated analytically (Wilton et al.), the rest by Gauss quadrature
Ports     : delta-gap sources on a transverse mesh line + travelling-wave extraction on the
            feed line (reference-plane de-embedding, multi-state S = B A^-1)
Geometry  : Manhattan (union of rectangles) so the mesh is a conforming structured triangulation

Time convention exp(+j w t).
"""
import numpy as np
from scipy import sparse
from scipy.special import jv
from scipy.interpolate import CubicSpline

MU0 = 4e-7 * np.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 ** 2)


# =============================================================================
# 1. Layered-media Green's function (source & observation on the air/substrate interface)
# =============================================================================
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


# =============================================================================
# 2. Mesh: Manhattan layout -> conforming structured triangulation, RWG edges
# =============================================================================
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


class RWG:
    """RWG basis functions on interior edges. f = alpha*(r - v) on each adjacent triangle."""
    def __init__(self, nodes, tris):
        self.nodes, self.tris = nodes, tris
        V = nodes[tris]
        e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
        self.area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        assert (self.area > 0).all(), "triangles must be counter-clockwise"
        self.cent = V.mean(1)
        d = {}
        for t, (a, b, c) in enumerate(tris):
            for i, (u, v) in enumerate(((a, b), (b, c), (c, a))):
                d.setdefault((min(u, v), max(u, v)), []).append((t, i))
        E = [(k, v) for k, v in d.items() if len(v) == 2]
        Ne = len(E)
        self.Ne = Ne
        tp = np.zeros(Ne, int); tm = np.zeros(Ne, int)
        vp = np.zeros((Ne, 2)); vm = np.zeros((Ne, 2))
        l = np.zeros(Ne); mid = np.zeros((Ne, 2)); nrm = np.zeros((Ne, 2)); ends = np.zeros((Ne, 2, 2))
        for e, ((u, v), ((t1, i1), (t2, i2))) in enumerate(E):
            tp[e], tm[e] = t1, t2
            vp[e] = nodes[tris[t1][(i1 + 2) % 3]]
            vm[e] = nodes[tris[t2][(i2 + 2) % 3]]
            pu, pv = nodes[u], nodes[v]
            l[e] = np.linalg.norm(pv - pu)
            mid[e] = 0.5 * (pu + pv)
            ends[e] = (pu, pv)
            tau = (pv - pu) / l[e]
            n = np.array([tau[1], -tau[0]])
            if n @ (self.cent[t2] - self.cent[t1]) < 0:
                n = -n
            nrm[e] = n                       # direction of positive current (T+ -> T-)
        self.tp, self.tm, self.l, self.mid, self.nrm, self.ends = tp, tm, l, mid, nrm, ends
        ap = l / (2 * self.area[tp])
        am = -l / (2 * self.area[tm])
        rows = np.concatenate([tp, tm]); cols = np.concatenate([np.arange(Ne)] * 2)
        Nt = len(tris)

        def sp(vals):
            return sparse.csr_matrix((vals, (rows, cols)), shape=(Nt, Ne))
        alpha = np.concatenate([ap, am])
        vv = np.vstack([vp, vm])
        self.Aal, self.Bx, self.By = sp(alpha), sp(alpha * vv[:, 0]), sp(alpha * vv[:, 1])
        self.D = sp(2 * alpha)               # divergence: +l/A on T+, -l/A on T-


def qform(S1, M, S2):
    """S1^T M S2 for sparse S1,S2 (Nt x Ne) and dense M (Nt x Nt)."""
    X = S1.T @ M
    return (S2.T @ X.T).T


def combine(rwg, Mrr, Mrx, Mry, M0):
    """Assemble the Ne x Ne matrix of  int int f_m . f_n K  from triangle-pair moments."""
    A, Bx, By = rwg.Aal, rwg.Bx, rwg.By
    Z = qform(A, Mrr, A)
    Z = Z - qform(A, Mrx, Bx) - qform(A, Mry, By)
    Z = Z - qform(Bx, Mrx.T, A) - qform(By, Mry.T, A)
    Z = Z + qform(Bx, M0, Bx) + qform(By, M0, By)
    return Z


# =============================================================================
# 3. Integrals
# =============================================================================
# 6-point degree-4 rule (weights sum to 1) for the analytic-inner / numerical-outer integrals
_a1, _a2 = 0.445948490915965, 0.091576213509771
_BAR6 = np.array([[1 - 2 * _a1, _a1, _a1], [_a1, 1 - 2 * _a1, _a1], [_a1, _a1, 1 - 2 * _a1],
                  [1 - 2 * _a2, _a2, _a2], [_a2, 1 - 2 * _a2, _a2], [_a2, _a2, 1 - 2 * _a2]])
_W6 = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
# 3-point degree-2 rule for the smooth remainder
_BAR3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
_W3 = np.ones(3) / 3


def quad_pts(rwg, bar):
    return np.einsum('qk,tkc->tqc', bar, rwg.nodes[rwg.tris])


def singular_moments(rwg, nb=16):
    """
    Frequency-independent 1/R integrals between all triangle pairs, inner integral analytic:
      M0[p,q]  = int_p int_q 1/R          Mr[p,q,:] = int_p r  int_q 1/R
      Mrr[p,q] = int_p int_q r.r'/R
    """
    nodes, tris, Ar = rwg.nodes, rwg.tris, rwg.area
    Nt = len(tris)
    V = nodes[tris]
    e = np.roll(V, -1, axis=1) - V
    L = np.linalg.norm(e, axis=2)
    tau = e / L[..., None]
    un = np.stack([tau[..., 1], -tau[..., 0]], -1)          # outward in-plane normal (CCW)
    P = quad_pts(rwg, _BAR6)
    M0 = np.zeros((Nt, Nt)); Mr = np.zeros((Nt, Nt, 2)); Mrr = np.zeros((Nt, Nt))
    for p0 in range(0, Nt, nb):
        sl = slice(p0, min(Nt, p0 + nb))
        r = P[sl]                                                    # (nb,Q,2)
        d = V[None, None] - r[:, :, None, None, :]                   # (nb,Q,Nt,3,2)
        sm = (d * tau[None, None]).sum(-1)
        sp_ = sm + L[None, None]
        t0 = (d * un[None, None]).sum(-1)
        tt = np.abs(t0)
        ok = tt > 1e-12 * L[None, None]
        ttS = np.where(ok, tt, 1.0)
        lg = np.where(ok, np.arcsinh(sp_ / ttS) - np.arcsinh(sm / ttS), 0.0)
        Rm = np.sqrt(t0 ** 2 + sm ** 2); Rp = np.sqrt(t0 ** 2 + sp_ ** 2)
        I1 = (t0 * lg).sum(-1)                                       # int_q 1/R
        eR = 0.5 * (sp_ * Rp - sm * Rm + t0 ** 2 * lg)
        IR = (eR[..., None] * un[None, None]).sum(-2)                # int_q (r'-r)/R
        W = IR + r[:, :, None, :] * I1[..., None]                    # int_q r'/R
        wq = _W6[None, :, None]
        a = Ar[sl, None]
        M0[sl] = a * (wq * I1).sum(1)
        Mr[sl] = a[..., None] * (wq[..., None] * r[:, :, None, :] * I1[..., None]).sum(1)
        Mrr[sl] = a * (wq * (r[:, :, None, :] * W).sum(-1)).sum(1)
    return M0, Mr, Mrr


def smooth_moments(rwg, gt, nb=40):
    """Gauss-quadrature moments of the smooth remainder kernels (A and q)."""
    Ar = rwg.area
    Nt = len(Ar)
    P = quad_pts(rwg, _BAR3)
    A0 = np.zeros((Nt, Nt), complex); Arx = np.zeros_like(A0); Ary = np.zeros_like(A0)
    Arr = np.zeros_like(A0); Q0 = np.zeros_like(A0)
    AA = Ar[:, None] * Ar[None, :] / 9.0
    for p0 in range(0, Nt, nb):
        sl = slice(p0, min(Nt, p0 + nb))
        pa = P[sl][:, None, :, None, :]
        pb = P[None, :, None, :, :]
        diff = pa - pb
        rho = np.sqrt((diff ** 2).sum(-1))                       # (nb,Nt,3,3)
        GA, GQ = gt.A(rho), gt.Q(rho)
        dot = (pa * pb).sum(-1)
        w = AA[sl]
        A0[sl] = w * GA.sum((2, 3))
        Q0[sl] = w * GQ.sum((2, 3))
        Gb = GA.sum(3)                                           # (nb,Nt,3)
        Arx[sl] = w * (Gb * P[sl][:, None, :, 0]).sum(2)
        Ary[sl] = w * (Gb * P[sl][:, None, :, 1]).sum(2)
        Arr[sl] = w * (dot * GA).sum((2, 3))
    return A0, Arx, Ary, Arr, Q0


def gram_matrix(rwg):
    """int f_m . f_n dS  (for a surface-impedance term Zs * Gram)."""
    Ar = rwg.area
    P = quad_pts(rwg, _BAR3)
    Mrr = np.diag(Ar * (_W3[None, :] * (P ** 2).sum(-1)).sum(1))
    Mrx = np.diag(Ar * (_W3[None, :] * P[..., 0]).sum(1))
    Mry = np.diag(Ar * (_W3[None, :] * P[..., 1]).sum(1))
    return combine(rwg, Mrr, Mrx, Mry, np.diag(Ar))


# =============================================================================
# 4. Ports and the solver
# =============================================================================
class Port:
    """
    Delta-gap port on a transverse mesh line with a straight feed line running toward the device.
      axis='x' : gap line is x = pos, feed runs along x, transverse extent span=(ymin, ymax)
      axis='y' : gap line is y = pos, feed runs along y, span=(xmin, xmax)
      direction: +1 / -1 = sense (along the axis) pointing INTO the device
      ref_dist : distance from the gap to the reference plane (positive, along the feed)
      s0       : closest sample distance from the gap (skip the gap's near field)
      zc       : reference impedance of the feed line
      step     : spacing of the current-sample cuts (a multiple of the mesh size; default = mesh size).
                 Needs beta*step of order 0.1-0.5 rad for a well-conditioned wave fit.
    """
    def __init__(self, axis, pos, span, direction, ref_dist, s0=None, zc=50.0, name='', step=None):
        self.step = step
        self.axis, self.pos, self.span = axis, pos, span
        self.dir, self.ref, self.s0, self.zc, self.name = direction, ref_dist, s0, zc, name

    @property
    def dvec(self):
        return np.array([self.dir, 0.0]) if self.axis == 'x' else np.array([0.0, self.dir])


class PlanarMoM:
    def __init__(self, stack, rects, ports, dx, dy, verbose=True):
        self.st, self.ports, self.verbose = stack, ports, verbose
        xl, yl = [], []
        self.samples = []
        for p in ports:
            step = p.step or (dx if p.axis == 'x' else dy)
            s0 = p.s0 if p.s0 is not None else 2 * step
            n = int(np.floor((p.ref - s0) / step + 1e-9)) + 1
            s = p.ref - step * np.arange(n)[::-1]            # uniform, ends on the reference plane
            self.samples.append(s)
            cuts = list(p.pos + p.dir * np.append(s, 0.0))
            (xl if p.axis == 'x' else yl).extend(cuts)
        nodes, tris = make_mesh(rects, dx, dy, xl, yl)
        self.rwg = RWG(nodes, tris)
        r = self.rwg
        if verbose:
            print(f"mesh: {len(tris)} triangles, {r.Ne} RWG unknowns")
        ext = nodes.max(0) - nodes.min(0)
        self.rho_max = 1.05 * np.hypot(*ext) + 1e-4
        M0, Mr, Mrr = singular_moments(r)
        sym = lambda M: 0.5 * (M + M.T)
        self.SA = combine(r, sym(Mrr), Mr[..., 0], Mr[..., 1], sym(M0))
        self.SA = 0.5 * (self.SA + self.SA.T)
        self.SQ = qform(r.D, sym(M0), r.D)
        self.SQ = 0.5 * (self.SQ + self.SQ.T)
        self._gram = None
        self._edge_sets()

    # ---- port bookkeeping -------------------------------------------------
    def _line_edges(self, p, c):
        r, tol = self.rwg, 1e-9
        k = 0 if p.axis == 'x' else 1
        t = 1 - k
        on = (np.abs(r.ends[:, 0, k] - c) < tol) & (np.abs(r.ends[:, 1, k] - c) < tol)
        on &= (r.mid[:, t] > p.span[0] - tol) & (r.mid[:, t] < p.span[1] + tol)
        idx = np.nonzero(on)[0]
        sig = np.sign(r.nrm[idx] @ p.dvec)
        return idx, sig

    def _edge_sets(self):
        self.gap, self.cuts = [], []
        for p, s in zip(self.ports, self.samples):
            self.gap.append(self._line_edges(p, p.pos))
            self.cuts.append([self._line_edges(p, p.pos + p.dir * x) for x in s])
            assert len(self.gap[-1][0]) > 0, f"port {p.name}: no mesh edges on gap line"

    # ---- solve ----------------------------------------------------------------
    def impedance(self, f, zs=0.0):
        w = 2 * np.pi * f
        gt = GreenTable(f, self.st, self.rho_max)
        A0, Arx, Ary, Arr, Q0 = smooth_moments(self.rwg, gt)
        sym = lambda M: 0.5 * (M + M.T)
        r = self.rwg
        ZA = gt.cA * self.SA + combine(r, sym(Arr), Arx, Ary, sym(A0))
        ZQ = gt.cq * self.SQ + qform(r.D, sym(Q0), r.D)
        Z = 1j * w * ZA + ZQ / (1j * w)
        if zs != 0:
            if self._gram is None:
                self._gram = gram_matrix(r)
            Z = Z + zs * self._gram
        return 0.5 * (Z + Z.T)

    def excitation(self, which=None):
        r = self.rwg
        ids = range(len(self.ports)) if which is None else which
        V = np.zeros((r.Ne, len(list(ids))), complex)
        for col, i in enumerate(ids):
            idx, sig = self.gap[i]
            V[idx, col] = sig * r.l[idx]                  # 1 V delta-gap, EMF along +dvec
        return V

    def cut_current(self, I, i):
        """Current along the feed direction through each sample cut: (K, n_excitations)."""
        out = []
        l = self.rwg.l
        for idx, sig in self.cuts[i]:                       # RWG coefficient is A/m -> times edge length
            out.append(((sig * l[idx])[:, None] * I[idx]).sum(0))
        return np.array(out)

    # ---- wave extraction --------------------------------------------------
    @staticmethod
    def fit_gamma(Ik, step):
        """Propagation constant from uniformly spaced samples of a 2-exponential current."""
        num = den = 0.0
        for col in range(Ik.shape[1]):
            v = Ik[:, col]
            num += np.sum((v[:-2] + v[2:]) * np.conj(v[1:-1]))
            den += 2 * np.sum(np.abs(v[1:-1]) ** 2)
        c = num / den
        g = np.arccosh(c + 0j) / step
        return abs(g.real) + 1j * abs(g.imag)

    @staticmethod
    def fit_waves(Ik, s, gamma):
        """I(s) = a e^{-g s} + b e^{+g s}  ->  (a, b) per excitation."""
        M = np.stack([np.exp(-gamma * s), np.exp(gamma * s)], 1)
        coef, *_ = np.linalg.lstsq(M, Ik, rcond=None)
        return coef[0], coef[1]

    def sparams(self, f, zs=0.0, gamma_out=False):
        """N-port S-parameters at the port reference planes (reference impedance = port.zc)."""
        Z = self.impedance(f, zs)
        V = self.excitation()
        I = np.linalg.solve(Z, V)
        n = len(self.ports)
        Am, Bm, gam = np.zeros((n, n), complex), np.zeros((n, n), complex), []
        for i, p in enumerate(self.ports):
            step = abs(self.samples[i][1] - self.samples[i][0]) if len(self.samples[i]) > 1 else 1.0
            Ik = self.cut_current(I, i)
            g = self.fit_gamma(Ik, step)
            gam.append(g)
            Am[i], Bm[i] = self.fit_waves(Ik, self.samples[i] - p.ref, g)
        zc = np.array([p.zc for p in self.ports], float)
        S = -np.diag(np.sqrt(zc)) @ Bm @ np.linalg.inv(Am) @ np.diag(1 / np.sqrt(zc))
        return (S, gam) if gamma_out else S


def line_characterization(stack, w, freqs, length=20e-3, dx=0.5e-3, dy=None, zs_fn=None, verbose=False):
    """
    Characteristic impedance and propagation constant of a uniform line of width w from a
    centre-fed MoM run (gap in the middle of a long line, travelling-wave fits both sides).
    Returns arrays (gamma, Zc).
    """
    dy = dy or w / 4
    L2 = length / 2
    margin = 2.0e-3
    pR = Port('x', 0.0, (-w / 2, w / 2), +1, L2 - margin, s0=1.0e-3, name='R')
    pL = Port('x', 0.0, (-w / 2, w / 2), -1, L2 - margin, s0=1.0e-3, name='L')
    sim = PlanarMoM(stack, [(-L2, -w / 2, L2, w / 2)], [pR, pL], dx, dy, verbose=verbose)
    gam_o, zc_o = [], []
    for f in freqs:
        zs = zs_fn(f) if zs_fn else 0.0
        Z = sim.impedance(f, zs)
        I = np.linalg.solve(Z, sim.excitation([0]))           # 1 V on the right-facing gap edges
        IR, IL = sim.cut_current(I, 0), sim.cut_current(I, 1)
        step = dx
        g = sim.fit_gamma(np.hstack([IR, IL]), step)
        A, B = sim.fit_waves(IR, sim.samples[0], g)
        Cp, Dp = sim.fit_waves(IL, sim.samples[1], g)
        zc = 1.0 / ((A[0] - B[0]) - (Cp[0] - Dp[0]))
        gam_o.append(g); zc_o.append(zc)
    return np.array(gam_o), np.array(zc_o)


def renormalize(S, zc, zref=50.0):
    """Convert S (reference impedances zc[i]) to a common real reference zref."""
    n = S.shape[0]
    zc = np.asarray(zc, float)
    I = np.eye(n)
    Zn = I * 0 + (I + S) @ np.linalg.inv(I - S)             # normalised impedance matrix
    Z = np.diag(np.sqrt(zc)) @ Zn @ np.diag(np.sqrt(zc))
    z = Z / zref
    return (z - I) @ np.linalg.inv(z + I)
