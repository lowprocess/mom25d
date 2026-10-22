"""
mom25d.base - the pieces of the 2.5D planar Method-of-Moments solver its layers share: RWG basis functions on
triangles, the frequency-independent 1/R moment integrals (the inner integral analytic, Wilton et al.), the
quadrature rules, the delta-gap port and the travelling-wave fits on a port's feed line.

Time convention exp(+j w t).
"""
import numpy as np
from scipy import sparse

MU0 = 4e-7 * np.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 ** 2)


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
# 2. Integrals
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


def gram_matrix(rwg):
    """int f_m . f_n dS  (for a surface-impedance term Zs * Gram)."""
    Ar = rwg.area
    P = quad_pts(rwg, _BAR3)
    Mrr = np.diag(Ar * (_W3[None, :] * (P ** 2).sum(-1)).sum(1))
    Mrx = np.diag(Ar * (_W3[None, :] * P[..., 0]).sum(1))
    Mry = np.diag(Ar * (_W3[None, :] * P[..., 1]).sum(1))
    return combine(rwg, Mrr, Mrx, Mry, np.diag(Ar))


# =============================================================================
# 3. Ports and the wave fits
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


def fit_waves(Ik, s, gamma):
    """I(s) = a e^{-g s} + b e^{+g s}  ->  (a, b) per excitation."""
    M = np.stack([np.exp(-gamma * s), np.exp(gamma * s)], 1)
    coef, *_ = np.linalg.lstsq(M, Ik, rcond=None)
    return coef[0], coef[1]
