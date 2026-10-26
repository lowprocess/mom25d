"""
mom25d.base - the pieces of the 2.5D planar Method-of-Moments solver its layers share: RWG basis functions on
triangles, the frequency-independent 1/R moment integrals (the inner integral analytic, Wilton et al.), the
quadrature rules and the travelling-wave fits on a port's feed line.

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
        self.Ne = Ne = len(E)
        u, v = np.array([k for k, _ in E], int).reshape(-1, 2).T                       # the edge's end nodes
        (t1, i1), (t2, i2) = np.array([s for _, s in E], int).reshape(-1, 2, 2).transpose(1, 2, 0)   # T+, T-
        self.tp, self.tm = t1, t2
        self.vp, self.vm = nodes[tris[t1, (i1 + 2) % 3]], nodes[tris[t2, (i2 + 2) % 3]]   # the free vertices
        pu, pv = nodes[u], nodes[v]
        self.l = np.linalg.norm(pv - pu, axis=1)
        self.mid, self.ends = 0.5 * (pu + pv), np.stack([pu, pv], 1)
        tau = (pv - pu) / self.l[:, None]
        n = np.c_[tau[:, 1], -tau[:, 0]]
        n[((self.cent[t2] - self.cent[t1]) * n).sum(1) < 0] *= -1
        self.nrm = n                         # direction of positive current (T+ -> T-)
        alpha = np.concatenate([self.l / (2 * self.area[t1]), -self.l / (2 * self.area[t2])])
        vv = np.vstack([self.vp, self.vm])
        sp = lambda vals: sparse.csr_matrix((vals, (np.r_[t1, t2], np.r_[np.arange(Ne), np.arange(Ne)])),
                                            shape=(len(tris), Ne))
        self.Aal, self.Bx, self.By = sp(alpha), sp(alpha * vv[:, 0]), sp(alpha * vv[:, 1])
        self.D = sp(2 * alpha)               # divergence: +l/A on T+, -l/A on T-


def qform(S1, M, S2):
    """S1^T M S2 for sparse S1,S2 (Nt x Ne) and dense M (Nt x Nt)."""
    return (S2.T @ (S1.T @ M).T).T


def combine(rwg, Mrr, Mrx, Mry, M0):
    """Assemble the Ne x Ne matrix of  int int f_m . f_n K  from triangle-pair moments."""
    A, Bx, By = rwg.Aal, rwg.Bx, rwg.By
    return (qform(A, Mrr, A) - qform(A, Mrx, Bx) - qform(A, Mry, By) - qform(Bx, Mrx.T, A) - qform(By, Mry.T, A)
            + qform(Bx, M0, Bx) + qform(By, M0, By))


# ---- Integrals --------------------------------------------------------------
# 6-point degree-4 rule (weights sum to 1) for the analytic-inner / numerical-outer integrals
_a1, _a2 = 0.445948490915965, 0.091576213509771
_BAR6 = np.array([[1 - 2 * _a1, _a1, _a1], [_a1, 1 - 2 * _a1, _a1], [_a1, _a1, 1 - 2 * _a1],
                  [1 - 2 * _a2, _a2, _a2], [_a2, 1 - 2 * _a2, _a2], [_a2, _a2, 1 - 2 * _a2]])
_W6 = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
# 3-point degree-2 rule for the smooth remainder
_BAR3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
_W3 = np.ones(3) / 3


def graded(u, grade):
    """u in 0..1 graded toward both ends (cosine) by `grade` (0: uniform): the cells across a strip."""
    return (1 - grade) * u + grade * (1 - np.cos(np.pi * u)) / 2


def quad_pts(rwg, bar):
    return np.einsum('qk,tkc->tqc', bar, rwg.nodes[rwg.tris])


def moments7(GA, GQ, ra, rb):
    """The seven moments of the kernels between points ra (observation) and rb (source), (..., 2) each:
    G_A x [1, x_a, y_a, x_b, y_b, r_a . r_b] and G_q (combine's triangle-pair moments, unsummed)."""
    return [GA, GA * ra[..., 0], GA * ra[..., 1], GA * rb[..., 0], GA * rb[..., 1], GA * (ra * rb).sum(-1), GQ]


def quad_moments(P, W, p, q, gt, chunk=20000):
    """(7, len(p)) moments7 of the smooth kernels (gt.AQ) between triangles p and q by quadrature (points P,
    weights W per triangle)."""
    out = np.empty((7, len(p)), complex)
    for c0 in range(0, len(p), chunk):
        s = slice(c0, c0 + chunk)
        pa, pb = P[p[s]][:, :, None, :], P[q[s]][:, None, :, :]
        Wt = W[p[s]][:, :, None] * W[q[s]][:, None, :]
        ga, gq = gt.AQ(np.sqrt(((pa - pb) ** 2).sum(-1)))
        out[:, s] = [m.sum((1, 2)) for m in moments7(ga * Wt, gq * Wt, pa, pb)]
    return out


def edge_frames(V):
    """Per triangle (vertices V, (Nt, 3, 2), counter-clockwise) and edge: vector, length, unit tangent, outward
    normal."""
    e = np.roll(V, -1, axis=1) - V
    L = np.linalg.norm(e, axis=2)
    tau = e / L[..., None]
    return e, L, tau, np.stack([tau[..., 1], -tau[..., 0]], -1)


def singular_moments(rwg, nb=16):
    """
    Frequency-independent 1/R integrals between all triangle pairs, inner integral analytic:
      M0[p,q]  = int_p int_q 1/R          Mr[p,q,:] = int_p r  int_q 1/R
      Mrr[p,q] = int_p int_q r.r'/R
    """
    nodes, tris, Ar = rwg.nodes, rwg.tris, rwg.area
    Nt = len(tris)
    V = nodes[tris]
    _, L, tau, un = edge_frames(V)
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
    Ar, P = rwg.area, quad_pts(rwg, _BAR3)
    m = lambda v: np.diag(Ar * (_W3[None, :] * v).sum(1))
    return combine(rwg, m((P ** 2).sum(-1)), m(P[..., 0]), m(P[..., 1]), np.diag(Ar))


# ---- The wave fits on a port's feed line ------------------------------------
def fit_gamma(Ik, step):
    """Propagation constant from uniformly spaced samples of a 2-exponential current."""
    c = np.sum((Ik[:-2] + Ik[2:]) * np.conj(Ik[1:-1])) / (2 * np.sum(np.abs(Ik[1:-1]) ** 2))
    g = np.arccosh(c + 0j) / step
    return abs(g.real) + 1j * abs(g.imag)


def fit_waves(Ik, s, gamma):
    """I(s) = a e^{-g s} + b e^{+g s}  ->  (a, b) per excitation."""
    M = np.stack([np.exp(-gamma * s), np.exp(gamma * s)], 1)
    coef, *_ = np.linalg.lstsq(M, Ik, rcond=None)
    return coef[0], coef[1]
