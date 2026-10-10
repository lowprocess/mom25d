"""
mom25d_sing.py - analytic integrals of 1/R over triangles with an out-of-plane offset d

For the near-singular kernel 1/sqrt(rho^2 + d^2) between metal on two closely spaced interfaces
(thin prepregs, the two sheets of a thick conductor). Inner (source) integral analytic
(Wilton et al. 1984 / Graglia 1993), outer integral by a 6-point rule. d = 0 reduces to the
in-plane formulas used by mom25d.singular_moments.
"""
import numpy as np

from mom25d import _BAR6, _W6


def offplane_pairs(nodes, tris, p_idx, q_idx, d, chunk=4000):
    """
    For pairs (p, q) of triangles (observation p, source q) at vertical offset d:
      m0  = int_p int_q 1/R
      mro = int_p r int_q 1/R          (observation-weighted, 2-vector)
      mrs = int_p int_q r'/R           (source-weighted, 2-vector)
      mrr = int_p int_q r.r'/R
    """
    p_idx, q_idx = np.asarray(p_idx), np.asarray(q_idx)
    n = len(p_idx)
    m0 = np.zeros(n); mro = np.zeros((n, 2)); mrs = np.zeros((n, 2)); mrr = np.zeros(n)
    V = nodes[tris]
    e = np.roll(V, -1, axis=1) - V
    L = np.linalg.norm(e, axis=2)
    tau = e / L[..., None]
    un = np.stack([tau[..., 1], -tau[..., 0]], -1)
    Ar = 0.5 * np.abs(e[:, 0, 0] * (-e[:, 2, 1]) - e[:, 0, 1] * (-e[:, 2, 0]))
    P = np.einsum('qk,tkc->tqc', _BAR6, V)
    ad = abs(d)
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        pp, qq = p_idx[sl], q_idx[sl]
        r = P[pp]                                          # (m,6,2)
        Vq, tq, uq, Lq = V[qq], tau[qq], un[qq], L[qq]     # (m,3,2)...
        dv = Vq[:, None, :, :] - r[:, :, None, :]          # (m,6,3,2)
        sm = (dv * tq[:, None]).sum(-1)
        sp = sm + Lq[:, None, :]
        t0 = (dv * uq[:, None]).sum(-1)
        R02 = t0 ** 2 + d * d
        R0 = np.sqrt(R02)
        Rm, Rp = np.sqrt(R02 + sm ** 2), np.sqrt(R02 + sp ** 2)
        ok = R0 > 1e-14 * Lq[:, None, :]
        R0s = np.where(ok, R0, 1.0)
        lg = np.where(ok, np.arcsinh(sp / R0s) - np.arcsinh(sm / R0s), 0.0)
        beta = np.arctan2(t0 * sp, R02 + ad * Rp) - np.arctan2(t0 * sm, R02 + ad * Rm)
        I1 = (t0 * lg - ad * beta).sum(-1)                                   # (m,6)
        eR = 0.5 * (R02 * lg + sp * Rp - sm * Rm)
        IR = (eR[..., None] * uq[:, None]).sum(-2)                           # (m,6,2)
        W = IR + r * I1[..., None]                                           # int r'/R
        wq = (_W6[None, :] * Ar[pp][:, None])                                # (m,6)
        m0[sl] = (wq * I1).sum(1)
        mro[sl] = (wq[..., None] * r * I1[..., None]).sum(1)
        mrs[sl] = (wq[..., None] * W).sum(1)
        mrr[sl] = (wq * (r * W).sum(-1)).sum(1)
    return m0, mro, mrs, mrr
