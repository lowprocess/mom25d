"""
mom25d.sing - analytic integrals of 1/R over triangles with an out-of-plane offset d

For the near-singular kernel 1/sqrt(rho^2 + d^2) between metal on two closely spaced interfaces
(thin prepregs, the two sheets of a thick conductor). One integral analytic (Wilton et al. 1984 /
Graglia 1993), the other by a 6-point rule. d = 0 reduces to the in-plane formulas used by
mom25d.singular_moments.

The rule runs over the smaller triangle of a pair and the analytic integral over the larger (the lower
index's rule where their areas are equal): a pair's moments are then one value whichever of the two is
the observation, so Z is symmetric, and they are the accurate ones. With the rule over a 0.9 mm plane
cell and a 0.3 mm trace cell 0.2 mm under it, the moment was up to 1.3 % off; the other way 0.003 %.
"""
import numpy as np

from .base import _BAR6, _W6, edge_frames


def offplane_pairs(nodes, tris, p_idx, q_idx, d, chunk=4000, outer_levels=0, _geometry=None, scalar_only=False):
    """
    For pairs (p, q) of triangles (observation p, source q) at vertical offset d:
      m0  = int_p int_q 1/R
      mro = int_p r int_q 1/R          (observation-weighted, 2-vector)
      mrs = int_p int_q r'/R           (source-weighted, 2-vector)
      mrr = int_p int_q r.r'/R
    d may be a scalar or one offset per pair. outer_levels subdivides the triangle the rule
    runs over (the smaller of the pair: rule_over) into four per level, improving self and
    near-singular terms.
    With scalar_only=True, returns just m0 without computing current moments.
    """
    p_idx, q_idx = np.asarray(p_idx), np.asarray(q_idx)
    n = len(p_idx)
    m0 = np.zeros(n)
    if not scalar_only:
        mro = np.zeros((n, 2)); mrs = np.zeros((n, 2)); mrr = np.zeros(n)
    V, L, tau, un, Ar, P, weights = (_offplane_geometry(nodes, tris, outer_levels)
                                     if _geometry is None else _geometry)
    swap = ~rule_over(Ar, p_idx, q_idx)                    # the rule over q: computed as (q, p), handed back as (p, q)
    p_idx, q_idx = np.where(swap, q_idx, p_idx), np.where(swap, p_idx, q_idx)
    offsets = np.broadcast_to(np.asarray(d, float), (n,))
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        pp, qq = p_idx[sl], q_idx[sl]
        dz = offsets[sl, None, None]
        ad = np.abs(dz)
        r = P[pp]                                          # (m,6,2)
        Vq, tq, uq, Lq = V[qq], tau[qq], un[qq], L[qq]     # (m,3,2)...
        dv = Vq[:, None, :, :] - r[:, :, None, :]          # (m,6,3,2)
        sm = (dv * tq[:, None]).sum(-1)
        sp = sm + Lq[:, None, :]
        t0 = (dv * uq[:, None]).sum(-1)
        R02 = t0 ** 2 + dz * dz
        R0 = np.sqrt(R02)
        Rm, Rp = np.sqrt(R02 + sm ** 2), np.sqrt(R02 + sp ** 2)
        ok = R0 > 1e-14 * Lq[:, None, :]
        R0s = np.where(ok, R0, 1.0)
        lg = np.where(ok, np.arcsinh(sp / R0s) - np.arcsinh(sm / R0s), 0.0)
        beta = np.arctan2(t0 * sp, R02 + ad * Rp) - np.arctan2(t0 * sm, R02 + ad * Rm)
        I1 = (t0 * lg - ad * beta).sum(-1)
        wq = (weights[None, :] * Ar[pp][:, None])
        m0[sl] = (wq * I1).sum(1)
        if scalar_only:
            continue
        eR = 0.5 * (R02 * lg + sp * Rp - sm * Rm)
        IR = (eR[..., None] * uq[:, None]).sum(-2)                           # (m,6,2)
        W = IR + r * I1[..., None]                                           # int r'/R
        mro[sl] = (wq[..., None] * r * I1[..., None]).sum(1)
        mrs[sl] = (wq[..., None] * W).sum(1)
        mrr[sl] = (wq * (r * W).sum(-1)).sum(1)
    if scalar_only:
        return m0
    mro[swap], mrs[swap] = mrs[swap], mro[swap]            # the moments weighted by p's and by q's coordinates
    return m0, mro, mrs, mrr


RULE_TIE = 1e-9         # triangles whose areas differ by less than this (relative) count as equal


def rule_over(area, p, q):
    """True where a pair's 6-point rule runs over p (and the analytic integral over q): p is the smaller triangle,
    or the lower index of two equal ones. The same answer for (p, q) and (q, p) read the other way round."""
    p, q = np.asarray(p), np.asarray(q)
    ap, aq = area[p], area[q]
    tie = np.abs(ap - aq) <= RULE_TIE * np.maximum(ap, aq)
    return np.where(tie, p <= q, ap < aq)


def _offplane_geometry(nodes, tris, outer_levels=0):
    V = nodes[tris]
    e, L, tau, un = edge_frames(V)
    Ar = 0.5 * np.abs(e[:, 0, 0] * (-e[:, 2, 1]) - e[:, 0, 1] * (-e[:, 2, 0]))
    bar, weights = _BAR6, _W6
    if outer_levels:
        from .biot import _split4
        ref = np.eye(3)[None]
        for _ in range(outer_levels):
            ref = _split4(ref)
        bar = np.einsum('qk,tkc->tqc', _BAR6, ref).reshape(-1, 3)
        weights = np.tile(_W6, len(ref)) / len(ref)
    P = np.einsum('qk,tkc->tqc', bar, V)
    return V, L, tau, un, Ar, P, weights


def offplane_moments(cA, cq, m):
    """The seven moments (base.moments7's order) of the extracted cA / R, cq / R from offplane_pairs' result m."""
    m0, mro, mrs, mrr = m
    return [cA * m0, cA * mro[:, 0], cA * mro[:, 1], cA * mrs[:, 0], cA * mrs[:, 1], cA * mrr, cq * m0]
