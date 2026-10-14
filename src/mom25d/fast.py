"""
mom25d.fast - large problems: hierarchical (H-matrix / ACA) far field + sparse near field + GMRES

  Z = Z_far + dZ_near
    Z_far   : every interaction with a 1-point (centroid) rule per triangle, never stored densely:
              admissible cluster pairs are compressed with adaptive cross approximation (ACA),
              the remaining (close) leaf blocks are stored dense
    dZ_near : sparse correction (accurate - centroid) for triangle pairs closer than near_fac cell
              sizes: analytic singular / near-singular parts + higher-order quadrature
  Vias and lumped elements add a few dense rows/columns and sparse entries.
  Solve: GMRES, preconditioned with a sparse LU of the near-field part.
Memory ~ O(N log N) instead of O(N^2); the dense path stays the reference for validation.
"""
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.spatial import cKDTree

from .multilayer import GreenTableML
from .sing import offplane_pairs

import os as _os
try:                                    # optional compiled kernels (python -m mom25d.core.build)
    from . import _core as _built
except ImportError:
    _built = None
_core = None if _os.environ.get('MOM25D_NO_CORE') else _built


def use_core(mode='auto'):
    """Which kernels mom25d uses from now on: 'auto' (the compiled core if built, unless MOM25D_NO_CORE is set),
    'compiled' (the core: RuntimeError if it is not built) or 'python' (NumPy). Returns the previous module (or
    None), for set_core to put back."""
    global _core
    prev = _core
    if mode == 'python':
        _core = None
    elif mode == 'compiled':
        if _built is None:
            raise RuntimeError("mom25d's compiled core is not built (python -m mom25d.core.build): it cannot be "
                               "chosen; 'python' or 'auto' run on NumPy")
        _core = _built
    elif mode == 'auto':
        _core = None if _os.environ.get('MOM25D_NO_CORE') else _built
    else:
        raise ValueError(f"core: auto, compiled or python, not {mode!r}")
    return prev


def set_core(module):
    """Put back what use_core returned."""
    global _core
    _core = module


# =============================================================================
# Cluster tree and H-matrix
# =============================================================================
class Cluster:
    __slots__ = ('idx', 'lo', 'hi', 'kids')

    def __init__(self, idx, pts, leaf):
        self.idx = idx
        p = pts[idx]
        self.lo, self.hi = p.min(0), p.max(0)
        self.kids = []
        if len(idx) > leaf:
            ax = np.argmax(self.hi - self.lo)
            o = np.argsort(p[:, ax], kind='stable')
            h = len(idx) // 2
            self.kids = [Cluster(idx[o[:h]], pts, leaf), Cluster(idx[o[h:]], pts, leaf)]

    @property
    def diam(self):
        return np.linalg.norm(self.hi - self.lo)


def _dist(a, b):
    d = np.maximum(0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return np.linalg.norm(d)


def aca(fill_rows, fill_cols, m, n, tol, max_rank, atol=0.0):
    """Adaptive cross approximation with partial pivoting: block ~ U @ V.
    Stops when the new cross is below tol x (block norm) or below atol per entry (global scale)."""
    U, V = [], []
    used_r, i = set(), 0
    norm2 = 0.0
    for _ in range(min(m, n, max_rank)):
        used_r.add(i)
        row = fill_rows(np.array([i]))[0]
        for u, v in zip(U, V):
            row = row - u[i] * v
        j = int(np.argmax(np.abs(row)))
        if abs(row[j]) < max(1e-300, 0.1 * atol):
            if not U and abs(row[j]) < 0.1 * atol:
                break                                  # block negligible at the global scale
            cand = [r for r in range(m) if r not in used_r]
            if not cand:
                break
            i = cand[0]
            continue
        v = row / row[j]
        col = fill_cols(np.array([j]))[:, 0]
        for uu, vv in zip(U, V):
            col = col - vv[j] * uu
        u = col
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cross = sum(2 * np.real(np.vdot(uu, u) * np.vdot(v, vv)) for uu, vv in zip(U, V))
        norm2 += cross + (nu * nv) ** 2
        U.append(u); V.append(v)
        if nu * nv <= max(tol * np.sqrt(abs(norm2)), atol * np.sqrt(m * n)):
            break
        a = np.abs(u)
        for r in used_r:
            a[r] = -1
        i = int(np.argmax(a))
    if not U:
        return np.zeros((m, 0), complex), np.zeros((0, n), complex)
    return np.array(U).T, np.array(V)


class HMatrix:
    def __init__(self, entry, pts, leaf=48, eta=1.5, tol=1e-4, max_rank=60, atol=0.0):
        self.n = len(pts)
        root = Cluster(np.arange(self.n), pts, leaf)
        self.lowrank, self.dense = [], []
        stack = [(root, root)]
        while stack:
            a, b = stack.pop()
            if max(a.diam, b.diam) < eta * _dist(a, b):
                U, V = aca(lambda r, a=a, b=b: entry(a.idx[r], b.idx),
                           lambda c, a=a, b=b: entry(a.idx, b.idx[c]),
                           len(a.idx), len(b.idx), tol, max_rank, atol)
                if U.shape[1] * (len(a.idx) + len(b.idx)) < len(a.idx) * len(b.idx):
                    self.lowrank.append((a.idx, b.idx, U, V))
                else:
                    self.dense.append((a.idx, b.idx, U @ V))
            elif not a.kids and not b.kids:
                self.dense.append((a.idx, b.idx, entry(a.idx, b.idx)))
            else:
                ka = a.kids or [a]
                kb = b.kids or [b]
                stack += [(x, y) for x in ka for y in kb]

    def finalize(self):
        """Merge all dense leaves into one sparse matrix; stack low-rank factors for batched products."""
        rows, cols, vals = [], [], []
        for r, c, M in self.dense:
            rr, cc = np.meshgrid(r, c, indexing='ij')
            rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(M.ravel())
        self.D = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(self.n, self.n)) if rows else sparse.csr_matrix((self.n, self.n))
        self.dense_nnz = self.D.nnz
        self.dense = []
        # low rank: y[r] += U (V x[c])  ->  sparse 'V' (all ranks stacked) then sparse 'U'
        if self.lowrank:
            vr, vc, vv, ur, uc, uv, k0 = [], [], [], [], [], [], 0
            for r, c, U, V in self.lowrank:
                k = U.shape[1]
                kk, cc = np.meshgrid(np.arange(k0, k0 + k), c, indexing='ij')
                vr.append(kk.ravel()); vc.append(cc.ravel()); vv.append(V.ravel())
                rr, kk2 = np.meshgrid(r, np.arange(k0, k0 + k), indexing='ij')
                ur.append(rr.ravel()); uc.append(kk2.ravel()); uv.append(U.ravel())
                k0 += k
            self.Vs = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(vr), np.concatenate(vc))), shape=(k0, self.n))
            self.Us = sparse.csr_matrix((np.concatenate(uv), (np.concatenate(ur), np.concatenate(uc))), shape=(self.n, k0))
            self.lr_size = sum(U.size + V.size for _, _, U, V in self.lowrank)
            self.ranks = [U.shape[1] for _, _, U, _ in self.lowrank]
            self.lowrank = []
        else:
            self.Vs = self.Us = None
            self.lr_size, self.ranks = 0, []

    def matvec(self, x):
        y = self.D @ x
        if self.Vs is not None:
            y = y + self.Us @ (self.Vs @ x)
        return y

    def memory(self):
        return 16 * (self.dense_nnz + self.lr_size)

    def stats(self):
        return dict(dense_entries=self.dense_nnz, lowrank_blocks=len(self.ranks),
                    mean_rank=round(float(np.mean(self.ranks)), 1) if self.ranks else 0.0,
                    MB=round(self.memory() / 2 ** 20, 1),
                    compression=round(self.memory() / (16 * self.n ** 2), 3))


# =============================================================================
# Fast operator for a Simulation
# =============================================================================
HYBRID_RETRY = 1.25      # after the GPU's mixed precision fails at f, double up to this x f
MIXED_RETRY = HYBRID_RETRY   # the same after the complex64 LU (CPU or GPU) fails to precondition
MIXED_MAX_ITER = 30      # refinement iterations before giving up on the complex64 LU (2 on the demo transition; a stall
                         # ran all 300 of restart 30 x 10: 150 s on 8.4k unknowns against 18 s for a double LU)


class FastOperator:
    """Builds and applies Z for a mom25d.sim.Simulation without forming it densely."""

    def __init__(self, sim, near_fac=None, tol=1e-4, leaf=64, eta=2.0, max_rank=80, atol_rel=1e-8):
        self.sim = sim
        self.trace = False
        self.mixed_tol = 1e-6
        self.atol_rel = atol_rel
        self.max_rank = max_rank
        r = sim.rwg
        self.near_fac = near_fac or sim.near_fac
        self.tol, self.leaf, self.eta = tol, leaf, eta
        Ne = r.Ne
        # half-basis description: (edge, triangle, alpha, free vertex)
        V = r.nodes[r.tris]
        hv = []
        for tri_of, sgn in ((r.tp, 1.0), (r.tm, -1.0)):
            Vt = V[tri_of]                                        # (Ne,3,2)
            on = np.zeros((Ne, 3), bool)
            for k in range(3):
                on[:, k] = (np.linalg.norm(Vt[:, k] - r.ends[:, 0], axis=1) < 1e-12) | \
                           (np.linalg.norm(Vt[:, k] - r.ends[:, 1], axis=1) < 1e-12)
            free = Vt[np.arange(Ne), np.argmin(on, 1)]
            hv.append((tri_of, sgn * r.l / (2 * r.area[tri_of]), free))
        self.h_tri = np.concatenate([hv[0][0], hv[1][0]])         # half h = e (plus) or e + Ne (minus)
        self.h_alpha = np.concatenate([hv[0][1], hv[1][1]])
        self.h_v = np.concatenate([hv[0][2], hv[1][2]])
        self.layer_of_tri = sim.tri_layer
        self.cent, self.area = r.cent, r.area
        self.layers = sim.layers
        self.lidx = {k: i for i, k in enumerate(self.layers)}
        self._near_pairs()
        self._static_cache = {}
        self.core = _core
        if self.core is not None:
            self._core_geometry()

    # ---- near pairs (geometry only) ---------------------------------------------------------------
    def _near_pairs(self):
        sim, r = self.sim, self.sim.rwg
        size = sim.tri_size
        if _core is not None and hasattr(_core, 'near_pairs'):     # the same pairs, in the same order
            P, Q = _core.near_pairs(np.ascontiguousarray(r.cent, float), np.ascontiguousarray(size, float),
                                    np.ascontiguousarray(sim.st.z[self.layer_of_tri], float), float(self.near_fac))
            self.nP, self.nQ = P, Q
            sim.log.note(f"fast solver: {len(self.nP)} near triangle pairs "
                         f"({len(self.nP) / len(size):.0f} per triangle)")
            return
        tree = cKDTree(r.cent)
        P, Q = [], []
        rad = self.near_fac * size
        lists = tree.query_ball_point(r.cent, rad)
        for p, qs in enumerate(lists):
            qs = np.asarray(qs, int)
            P.append(np.full(len(qs), p)); Q.append(qs)
        P, Q = np.concatenate(P), np.concatenate(Q)
        # symmetrise: near if within either triangle's radius
        key = np.unique(np.concatenate([P * len(size) + Q, Q * len(size) + P]))
        P, Q = key // len(size), key % len(size)
        la, lb = self.layer_of_tri[P], self.layer_of_tri[Q]
        rho = np.linalg.norm(r.cent[P] - r.cent[Q], axis=1)
        st = sim.st
        dz = np.abs(st.z[la] - st.z[lb])
        keep = rho ** 2 + dz ** 2 < (self.near_fac * np.maximum(size[P], size[Q])) ** 2
        self.nP, self.nQ = P[keep], Q[keep]
        sim.log.note(f"fast solver: {len(self.nP)} near triangle pairs "
                     f"({len(self.nP) / len(size):.0f} per triangle)")

    # ---- kernels ----------------------------------------------------------------------------------
    def _kern(self, a, b, rho):
        gt = self.tables[(min(a, b), max(a, b))]
        GA, GQ = gt.AQ(rho)
        if a == b:
            with np.errstate(divide='ignore'):
                inv = np.where(rho > 0, 1.0 / np.where(rho > 0, rho, 1.0), 0.0)
            GA, GQ = GA + gt.cA * inv, GQ + gt.cq * inv
            if gt.dimg:
                Ri = 1.0 / np.sqrt(rho ** 2 + gt.dimg ** 2)
                GA, GQ = GA + gt.cAi * Ri, GQ + gt.cqi * Ri
        elif gt.cA != 0:
            R3 = 1.0 / np.sqrt(rho ** 2 + gt.d ** 2)
            GA, GQ = GA + gt.cA * R3, GQ + gt.cq * R3
        return GA, GQ

    def _far_halves(self, hr, hc):
        """Centroid-rule interaction between half-bases hr (rows) and hc (cols)."""
        w = self.w
        tr, tc = self.h_tri[hr], self.h_tri[hc]
        cr, cc = self.cent[tr], self.cent[tc]
        rho = np.sqrt(((cr[:, None, :] - cc[None, :, :]) ** 2).sum(-1))
        lr, lc = self.layer_of_tri[tr], self.layer_of_tri[tc]
        GA = np.zeros(rho.shape, complex); GQ = np.zeros(rho.shape, complex)
        for a in np.unique(lr):
            ma = lr == a
            for b in np.unique(lc):
                mb = lc == b
                ga, gq = self._kern(a, b, rho[np.ix_(ma, mb)])
                GA[np.ix_(ma, mb)], GQ[np.ix_(ma, mb)] = ga, gq
        ar, ac = self.h_alpha[hr] * self.area[tr], self.h_alpha[hc] * self.area[tc]
        dr, dc = cr - self.h_v[hr], cc - self.h_v[hc]
        dot = dr @ dc.T
        return np.outer(ar, ac) * (1j * w * dot * GA + 4 * GQ / (1j * w))

    def entry_pairs(self, rows, cols, chunk=200000):
        """Centroid-rule Z for individual (row, col) edge pairs."""
        Ne, w = self.sim.Ne, self.w
        out = np.zeros(len(rows), complex)
        for c0 in range(0, len(rows), chunk):
            rr, cc = rows[c0:c0 + chunk], cols[c0:c0 + chunk]
            for hr_off in (0, Ne):
                for hc_off in (0, Ne):
                    hr, hc = rr + hr_off, cc + hc_off
                    tr, tc = self.h_tri[hr], self.h_tri[hc]
                    rho = np.linalg.norm(self.cent[tr] - self.cent[tc], axis=1)
                    lr, lc = self.layer_of_tri[tr], self.layer_of_tri[tc]
                    GA = np.zeros(len(rr), complex); GQ = np.zeros(len(rr), complex)
                    for a in np.unique(lr):
                        for b in np.unique(lc):
                            m = (lr == a) & (lc == b)
                            if m.any():
                                GA[m], GQ[m] = self._kern(a, b, rho[m])
                    ar = self.h_alpha[hr] * self.area[tr]; ac = self.h_alpha[hc] * self.area[tc]
                    dot = ((self.cent[tr] - self.h_v[hr]) * (self.cent[tc] - self.h_v[hc])).sum(1)
                    out[c0:c0 + chunk] += ar * ac * (1j * w * dot * GA + 4 * GQ / (1j * w))
        return out

    def entry(self, rows, cols):
        if self.core is not None:
            return self.core.far_block(np.asarray(rows, np.int32), np.asarray(cols, np.int32), self.g, self.tb, self.w)
        Ne = self.sim.Ne
        hr = np.concatenate([rows, rows + Ne]); hc = np.concatenate([cols, cols + Ne])
        H = self._far_halves(hr, hc)
        m, n = len(rows), len(cols)
        return H[:m, :n] + H[:m, n:] + H[m:, :n] + H[m:, n:]

    # ---- near correction ----------------------------------------------------------------------------
    def _near_moments(self, P=None, Q=None, cache=True):
        """Accurate minus centroid moments for triangle pairs (default: all near pairs)."""
        sim, r = self.sim, self.sim.rwg
        if P is None:
            P, Q = self.nP, self.nQ
        else:
            cache = False
        la, lb = self.layer_of_tri[P], self.layer_of_tri[Q]
        n = len(P)
        out = np.zeros((n, 7), complex)        # A0, Axo, Ayo, Axs, Ays, Arr, Q0
        for a in self.layers:
            for b in self.layers:
                m = np.nonzero((la == a) & (lb == b))[0]
                if not len(m):
                    continue
                gt = self.tables[(min(a, b), max(a, b))]
                p, q = P[m], Q[m]
                # smooth rest by quadrature
                Pq, Wq = (sim.P3, sim.W3) if a == b else (sim.P6, sim.W6)
                for c0 in range(0, len(m), 20000):
                    s = slice(c0, c0 + 20000)
                    pa, pb = Pq[p[s]][:, :, None, :], Pq[q[s]][:, None, :, :]
                    Wt = Wq[p[s]][:, :, None] * Wq[q[s]][:, None, :]
                    ga, gq = gt.AQ(np.sqrt(((pa - pb) ** 2).sum(-1)))
                    ga, gq = ga * Wt, gq * Wt
                    out[m[s]] = np.stack([ga.sum((1, 2)), (ga * pa[..., 0]).sum((1, 2)),
                                          (ga * pa[..., 1]).sum((1, 2)), (ga * pb[..., 0]).sum((1, 2)),
                                          (ga * pb[..., 1]).sum((1, 2)), (ga * (pa * pb).sum(-1)).sum((1, 2)),
                                          gq.sum((1, 2))], 1)
                # analytic singular / near-singular parts
                terms = []
                if a == b:
                    terms.append((0.0, gt.cA, gt.cq))
                    if gt.dimg:
                        terms.append((gt.dimg, gt.cAi, gt.cqi))
                elif gt.cA != 0:
                    terms.append((gt.d, gt.cA, gt.cq))
                for d, cA, cq in terms:
                    key = (a, b, d)
                    if not cache:
                        m0, mro, mrs, mrr = offplane_pairs(r.nodes, r.tris, p, q, d)
                    else:
                        if key not in self._static_cache:
                            self._static_cache[key] = offplane_pairs(r.nodes, r.tris, p, q, d)
                        m0, mro, mrs, mrr = self._static_cache[key]
                    out[m] += np.stack([cA * m0, cA * mro[:, 0], cA * mro[:, 1], cA * mrs[:, 0],
                                        cA * mrs[:, 1], cA * mrr, cq * m0], 1)
                # subtract the centroid-rule value used by the far operator
                rho = np.linalg.norm(r.cent[p] - r.cent[q], axis=1)
                GA, GQ = self._kern(a, b, rho)
                wgt = r.area[p] * r.area[q]
                cp, cq_ = r.cent[p], r.cent[q]
                out[m] -= np.stack([GA * wgt, GA * wgt * cp[:, 0], GA * wgt * cp[:, 1], GA * wgt * cq_[:, 0],
                                    GA * wgt * cq_[:, 1], GA * wgt * (cp * cq_).sum(1), GQ * wgt], 1)
        return out

    def _fan_columns(self):
        """Accurate G_q moments between every triangle and the via fan triangles (Nt x |F|)."""
        sim, r = self.sim, self.sim.rwg
        F = sim.F
        lt = self.layer_of_tri
        if self.core is not None and hasattr(self.core, 'fan_q') and getattr(self, 'tb', None) is not None:
            QF = self.core.fan_q(self.g, self.tb, np.asarray(F, np.int32))           # compiled, OpenMP
        else:
            QF = np.zeros((len(r.tris), len(F)), complex)
            for j, q in enumerate(F):
                rho = np.linalg.norm(r.cent - r.cent[q], axis=1)
                for a in self.layers:
                    m = lt == a
                    _, GQ = self._kern(a, lt[q], rho[m])
                    QF[m, j] = GQ * r.area[m] * r.area[q]
        # exact correction for near pairs only (same split as the rest of the operator)
        pos = {q: j for j, q in enumerate(F)}
        sel = np.isin(self.nQ, F)
        cols = np.array([pos[q] for q in self.nQ[sel]], int)
        np.add.at(QF, (self.nP[sel], cols), self._qcorr[sel])
        return QF

    def _near_matrix(self):
        """Sparse (Ne x Ne) correction from triangle-pair moments."""
        r, Ne, w = self.sim.rwg, self.sim.Ne, self.w
        M = self._near_moments()
        self._Mnear = M
        # halves on each triangle
        h_of_tri = [[] for _ in range(len(r.tris))]
        for h, t in enumerate(self.h_tri):
            h_of_tri[t].append(h)
        rows, cols, vals = [], [], []
        P, Q = self.nP, self.nQ
        maxh = max(len(x) for x in h_of_tri)
        Hp = -np.ones((len(r.tris), maxh), int)
        for t, hs in enumerate(h_of_tri):
            Hp[t, :len(hs)] = hs
        for i in range(maxh):
            for j in range(maxh):
                hi, hj = Hp[P, i], Hp[Q, j]
                ok = (hi >= 0) & (hj >= 0)
                hi, hj, mm = hi[ok], hj[ok], M[ok]
                ai, aj = self.h_alpha[hi], self.h_alpha[hj]
                vi, vj = self.h_v[hi], self.h_v[hj]
                A = (mm[:, 5] - (vj[:, 0] * mm[:, 1] + vj[:, 1] * mm[:, 2])
                     - (vi[:, 0] * mm[:, 3] + vi[:, 1] * mm[:, 4]) + (vi * vj).sum(1) * mm[:, 0])
                z = ai * aj * (1j * w * A + 4 * mm[:, 6] / (1j * w))
                rows.append(hi % Ne); cols.append(hj % Ne); vals.append(z)
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(Ne, Ne))

    # ---- assemble & solve ---------------------------------------------------------------------------------
    # ---- compiled-core data ---------------------------------------------------------------------
    def _core_geometry(self):
        sim, r = self.sim, self.sim.rwg
        Ne, Nt = sim.Ne, len(r.tris)
        te = -np.ones((Nt, 3), np.int32); ta = np.zeros((Nt, 3)); tv = np.zeros((Nt, 3, 2))
        fill = np.zeros(Nt, int)
        for h in range(2 * Ne):
            t = self.h_tri[h]; k = fill[t]; fill[t] += 1
            te[t, k] = h % Ne; ta[t, k] = self.h_alpha[h]; tv[t, k] = self.h_v[h]
        lt = np.array([self.lidx[k] for k in self.layer_of_tri], np.int32)
        self.g = dict(ep=r.tp.astype(np.int32), em=r.tm.astype(np.int32), lt=lt, te=te, ta=ta, tv=tv,
                      ap=self.h_alpha[:Ne].copy(), am=self.h_alpha[Ne:].copy(),
                      vp=np.ascontiguousarray(self.h_v[:Ne]), vm=np.ascontiguousarray(self.h_v[Ne:]),
                      cent=np.ascontiguousarray(r.cent), area=r.area.copy())
        self.q = dict(nodes=np.ascontiguousarray(r.nodes), tris=r.tris.astype(np.int32),
                      P3=np.ascontiguousarray(sim.P3), W3=np.ascontiguousarray(sim.W3),
                      P6=np.ascontiguousarray(sim.P6), W6=np.ascontiguousarray(sim.W6))
        self.nP32, self.nQ32 = self.nP.astype(np.int32), self.nQ.astype(np.int32)

    def _core_tables(self):
        L = len(self.layers)
        keys = sorted(self.tables)
        tab = np.zeros((L, L), np.int32)
        kind = np.zeros(len(keys), np.int32)
        for i, (a, b) in enumerate(keys):
            tab[self.lidx[a], self.lidx[b]] = tab[self.lidx[b], self.lidx[a]] = i
            gt = self.tables[(a, b)]
            kind[i] = 0 if a == b else (1 if gt.cA != 0 else 2)
        T = [self.tables[k] for k in keys]
        c = lambda v: np.array([complex(x) for x in v])
        self.tb = dict(A=np.stack([t.A_rest for t in T]).astype(complex), Q=np.stack([t.Q_rest for t in T]).astype(complex),
                       dr=np.array([t.rho[1] for t in T]), d=np.array([t.d for t in T]),
                       dimg=np.array([t.dimg for t in T]), cA=c(t.cA for t in T), cq=c(t.cq for t in T),
                       cAi=c(t.cAi for t in T), cqi=c(t.cqi for t in T), tab=tab, kind=kind)

    def matvec_exact(self, x, Sx=None):
        """y = Z x in double precision without storing Z (far: triangle-aggregated; near: on the fly)."""
        sim, Ne = self.sim, self.sim.Ne
        xe = np.ascontiguousarray(x[:Ne], complex)
        y = np.empty(len(x), complex)
        near = (self.dZ_exact @ xe) if getattr(self, 'dZ_exact', None) is not None else \
            self.core.near_matvec(xe, self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
        y[:Ne] = self.core.matvec_far(xe, self.g, self.tb, self.w) + near
        if Sx is not None:
            y[:Ne] += Sx[:Ne, :Ne] @ xe
        if sim.Nv:
            y[:Ne] += self.B @ x[Ne:]
            y[Ne:] = self.B.T @ xe + self.C @ x[Ne:]
        return y

    def matvec_exact_multi(self, X, Sx=None):
        """Y = Z X for several columns (N x k); one far-field kernel pass serves all columns."""
        sim, Ne = self.sim, self.sim.Ne
        X = np.asarray(X, complex)
        if X.ndim == 1:
            return self.matvec_exact(X, Sx)
        if not hasattr(self.core, 'matvec_far_multi'):
            return np.stack([self.matvec_exact(X[:, j], Sx) for j in range(X.shape[1])], 1)
        Xe = np.ascontiguousarray(X[:Ne])
        Y = np.empty(X.shape, complex)
        if getattr(self, 'dZ_exact', None) is not None:
            near = self.dZ_exact @ Xe
        else:
            near = np.stack([self.core.near_matvec(np.ascontiguousarray(Xe[:, j]), self.g, self.tb, self.q,
                                                   self.nP32, self.nQ32, self.w) for j in range(X.shape[1])], 1)
        Y[:Ne] = self.core.matvec_far_multi(Xe, self.g, self.tb, self.w) + near
        if Sx is not None:
            Y[:Ne] += Sx[:Ne, :Ne] @ Xe
        if sim.Nv:
            Y[:Ne] += self.B @ X[Ne:]
            Y[Ne:] = self.B.T @ Xe + self.C @ X[Ne:]
        return Y

    def setup(self, f, hmatrix=True, mode=None):
        """mode: 'aca' (sparse near field + H-matrix), 'direct' (tables only when the core fills Z),
        'sparse' (sparse near field, no H-matrix)."""
        mode = mode or ('aca' if hmatrix else 'sparse')
        sim = self.sim
        self.f, self.w = f, 2 * np.pi * f
        self.tables = {}
        for i, a in enumerate(self.layers):
            for b in self.layers[i:]:
                xs = a != b and abs(sim.st.z[a] - sim.st.z[b]) < 3 * sim.dmax
                im = a == b and sim._near_layer(a) < 1.5 * sim.dmax
                with sim.log.timer('Green tables', f"M{a}-M{b} @ {f / 1e9:.2f} GHz"):
                    self.tables[(a, b)] = GreenTableML(f, sim.st, a, b, sim.rho_max, cross_sing=xs, image_sing=im)
        sim.tables = self.tables
        if self.core is not None:
            self._core_tables()
            if mode == 'direct':
                return
        with sim.log.timer('near-field correction'):
            if self.core is not None:
                ri, ci, vv, qc = self.core.near_coo(self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
                ok = ri >= 0
                dZ = sparse.csr_matrix((vv[ok], (ri[ok], ci[ok])), shape=(sim.Ne, sim.Ne))
                self._qcorr = qc
                del ri, ci, vv
            else:
                dZ = self._near_matrix()
                self._qcorr = self._Mnear[:, 6]
        if sim.Nv:
            with sim.log.timer('via kernels', f"{sim.Nv} chains"):
                self.B, self.C = sim._via_blocks(f, self._fan_columns())
        if mode != 'aca':
            self.dZ = dZ.tocsr()
            return
        with sim.log.timer('H-matrix (ACA)'):
            # global scale: typical self term; far blocks below tol x that per entry are dropped/truncated
            dg = np.abs(self.entry_pairs(np.arange(sim.Ne), np.arange(sim.Ne)) + dZ.diagonal())
            self.atol = self.atol_rel * float(np.median(dg))
            self.H = HMatrix(self.entry, sim.rwg.mid, leaf=self.leaf, eta=self.eta, tol=self.tol,
                             max_rank=self.max_rank, atol=self.atol)
            self.H.finalize()
        # preconditioner: the full operator restricted to the near-field sparsity pattern
        with sim.log.timer('preconditioner build'):
            dZ = dZ.tocoo()
            vals = self.entry_pairs(dZ.row, dZ.col) + dZ.data
            self.Pnear = sparse.csc_matrix((vals, (dZ.row, dZ.col)), shape=dZ.shape)
            self.dZ = dZ.tocsr()
        sim.log.note(f"H-matrix: {self.H.stats()}")

    def extra_blocks(self, f):
        """Via rows/columns, lumped elements and metal loss (dense in the few via unknowns)."""
        sim = self.sim
        N = sim.Ne + sim.Nv
        Ne = sim.Ne
        B = np.zeros((N, sim.Nv), complex) if sim.Nv else None
        S = sparse.lil_matrix((N, N), dtype=complex)
        for el, (idx, sig) in zip(sim.lumped, sim.lumped_edges):
            u = sig * sim.rwg.l[idx]
            S[np.ix_(idx, idx)] = S[np.ix_(idx, idx)].toarray() + el.Z(f) * np.outer(u, u)
        if sim.metal is not None:
            from .base import gram_matrix
            if sim._gram is None:
                sim._gram = sparse.csr_matrix(gram_matrix(sim.rwg))
            G = sparse.csr_matrix(sim._gram)
            S[:Ne, :Ne] = S[:Ne, :Ne] + sim.metal.zs(f) * G
        return B, S.tocsr()

    def solve(self, f, E, tol=1e-6, restart=200, maxiter=10):
        sim = self.sim
        self.setup(f)
        _, Sx = self.extra_blocks(f)
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        Pn = self.Pnear + Sx[:Ne, :Ne]
        if Nv:                                   # bordered system: dense via rows/columns
            Pn = sparse.bmat([[Pn, sparse.csr_matrix(self.B)],
                              [sparse.csr_matrix(self.B.T), sparse.csr_matrix(self.C)]])
        Pn = sparse.csr_matrix(Pn)
        # symmetric diagonal equilibration: via (fan-charge) rows are ~1e6 x larger than strip rows
        dsc = 1.0 / np.sqrt(np.abs(Pn.diagonal()) + 1e-300)
        Ds = sparse.diags(dsc)
        with sim.log.timer('preconditioner LU'):
            lu = splu(sparse.csc_matrix(Ds @ Pn @ Ds), permc_spec='COLAMD')

        def mv(xs):
            x = dsc * xs
            xe = x[:Ne]
            y = np.empty(N, complex)
            y[:Ne] = self.H.matvec(xe) + self.dZ @ xe + Sx[:Ne, :Ne] @ xe
            if Nv:
                y[:Ne] += self.B @ x[Ne:]
                y[Ne:] = self.B.T @ xe + self.C @ x[Ne:]
            return dsc * y
        A = LinearOperator((N, N), matvec=mv, dtype=complex)
        M = LinearOperator((N, N), matvec=lu.solve, dtype=complex)
        X = np.zeros((N, E.shape[1]), complex)
        self.iters = []
        with sim.log.timer('GMRES', f"{E.shape[1]} rhs"):
            for j in range(E.shape[1]):
                it = [0]
                t0 = time.time()

                def cb(rn):
                    it[0] += 1
                    if self.trace and it[0] % 25 == 0:
                        print(f"      GMRES it {it[0]}: residual {rn:.2e}  [{time.time() - t0:.1f}s]", flush=True)
                y, info = gmres(A, dsc * E[:, j], M=M, rtol=tol, restart=restart, maxiter=maxiter,
                                callback=cb, callback_type='pr_norm')
                X[:, j] = dsc * y
                self.iters.append(it[0])
                if info > 0:
                    sim.log.note(f"WARNING GMRES did not converge for port {j + 1} ({it[0]} iterations)")
        sim.log.note(f"GMRES iterations per port: {self.iters}")
        return X

    def solve_direct(self, f, E, block=1024, precision='double'):
        """Dense LU of the fast-operator entries (centroid far field + exact near correction).
        precision='mixed': Z stored in complex64 (half the memory), LU in single precision used as a
        preconditioner for GMRES on the exact double-precision matrix-free operator."""
        sim = self.sim
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        if self.core is None:
            if precision == 'mixed':
                sim.log.note("mixed precision needs the compiled core - using double")
            return self._solve_direct_numpy(f, E, block)
        self.setup(f, mode='direct')
        _, Sx = self.extra_blocks(f)
        # mixed precision with a GPU: Z filled in double and kept on the host (24 N^2 bytes with the complex64
        # copy the GPU factors), the refinement products dense: 5.7 s a solve on the via transition (13.5k
        # unknowns) against 8.7 s with the sparse exact near field (its CSR build alone 3.1 s) and 14.9 s double
        hybrid = precision == 'mixed' and self._gpu_for(N) is not None and self._fits(24.0 * N * N)
        quads = getattr(sim, 'qT', None) is not None
        if quads:
            hybrid = False                          # quad cells: the projected LU on the CPU (for now)
        if hybrid and f <= HYBRID_RETRY * getattr(self, '_hybrid_failed', 0.0):
            # the complex64 LU failed to precondition at or above this frequency: double up to HYBRID_RETRY x
            # the highest failure, the GPU again above it (it fails at the low end of a band, where the MPIE
            # matrix is worst conditioned; a sweep that gave up for good after its first 1 GHz solve ran the via
            # transition, 38 solves, on the CPU: 621 s, against 614 s without a GPU)
            precision, hybrid = 'double', False
        if precision == 'mixed' and not hybrid and f <= MIXED_RETRY * getattr(self, '_mixed_failed', 0.0):
            # the same for the complex64 LU (CPU or GPU): a failure costs the mixed solve and a double one
            precision = 'double'
        tiled = getattr(self, 'tiled', None) or getattr(sim, 'tiled', 'auto')
        if precision == 'mixed' and tiled != 'off' and not getattr(self, '_tiled_failed', False) and \
                (tiled == 'on' or not hybrid) and not quads:
            # no dense Z: the GPU holds half a complex64 matrix (or streams it), the host a block column
            from .gpu import get_backend
            gpu = get_backend(sim.gpu) if getattr(sim, 'gpu', 'off') != 'off' else None
            blk = getattr(self, 'tiled_block', 1024)
            if gpu is not None and hasattr(gpu, 'cp') and 0.9 * gpu.mem_free() >= self._tiled_work(N, blk):
                X = self._solve_tiled(f, E, Sx, gpu, blk)
                if X is not None:
                    return X
        dt = np.complex64 if precision == 'mixed' and not hybrid else np.complex128
        with sim.log.timer('direct fill', f"N={N}, {np.dtype(dt).name}, {self.core.num_threads()} threads"):
            Z = np.zeros((N, N), dt)
            (self.core.fill_far_c64 if dt == np.complex64 else self.core.fill_far)(Z, self.g, self.tb, self.w)
        with sim.log.timer('near-field correction'):
            if precision == 'mixed' and not hybrid:
                # keep the exact near correction (sparse, double) for the refinement products
                ri, ci, vv, qc = self.core.near_coo(self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
                ok = ri >= 0
                self.dZ_exact = sparse.csr_matrix((vv[ok], (ri[ok], ci[ok])), shape=(Ne, Ne))
                del ri, ci, vv, ok
                Dc = self.dZ_exact.tocoo()
                np.add.at(Z, (Dc.row, Dc.col), Dc.data.astype(dt))
                del Dc
                self._qcorr = qc
            else:
                self.dZ_exact = None
                self._qcorr = self.core.near_into_dense(Z, self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
        if Nv:
            with sim.log.timer('via kernels', f"{Nv} chains"):
                self.B, self.C = sim._via_blocks(f, self._fan_columns())
            Z[:Ne, Ne:] = self.B; Z[Ne:, :Ne] = self.B.T; Z[Ne:, Ne:] = self.C
        if Sx.nnz:
            Sc = Sx.tocoo()
            np.add.at(Z, (Sc.row, Sc.col), Sc.data.astype(dt))
        import scipy.linalg as sla
        if quads:
            return self._solve_quads(f, Z, E, Sx, precision, block)
        dsc = (1.0 / np.sqrt(np.abs(np.diagonal(Z)).astype(float) + 1e-300))
        if hybrid:
            return self._solve_hybrid(f, Z, dsc, E)
        gpu = None
        if precision == 'mixed' and getattr(sim, 'gpu', 'off') != 'off':
            from .gpu import get_backend
            gpu = get_backend(sim.gpu)
            if not getattr(self, '_gpu_logged', False):
                self._gpu_logged = True
                if gpu is None:
                    if sim.gpu not in ('auto', None):
                        sim.log.note("GPU requested but not available: " + "; ".join(get_backend.why))
                else:
                    sim.log.note(f"GPU: {gpu.device} via {gpu.name}, {gpu.mem_free() / 1e9:.1f} GB free, "
                                 f"LU needs ~{gpu.vram_needed(N) / 1e9:.1f} GB")
            if gpu is not None and not gpu.fits(N):
                sim.log.note(f"GPU: LU needs ~{gpu.vram_needed(N) / 1e9:.1f} GB, only {gpu.mem_free() / 1e9:.1f} GB "
                             f"free - factoring on the CPU")
                gpu = None
        if gpu is not None:
            with sim.log.timer('LU factor (GPU)', f"N={N}, {gpu.name}"):
                Z *= dsc[:, None].astype(dt)
                Z *= dsc[None, :].astype(dt)
                try:
                    gpu.factor(Z)
                except Exception as e:                      # driver / unsupported dtype: fall back
                    sim.log.note(f"GPU LU failed ({type(e).__name__}: {e}) - factoring on the CPU")
                    gpu.release()
                    gpu = None
                    lu, piv = sla.lu_factor(Z.T, overwrite_a=True, check_finite=False)
                del Z
        else:
            with sim.log.timer('LU factor', f"N={N}"):
                Z *= dsc[:, None].astype(dt)
                Z *= dsc[None, :].astype(dt)
                # factor Z^T: it is Fortran-contiguous, so LAPACK works in place (no second N^2 copy);
                # Z x = b is then solved with trans=1
                lu, piv = sla.lu_factor(Z.T, overwrite_a=True, check_finite=False)
                del Z
        self.iters = []
        if precision != 'mixed':
            with sim.log.timer('LU solve'):
                return dsc[:, None] * sla.lu_solve((lu, piv), dsc[:, None] * E, trans=1, check_finite=False)
        # mixed: GMRES on the exact (double, matrix-free) equilibrated operator, LU(complex64) preconditioner
        if gpu is not None:
            lu_apply = gpu.solve
        else:
            def lu_apply(R):
                return sla.lu_solve((lu, piv), R.astype(np.complex64), trans=1, check_finite=False).astype(complex)
        with sim.log.timer('refinement (GMRES, exact matvec)'):
            Y, self.iters = gmres_batched(
                lambda V: dsc[:, None] * self.matvec_exact_multi(dsc[:, None] * V, Sx),
                dsc[:, None] * E, M=lambda R: lu_apply(R), rtol=self.mixed_tol, restart=MIXED_MAX_ITER, maxiter=1)
            X = dsc[:, None] * Y
        if gpu is not None:
            gpu.release()
        sim.log.note(f"mixed precision: GMRES iterations per port {self.iters}")
        # the single-precision LU can be too weak a preconditioner (fine meshes at low frequency: the MPIE
        # matrix is ill-conditioned); an unconverged refinement returns garbage, so check and redo in double
        R = dsc[:, None] * E - dsc[:, None] * self.matvec_exact_multi(X, Sx)
        rel = np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(dsc[:, None] * E, axis=0), 1e-300)
        if (rel > 100 * self.mixed_tol).any():
            sim.log.note(f"WARNING mixed precision did not converge at {f / 1e9:.4g} GHz (residual {rel.max():.1e}) "
                         f"- solving in double precision up to {MIXED_RETRY * f / 1e9:.4g} GHz")
            self._mixed_failed = max(getattr(self, '_mixed_failed', 0.0), f)
            return self.solve_direct(f, E, block, precision='double')
        return X

    def _solve_quads(self, f, Z, E, Sx, precision, block=1024):
        """The direct solve on the quad-cell subspace (sim.qT, quads.py): Z_q = T^T Z T, LU of Z_q, I = T I_q.
        precision='mixed': complex64 LU of Z_q preconditioning GMRES on T^T A T (A the exact operator)."""
        import scipy.linalg as sla
        sim, T = self.sim, self.sim.qT
        Nq = T.shape[1]
        from .quads import project
        qi = sim.quad_info
        with sim.log.timer('quad projection', f"{T.shape[0]} -> {Nq}"):
            if not hasattr(self, '_qC'):
                self._qC = T[qi['elim']].tocsr()
            Zq = project(Z, qi['keep'], qi['elim'], self._qC)
            del Z
        dt = np.complex64 if precision == 'mixed' else np.complex128
        Zq = Zq.T                                     # Z_q^T, Fortran order: factored in place (trans=1)
        Eq = np.asarray(T.T @ E)
        dsc = 1.0 / np.sqrt(np.abs(np.diagonal(Zq)).astype(float) + 1e-300)
        with sim.log.timer('LU factor', f"N={Nq} (quads)"):
            Zq *= dsc[:, None].astype(dt)
            Zq *= dsc[None, :].astype(dt)
            lu, piv = sla.lu_factor(Zq, overwrite_a=True, check_finite=False)
            del Zq
        self.iters = []
        if precision != 'mixed':
            with sim.log.timer('LU solve'):
                return T @ (dsc[:, None] * sla.lu_solve((lu, piv), dsc[:, None] * Eq, trans=1, check_finite=False))
        with sim.log.timer('refinement (GMRES, exact matvec)'):
            Y, self.iters = gmres_batched(
                lambda V: dsc[:, None] * (T.T @ self.matvec_exact_multi(T @ (dsc[:, None] * V), Sx)),
                dsc[:, None] * Eq,
                M=lambda R: sla.lu_solve((lu, piv), R.astype(np.complex64), trans=1, check_finite=False).astype(complex),
                rtol=self.mixed_tol, restart=MIXED_MAX_ITER, maxiter=1)
            Xq = dsc[:, None] * Y
        sim.log.note(f"mixed precision: GMRES iterations per port {self.iters}")
        X = T @ Xq
        R = T.T @ (E - self.matvec_exact_multi(X, Sx))
        rel = np.linalg.norm(dsc[:, None] * R, axis=0) / np.maximum(np.linalg.norm(dsc[:, None] * Eq, axis=0), 1e-300)
        if (rel > 100 * self.mixed_tol).any():
            sim.log.note(f"WARNING mixed precision did not converge at {f / 1e9:.4g} GHz (residual {rel.max():.1e}) "
                         f"- solving in double precision up to {MIXED_RETRY * f / 1e9:.4g} GHz")
            self._mixed_failed = max(getattr(self, '_mixed_failed', 0.0), f)
            return self.solve_direct(f, E, block, precision='double')
        return X

    def _solve_tiled(self, f, E, Sx, gpu, block=1024):
        """Mixed precision without a dense Z anywhere: the symmetric part of the equilibrated Z factored as
        L L^T in complex64 tiles on the GPU (ldlt.SymFactor: block columns filled on the host one at a time,
        4 N^2 bytes on the device, streamed from the host past its memory), GMRES on the matrix-free double
        operator of that symmetric part with the factor as the preconditioner. The near-field correction is not
        quite symmetric (its singular parts are integrated analytically over one triangle of a pair and by
        quadrature over the other: up to 0.07 of the unit diagonal between a plane triangle and a strip one), the
        exact Galerkin Z is: this path solves (Z + Z^T) / 2, |dS| <= 1.2e-3 on the SMA launch (1-6 GHz) against
        Z itself. Preconditioning Z with the factor of its symmetric part did not converge (300 iterations)."""
        from .ldlt import SymFactor
        sim, core = self.sim, self.core
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        with sim.log.timer('near-field correction'):
            ri, ci, vv, qc = core.near_coo(self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
            ok = ri >= 0
            self.dZ_exact = sparse.csr_matrix((vv[ok], (ri[ok], ci[ok])), shape=(Ne, Ne))
            self._qcorr = qc
            del ri, ci, vv, ok
        if Nv:
            with sim.log.timer('via kernels', f"{Nv} chains"):
                self.B, self.C = sim._via_blocks(f, self._fan_columns())
        M = sparse.bmat([[self.dZ_exact, None], [None, sparse.csr_matrix((Nv, Nv))]]) if Nv else self.dZ_exact
        M = M + Sx
        Ms = ((M + M.T) * 0.5).tocsc()
        Cs = 0.5 * (self.C + self.C.T) if Nv else None
        # the operator of the refinement: the same symmetric part
        self.dZ_exact = (0.5 * (self.dZ_exact + self.dZ_exact.T)).tocsr()
        Sx = (0.5 * (Sx + Sx.T)).tocsr()
        if Nv:
            self.C = Cs
        diag = Ms.diagonal().astype(complex)
        for c0 in range(0, Ne, block):
            idx = np.arange(c0, min(c0 + block, Ne), dtype=np.int32)
            diag[idx] += np.diagonal(core.far_block(idx, idx, self.g, self.tb, self.w))
        if Nv:
            diag[Ne:] += np.diagonal(Cs)
        dsc = 1.0 / np.sqrt(np.abs(diag) + 1e-300)

        def fill(j0, j1):
            out = np.zeros((N - j0, j1 - j0), complex)
            c1 = min(j1, Ne)
            if j0 < c1:
                out[:Ne - j0, :c1 - j0] = core.far_block(np.arange(j0, Ne, dtype=np.int32),
                                                         np.arange(j0, c1, dtype=np.int32), self.g, self.tb, self.w)
                if Nv:
                    out[Ne - j0:, :c1 - j0] = self.B[j0:c1].T
            if j1 > Ne:
                v0, off = max(j0, Ne) - Ne, max(j0, Ne) - j0
                if j0 < Ne:
                    out[:Ne - j0, off:] = self.B[j0:Ne, v0:j1 - Ne]
                out[off:, off:] = Cs[v0:, v0:j1 - Ne]
            out += Ms[j0:, j0:j1].toarray()
            out *= dsc[j0:, None]
            out *= dsc[None, j0:j1]
            return out
        cp = gpu.cp
        fac = SymFactor(N, fill, xp=cp, block=block, budget=max(0.0, 0.9 * gpu.mem_free() - self._tiled_work(N, block)),
                        log=sim.log.note)
        with sim.log.timer('LDLT factor (GPU, tiled)', f"N={N}, block {block}"):
            fac.factor()
            cp.cuda.Device().synchronize()
        streamed = fac.on_device.count(False)
        sim.log.note(f"tiled LDL^T: {fac.held / 1e9:.1f} GB of panels on the GPU"
                     + (f", {streamed} of {len(fac.panels)} streamed from the host" if streamed else '')
                     + f", smallest pivot {fac.small_pivot:.1e}")
        B = dsc[:, None] * E
        with sim.log.timer('refinement (GMRES, exact matvec)'):
            Y, self.iters = gmres_batched(lambda V: dsc[:, None] * self.matvec_exact_multi(dsc[:, None] * V, Sx), B,
                                          M=fac.solve, rtol=self.mixed_tol, restart=30, maxiter=10)
            X = dsc[:, None] * Y
            R = B - dsc[:, None] * self.matvec_exact_multi(X, Sx)
            rel = np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-300)
        fac.release()
        sim.log.note(f"tiled mixed precision: GMRES iterations per port {self.iters}")
        if (rel > 100 * self.mixed_tol).any():
            sim.log.note(f"WARNING tiled mixed precision did not converge at {f / 1e9:.4g} GHz (residual "
                         f"{rel.max():.1e}): the pivoted LU from here on")
            self._tiled_failed = True
            return None
        return X

    @staticmethod
    def _tiled_work(N, block=1024):
        """GPU bytes the tiled factor needs besides its panels: the column being updated, a streamed panel, the
        right-hand sides of a solve."""
        return 3 * 8.0 * N * block + 0.5e9

    def _gpu_for(self, N):
        """The GPU backend when one is asked for (sim.gpu), available and the LU of N unknowns fits, else None."""
        if getattr(self.sim, 'gpu', 'off') == 'off':
            return None
        from .gpu import get_backend
        gpu = get_backend(self.sim.gpu)
        return gpu if gpu is not None and gpu.fits(N) else None

    def _fits(self, nbytes):
        from .sim import available_memory
        return nbytes + 0.9e9 <= 0.8 * available_memory()

    def _solve_hybrid(self, f, Z, dsc, E):
        """Mixed precision with the LU on the GPU: Z (double, host) equilibrated, its complex64 copy factored on
        the GPU, GMRES on the double Z with that LU as the preconditioner; double precision on the CPU when the
        refinement does not converge."""
        import scipy.linalg as sla
        sim = self.sim
        gpu = self._gpu_for(Z.shape[0])
        if self.core is not None and hasattr(self.core, 'scale_c64') and Z.dtype == complex and Z.flags.c_contiguous:
            Z64 = self.core.scale_c64(Z, np.ascontiguousarray(dsc, float))
        else:
            Z *= dsc[:, None]
            Z *= dsc[None, :]
            Z64 = None
        if not getattr(self, '_gpu_logged', False):
            self._gpu_logged = True
            sim.log.note(f"GPU: {gpu.device} via {gpu.name}, {gpu.mem_free() / 1e9:.1f} GB free: complex64 LU there, "
                         f"double-precision refinement on the host")
        with sim.log.timer('LU factor (GPU)', f"N={Z.shape[0]}, {gpu.name}"):
            try:
                gpu.factor(Z.astype(np.complex64) if Z64 is None else Z64)
                del Z64
            except Exception as e:                          # driver / unsupported dtype: double on the CPU
                sim.log.note(f"GPU LU failed ({type(e).__name__}: {e}) - factoring on the CPU in double")
                gpu.release()
                lu, piv = sla.lu_factor(Z.T, overwrite_a=True, check_finite=False)
                return dsc[:, None] * sla.lu_solve((lu, piv), dsc[:, None] * E, trans=1, check_finite=False)
        B = dsc[:, None] * E
        with sim.log.timer('refinement (GMRES, dense double)'):
            # 2 restarts at most: where the complex64 LU is too weak (many via chains: the SMA launch, 536 via
            # unknowns) 300 iterations had not converged and cost 55 s, three times a double LU
            Y, self.iters = gmres_batched(lambda V: Z @ V, B, M=lambda R: gpu.solve(R), rtol=self.mixed_tol,
                                          restart=30, maxiter=2)
            rel = np.linalg.norm(B - Z @ Y, axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-300)
        gpu.release()
        sim.log.note(f"mixed precision (GPU): GMRES iterations per port {self.iters}")
        if (rel > 100 * self.mixed_tol).any():
            sim.log.note(f"mixed precision (GPU) did not converge at {f / 1e9:.4g} GHz (residual {rel.max():.1e}): "
                         f"double precision up to {HYBRID_RETRY * f / 1e9:.4g} GHz")
            self._hybrid_failed = max(getattr(self, '_hybrid_failed', 0.0), f)
            with sim.log.timer('LU factor', f"N={Z.shape[0]}"):
                lu, piv = sla.lu_factor(Z.T, overwrite_a=True, check_finite=False)
            Y = sla.lu_solve((lu, piv), B, trans=1, check_finite=False)
        return dsc[:, None] * Y

    def _solve_direct_numpy(self, f, E, block=1024):
        """Same entries as the fast operator (centroid far field + exact near correction), written
        into a dense Z block by block and LU-solved: memory ~ N^2 complex, robust at resonances."""
        sim = self.sim
        self.setup(f, mode='sparse')
        _, Sx = self.extra_blocks(f)
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        Z = np.zeros((N, N), complex)
        with sim.log.timer('direct fill', f"N={N}"):
            idx = np.arange(Ne)
            for r0 in range(0, Ne, block):
                rr = idx[r0:r0 + block]
                for c0 in range(0, Ne, block):
                    cc = idx[c0:c0 + block]
                    Z[r0:r0 + len(rr), c0:c0 + len(cc)] = self.entry(rr, cc)
            Z[:Ne, :Ne] += (self.dZ + Sx[:Ne, :Ne]).toarray() if Ne < 4000 else 0
            if Ne >= 4000:
                Dz = (self.dZ + Sx[:Ne, :Ne]).tocoo()
                np.add.at(Z, (Dz.row, Dz.col), Dz.data)
            if Nv:
                Z[:Ne, Ne:] = self.B
                Z[Ne:, :Ne] = self.B.T
                Z[Ne:, Ne:] = self.C
        with sim.log.timer('LU solve', f"N={N}"):
            import scipy.linalg as sla
            # symmetric diagonal equilibration (via fan-charge rows dwarf strip rows, esp. at low f)
            dsc = 1.0 / np.sqrt(np.abs(np.diagonal(Z)) + 1e-300)
            Z *= dsc[:, None]
            Z *= dsc[None, :]
            lu, piv = sla.lu_factor(Z.T, overwrite_a=True, check_finite=False)     # in place (see above)
            del Z
            X = dsc[:, None] * sla.lu_solve((lu, piv), dsc[:, None] * E, trans=1, check_finite=False)
        self.iters = []
        return X

    def solve_blr(self, f, E, tol=1e-4, tile=256, eta=1.0, dtype=np.complex64):
        """Block low-rank LU preconditioner + GMRES with the exact matrix-free operator (no stored Z)."""
        from .blr import BLRFactor, cluster_tiles
        assert self.core is not None, "solver='blr' needs the compiled core (python -m mom25d.core.build)"
        sim = self.sim
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        self.setup(f, mode='sparse')
        self.dZ_exact = self.dZ
        _, Sx = self.extra_blocks(f)
        Sx = Sx.tocsr()
        dZx = (self.dZ + Sx[:Ne, :Ne]).tocsr()
        pts = np.vstack([sim.rwg.mid] + ([np.array([[v.x, v.y] for v, _, _ in sim.chains])] if Nv else []))
        tiles, boxes = cluster_tiles(pts, tile)

        def raw(rows, cols):
            M = np.zeros((len(rows), len(cols)), complex)
            re, ce = rows < Ne, cols < Ne
            if re.any() and ce.any():
                r_, c_ = rows[re], cols[ce]
                M[np.ix_(re, ce)] = self.core.far_block(r_.astype(np.int32), c_.astype(np.int32), self.g, self.tb, self.w) \
                    + dZx[r_][:, c_].toarray()
            if Nv:
                if re.any() and (~ce).any():
                    M[np.ix_(re, ~ce)] = self.B[rows[re]][:, cols[~ce] - Ne]
                if (~re).any() and ce.any():
                    M[np.ix_(~re, ce)] = self.B[cols[ce]][:, rows[~re] - Ne].T
                if (~re).any() and (~ce).any():
                    M[np.ix_(~re, ~ce)] = self.C[np.ix_(rows[~re] - Ne, cols[~ce] - Ne)]
            return M
        with sim.log.timer('BLR assemble + factor', f"N={N}, {len(tiles)} tiles"):
            diag = np.empty(N, complex)
            for t in tiles:
                diag[t] = np.diagonal(raw(t, t))
            dsc = 1.0 / np.sqrt(np.abs(diag) + 1e-300)
            blr = BLRFactor(lambda r, c: raw(r, c) * dsc[r][:, None] * dsc[c][None, :], tiles, boxes,
                            tol=tol, eta=eta, dtype=dtype, scale=1.0)
        dense_eq = 16 * N * N
        sim.log.note(f"BLR: {blr.memory / 2 ** 20:.0f} MB ({100 * blr.memory / (8 * N * N):.0f}% of complex64 dense, "
                     f"{100 * blr.memory / dense_eq:.0f}% of complex128), {blr.stats['lowrank']} low-rank tiles, "
                     f"mean rank {blr.stats['rank_sum'] / max(1, blr.stats['lowrank']):.0f}")
        self.blr = blr
        A = LinearOperator((N, N), matvec=lambda y: dsc * self.matvec_exact(dsc * y, Sx), dtype=complex)
        M = LinearOperator((N, N), matvec=blr.solve, dtype=complex)
        X = np.zeros((N, E.shape[1]), complex)
        self.iters = []
        with sim.log.timer('GMRES (BLR preconditioned)'):
            for j in range(E.shape[1]):
                it = [0]
                y, info = gmres(A, dsc * E[:, j], M=M, rtol=self.mixed_tol, restart=50, maxiter=10,
                                callback=lambda _: it.__setitem__(0, it[0] + 1), callback_type='pr_norm')
                X[:, j] = dsc * y
                self.iters.append(it[0])
        sim.log.note(f"BLR: GMRES iterations per port {self.iters}")
        return X

def gmres_batched(A, B, M=None, x0=None, rtol=1e-6, restart=30, maxiter=10):
    """
    Right-preconditioned restarted GMRES for k independent systems A x_j = b_j run in lockstep, so
    every operator / preconditioner call is one batched (N x k_active) product. A, M: callables on
    (N x k) arrays. Converged columns drop out. Returns (X, iterations per column).
    """
    B = np.asarray(B, complex)
    N, k = B.shape
    X = np.zeros((N, k), complex) if x0 is None else np.array(x0, complex)
    M = M or (lambda R: R)
    bn = np.linalg.norm(B, axis=0)
    bn[bn == 0] = 1.0
    iters = np.zeros(k, int)
    act = np.arange(k)
    for _ in range(maxiter):
        R = B[:, act] - (A(X[:, act]) if np.any(X[:, act]) else 0)
        beta = np.linalg.norm(R, axis=0)
        done = beta <= rtol * bn[act]
        act, R, beta = act[~done], R[:, ~done], beta[~done]
        if not len(act):
            break
        ka = len(act)
        V = np.zeros((restart + 1, N, ka), complex)
        Z = np.zeros((restart, N, ka), complex)
        H = np.zeros((restart + 1, restart, ka), complex)
        cs, sn = np.zeros((restart, ka), complex), np.zeros((restart, ka), complex)
        g = np.zeros((restart + 1, ka), complex)
        g[0] = beta
        V[0] = R / beta
        live = np.ones(ka, bool)
        last = np.full(ka, restart - 1)
        for j in range(restart):
            Z[j] = M(V[j])
            Wv = A(Z[j])
            for i in range(j + 1):                                   # modified Gram-Schmidt
                hij = np.einsum('nk,nk->k', V[i].conj(), Wv)
                H[i, j] = hij
                Wv = Wv - V[i] * hij
            hn = np.linalg.norm(Wv, axis=0)
            H[j + 1, j] = hn
            V[j + 1] = Wv / np.where(hn > 0, hn, 1)
            for i in range(j):                                       # apply previous Givens rotations
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -np.conj(sn[i]) * H[i, j] + np.conj(cs[i]) * H[i + 1, j]
                H[i, j] = t
            den = np.sqrt(np.abs(H[j, j]) ** 2 + np.abs(H[j + 1, j]) ** 2)
            den[den == 0] = 1.0
            cs[j] = H[j, j] / den
            sn[j] = H[j + 1, j] / den
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0
            g[j + 1] = -np.conj(sn[j]) * g[j]
            g[j] = cs[j] * g[j]
            iters[act[live]] += 1
            conv = live & (np.abs(g[j + 1]) <= rtol * bn[act])
            last[conv] = j
            live &= ~conv
            if not live.any():
                break
        for c in range(ka):                                           # update each column
            m = last[c] + 1
            y = np.linalg.solve(np.triu(H[:m, :m, c]), g[:m, c]) if m else np.zeros(0)
            X[:, act[c]] += Z[:m, :, c].T @ y
    return X, iters.tolist()
