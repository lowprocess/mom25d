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
import os
import time

import numpy as np
import scipy.linalg as sla
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.spatial import cKDTree

from .base import gram_matrix, moments7, quad_moments
from .sing import offplane_pairs, offplane_moments

try:                                    # optional compiled kernels (python -m mom25d.core.build)
    from . import _core as _built
except ImportError:
    _built = None
_core = None if os.environ.get('MOM25D_NO_CORE') else _built


def use_core(mode='auto'):
    """Which kernels mom25d uses from now on: 'auto' (the compiled core if built, unless MOM25D_NO_CORE is set),
    'compiled' (the core: RuntimeError if it is not built) or 'python' (NumPy). Returns the previous module (or
    None), for set_core to put back."""
    global _core
    if mode not in ('auto', 'compiled', 'python'):
        raise ValueError(f"core: auto, compiled or python, not {mode!r}")
    if mode == 'compiled' and _built is None:
        raise RuntimeError("mom25d's compiled core is not built (python -m mom25d.core.build): it cannot be "
                           "chosen; 'python' or 'auto' run on NumPy")
    prev, _core = _core, None if mode == 'python' or (mode == 'auto' and os.environ.get('MOM25D_NO_CORE')) else _built
    return prev


def set_core(module):
    """Put back what use_core returned."""
    global _core
    _core = module


def fast_cores():
    """The physical cores of the fastest class (a hybrid CPU's P-cores; every core of the others), from Linux's
    /sys; None where it cannot tell."""
    import glob
    cores = {}
    for d in glob.glob('/sys/devices/system/cpu/cpu[0-9]*'):
        try:
            f = int(open(d + '/cpufreq/cpuinfo_max_freq').read())
            key = tuple(open(f'{d}/topology/{k}').read().strip() for k in ('physical_package_id', 'core_id'))
        except (OSError, ValueError):
            return None
        cores[key] = max(cores.get(key, 0), f)
    top = max(cores.values(), default=0)
    return sum(f >= 0.9 * top for f in cores.values()) or None


_blas_capped = False


def cap_blas_threads():
    """Once: the LU's BLAS (OpenBLAS / MKL, through threadpoolctl) on the fast physical cores only. A BLAS LU splits
    each step evenly over its threads and waits for the slowest: on an i7-13700K (8 P-cores with their hyperthreads,
    8 E-cores) its 24 threads made rf_stubnotch's 1.5k-unknown LU 0.62 s against 0.047 s on 8 (a whole kicad2ems
    run 10.8 s against 3.3 s), as fast at 5k unknowns. MOM25D_BLAS_THREADS (or OPENBLAS_NUM_THREADS / MKL_NUM_THREADS)
    sets it instead; without threadpoolctl nothing is changed."""
    global _blas_capped
    if _blas_capped:
        return
    _blas_capped = True
    env = os.environ.get('MOM25D_BLAS_THREADS')
    if not env and (os.environ.get('OPENBLAS_NUM_THREADS') or os.environ.get('MKL_NUM_THREADS')):
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    n = int(env) if env else fast_cores()
    if n:
        threadpool_limits(limits=n, user_api='blas')


NEAR_CHUNK = 8_000_000      # near-field triplets per piece of the sum into the sparse matrix (_near_csr)


def _near_csr(ri, ci, vv, n, owner):
    """The n x n CSR of the core's near-field triplets, duplicates summed. The structure is built once and
    kept on `owner` (3.4 s on the SMA launch's 84 M triplets); each frequency then only sums
    its values into it, a piece at a time (0.6 s, against 3.1 s for SciPy's COO -> CSR every frequency)."""
    pat = getattr(owner, '_near_pat', None)
    if pat is None or pat[0] != (len(ri), n):
        # the CSR structure of the triplets (row < 0: unused), the same at every frequency (a pair's nine slots are
        # the edges of its two triangles): the used slots in CSR order, and where each entry's run of them starts
        slot = np.flatnonzero(ri >= 0)
        key = ri[slot].astype(np.int64) * n + ci[slot]
        o = np.argsort(key)
        slot, key = slot[o].astype(np.int32), key[o]
        del o
        first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        key = key[first]
        indptr = np.r_[0, np.cumsum(np.bincount(key // n, minlength=n))].astype(np.int32)
        pat = owner._near_pat = ((len(ri), n), (indptr, (key % n).astype(np.int32), slot, first.astype(np.int32)))
    indptr, indices, slot, first = pat[1]
    data = np.empty(len(first), vv.dtype)
    a = 0
    while a < len(first):                              # whole entries per piece: about NEAR_CHUNK slots
        b = min(len(first), int(np.searchsorted(first, first[a] + NEAR_CHUNK)))
        s1 = first[b] if b < len(first) else len(slot)
        data[a:b] = np.add.reduceat(vv[slot[first[a]:s1]], first[a:b] - first[a])
        a = b
    M = sparse.csr_matrix((data, indices, indptr), shape=(n, n))
    M.has_canonical_format = True
    return M


# ---- Cluster tree and H-matrix ----------------------------------------------
class Cluster:
    __slots__ = ('idx', 'lo', 'hi', 'kids')

    def __init__(self, idx, pts, leaf):
        p = pts[idx]
        self.idx, self.lo, self.hi, self.kids = idx, p.min(0), p.max(0), []
        if len(idx) > leaf:                             # bisect along the longest side
            o, h = np.argsort(p[:, np.argmax(self.hi - self.lo)], kind='stable'), len(idx) // 2
            self.kids = [Cluster(idx[o[:h]], pts, leaf), Cluster(idx[o[h:]], pts, leaf)]

    @property
    def diam(self):
        return np.linalg.norm(self.hi - self.lo)


def aca(fill_rows, fill_cols, m, n, tol, max_rank, atol=0.0):
    """Adaptive cross approximation with partial pivoting: block ~ U @ V.
    Stops when the new cross is below tol x (block norm) or below atol per entry (global scale)."""
    U, V, used_r, i, norm2 = [], [], set(), 0, 0.0
    for _ in range(min(m, n, max_rank)):
        used_r.add(i)
        row = fill_rows(np.array([i]))[0]
        for u, v in zip(U, V):
            row = row - u[i] * v
        j = int(np.argmax(np.abs(row)))
        if abs(row[j]) < max(1e-300, 0.1 * atol):
            if not U and abs(row[j]) < 0.1 * atol:
                break                                  # block negligible at the global scale
            i = next((r for r in range(m) if r not in used_r), None)
            if i is None:
                break
            continue
        v, u = row / row[j], fill_cols(np.array([j]))[:, 0]
        for uu, vv in zip(U, V):
            u = u - vv[j] * uu
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cross = sum(2 * np.real(np.vdot(uu, u) * np.vdot(v, vv)) for uu, vv in zip(U, V))
        norm2 += cross + (nu * nv) ** 2
        U.append(u); V.append(v)
        if nu * nv <= max(tol * np.sqrt(abs(norm2)), atol * np.sqrt(m * n)):
            break
        a = np.abs(u)
        a[list(used_r)] = -1
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
            if max(a.diam, b.diam) < eta * np.linalg.norm(np.maximum(0, np.maximum(a.lo - b.hi, b.lo - a.hi))):
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
                stack += [(x, y) for x in (a.kids or [a]) for y in (b.kids or [b])]

    def finalize(self):
        """Merge all dense leaves into one sparse matrix; stack low-rank factors for batched products."""
        def blocks(items, shape):                   # [(rows, cols, block)] -> one sparse matrix of their sum
            ij = [np.meshgrid(r, c, indexing='ij') for r, c, _ in items]
            return sparse.csr_matrix((np.concatenate([M.ravel() for _, _, M in items] or [[]]),
                                      (np.concatenate([i.ravel() for i, _ in ij] or [[]]).astype(int),
                                       np.concatenate([j.ravel() for _, j in ij] or [[]]).astype(int))), shape=shape)
        self.D = blocks(self.dense, (self.n, self.n))
        self.dense_nnz, self.dense = self.D.nnz, []
        # low rank: y[r] += U (V x[c])  ->  sparse 'V' (all ranks stacked) then sparse 'U'
        lr = self.lowrank
        k0 = np.cumsum([0] + [U.shape[1] for _, _, U, _ in lr])
        ks = [np.arange(a, b) for a, b in zip(k0[:-1], k0[1:])]
        self.Vs = blocks([(k, c, V) for k, (_, c, _, V) in zip(ks, lr)], (k0[-1], self.n)) if lr else None
        self.Us = blocks([(r, k, U) for k, (r, _, U, _) in zip(ks, lr)], (self.n, k0[-1])) if lr else None
        self.lr_size = sum(U.size + V.size for _, _, U, V in lr)
        self.ranks, self.lowrank = [U.shape[1] for _, _, U, _ in lr], []

    def matvec(self, x):
        y = self.D @ x
        if self.Vs is not None:
            y = y + self.Us @ (self.Vs @ x)
        return y

    def stats(self):
        mem = 16 * (self.dense_nnz + self.lr_size)
        return dict(dense_entries=self.dense_nnz, lowrank_blocks=len(self.ranks),
                    mean_rank=round(float(np.mean(self.ranks)), 1) if self.ranks else 0.0,
                    MB=round(mem / 2 ** 20, 1), compression=round(mem / (16 * self.n ** 2), 3))


# ---- Fast operator for a Simulation -----------------------------------------
HYBRID_RETRY = 1.25      # after the GPU's mixed precision fails at f, double up to this x f
MIXED_RETRY = HYBRID_RETRY   # the same after the complex64 LU (CPU or GPU) fails to precondition
MIXED_MAX_ITER = 30      # refinement iterations before giving up on the complex64 LU (2 on the demo transition; a stall
                         # ran all 300 of restart 30 x 10: 150 s on 8.4k unknowns against 18 s for a double LU)




def _equilibrate(Z, dsc=None):
    """Symmetric diagonal equilibration of Z in place (via fan-charge rows dwarf strip rows, most at low f), by
    dsc or 1 / sqrt|Z_ii|; returns the scales."""
    if dsc is None:
        dsc = 1.0 / np.sqrt(np.abs(np.diagonal(Z)).astype(float) + 1e-300)
    Z *= dsc[:, None].astype(Z.dtype)
    Z *= dsc[None, :].astype(Z.dtype)
    return dsc


def _lu(Z):
    """LU of Z^T, in place: Z^T of a C-ordered Z is Fortran-ordered, so LAPACK needs no second N^2 copy."""
    return sla.lu_factor(Z.T, overwrite_a=True, check_finite=False)


def _lu_solve(lu, R):
    """Z^-1 R with _lu's factor (trans=1), in the factor's precision; complex128 out."""
    return sla.lu_solve(lu, R.astype(lu[0].dtype), trans=1, check_finite=False).astype(complex)


class FastOperator:
    """Builds and applies Z for a mom25d.sim.Simulation without forming it densely."""

    def __init__(self, sim, near_fac=None, tol=1e-4, leaf=64, eta=2.0, max_rank=80, atol_rel=1e-8):
        r = sim.rwg
        self.sim, self.trace, self.mixed_tol, self.atol_rel, self.max_rank = sim, False, 1e-6, atol_rel, max_rank
        self.near_fac, self.tol, self.leaf, self.eta = near_fac or sim.near_fac, tol, leaf, eta
        self.tiled, self.tiled_block = None, 1024          # None: sim.tiled
        self._mixed_failed = self._hybrid_failed = 0.0      # the highest frequency the complex64 LU failed at
        self._tiled_failed = self._gpu_logged = False
        # half-basis description: half h is edge h on its T+ (h < Ne), edge h - Ne on its T-: triangle, alpha,
        # free vertex; and the halves on each triangle (-1: none)
        self.h_tri = np.concatenate([r.tp, r.tm])
        self.h_alpha = np.concatenate([r.l / (2 * r.area[r.tp]), -r.l / (2 * r.area[r.tm])])
        self.h_v = np.vstack([r.vp, r.vm])
        self.tri_halves, fill = -np.ones((len(r.tris), 3), int), np.zeros(len(r.tris), int)
        for h, t in enumerate(self.h_tri):
            self.tri_halves[t, fill[t]] = h
            fill[t] += 1
        self.layer_of_tri, self.cent, self.area, self.layers = sim.tri_layer, r.cent, r.area, sim.layers
        self.lidx = {k: i for i, k in enumerate(self.layers)}
        self._static_cache, self.dZ_exact, self.core = {}, None, _core
        self._near_pairs()
        if self.core is not None:                       # the compiled core's geometry
            Ne, H, c = sim.Ne, self.tri_halves, np.ascontiguousarray
            self.g = dict(ep=r.tp.astype(np.int32), em=r.tm.astype(np.int32),
                          lt=np.array([self.lidx[k] for k in self.layer_of_tri], np.int32),
                          te=np.where(H >= 0, H % Ne, -1).astype(np.int32), ta=np.where(H >= 0, self.h_alpha[H], 0.0),
                          tv=np.where((H >= 0)[..., None], self.h_v[H], 0.0),
                          ap=self.h_alpha[:Ne].copy(), am=self.h_alpha[Ne:].copy(), vp=c(self.h_v[:Ne]),
                          vm=c(self.h_v[Ne:]), cent=c(r.cent), area=r.area.copy())
            self.q = dict(nodes=c(r.nodes), tris=r.tris.astype(np.int32), P3=c(sim.P3), W3=c(sim.W3), P6=c(sim.P6),
                          W6=c(sim.W6))
            self.nP32, self.nQ32 = self.nP.astype(np.int32), self.nQ.astype(np.int32)
            self._near_pat = None                       # _near_csr: the structure of these pairs

    # ---- near pairs (geometry only) ---------------------------------------------------------------
    def _near_pairs(self):
        sim, r = self.sim, self.sim.rwg
        size = sim.tri_size
        if _core is not None and hasattr(_core, 'near_pairs'):     # the same pairs, in the same order
            self.nP, self.nQ = _core.near_pairs(np.ascontiguousarray(r.cent, float), np.ascontiguousarray(size, float),
                                                np.ascontiguousarray(sim.st.z[self.layer_of_tri], float),
                                                float(self.near_fac))
        else:
            lists = cKDTree(r.cent).query_ball_point(r.cent, self.near_fac * size)
            P = np.concatenate([np.full(len(qs), p) for p, qs in enumerate(lists)])
            Q = np.concatenate([np.asarray(qs, int) for qs in lists])
            n = len(size)
            key = np.unique(np.concatenate([P * n + Q, Q * n + P]))     # symmetrise: near if within either's radius
            P, Q = key // n, key % n
            rho = np.linalg.norm(r.cent[P] - r.cent[Q], axis=1)
            dz = np.abs(sim.st.z[self.layer_of_tri[P]] - sim.st.z[self.layer_of_tri[Q]])
            keep = rho ** 2 + dz ** 2 < (self.near_fac * np.maximum(size[P], size[Q])) ** 2
            self.nP, self.nQ = P[keep], Q[keep]
        sim.log.note(f"fast solver: {len(self.nP)} near triangle pairs ({len(self.nP) / len(size):.0f} per triangle)")

    # ---- kernels ----------------------------------------------------------------------------------
    def _kern(self, a, b, rho):
        return self.tables[(min(a, b), max(a, b))].total(rho)

    def _far_halves(self, hr, hc, outer=True):
        """Centroid-rule interaction between half-bases hr and hc: every row with every column (outer), or pairwise."""
        w, tr, tc = self.w, self.h_tri[hr], self.h_tri[hc]
        row, col = ((lambda a: a[:, None]), (lambda a: a[None])) if outer else ((lambda a: a), (lambda a: a))
        cr, cc = row(self.cent[tr]), col(self.cent[tc])
        rho = np.sqrt(((cr - cc) ** 2).sum(-1))
        lr, lc = row(self.layer_of_tri[tr]), col(self.layer_of_tri[tc])
        GA, GQ = np.zeros(rho.shape, complex), np.zeros(rho.shape, complex)
        for a in np.unique(lr):
            for b in np.unique(lc):
                m = (lr == a) & (lc == b)
                if m.any():
                    GA[m], GQ[m] = self._kern(a, b, rho[m])
        dot = ((cr - row(self.h_v[hr])) * (cc - col(self.h_v[hc]))).sum(-1)
        return (row(self.h_alpha[hr] * self.area[tr]) * col(self.h_alpha[hc] * self.area[tc])
                * (1j * w * dot * GA + 4 * GQ / (1j * w)))

    def entry_pairs(self, rows, cols, chunk=200000):
        """Centroid-rule Z for individual (row, col) edge pairs."""
        Ne, out = self.sim.Ne, np.zeros(len(rows), complex)
        for c0 in range(0, len(rows), chunk):
            rr, cc = rows[c0:c0 + chunk], cols[c0:c0 + chunk]
            out[c0:c0 + chunk] = sum(self._far_halves(rr + i, cc + j, outer=False) for i in (0, Ne) for j in (0, Ne))
        return out

    def entry(self, rows, cols):
        if self.core is not None:
            return self.core.far_block(np.asarray(rows, np.int32), np.asarray(cols, np.int32), self.g, self.tb, self.w)
        Ne, m, n = self.sim.Ne, len(rows), len(cols)
        H = self._far_halves(np.concatenate([rows, rows + Ne]), np.concatenate([cols, cols + Ne]))
        return H[:m, :n] + H[:m, n:] + H[m:, :n] + H[m:, n:]

    # ---- near correction ----------------------------------------------------------------------------
    def _near_moments(self):
        """Accurate minus centroid moments (base.moments7's order) for the near triangle pairs."""
        sim, r, P, Q = self.sim, self.sim.rwg, self.nP, self.nQ
        la, lb = self.layer_of_tri[P], self.layer_of_tri[Q]
        out = np.zeros((len(P), 7), complex)
        for a in self.layers:
            for b in self.layers:
                m = np.nonzero((la == a) & (lb == b))[0]
                if not len(m):
                    continue
                gt, p, q = self.tables[(min(a, b), max(a, b))], P[m], Q[m]
                # smooth rest by quadrature, plus the analytic singular / near-singular parts
                out[m] = quad_moments(*((sim.P3, sim.W3) if a == b else (sim.P6, sim.W6)), p, q, gt).T
                for d, cA, cq in gt.extracted():
                    if (a, b, d) not in self._static_cache:
                        self._static_cache[(a, b, d)] = offplane_pairs(r.nodes, r.tris, p, q, d)
                    out[m] += np.stack(offplane_moments(cA, cq, self._static_cache[(a, b, d)]), 1)
                # less the centroid-rule value used by the far operator
                GA, GQ = self._kern(a, b, np.linalg.norm(r.cent[p] - r.cent[q], axis=1))
                wgt = r.area[p] * r.area[q]
                out[m] -= np.stack(moments7(GA * wgt, GQ * wgt, r.cent[p], r.cent[q]), 1)
        return out

    def _fan_columns(self):
        """Accurate G_q moments between every triangle and the via fan triangles (Nt x |F|)."""
        r, F, lt = self.sim.rwg, self.sim.F, self.layer_of_tri
        if self.core is not None and hasattr(self.core, 'fan_q') and getattr(self, 'tb', None) is not None:
            QF = self.core.fan_q(self.g, self.tb, np.asarray(F, np.int32))           # compiled, OpenMP
        else:
            QF = np.zeros((len(r.tris), len(F)), complex)
            for j, q in enumerate(F):
                rho = np.linalg.norm(r.cent - r.cent[q], axis=1)
                for a in self.layers:
                    m = lt == a
                    QF[m, j] = self._kern(a, lt[q], rho[m])[1] * r.area[m] * r.area[q]
        # exact correction for near pairs only (same split as the rest of the operator)
        pos, sel = {q: j for j, q in enumerate(F)}, np.isin(self.nQ, F)
        np.add.at(QF, (self.nP[sel], np.array([pos[q] for q in self.nQ[sel]], int)), self._qcorr[sel])
        return QF

    def _near_matrix(self):
        """Sparse (Ne x Ne) correction from triangle-pair moments; keeps their G_q part (_qcorr: the via fans')."""
        Ne, w, P, Q = self.sim.Ne, self.w, self.nP, self.nQ
        M = self._near_moments()
        self._qcorr = M[:, 6]
        rows, cols, vals = [], [], []
        for i in range(3):
            for j in range(3):
                hi, hj = self.tri_halves[P, i], self.tri_halves[Q, j]
                ok = (hi >= 0) & (hj >= 0)
                hi, hj, mm = hi[ok], hj[ok], M[ok]
                vi, vj = self.h_v[hi], self.h_v[hj]
                A = (mm[:, 5] - (vj[:, 0] * mm[:, 1] + vj[:, 1] * mm[:, 2])
                     - (vi[:, 0] * mm[:, 3] + vi[:, 1] * mm[:, 4]) + (vi * vj).sum(1) * mm[:, 0])
                rows.append(hi % Ne); cols.append(hj % Ne)
                vals.append(self.h_alpha[hi] * self.h_alpha[hj] * (1j * w * A + 4 * mm[:, 6] / (1j * w)))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(Ne, Ne))

    def _near_exact(self):
        """The core's exact near-field correction (sparse, double); keeps its via-fan part (_qcorr)."""
        ri, ci, vv, self._qcorr = self.core.near_coo(self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
        return _near_csr(ri, ci, vv, self.sim.Ne, self)

    def _vias(self, f):
        """The via rows / columns B and block C (sim._via_blocks with this operator's fan columns)."""
        if self.sim.Nv:
            with self.sim.log.timer('via kernels', f"{self.sim.Nv} chains"):
                self.B, self.C = self.sim._via_blocks(f, self._fan_columns())

    # ---- compiled-core data ---------------------------------------------------------------------
    def _core_tables(self):
        keys = sorted(self.tables)
        T = [self.tables[k] for k in keys]
        tab = np.zeros((len(self.layers),) * 2, np.int32)
        for i, (a, b) in enumerate(keys):
            tab[self.lidx[a], self.lidx[b]] = tab[self.lidx[b], self.lidx[a]] = i
        get = lambda attr, dt=float: np.array([dt(getattr(t, attr)) for t in T])
        self.tb = dict(A=np.stack([t.A_rest for t in T]).astype(complex), Q=np.stack([t.Q_rest for t in T]).astype(complex),
                       dr=np.array([t.rho[1] for t in T]), d=get('d'), dimg=get('dimg'), cA=get('cA', complex),
                       cq=get('cq', complex), cAi=get('cAi', complex), cqi=get('cqi', complex), tab=tab,
                       kind=np.array([0 if a == b else (1 if t.cA != 0 else 2) for (a, b), t in zip(keys, T)], np.int32))

    # ---- assemble & solve ---------------------------------------------------------------------------------
    def matvec_exact(self, X, Sx=None):
        """Z X in double precision without storing Z, X: N or N x k (far: triangle-aggregated, one kernel pass for
        all columns; near: dZ_exact, or on the fly)."""
        Ne, core, X = self.sim.Ne, self.core, np.asarray(X, complex)
        Xe = np.ascontiguousarray(X[:Ne])
        per_column = lambda fn: fn(Xe) if X.ndim == 1 else \
            np.stack([fn(np.ascontiguousarray(Xe[:, j])) for j in range(X.shape[1])], 1)
        if X.ndim == 2 and hasattr(core, 'matvec_far_multi'):
            far = core.matvec_far_multi(Xe, self.g, self.tb, self.w)
        else:
            far = per_column(lambda x: core.matvec_far(x, self.g, self.tb, self.w))
        near = self.dZ_exact @ Xe if self.dZ_exact is not None else \
            per_column(lambda x: core.near_matvec(x, self.g, self.tb, self.q, self.nP32, self.nQ32, self.w))
        return self._bordered(X, far + near + (Sx[:Ne, :Ne] @ Xe if Sx is not None else 0))

    def _bordered(self, X, Ye):
        """Z X from Ye, the edge-edge block's product: the via rows and columns (B, C) added."""
        Ne, Y = self.sim.Ne, np.empty(X.shape, complex)
        Y[:Ne] = Ye
        if self.sim.Nv:
            Y[:Ne] += self.B @ X[Ne:]
            Y[Ne:] = self.B.T @ X[:Ne] + self.C @ X[Ne:]
        return Y

    def setup(self, f, hmatrix=True, mode=None):
        """mode: 'aca' (sparse near field + H-matrix), 'direct' (tables only when the core fills Z),
        'sparse' (sparse near field, no H-matrix)."""
        mode, sim = mode or ('aca' if hmatrix else 'sparse'), self.sim
        self.f, self.w, self.tables = f, 2 * np.pi * f, sim.green_tables(f)
        if self.core is not None:
            self._core_tables()
            if mode == 'direct':
                return
        with sim.log.timer('near-field correction'):
            dZ = self._near_exact() if self.core is not None else self._near_matrix()
        self._vias(f)
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
        """Lumped elements and metal loss: a sparse N x N addition to Z."""
        sim, Ne, N = self.sim, self.sim.Ne, self.sim.Ne + self.sim.Nv
        S = sparse.lil_matrix((N, N), dtype=complex)
        for el, (idx, sig) in zip(sim.lumped, sim.lumped_edges):
            u = sig * sim.rwg.l[idx]
            S[np.ix_(idx, idx)] = S[np.ix_(idx, idx)].toarray() + el.Z(f) * np.outer(u, u)
        if sim.metal is not None:
            if sim._gram is None:
                sim._gram = sparse.csr_matrix(gram_matrix(sim.rwg))
            S[:Ne, :Ne] = S[:Ne, :Ne] + sim.metal.zs(f) * sparse.csr_matrix(sim._gram)
        return S.tocsr()

    def _gmres_columns(self, A, M, B, tol, restart, maxiter):
        """SciPy's GMRES column by column: (X, convergence info per column); iteration counts in self.iters."""
        X, infos, self.iters = np.zeros(B.shape, complex), [], []
        for j in range(B.shape[1]):
            it, t0 = [0], time.time()

            def cb(rn):
                it[0] += 1
                if self.trace and it[0] % 25 == 0:
                    print(f"      GMRES it {it[0]}: residual {rn:.2e}  [{time.time() - t0:.1f}s]", flush=True)
            X[:, j], info = gmres(A, B[:, j], M=M, rtol=tol, restart=restart, maxiter=maxiter, callback=cb,
                                  callback_type='pr_norm')
            self.iters.append(it[0]); infos.append(info)
        return X, infos

    def _refine(self, A, B, M, restart, maxiter):
        """Batched GMRES on A Y = B preconditioned with M, to mixed_tol: (Y, converged, largest relative residual).
        Not converged includes a breakdown (NaN)."""
        Y, self.iters = gmres_batched(A, B, M=M, rtol=self.mixed_tol, restart=restart, maxiter=maxiter)
        rel = np.linalg.norm(B - A(Y), axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-300)
        return Y, bool((rel <= 100 * self.mixed_tol).all()), rel.max()

    def solve(self, f, E, tol=1e-6, restart=200, maxiter=10):
        sim = self.sim
        self.setup(f)
        Sx = self.extra_blocks(f)
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
            return dsc * self._bordered(x, self.H.matvec(x[:Ne]) + self.dZ @ x[:Ne] + Sx[:Ne, :Ne] @ x[:Ne])
        with sim.log.timer('GMRES', f"{E.shape[1]} rhs"):
            Y, infos = self._gmres_columns(LinearOperator((N, N), matvec=mv, dtype=complex),
                                           LinearOperator((N, N), matvec=lu.solve, dtype=complex),
                                           dsc[:, None] * E, tol, restart, maxiter)
        for j, info in enumerate(infos):
            if info > 0:
                sim.log.note(f"WARNING GMRES did not converge for port {j + 1} ({self.iters[j]} iterations)")
        sim.log.note(f"GMRES iterations per port: {self.iters}")
        return dsc[:, None] * Y

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
        Sx, self.iters = self.extra_blocks(f), []
        T = sim.qT
        quads = T is not None                       # quad cells / line functions: Z projected before the LU
        Nq = T.shape[1] if quads else N
        # mixed precision with a GPU: Z filled in double and kept on the host (24 N^2 bytes with the complex64
        # copy the GPU factors), the refinement products dense: 5.7 s a solve on the via transition (13.5k
        # unknowns) against 8.7 s with the sparse exact near field (its CSR build alone 3.1 s) and 14.9 s double
        from .sim import available_memory
        # quads: the projection holds Z and the reduced matrix (16 (N^2 + Nq^2) bytes), then the GPU's copy of it
        gpu = self._backend() if precision == 'mixed' else None
        host = max(16.0 * (N * N + Nq * Nq), 24.0 * Nq * Nq) if quads else 24.0 * N * N
        hybrid = gpu is not None and gpu.fits(Nq) and host + 0.9e9 <= 0.8 * available_memory()
        if hybrid and f <= HYBRID_RETRY * self._hybrid_failed:
            # the complex64 LU failed to precondition at or above this frequency: double up to HYBRID_RETRY x
            # the highest failure, the GPU again above it (it fails at the low end of a band, where the MPIE
            # matrix is worst conditioned; a sweep that gave up for good after its first 1 GHz solve ran the via
            # transition, 38 solves, on the CPU: 621 s, against 614 s without a GPU)
            precision, hybrid = 'double', False
        if precision == 'mixed' and not hybrid and f <= MIXED_RETRY * self._mixed_failed:
            # the same for the complex64 LU (CPU or GPU): a failure costs the mixed solve and a double one
            precision = 'double'
        tiled = self.tiled or sim.tiled
        if precision == 'mixed' and tiled != 'off' and not self._tiled_failed and (tiled == 'on' or not hybrid) \
                and gpu is not None and hasattr(gpu, 'cp') and not quads \
                and 0.9 * gpu.mem_free() >= self._tiled_work(N, self.tiled_block):
            # no dense Z: the GPU holds half a complex64 matrix (or streams it), the host a block column
            X = self._solve_tiled(f, E, Sx, gpu, self.tiled_block)
            if X is not None:
                return X
        dt = np.complex64 if precision == 'mixed' and not hybrid else np.complex128
        with sim.log.timer('direct fill', f"N={N}, {np.dtype(dt).name}, {self.core.num_threads()} threads"):
            Z = np.zeros((N, N), dt)
            (self.core.fill_far_c64 if dt == np.complex64 else self.core.fill_far)(Z, self.g, self.tb, self.w)
        with sim.log.timer('near-field correction'):
            if dt == np.complex64:
                # keep the exact near correction (sparse, double) for the refinement products
                self.dZ_exact = self._near_exact()
                Dc = self.dZ_exact.tocoo()
                Z[Dc.row, Dc.col] += Dc.data.astype(dt)            # one entry per (row, col): no add.at needed
                del Dc
            else:
                self.dZ_exact = None
                self._qcorr = self.core.near_into_dense(Z, self.g, self.tb, self.q, self.nP32, self.nQ32, self.w)
        self._vias(f)
        if Nv:
            Z[:Ne, Ne:], Z[Ne:, :Ne], Z[Ne:, Ne:] = self.B, self.B.T, self.C
        if Sx.nnz:
            Sc = Sx.tocoo()
            np.add.at(Z, (Sc.row, Sc.col), Sc.data.astype(dt))
        up, down = ((lambda X: X), (lambda X: X)) if T is None else ((lambda X: T @ X), (lambda X: np.asarray(T.T @ X)))
        if quads:
            # the quad-cell / line-function subspace (quads.py): Z_q = T^T Z T, E_q = T^T E, I = T I_q; mixed
            # precision refines on T^T A T (A the exact operator)
            from .quads import project
            with sim.log.timer('quad projection', f"{T.shape[0]} -> {T.shape[1]}"):
                if not hasattr(self, '_qC'):
                    self._qC = T[sim.quad_info['elim']].tocsr()
                Z = project(Z, sim.quad_info['keep'], sim.quad_info['elim'], self._qC, core=self.core)
        dsc = 1.0 / np.sqrt(np.abs(np.diagonal(Z)).astype(float) + 1e-300)
        if hybrid:
            return up(self._solve_hybrid(f, Z, dsc, down(E), gpu))
        gpu = self._mixed_gpu(Nq) if precision == 'mixed' else None
        _equilibrate(Z, dsc)
        if gpu is not None:
            with sim.log.timer('LU factor (GPU)', f"N={N}, {gpu.name}"):
                try:
                    gpu.factor(Z)
                except Exception as e:                      # driver / unsupported dtype: fall back
                    sim.log.note(f"GPU LU failed ({type(e).__name__}: {e}) - factoring on the CPU")
                    gpu.release()
                    gpu = None
        if gpu is None:
            with sim.log.timer('LU factor', f"N={len(dsc)}" + (' (quads)' if quads else '')):
                lu = _lu(Z)
        del Z
        B = dsc[:, None] * down(E)
        if precision != 'mixed':
            with sim.log.timer('LU solve'):
                return up(dsc[:, None] * _lu_solve(lu, B))
        # mixed: GMRES on the exact (double, matrix-free) equilibrated operator, LU(complex64) preconditioner
        with sim.log.timer('refinement (GMRES, exact matvec)'):
            Y, ok, rel = self._refine(lambda V: dsc[:, None] * down(self.matvec_exact(up(dsc[:, None] * V), Sx)), B,
                                      gpu.solve if gpu is not None else (lambda R: _lu_solve(lu, R)), MIXED_MAX_ITER, 1)
        if gpu is not None:
            gpu.release()
        sim.log.note(f"mixed precision: GMRES iterations per port {self.iters}")
        if ok:
            return up(dsc[:, None] * Y)
        # the single-precision LU can be too weak a preconditioner (fine meshes at low frequency: the MPIE matrix is
        # ill-conditioned); an unconverged refinement returns garbage: this solve, and those up to MIXED_RETRY x f,
        # in double
        sim.log.note(f"WARNING mixed precision did not converge at {f / 1e9:.4g} GHz (residual {rel:.1e}) "
                     f"- solving in double precision up to {MIXED_RETRY * f / 1e9:.4g} GHz")
        self._mixed_failed = max(self._mixed_failed, f)
        return self.solve_direct(f, E, block, precision='double')

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
            self.dZ_exact = self._near_exact()
        self._vias(f)
        M = (sparse.bmat([[self.dZ_exact, None], [None, sparse.csr_matrix((Nv, Nv))]]) if Nv else self.dZ_exact) + Sx
        Ms = ((M + M.T) * 0.5).tocsc()
        # the operator of the refinement: the same symmetric part
        self.dZ_exact = (0.5 * (self.dZ_exact + self.dZ_exact.T)).tocsr()
        Sx = (0.5 * (Sx + Sx.T)).tocsr()
        if Nv:
            self.C = Cs = 0.5 * (self.C + self.C.T)
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
        with sim.log.timer('refinement (GMRES, exact matvec)'):
            Y, ok, rel = self._refine(lambda V: dsc[:, None] * self.matvec_exact(dsc[:, None] * V, Sx), dsc[:, None] * E,
                                      fac.solve, 30, 10)
        fac.release()
        sim.log.note(f"tiled mixed precision: GMRES iterations per port {self.iters}")
        if not ok:
            sim.log.note(f"WARNING tiled mixed precision did not converge at {f / 1e9:.4g} GHz (residual "
                         f"{rel:.1e}): the pivoted LU from here on")
            self._tiled_failed = True
            return None
        return dsc[:, None] * Y

    @staticmethod
    def _tiled_work(N, block=1024):
        """GPU bytes the tiled factor needs besides its panels: the column being updated, a streamed panel, the
        right-hand sides of a solve."""
        return 3 * 8.0 * N * block + 0.5e9

    def _backend(self):
        """The GPU backend sim.gpu asks for, when it is available (None for 'off')."""
        if self.sim.gpu == 'off':
            return None
        from .gpu import get_backend
        return get_backend(self.sim.gpu)

    def _mixed_gpu(self, N):
        """The GPU for the complex64 LU of N unknowns (logged once), or None: none asked for or available, or too
        small for it."""
        sim, gpu = self.sim, self._backend()
        if sim.gpu != 'off' and not self._gpu_logged:
            self._gpu_logged = True
            if gpu is not None:
                sim.log.note(f"GPU: {gpu.device} via {gpu.name}, {gpu.mem_free() / 1e9:.1f} GB free, "
                             f"LU needs ~{gpu.vram_needed(N) / 1e9:.1f} GB")
            elif sim.gpu not in ('auto', None):
                from .gpu import get_backend
                sim.log.note("GPU requested but not available: " + "; ".join(get_backend.why))
        if gpu is not None and not gpu.fits(N):
            sim.log.note(f"GPU: LU needs ~{gpu.vram_needed(N) / 1e9:.1f} GB, only {gpu.mem_free() / 1e9:.1f} GB "
                         f"free - factoring on the CPU")
            return None
        return gpu

    def _solve_hybrid(self, f, Z, dsc, E, gpu):
        """Mixed precision with the LU on the GPU: Z (double, host) equilibrated, its complex64 copy factored on
        the GPU, GMRES on the double Z with that LU as the preconditioner; double precision on the CPU when the
        refinement does not converge."""
        sim, Z64 = self.sim, None
        if self.core is not None and hasattr(self.core, 'scale_c64') and Z.dtype == complex and Z.flags.c_contiguous:
            Z64 = self.core.scale_c64(Z, np.ascontiguousarray(dsc, float))         # Z scaled, and its complex64 copy
        else:
            _equilibrate(Z, dsc)
        if not self._gpu_logged:
            self._gpu_logged = True
            sim.log.note(f"GPU: {gpu.device} via {gpu.name}, {gpu.mem_free() / 1e9:.1f} GB free: complex64 LU there, "
                         f"double-precision refinement on the host")
        B = dsc[:, None] * E
        with sim.log.timer('LU factor (GPU)', f"N={Z.shape[0]}, {gpu.name}"):
            try:
                gpu.factor(Z.astype(np.complex64) if Z64 is None else Z64)
                del Z64
            except Exception as e:                          # driver / unsupported dtype: double on the CPU
                sim.log.note(f"GPU LU failed ({type(e).__name__}: {e}) - factoring on the CPU in double")
                gpu.release()
                return dsc[:, None] * _lu_solve(_lu(Z), B)
        with sim.log.timer('refinement (GMRES, dense double)'):
            # 2 restarts at most: where the complex64 LU is too weak (many via chains: the SMA launch, 536 via
            # unknowns) 300 iterations had not converged and cost 55 s, three times a double LU
            Y, ok, rel = self._refine(lambda V: Z @ V, B, gpu.solve, 30, 2)
        gpu.release()
        sim.log.note(f"mixed precision (GPU): GMRES iterations per port {self.iters}")
        if not ok:
            sim.log.note(f"mixed precision (GPU) did not converge at {f / 1e9:.4g} GHz (residual {rel:.1e}): "
                         f"double precision up to {HYBRID_RETRY * f / 1e9:.4g} GHz")
            self._hybrid_failed = max(self._hybrid_failed, f)
            with sim.log.timer('LU factor', f"N={Z.shape[0]}"):
                Y = _lu_solve(_lu(Z), B)
        return dsc[:, None] * Y

    def _solve_direct_numpy(self, f, E, block=1024):
        """Same entries as the fast operator (centroid far field + exact near correction), written
        into a dense Z block by block and LU-solved: memory ~ N^2 complex, robust at resonances."""
        sim = self.sim
        self.setup(f, mode='sparse')
        Sx = self.extra_blocks(f)
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        Z = np.zeros((N, N), complex)
        with sim.log.timer('direct fill', f"N={N}"):
            idx = np.arange(Ne)
            for r0 in range(0, Ne, block):
                for c0 in range(0, Ne, block):
                    Z[r0:r0 + block, c0:c0 + block] = self.entry(idx[r0:r0 + block], idx[c0:c0 + block])
            Dz = self.dZ + Sx[:Ne, :Ne]
            if Ne < 4000:
                Z[:Ne, :Ne] += Dz.toarray()
            else:
                Dz = Dz.tocoo()
                np.add.at(Z, (Dz.row, Dz.col), Dz.data)
            if Nv:
                Z[:Ne, Ne:], Z[Ne:, :Ne], Z[Ne:, Ne:] = self.B, self.B.T, self.C
        with sim.log.timer('LU solve', f"N={N}"):
            dsc = _equilibrate(Z)
            X = dsc[:, None] * _lu_solve(_lu(Z), dsc[:, None] * E)
        self.iters = []
        return X

    def _points(self):
        """Every unknown's place: the edges' midpoints, the via chains' axes."""
        sim = self.sim
        return np.vstack([sim.rwg.mid] + ([np.array([[v.x, v.y] for v, _, _ in sim.chains])] if sim.Nv else []))

    def _raw_entry(self, dZx):
        """entry(rows, cols): the dense block of Z (far field from the core, the sparse near correction and extra
        blocks dZx, the via rows and columns), any unknowns, original numbering."""
        Ne, Nv = self.sim.Ne, self.sim.Nv

        def raw(rows, cols):
            M = np.zeros((len(rows), len(cols)), complex)
            re, ce = rows < Ne, cols < Ne
            if re.any() and ce.any():
                r_, c_ = rows[re], cols[ce]
                M[np.ix_(re, ce)] = self.core.far_block(r_.astype(np.int32), c_.astype(np.int32), self.g, self.tb, self.w) \
                    + dZx[r_][:, c_].toarray()
            if Nv:                                                  # (empty selections assign nothing)
                M[np.ix_(re, ~ce)] = self.B[rows[re]][:, cols[~ce] - Ne]
                M[np.ix_(~re, ce)] = self.B[cols[ce]][:, rows[~re] - Ne].T
                M[np.ix_(~re, ~ce)] = self.C[np.ix_(rows[~re] - Ne, cols[~ce] - Ne)]
            return M
        return raw

    def solve_hodlr(self, f, E, tol=1e-6, leaf=256, refine=False):
        """The hierarchical direct solver (hodlr.HODLR): Z compressed from its entries, factored recursively; no
        dense Z. refine: GMRES on the exact matrix-free operator, the HODLR solve its preconditioner."""
        from .hodlr import HODLR
        assert self.core is not None, "solver='hodlr' needs the compiled core (python -m mom25d.core.build)"
        sim = self.sim
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        self.setup(f, mode='sparse')
        self.dZ_exact = self.dZ
        Sx = self.extra_blocks(f)
        dZx = (self.dZ + Sx[:Ne, :Ne]).tocsr()
        raw = self._raw_entry(dZx)
        pts = self._points()
        # equilibration from the diagonal (via rows are ~1e6 x the strip rows)
        d = np.empty(N, complex)
        for s0 in range(0, N, 512):
            idx = np.arange(s0, min(N, s0 + 512))
            d[idx] = np.diagonal(raw(idx, idx))
        dsc = 1.0 / np.sqrt(np.abs(d) + 1e-300)
        # the near-field entries that matter at tol mark ACA's hot rows (hodlr.aca_checked): the correction's pattern
        # reaches 5 cells, 15 mm under 3 mm plane cells, but beyond ~3 mm it is under 5e-4 of its largest
        sig = sparse.diags(dsc[:Ne]) @ dZx @ sparse.diags(dsc[:Ne])
        sig.data[np.abs(sig.data) < 0.1 * tol] = 0
        sig.eliminate_zeros()
        with sim.log.timer('HODLR compress + factor', f"N={N}, tol {tol:g}"):
            self.hodlr = H = HODLR(lambda r, c: raw(r, c) * dsc[r][:, None] * dsc[c][None, :], pts, leaf=leaf, tol=tol,
                                   near=sig, log=sim.log.note)
        sim.log.note(f"HODLR: {H.stats()}")
        B = dsc[:, None] * E
        if not refine:
            with sim.log.timer('HODLR solve'):
                return dsc[:, None] * H.solve(B)
        with sim.log.timer('refinement (GMRES, exact matvec)'):
            Y, ok, rel = self._refine(lambda V: dsc[:, None] * self.matvec_exact(dsc[:, None] * V, Sx), B, H.solve,
                                      MIXED_MAX_ITER, 1)
        sim.log.note(f"HODLR: GMRES iterations per port {self.iters}" + ('' if ok else f", NOT converged ({rel:.1e})"))
        if not ok and tol > 1e-9:
            # too loose a factorisation for this matrix (fine meshes at low frequency are ill-conditioned): again at
            # 100 x tighter, rather than an unconverged answer
            sim.log.note(f"WARNING HODLR refinement did not converge at {f / 1e9:.4g} GHz with tol {tol:g} - "
                         f"factoring again at {tol / 100:g}")
            return self.solve_hodlr(f, E, tol=tol / 100, leaf=leaf, refine=True)
        return dsc[:, None] * Y

    def solve_blr(self, f, E, tol=1e-4, tile=256, eta=1.0, dtype=np.complex64):
        """Block low-rank LU preconditioner + GMRES with the exact matrix-free operator (no stored Z)."""
        from .blr import BLRFactor, cluster_tiles
        assert self.core is not None, "solver='blr' needs the compiled core (python -m mom25d.core.build)"
        sim = self.sim
        Ne, Nv = sim.Ne, sim.Nv
        N = Ne + Nv
        self.setup(f, mode='sparse')
        self.dZ_exact = self.dZ
        Sx = self.extra_blocks(f)
        dZx = (self.dZ + Sx[:Ne, :Ne]).tocsr()
        tiles, boxes = cluster_tiles(self._points(), tile)
        raw = self._raw_entry(dZx)
        with sim.log.timer('BLR assemble + factor', f"N={N}, {len(tiles)} tiles"):
            diag = np.empty(N, complex)
            for t in tiles:
                diag[t] = np.diagonal(raw(t, t))
            dsc = 1.0 / np.sqrt(np.abs(diag) + 1e-300)
            self.blr = blr = BLRFactor(lambda r, c: raw(r, c) * dsc[r][:, None] * dsc[c][None, :], tiles, boxes,
                                       tol=tol, eta=eta, dtype=dtype, scale=1.0)
        sim.log.note(f"BLR: {blr.memory / 2 ** 20:.0f} MB ({100 * blr.memory / (8 * N * N):.0f}% of complex64 dense, "
                     f"{100 * blr.memory / (16 * N * N):.0f}% of complex128), {blr.stats['lowrank']} low-rank tiles, "
                     f"mean rank {blr.stats['rank_sum'] / max(1, blr.stats['lowrank']):.0f}")
        with sim.log.timer('GMRES (BLR preconditioned)'):
            Y, _ = self._gmres_columns(LinearOperator((N, N), matvec=lambda y: dsc * self.matvec_exact(dsc * y, Sx),
                                                      dtype=complex),
                                       LinearOperator((N, N), matvec=blr.solve, dtype=complex),
                                       dsc[:, None] * E, self.mixed_tol, 50, 10)
        sim.log.note(f"BLR: GMRES iterations per port {self.iters}")
        return dsc[:, None] * Y


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
    iters, act = np.zeros(k, int), np.arange(k)
    for _ in range(maxiter):
        R = B[:, act] - (A(X[:, act]) if np.any(X[:, act]) else 0)
        beta = np.linalg.norm(R, axis=0)
        done = beta <= rtol * bn[act]
        act, R, beta = act[~done], R[:, ~done], beta[~done]
        if not len(act):
            break
        ka = len(act)
        V, Z = np.zeros((restart + 1, N, ka), complex), np.zeros((restart, N, ka), complex)
        H, g = np.zeros((restart + 1, restart, ka), complex), np.zeros((restart + 1, ka), complex)
        cs, sn = np.zeros((restart, ka), complex), np.zeros((restart, ka), complex)
        g[0], V[0] = beta, R / beta
        live, last = np.ones(ka, bool), np.full(ka, restart - 1)
        for j in range(restart):
            Z[j] = M(V[j])
            Wv = A(Z[j])
            for i in range(j + 1):                                   # modified Gram-Schmidt
                H[i, j] = hij = np.einsum('nk,nk->k', V[i].conj(), Wv)
                Wv = Wv - V[i] * hij
            H[j + 1, j] = hn = np.linalg.norm(Wv, axis=0)
            V[j + 1] = Wv / np.where(hn > 0, hn, 1)
            for i in range(j):                                       # apply previous Givens rotations
                H[i, j], H[i + 1, j] = (cs[i] * H[i, j] + sn[i] * H[i + 1, j],
                                        -np.conj(sn[i]) * H[i, j] + np.conj(cs[i]) * H[i + 1, j])
            den = np.sqrt(np.abs(H[j, j]) ** 2 + np.abs(H[j + 1, j]) ** 2)
            den[den == 0] = 1.0
            cs[j], sn[j] = H[j, j] / den, H[j + 1, j] / den
            H[j, j], H[j + 1, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j], 0
            g[j + 1], g[j] = -np.conj(sn[j]) * g[j], cs[j] * g[j]
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
