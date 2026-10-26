"""
mom25d.blr - block low-rank (BLR) LU preconditioner for the MoM matrix

The impedance matrix is never stored densely. Unknowns are ordered by a geometric cluster tree
and the matrix is cut into tiles; tiles between well-separated clusters are kept as truncated
low-rank factors U V (SVD at a relative tolerance), close tiles dense. A right-looking tile LU
with recompression of the low-rank updates gives an approximate factorisation (default 1e-4),
used as the preconditioner of GMRES whose products are exact (compiled matrix-free operator).
Memory ~ dense tiles + sum of ranks, typically a fraction of N^2; storage in complex64 by default.
"""
import numpy as np
import scipy.linalg as sla


def _compress(M, atol, max_rank, dtype):
    """Truncated SVD: ('L', U, V) with M ~ U V (the singular values above atol), or ('D', M) where low rank does not
    pay; in dtype."""
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > atol))
    if r * sum(M.shape) >= 0.6 * M.size or r > max_rank:
        return ('D', M.astype(dtype))
    return ('L', (U[:, :r] * s[:r]).astype(dtype), Vh[:r].astype(dtype))


def _recompress(U, V, atol, max_rank, dtype):
    """_compress of the low-rank sum U V, through QR factors of U and V^T."""
    if U.shape[1] == 0:
        return ('L', U, V)
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V.T)
    W, s, Zh = np.linalg.svd(Ru @ Rv.T, full_matrices=False)
    r = int(np.sum(s > atol))
    m, n = U.shape[0], V.shape[1]
    if r * (m + n) >= 0.6 * m * n or r > max_rank:
        return ('D', (U @ V).astype(dtype))
    return ('L', (Qu @ (W[:, :r] * s[:r])).astype(dtype), (Zh[:r] @ Qv.T).astype(dtype))


class BLRFactor:
    """
    entry(rows, cols) -> dense block of the (equilibrated) matrix, rows/cols in ORIGINAL numbering.
    tiles: list of index arrays (original numbering), in elimination order.
    boxes: per tile (lo, hi) bounding boxes for the admissibility test.
    Tiles are ('D', M) dense or ('L', U, V) low rank (M ~ U V).
    """
    def __init__(self, entry, tiles, boxes, tol=1e-4, eta=1.0, dtype=np.complex64, max_rank=None,
                 scale=1.0, log=None):
        self.tiles, self.dtype, self.nb = tiles, dtype, len(tiles)
        nb, atol = self.nb, tol * scale
        T = [[None] * nb for _ in range(nb)]
        for i in range(nb):
            for j in range(nb):
                M = entry(tiles[i], tiles[j])
                (lo_i, hi_i), (lo_j, hi_j) = boxes[i], boxes[j]
                gap = np.linalg.norm(np.maximum(0, np.maximum(lo_i - hi_j, lo_j - hi_i)))
                far = i != j and gap > eta * max(np.linalg.norm(hi_i - lo_i), np.linalg.norm(hi_j - lo_j))
                T[i][j] = _compress(M, atol, max_rank or min(M.shape) // 2, dtype) if far else ('D', M.astype(dtype))
        # ---- right-looking tile LU with pivoting inside the diagonal tiles --------------------
        self.lu = [None] * nb
        trsm = lambda A, B, **kw: sla.solve_triangular(A, B, check_finite=False, **kw)
        for k in range(nb):
            lu, piv = sla.lu_factor(T[k][k][1], check_finite=False)
            perm = np.arange(len(piv))                       # the row permutation of LAPACK's pivots
            for i, j in enumerate(piv):
                perm[i], perm[j] = perm[j], perm[i]
            self.lu[k], T[k][k] = (lu, piv, perm), None
            L, Uk = np.tril(lu, -1) + np.eye(len(lu), dtype=lu.dtype), np.triu(lu)
            for j in range(nb):                              # the row permutation on the whole tile row, then
                t = T[k][j]                                  # the row panel L^-1 A_kj
                if j != k and t is not None:
                    t = (t[0], t[1][perm], *t[2:])
                    T[k][j] = (t[0], trsm(L, t[1], lower=True, unit_diagonal=True), *t[2:]) if j > k else t
            for i in range(k + 1, nb):                       # column panel: A_ik U^-1
                t = T[i][k]
                if t[0] == 'D':
                    T[i][k] = ('D', trsm(Uk, t[1].T, trans='T', lower=False).T)
                else:
                    T[i][k] = ('L', t[1], trsm(Uk, t[2].T, trans='T', lower=False).T)
            for i in range(k + 1, nb):                       # Schur complement updates
                a = T[i][k]
                for j in range(k + 1, nb):
                    b = T[k][j]
                    if a[0] == 'D' and b[0] == 'D':
                        upd = ('D', a[1] @ b[1])
                    elif a[0] == 'L' and b[0] == 'D':
                        upd = ('L', a[1], a[2] @ b[1])
                    elif a[0] == 'D' and b[0] == 'L':
                        upd = ('L', a[1] @ b[1], b[2])
                    else:
                        upd = ('L', a[1], (a[2] @ b[1]) @ b[2])
                    c = T[i][j]
                    if c[0] == 'D':
                        c[1][...] -= (upd[1] if upd[0] == 'D' else upd[1] @ upd[2])
                    elif upd[0] == 'D':
                        T[i][j] = ('D', (c[1] @ c[2] - upd[1]).astype(dtype))
                    else:
                        T[i][j] = _recompress(np.hstack([c[1], -upd[1]]), np.vstack([c[2], upd[2]]), atol,
                                              max_rank or min(c[1].shape[0], c[2].shape[1]) // 2, dtype)
            if log and (k % max(1, nb // 10) == 0):
                log(f"    BLR LU: tile {k + 1}/{nb}, memory {self._memory(T) / 2 ** 20:.0f} MB")
        self.T = T
        self.memory = self._memory(T) + sum(l[0].nbytes for l in self.lu)
        kept = [t for row in T for t in row if t is not None]
        self.stats = dict(dense=sum(t[0] == 'D' for t in kept), lowrank=sum(t[0] == 'L' for t in kept),
                          rank_sum=sum(t[1].shape[1] for t in kept if t[0] == 'L'))

    @staticmethod
    def _memory(T):
        return sum(sum(a.nbytes for a in t[1:]) for row in T for t in row if t is not None)

    @staticmethod
    def _mul(t, x):
        return t[1] @ x if t[0] == 'D' else t[1] @ (t[2] @ x)

    def solve(self, b):
        """Apply the approximate inverse (vector or matrix right-hand side, original numbering)."""
        nb, T, x = self.nb, self.T, np.asarray(b)
        y = [None] * nb
        for k in range(nb):
            lu, piv, perm = self.lu[k]
            r = x[self.tiles[k]].astype(lu.dtype)[perm]
            for j in range(k):
                r = r - self._mul(T[k][j], y[j])
            y[k] = sla.solve_triangular(lu, r, lower=True, unit_diagonal=True, check_finite=False)
        z = [None] * nb
        for k in range(nb - 1, -1, -1):
            r = y[k]
            for j in range(k + 1, nb):
                r = r - self._mul(T[k][j], z[j])
            z[k] = sla.solve_triangular(self.lu[k][0], r, lower=False, check_finite=False)
        out = np.zeros(x.shape, complex)
        for k in range(nb):
            out[self.tiles[k]] = z[k]
        return out


def cluster_tiles(pts, tile):
    """Geometric bisection (fast.Cluster): list of index arrays (leaves of <= tile points) and their bounding boxes."""
    from .fast import Cluster
    leaves = lambda c: [c] if not c.kids else leaves(c.kids[0]) + leaves(c.kids[1])
    tiles = leaves(Cluster(np.arange(len(pts)), pts, tile))
    return [c.idx for c in tiles], [(c.lo, c.hi) for c in tiles]
