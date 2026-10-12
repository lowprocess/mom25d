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


def _perm_from_piv(piv):
    p = np.arange(len(piv))
    for i, j in enumerate(piv):
        p[i], p[j] = p[j], p[i]
    return p


def _compress(M, atol, max_rank):
    """Truncated SVD: M ~ U V with singular values above atol. Returns ('L', U, V) or ('D', M)."""
    m, n = M.shape
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > atol))
    if r * (m + n) >= 0.6 * m * n or r > max_rank:
        return ('D', M)
    return ('L', U[:, :r] * s[:r], Vh[:r])


def _recompress(U, V, atol, max_rank, dtype):
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
    """
    def __init__(self, entry, tiles, boxes, tol=1e-4, eta=1.0, dtype=np.complex64, max_rank=None,
                 scale=1.0, log=None):
        self.tiles, self.dtype = tiles, dtype
        nb = len(tiles)
        self.nb = nb
        atol = tol * scale
        T = [[None] * nb for _ in range(nb)]
        self.stats = dict(dense=0, lowrank=0, rank_sum=0)
        for i in range(nb):
            for j in range(nb):
                M = entry(tiles[i], tiles[j])
                lo_i, hi_i = boxes[i]; lo_j, hi_j = boxes[j]
                gap = np.linalg.norm(np.maximum(0, np.maximum(lo_i - hi_j, lo_j - hi_i)))
                diam = max(np.linalg.norm(hi_i - lo_i), np.linalg.norm(hi_j - lo_j))
                mr = max_rank or min(M.shape) // 2
                if i != j and gap > eta * diam:
                    T[i][j] = _compress(M, atol, mr)
                    if T[i][j][0] == 'L':
                        T[i][j] = ('L', T[i][j][1].astype(dtype), T[i][j][2].astype(dtype))
                    else:
                        T[i][j] = ('D', M.astype(dtype))
                else:
                    T[i][j] = ('D', M.astype(dtype))
        # ---- right-looking tile LU with pivoting inside the diagonal tiles --------------------
        self.lu = [None] * nb
        for k in range(nb):
            lu, piv = sla.lu_factor(T[k][k][1], check_finite=False)
            perm = _perm_from_piv(piv)
            self.lu[k] = (lu, piv, perm)
            T[k][k] = None
            L = np.tril(lu, -1) + np.eye(len(lu), dtype=lu.dtype)
            Uk = np.triu(lu)
            for j in range(nb):                              # apply the row permutation to the whole tile row
                if j == k or T[k][j] is None:
                    continue
                t = T[k][j]
                T[k][j] = ('D', t[1][perm]) if t[0] == 'D' else ('L', t[1][perm], t[2])
            for j in range(k + 1, nb):                       # row panel: L^-1 A_kj
                t = T[k][j]
                if t[0] == 'D':
                    T[k][j] = ('D', sla.solve_triangular(L, t[1], lower=True, unit_diagonal=True, check_finite=False))
                else:
                    T[k][j] = ('L', sla.solve_triangular(L, t[1], lower=True, unit_diagonal=True, check_finite=False), t[2])
            for i in range(k + 1, nb):                       # column panel: A_ik U^-1
                t = T[i][k]
                if t[0] == 'D':
                    T[i][k] = ('D', sla.solve_triangular(Uk, t[1].T, trans='T', lower=False, check_finite=False).T)
                else:
                    T[i][k] = ('L', t[1], sla.solve_triangular(Uk, t[2].T, trans='T', lower=False, check_finite=False).T)
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
        for i in range(nb):
            for j in range(nb):
                if T[i][j] is not None:
                    if T[i][j][0] == 'D':
                        self.stats['dense'] += 1
                    else:
                        self.stats['lowrank'] += 1; self.stats['rank_sum'] += T[i][j][1].shape[1]

    @staticmethod
    def _memory(T):
        tot = 0
        for row in T:
            for t in row:
                if t is None:
                    continue
                tot += t[1].nbytes + (t[2].nbytes if t[0] == 'L' else 0)
        return tot

    @staticmethod
    def _mul(t, x):
        return t[1] @ x if t[0] == 'D' else t[1] @ (t[2] @ x)

    def solve(self, b):
        """Apply the approximate inverse (vector or matrix right-hand side, original numbering)."""
        nb, T = self.nb, self.T
        x = np.asarray(b)
        y = [None] * nb
        for k in range(nb):
            lu, piv, perm = self.lu[k]
            r = x[self.tiles[k]].astype(lu.dtype)[perm]
            for j in range(k):
                r = r - self._mul(T[k][j], y[j])
            y[k] = sla.solve_triangular(lu, r, lower=True, unit_diagonal=True, check_finite=False)
        z = [None] * nb
        for k in range(nb - 1, -1, -1):
            lu = self.lu[k][0]
            r = y[k]
            for j in range(k + 1, nb):
                r = r - self._mul(T[k][j], z[j])
            z[k] = sla.solve_triangular(lu, r, lower=False, check_finite=False)
        out = np.zeros(x.shape, complex)
        for k in range(nb):
            out[self.tiles[k]] = z[k]
        return out


def cluster_tiles(pts, tile):
    """Geometric bisection: list of index arrays (leaves of <= tile points) and their bounding boxes."""
    out = []

    def rec(idx):
        p = pts[idx]
        if len(idx) <= tile:
            out.append((idx, (p.min(0), p.max(0))))
            return
        ax = np.argmax(p.max(0) - p.min(0))
        o = np.argsort(p[:, ax], kind='stable')
        h = len(idx) // 2
        rec(idx[o[:h]]); rec(idx[o[h:]])
    rec(np.arange(len(pts)))
    return [t for t, _ in out], [b for _, b in out]
