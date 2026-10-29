"""
mom25d.hodlr - a hierarchical direct solver (HODLR) for the MoM matrix, from its entries alone.

The unknowns are ordered by a binary geometric cluster tree: each node a contiguous range of the order, split in
two along its longest side. A node's two off-diagonal blocks, Z[a, b] and Z[b, a] between its children, are kept
as low-rank products U V (adaptive cross approximation, fast.aca: only O(r (|a| + |b|)) entries evaluated); the
leaves' diagonal blocks are dense. The factorisation is recursive, Sherman-Morrison-Woodbury at each node: with
D = diag(Z[a, a], Z[b, b]) and the coupling W Q, W = diag(U_ab, U_ba), Q = [[0, V_ab], [V_ba, 0]],

    Z^-1 = D^-1 - D^-1 W (I + Q D^-1 W)^-1 Q D^-1,

D^-1 from the children's factorisations. For ranks r bounded independently of N: O(N r log N) entries,
O(N r^2 log^2 N) work and O(N r log N) memory, against N^2 entries and N^3 / 3 for the dense LU.

    H = HODLR(entry, pts, leaf=256, tol=1e-6)      # entry(rows, cols): the dense block, original numbering
    X = H.solve(B)                                   # B: N or N x k, original numbering
"""
import time

import numpy as np
import scipy.linalg as sla



def aca_checked(rows, cols, m, n, tol, max_rank, rng, hot=(), check=16):
    """Adaptive cross approximation, partial pivoting, M ~ U V from rows(i) and cols(j) of the block, with a check
    before it stops. ACA's own criterion (the last cross below tol x the approximation's norm) can end early,
    missing the part of the block its pivots never reached: here the rows next to the other cluster, whose entries
    are near-singular (a filter's top block 4 % off at 1e-3). So the rows `hot` (those with near-field entries in
    the block) are fetched once and checked every time, with `check` random others; the worst row off by more than
    tol x its own norm is the next pivot."""
    U, V, used, norm2 = [], [], np.zeros(m, bool), 0.0
    hot = np.asarray(hot, int)
    Rhot = rows(hot) if len(hot) else None
    nhot = np.linalg.norm(Rhot, axis=1) if len(hot) else None
    i = int(hot[np.argmax(nhot)]) if len(hot) else 0
    while len(U) < min(m, n, max_rank):
        used[i] = True
        k = np.flatnonzero(hot == i)
        row = Rhot[k[0]].copy() if len(k) else rows(np.array([i]))[0]
        if U:
            row = row - np.array(U)[:, i] @ np.array(V)
        j = int(np.argmax(np.abs(row)))
        small = abs(row[j]) <= 1e-300
        if not small:
            v = row / row[j]
            u = cols(np.array([j]))[:, 0]
            if U:
                u = u - np.array(U).T @ np.array(V)[:, j]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if U:
                Ua, Va = np.array(U), np.array(V)
                norm2 += 2 * np.real(np.sum((Ua.conj() @ u) * (Va.conj() @ v).conj())) + (nu * nv) ** 2
            else:
                norm2 = (nu * nv) ** 2
            U.append(u); V.append(v)
            small = nu * nv <= tol * np.sqrt(abs(norm2))
            a = np.abs(u)
            a[used] = -1
            i = int(np.argmax(a))
            if a[i] < 0:
                small = True
        if small:
            Ua, Va = np.array(U), np.array(V)
            free = np.flatnonzero(~used)
            if not len(free):
                break
            pick = rng.choice(free, min(check, len(free)), replace=False)
            R, ref = rows(pick), None
            ref = np.linalg.norm(R, axis=1)
            R = R - Ua[:, pick].T @ Va
            idx, err, scale = pick, np.linalg.norm(R, axis=1), ref
            if len(hot):
                Rh = Rhot - Ua[:, hot].T @ Va
                idx, err = np.concatenate([idx, hot]), np.concatenate([err, np.linalg.norm(Rh, axis=1)])
                scale = np.concatenate([scale, nhot])
            # off by more than tol of the row itself, or of the block's typical row
            bad = err > tol * np.maximum(scale, np.sqrt(abs(norm2) / m))
            if not bad.any():
                break
            w = np.where(bad, err, -1.0)
            i = int(idx[np.argmax(w)])
            used[i] = False
    if not U:
        return np.zeros((m, 0), complex), np.zeros((0, n), complex)
    return _recompress(np.array(U).T, np.array(V), tol)


def _recompress(U, V, tol):
    """U V with the fewest columns for tol (QR of both factors, SVD of the small core)."""
    if U.shape[1] == 0:
        return U, V
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V.T)
    W, sv, Zh = np.linalg.svd(Ru @ Rv.T)
    r = max(1, int(np.sum(sv > tol * sv[0])))
    return Qu @ (W[:, :r] * sv[:r]), Zh[:r] @ Qv.T


def panel_compress(block, m, n, tol, rng):
    """A block too large to hold (m x n > PANEL) from block(row range) a row panel at a time, each panel compressed
    (svd_compress), then the stack recompressed: M ~ diag(Q_i) [R_i V_i] = diag(Q_i) W, W's SVD truncated. One pass
    over the entries, memory a panel and the factors."""
    rows = max(1, PANEL // n)
    Qs, Ws = [], []
    for s0 in range(0, m, rows):
        U, V = svd_compress(block(s0, min(m, s0 + rows)), tol / 4, rng)
        if U.shape[1]:
            Q, R = np.linalg.qr(U)
            Qs.append((s0, Q)); Ws.append(R @ V)
    if not Ws:
        return np.zeros((m, 0), complex), np.zeros((0, n), complex)
    W = np.vstack(Ws)
    X, sv, Vh = np.linalg.svd(W, full_matrices=False)
    r = int(np.sum(sv > tol * sv[0]))
    U, k = np.zeros((m, r), complex), 0
    for (s0, Q), Wi in zip(Qs, Ws):
        U[s0:s0 + len(Q)] = Q @ (X[k:k + len(Wi), :r] * sv[:r])
        k += len(Wi)
    return U, Vh[:r]


def svd_compress(M, tol, rng=None, step=32, probes=8):
    """The exact block truncated at tol x its largest singular value: (U, V). Small blocks by a full SVD; large ones
    by an adaptive randomized range finder (Halko, Martinsson & Tropp 2011): Q doubled from `step` columns until
    `probes` fresh random products are left with less than tol x ||M|| outside it, then the SVD of Q^H M -
    O(m n r) instead of the full SVD's O(m n min(m, n)) (a 6.7k block: ~0.5 s against ~30 s)."""
    m, n = M.shape

    def full():
        U, sv, Vh = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(sv > tol * sv[0])) if len(sv) and sv[0] > 0 else 0
        return U[:, :r] * sv[:r], Vh[:r]
    if min(m, n) <= 4 * step:
        return full()
    rng = rng or np.random.default_rng(0)
    gauss = lambda k: rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
    Y = M @ gauss(step)
    norm = np.linalg.norm(Y, 2) / np.sqrt(2 * step)         # ~ ||M||_2 (a Gaussian probe's gain)
    Q = np.linalg.qr(Y)[0]
    while True:
        if 2 * Q.shape[1] + probes > min(m, n):
            return full()        # near full rank: the basis would outgrow the space (no longer orthogonal)
        T = M @ gauss(probes)
        T -= Q @ (Q.conj().T @ T)
        if np.linalg.norm(T, axis=0).max() <= tol * norm * np.sqrt(2 * probes) / 10:
            break
        Y = M @ gauss(Q.shape[1])                             # double the basis each time: O(log r) passes
        Y = np.hstack([Y, T])
        Y -= Q @ (Q.conj().T @ Y)
        Y -= Q @ (Q.conj().T @ Y)                             # twice: orthogonal to Q in floating point
        Q = np.hstack([Q, np.linalg.qr(Y)[0]])
    W, sv, Vh = np.linalg.svd(Q.conj().T @ M, full_matrices=False)
    r = int(np.sum(sv > tol * sv[0])) if sv[0] > 0 else 0
    return (Q @ W[:, :r]) * sv[:r], Vh[:r]

DENSE_BLOCK = 512 * 512      # off-diagonal blocks up to this many entries: evaluated whole, truncated SVD
HOT_MAX = 0.1                # ... and those with more than this fraction of rows holding near-field entries
PANEL = 2 ** 25              # entries: a larger block evaluated a row panel at a time (512 MB complex128)


class _Node:
    __slots__ = ('s', 'e', 'kids', 'lu', 'U_ab', 'V_ab', 'U_ba', 'V_ba', 'Ya', 'Yb', 'S')

    def __init__(self, s, e):
        self.s, self.e, self.kids, self.lu = s, e, [], None


def _tree(pts, leaf):
    """(order, root): the unknowns in tree order and the root, each node covering order[s:e], split at the median
    of its longest side until it holds at most `leaf`."""
    order = np.arange(len(pts))

    def build(s, e):
        node = _Node(s, e)
        if e - s > leaf:
            p = pts[order[s:e]]
            ax = int(np.argmax(p.max(0) - p.min(0)))
            order[s:e] = order[s:e][np.argsort(p[:, ax], kind='stable')]
            h = s + (e - s) // 2
            node.kids = [build(s, h), build(h, e)]
        return node
    root = build(0, len(pts))
    return order, root


class HODLR:
    def __init__(self, entry, pts, leaf=256, tol=1e-6, max_rank=None, atol=0.0, log=None, compress='aca', near=None,
                 symmetric=False):
        t0 = time.time()
        self.n, self.tol = len(pts), tol
        self.order, self.root = _tree(np.asarray(pts, float), leaf)
        self.entry, self.atol, self.log = entry, atol, log
        self.max_rank, self.compress, self.rng = max_rank, compress, np.random.default_rng(0)
        self.symmetric = symmetric           # Z^T = Z (reciprocal): Z[b, a] from Z[a, b], half the compression
        # near: a sparse matrix (original numbering) whose pattern marks the near-field entries; None: no hot rows
        self.near = None if near is None else near.tocsr()
        self.ranks, self.entries, self.paths = {}, 0, {}     # level -> [ranks]; entries evaluated; how each was compressed
        self._factor(self.root, 0)
        self.seconds = time.time() - t0
        del self.entry

    # ---- build ------------------------------------------------------------------------------------
    def _block(self, rows, cols):
        self.entries += len(rows) * len(cols)
        return self.entry(self.order[rows], self.order[cols])

    def _lowrank(self, a, b):
        ra, rb = np.arange(a.s, a.e), np.arange(b.s, b.e)
        m, n = len(ra), len(rb)
        path = lambda k, frac=None: self.paths.setdefault(self._level, []).append((k, frac))
        whole = lambda: svd_compress(self._block(ra, rb), self.tol, self.rng) if m * n <= PANEL else \
            panel_compress(lambda s0, s1: self._block(ra[s0:s1], rb), m, n, self.tol, self.rng)
        if self.compress == 'svd' or m * n <= DENSE_BLOCK:
            path('small')
            return whole()
        hot = ()
        if self.near is not None:
            ga, gb = self.order[ra], self.order[rb]
            ok_a, ok_b = ga < self.near.shape[0], gb < self.near.shape[1]
            sub = self.near[ga[ok_a]][:, gb[ok_b]]
            hot = np.flatnonzero(ok_a)[np.flatnonzero(np.diff(sub.indptr))]
            if len(hot) > HOT_MAX * m:                       # mostly near-field rows: the block itself is cheaper
                path('hot', len(hot) / m)
                return whole()
            path('aca', len(hot) / m)
        return aca_checked(lambda r: self._block(ra[r], rb), lambda c: self._block(ra, rb[c]), m, n, self.tol,
                           self.max_rank or min(m, n), self.rng, hot=hot)

    def _factor(self, node, level):
        if not node.kids:
            node.lu = sla.lu_factor(self._block(np.arange(node.s, node.e), np.arange(node.s, node.e)),
                                    check_finite=False)
            return
        a, b = node.kids
        self._factor(a, level + 1)
        self._factor(b, level + 1)
        self._level = level
        node.U_ab, node.V_ab = self._lowrank(a, b)
        node.U_ba, node.V_ba = (node.V_ab.T.copy(), node.U_ab.T.copy()) if self.symmetric else self._lowrank(b, a)
        self.ranks.setdefault(level, []).extend([node.U_ab.shape[1], node.U_ba.shape[1]])
        node.Ya, node.Yb = self._solve(a, node.U_ab), self._solve(b, node.U_ba)
        r1, r2 = node.U_ab.shape[1], node.U_ba.shape[1]
        S = np.eye(r1 + r2, dtype=complex)
        S[:r1, r1:] = node.V_ab @ node.Yb
        S[r1:, :r1] = node.V_ba @ node.Ya
        node.S = sla.lu_factor(S, check_finite=False) if r1 + r2 else None
        if self.log and level <= 2:
            self.log(f"    HODLR level {level}: ranks {r1}, {r2} ({a.e - a.s} + {b.e - b.s} unknowns)")

    # ---- solve ------------------------------------------------------------------------------------
    def _solve(self, node, B):
        """Z[node, node]^-1 B, B in the node's local (tree) order."""
        if not node.kids:
            return sla.lu_solve(node.lu, B, check_finite=False)
        a, b = node.kids
        na = a.e - a.s
        Xa, Xb = self._solve(a, B[:na]), self._solve(b, B[na:])
        if node.S is None:
            return np.concatenate([Xa, Xb])
        r1 = node.U_ab.shape[1]
        z = sla.lu_solve(node.S, np.concatenate([node.V_ab @ Xb, node.V_ba @ Xa]), check_finite=False)
        return np.concatenate([Xa - node.Ya @ z[:r1], Xb - node.Yb @ z[r1:]])

    def solve(self, B):
        B = np.asarray(B)
        X = np.empty(B.shape, complex)
        X[self.order] = self._solve(self.root, B[self.order].astype(complex))
        return X

    def matvec(self, x):
        """Z x from the compressed form (the leaves' dense blocks are kept only as LU: reconstructed here)."""
        x = np.asarray(x, complex)
        y = np.empty(x.shape, complex)
        y[self.order] = self._mul(self.root, x[self.order])
        return y

    def _mul(self, node, x):
        if not node.kids:
            lu, piv = node.lu
            L, U = np.tril(lu, -1) + np.eye(len(lu)), np.triu(lu)
            P = np.arange(len(piv))
            for i, j in enumerate(piv):
                P[i], P[j] = P[j], P[i]
            out = np.empty(x.shape, complex)
            out[P] = L @ (U @ x)
            return out
        a, b = node.kids
        na = a.e - a.s
        xa, xb = x[:na], x[na:]
        return np.concatenate([self._mul(a, xa) + node.U_ab @ (node.V_ab @ xb),
                               self._mul(b, xb) + node.U_ba @ (node.V_ba @ xa)])

    # ---- statistics -------------------------------------------------------------------------------
    @property
    def memory(self):
        tot, stack = 0, [self.root]
        while stack:
            nd = stack.pop()
            if nd.kids:
                tot += sum(m.nbytes for m in (nd.U_ab, nd.V_ab, nd.U_ba, nd.V_ba, nd.Ya, nd.Yb))
                tot += nd.S[0].nbytes if nd.S is not None else 0
                stack += nd.kids
            else:
                tot += nd.lu[0].nbytes
        return tot

    def stats(self):
        return dict(n=self.n, seconds=round(self.seconds, 1), MB=round(self.memory / 2 ** 20),
                    dense_MB=round(16 * self.n ** 2 / 2 ** 20), entries=f"{self.entries / self.n ** 2:.2f} N^2",
                    ranks={lv: (int(np.max(r)), int(np.mean(r))) for lv, r in sorted(self.ranks.items())})
