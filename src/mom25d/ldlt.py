"""
mom25d.ldlt - a tiled complex-symmetric factorisation Z = L L^T (complex64) on the GPU, for problems whose
dense Z does not fit the host or the GPU as a whole.

    fac = SymFactor(N, fill, gpu, block=1024)     # fill(j0, j1) -> rows j0.. of columns j0..j1 (host)
    fac.factor()                                  # left-looking: one block column at a time
    X = fac.solve(R)                              # Z^-1 R, complex64 arithmetic, complex128 out

The MoM matrix is symmetric (Z = Z^T: Galerkin, reciprocal media), so only its lower half is kept: the panel
of block column j holds rows j*b.. (L_jj on top, L_ij below), 4 N^2 bytes in all against 8 N^2 for a
complex64 LU and 16 N^2 for the double Z. Each column is filled on the host only when it is factored (the
host never holds Z), updated with every earlier panel (P_k[rows >= j] P_k[j]^T: one GEMM each), its diagonal
block factored on the host in double (b x b) and the rest solved against it (one TRSM). Panels stay on the
GPU while they fit and are streamed from the host after that (out of core: the factor then needs host RAM
of 4 N^2 bytes and the GPU only a column and a panel).

No pivoting: an LDL^T with Bunch-Kaufman pivots would mix rows across tiles. The equilibrated MoM matrix
(unit diagonal) factors without pivots in the cases tried; the factor is a preconditioner of a refinement
on the double-precision operator, and a stalled refinement falls back to a pivoted LU (fast.py).
"""
import numpy as np


def chol_sym(A, base=64):
    """Unpivoted complex-symmetric Cholesky (A = L L^T, not L L^H) of a small dense block, in place (lower)."""
    import scipy.linalg as sla
    n = A.shape[0]
    if n <= base:
        for i in range(n):
            A[i, i] = np.sqrt(A[i, i])
            A[i + 1:, i] /= A[i, i]
            A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i + 1:, i])
        return _tril(A)
    h = n // 2
    chol_sym(A[:h, :h], base)
    A[h:, :h] = sla.solve_triangular(A[:h, :h], A[h:, :h].T, lower=True, check_finite=False).T
    A[h:, h:] -= A[h:, :h] @ A[h:, :h].T
    chol_sym(A[h:, h:], base)
    A[:h, h:] = 0
    return A


def _tril(A):
    A[np.triu_indices(A.shape[0], 1)] = 0
    return A


class SymFactor:
    """Z = L L^T in complex64 tiles. xp: the array module of the device (cupy, or numpy to emulate)."""

    def __init__(self, N, fill, xp=np, block=1024, budget=None, log=None):
        self.N, self.fill, self.xp, self.b = N, fill, xp, block
        self.starts = list(range(0, N, block))
        self.budget = budget                    # bytes of panels kept on the device (None: all)
        self.log = log or (lambda *a: None)
        self.panels, self.on_device = [], []
        self.small_pivot = np.inf

    def _dev(self, k):
        p = self.panels[k]
        return p if self.on_device[k] else self.xp.asarray(p)

    def factor(self):
        xp, b, N = self.xp, self.b, self.N
        held = 0
        for j, j0 in enumerate(self.starts):
            j1 = min(j0 + b, N)
            C = xp.asarray(np.asarray(self.fill(j0, j1), np.complex64))           # (N - j0) x (j1 - j0)
            for k in range(j):
                k0 = self.starts[k]
                P = self._dev(k)
                C -= P[j0 - k0:] @ P[j0 - k0:j1 - k0].T
                del P
            D = np.asarray(self._host(C[:j1 - j0]), np.complex128)
            chol_sym(D)
            self.small_pivot = min(self.small_pivot, float(np.abs(np.diag(D)).min()))
            Ld = xp.asarray(D.astype(np.complex64))
            C[:j1 - j0] = Ld
            if j1 < N:
                C[j1 - j0:] = self._trsm_right(Ld, C[j1 - j0:])
            nbytes = C.size * 8
            keep = self.budget is None or held + nbytes <= self.budget
            if keep:
                held += nbytes
                self.panels.append(C)
            else:
                self.panels.append(self._host(C))
                del C
            self.on_device.append(keep)
        self.held = held
        return self

    def _host(self, a):
        return a if self.xp is np else self.xp.asnumpy(a)

    def _trsm_right(self, L, B):
        """B L^-T (L lower triangular)."""
        if self.xp is np:
            import scipy.linalg as sla
            return sla.solve_triangular(L, B.T, lower=True, check_finite=False).T
        import cupyx.scipy.linalg as cla
        return cla.solve_triangular(L, B.T, lower=True, check_finite=False).T

    def _trsm(self, L, B, trans=False):
        if self.xp is np:
            import scipy.linalg as sla
            return sla.solve_triangular(L, B, lower=True, trans='T' if trans else 'N', check_finite=False)
        import cupyx.scipy.linalg as cla
        return cla.solve_triangular(L, B, lower=True, trans='T' if trans else 'N', check_finite=False)

    def solve(self, R):
        """Z^-1 R (host, N or N x k), complex64 arithmetic on the device."""
        xp, b = self.xp, self.b
        R = np.asarray(R)
        vec = R.ndim == 1
        Y = xp.asarray(R.reshape(self.N, -1).astype(np.complex64))
        for j, j0 in enumerate(self.starts):                    # L y = r
            P = self._dev(j)
            n = P.shape[1]
            Y[j0:j0 + n] = self._trsm(P[:n], Y[j0:j0 + n])
            if j0 + n < self.N:
                Y[j0 + n:] -= P[n:] @ Y[j0:j0 + n]
        for j in range(len(self.starts) - 1, -1, -1):             # L^T x = y
            j0 = self.starts[j]
            P = self._dev(j)
            n = P.shape[1]
            if j0 + n < self.N:
                Y[j0:j0 + n] -= P[n:].T @ Y[j0 + n:]
            Y[j0:j0 + n] = self._trsm(P[:n], Y[j0:j0 + n], trans=True)
        out = np.asarray(self._host(Y)).astype(complex)
        return out[:, 0] if vec else out

    def release(self):
        self.panels, self.on_device = [], []
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()
