"""
mom25d.gpu_vk - the complex64 LU on a Vulkan GPU (slangpy): mom25d.gpu's backend interface where CuPy has no GPU (AMD
on Windows: no ROCm CuPy there). Vulkan has no LAPACK, so the factorisation is the hybrid one of MAGMA's getrf:

  the CPU factors each panel (N - k0 rows, NB columns) with LAPACK's cgetrf (partial pivoting, the same pivots and
  layout as scipy's lu_factor), and inverts its unit lower triangle (ctrtri);
  the GPU applies the panel's row interchanges to the other columns, forms U12 = L11^-1 A12 (a GEMM with that
  inverse) and the trailing update A22 -= L21 U12 (the O(N^3) part: a register-tiled complex GEMM).

With look-ahead: the GPU updates the next panel's columns first and that panel goes to the CPU while the GPU runs
the rest of the trailing update, so the CPU's panel and the GPU's update overlap.

The matrix is A = Z^T, column-major (Z's rows are its columns, as the CuPy backend factors it): column c in buffer
c >> sh at (c & mask) * N, the buffers at most MAX_BUFFER bytes each (Vulkan drivers cap one buffer at 2-4 GB; 16 GB
of matrix is several). The triangular solves of solve() run on the GPU too: blocked substitution, the diagonal
blocks' inverses from the factorisation, the off-diagonal blocks as matrix-vector products.

    b = VulkanBackend(); b.factor(Z); X = b.solve(R); lu, piv = b.download(); b.release()
"""
import os
import time

import numpy as np
import scipy.linalg as sla
from scipy.linalg import lapack

MAX_BUFFER = 2 ** 31            # bytes of one matrix buffer (under every driver's maxBufferSize)
TABLE = 32                      # matrix buffers at most (the descriptor's table): 64 GB
NB = 256                        # panel width (MOM25D_VK_NB)

SOURCE = r'''
struct Mat {                    // a column-major complex matrix: column c in buffer c >> sh, at (c & mask) * ld
    float2* b[TABLE];
    int sh, mask, ld, r0, c0;   // r0, c0: the submatrix's corner
};

float2* colp(Mat m, int c)
{
    const int cc = c + m.c0;
    return m.b[cc >> m.sh] + (int64_t)(cc & m.mask) * m.ld + m.r0;
}

float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

uniform Mat A_, B_, C_;
uniform int m_, n_, k_;
uniform float alpha_, beta_;

// C = beta C + alpha A B (complex64): 16 x 16 threads, each GT x GT outputs of a TM x TM tile, TK deep per stage
// (GT = 4: 64 x 64 tiles ran 13.6 TFLOP/s on an RX 7800 XT with the prefetch, 128 x 128 ones 9.3; TK = 32 crashed
// RADV's shader compiler).
// gemm_flat: A and B each in one buffer (the host checks), addressed without the buffer table. Whole tiles load
// without bounds tests (a workgroup-uniform branch).
#define TM (16 * GT)
#define TK 16
groupshared float2 As[TK][TM + 1];
groupshared float2 Bs[TK][TM + 1];

float2* flatp(Mat m, int c)     // m's column c, m in one buffer: b[0] is that buffer
{
    return m.b[0] + (int64_t)((c + m.c0) & m.mask) * m.ld + m.r0;
}

#define GEMM_BODY(COLA, COLB)                                                                           \
    const int tx = (int)lid.x, ty = (int)lid.y, l = ty * 16 + tx;                                       \
    const int m0 = (int)gid.x * TM, n0 = (int)gid.y * TM;                                               \
    const bool whole = m0 + TM <= m_ && n0 + TM <= n_;                                                  \
    float2 acc[GT][GT];                                                                                 \
    for (int a = 0; a < GT; a++)                                                                        \
        for (int b = 0; b < GT; b++)                                                                    \
            acc[a][b] = float2(0.0, 0.0);                                                               \
    for (int k0 = 0; k0 < k_; k0 += TK) {                                                               \
        if (whole && k0 + TK <= k_) {                                                                   \
            for (int q = l; q < TM * TK; q += 256) {                                                    \
                const int r = q % TM, c = q / TM;                                                       \
                As[c][r] = COLA(A_, k0 + c)[m0 + r];                                                    \
                const int kk = q % TK, nn = q / TK;                                                     \
                Bs[kk][nn] = COLB(B_, n0 + nn)[k0 + kk];                                                \
            }                                                                                           \
        } else {                                                                                        \
            for (int q = l; q < TM * TK; q += 256) {                                                    \
                const int r = q % TM, c = q / TM;                                                       \
                As[c][r] = (m0 + r < m_ && k0 + c < k_) ? COLA(A_, k0 + c)[m0 + r] : float2(0.0, 0.0);  \
                const int kk = q % TK, nn = q / TK;                                                     \
                Bs[kk][nn] = (k0 + kk < k_ && n0 + nn < n_) ? COLB(B_, n0 + nn)[k0 + kk] : float2(0.0, 0.0); \
            }                                                                                           \
        }                                                                                               \
        GroupMemoryBarrierWithGroupSync();                                                              \
        for (int kk = 0; kk < TK; kk++) {                                                               \
            float2 av[GT], bv[GT];                                                                      \
            for (int a = 0; a < GT; a++) av[a] = As[kk][tx + 16 * a];                                   \
            for (int b = 0; b < GT; b++) bv[b] = Bs[kk][ty + 16 * b];                                   \
            for (int a = 0; a < GT; a++)                                                                \
                for (int b = 0; b < GT; b++) {                                                          \
                    acc[a][b].x = fma(av[a].x, bv[b].x, fma(-av[a].y, bv[b].y, acc[a][b].x));           \
                    acc[a][b].y = fma(av[a].x, bv[b].y, fma(av[a].y, bv[b].x, acc[a][b].y));            \
                }                                                                                       \
        }                                                                                               \
        GroupMemoryBarrierWithGroupSync();                                                              \
    }                                                                                                   \
    for (int b = 0; b < GT; b++) {                                                                      \
        const int c = n0 + ty + 16 * b;                                                                 \
        if (c >= n_) continue;                                                                          \
        float2* col = colp(C_, c);                                                                      \
        for (int a = 0; a < GT; a++) {                                                                  \
            const int r = m0 + tx + 16 * a;                                                             \
            if (r >= m_) continue;                                                                      \
            col[r] = (beta_ != 0.0 ? beta_ * col[r] : float2(0.0, 0.0)) + alpha_ * acc[a][b];           \
        }                                                                                               \
    }

[shader("compute")] [numthreads(16, 16, 1)]
void gemm(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    GEMM_BODY(colp, colp)
}

// a whole tile of gemm_flat: the next stage's loads into registers before this stage's products (their latency
// behind the arithmetic); k past k_ (a last panel narrower than TK stages) loads zeros
void gemm_whole(int tx, int ty, int m0, int n0)
{
    const int l = ty * 16 + tx;
    float2 acc[GT][GT];
    for (int a = 0; a < GT; a++)
        for (int b = 0; b < GT; b++)
            acc[a][b] = float2(0.0, 0.0);
    const int R = TM * TK / 256;
    float2 ra[R], rb[R];
    for (int u = 0; u < R; u++) {
        const int q = l + 256 * u;
        ra[u] = q / TM < k_ ? flatp(A_, q / TM)[m0 + q % TM] : float2(0.0, 0.0);
        rb[u] = q % TK < k_ ? flatp(B_, n0 + q / TK)[q % TK] : float2(0.0, 0.0);
    }
    for (int k0 = 0; k0 < k_; k0 += TK) {
        for (int u = 0; u < R; u++) {
            const int q = l + 256 * u;
            As[q / TM][q % TM] = ra[u];
            Bs[q % TK][q / TK] = rb[u];
        }
        GroupMemoryBarrierWithGroupSync();
        if (k0 + TK < k_) {
            for (int u = 0; u < R; u++) {
                const int q = l + 256 * u;
                const int ka = k0 + TK + q / TM, kb = k0 + TK + q % TK;
                ra[u] = ka < k_ ? flatp(A_, ka)[m0 + q % TM] : float2(0.0, 0.0);
                rb[u] = kb < k_ ? flatp(B_, n0 + q / TK)[kb] : float2(0.0, 0.0);
            }
        }
        for (int kk = 0; kk < TK; kk++) {
            float2 av[GT], bv[GT];
            for (int a = 0; a < GT; a++) av[a] = As[kk][tx + 16 * a];
            for (int b = 0; b < GT; b++) bv[b] = Bs[kk][ty + 16 * b];
            for (int a = 0; a < GT; a++)
                for (int b = 0; b < GT; b++) {
                    acc[a][b].x = fma(av[a].x, bv[b].x, fma(-av[a].y, bv[b].y, acc[a][b].x));
                    acc[a][b].y = fma(av[a].x, bv[b].y, fma(av[a].y, bv[b].x, acc[a][b].y));
                }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (int b = 0; b < GT; b++) {
        float2* col = colp(C_, n0 + ty + 16 * b);
        for (int a = 0; a < GT; a++) {
            const int r = m0 + tx + 16 * a;
            col[r] = (beta_ != 0.0 ? beta_ * col[r] : float2(0.0, 0.0)) + alpha_ * acc[a][b];
        }
    }
}

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_flat(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    if ((int)gid.x * TM + TM <= m_ && (int)gid.y * TM + TM <= n_) {     // uniform in the workgroup
        gemm_whole((int)lid.x, (int)lid.y, (int)gid.x * TM, (int)gid.y * TM);
        return;
    }
    GEMM_BODY(flatp, flatp)
}

// C = A (m x n): panels to and from the staging buffers, U12 into place
[shader("compute")] [numthreads(256, 1, 1)]
void copy(uint3 tid: SV_DispatchThreadID)
{
    const int64_t t = (int64_t)tid.x + (int64_t)tid.y * 65535 * 256;
    if (t >= (int64_t)m_ * n_) return;
    const int r = (int)(t % m_), c = (int)(t / m_);
    colp(C_, c)[r] = colp(A_, c)[r];
}

// the panel's row interchanges on columns [ca_, ca_ + na_) and [cb_, cb_ + n_ - na_) of C_ (C_.c0 = 0), in two
// passes through tmp: rows_[i] <- src_[i]
uniform int* rows_;
uniform int* src_;
uniform float2* tmp_;
uniform int cnt_, ca_, na_, cb_;

int swap_col(int j) { return j < na_ ? ca_ + j : cb_ + (j - na_); }

[shader("compute")] [numthreads(256, 1, 1)]
void swap_gather(uint3 tid: SV_DispatchThreadID)
{
    const int64_t t = (int64_t)tid.x + (int64_t)tid.y * 65535 * 256;
    if (t >= (int64_t)cnt_ * n_) return;
    const int i = (int)(t % cnt_), j = (int)(t / cnt_);
    tmp_[t] = colp(C_, swap_col(j))[src_[i]];
}

[shader("compute")] [numthreads(256, 1, 1)]
void swap_scatter(uint3 tid: SV_DispatchThreadID)
{
    const int64_t t = (int64_t)tid.x + (int64_t)tid.y * 65535 * 256;
    if (t >= (int64_t)cnt_ * n_) return;
    const int i = (int)(t % cnt_), j = (int)(t / cnt_);
    colp(C_, swap_col(j))[rows_[i]] = tmp_[t];
}

// U12 = L11^-1 A12 in place (and into C_, U12's work array): forward substitution with the unit lower L11 (A_, b x b,
// k_ = b) on n_ columns of B_, TRSM_COLS columns a workgroup, the column block in shared memory, TRSM_LANES threads
// a column; LAPACK's trsm, in order
groupshared float2 Xs[NBMAX][TRSM_COLS + 1];

[shader("compute")] [numthreads(TRSM_COLS * TRSM_LANES, 1, 1)]
void trsm(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    const int cl = (int)lid.x % TRSM_COLS, lane = (int)lid.x / TRSM_COLS;
    const int col = (int)gid.x * TRSM_COLS + cl;
    const bool ok = col < n_;
    float2* bc = colp(B_, ok ? col : 0);
    for (int i = lane; i < k_; i += TRSM_LANES) Xs[i][cl] = ok ? bc[i] : float2(0.0, 0.0);
    GroupMemoryBarrierWithGroupSync();
    for (int i = 0; i < k_ - 1; i++) {
        const float2 xi = Xs[i][cl];
        float2* li = colp(A_, i);                               // L11[:, i]
        for (int j = i + 1 + lane; j < k_; j += TRSM_LANES) Xs[j][cl] -= cmul(li[j], xi);
        GroupMemoryBarrierWithGroupSync();
    }
    if (!ok) return;
    float2* uc = colp(C_, col);
    for (int i = lane; i < k_; i += TRSM_LANES) { bc[i] = Xs[i][cl]; uc[i] = Xs[i][cl]; }
}

// the solves, Z x = R with A = Z^T = P L U:  U^T y = R (forward), L^T w = y (backward), x = P w (the host).
// X_: the right-hand sides, N x nrhs column-major (ld N). One block of NB rows at a time:
//   tri:  x_b = Dinv_b^T x_b  (Dinv_b: the block's inverse, NB x NB column-major, from the factorisation)
//   upd:  forward: x_i -= sum_{j in b} A(j, i) x_j for i past the block; backward: for i before it
//   (A(j, i) = U^T(i, j) for j <= i, L^T(i, j) for j > i: row i of the transposed factor is column i of A)
uniform float2* X_;
uniform float2* D_;
uniform int nrhs_, blk_, bn_, fwd_;
groupshared float2 xb[NBMAX];
groupshared float2 red[64];

[shader("compute")] [numthreads(256, 1, 1)]
void solve_tri(uint3 tid: SV_DispatchThreadID)
{
    // one workgroup per right-hand side: y = Dinv^T x_b through shared memory
    const int c = (int)tid.y, t = (int)tid.x;
    float2* x = X_ + (int64_t)c * k_ + blk_;
    for (int i = t; i < bn_; i += 256) xb[i] = x[i];
    GroupMemoryBarrierWithGroupSync();
    for (int i = t; i < bn_; i += 256) {
        float2 s = float2(0.0, 0.0);
        float2* d = D_ + (int64_t)blk_ * NBMAX + (int64_t)i * bn_;    // block's Dinv, column i
        for (int j = 0; j < bn_; j++) s += cmul(d[j], xb[j]);
        x[i] = s;
    }
}

[shader("compute")] [numthreads(64, 1, 1)]
void solve_upd(uint3 tid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    // one wave of 64 per row i (outside the block), all right-hand sides: x_i -= A(blk.., i) . x_blk
    const int g = (int)gid.x + (int)gid.y * 65535;
    const int i = fwd_ != 0 ? blk_ + bn_ + g : g;
    if (fwd_ != 0 ? i >= k_ : i >= blk_) return;
    float2* col = colp(C_, i) + blk_;                         // A(blk.., i): contiguous
    for (int c = 0; c < nrhs_; c++) {
        float2* x = X_ + (int64_t)c * k_;
        float2 s = float2(0.0, 0.0);
        for (int j = (int)lid.x; j < bn_; j += 64) s += cmul(col[j], x[blk_ + j]);
        red[lid.x] = s;
        GroupMemoryBarrierWithGroupSync();
        for (int w = 32; w > 0; w >>= 1) {
            if ((int)lid.x < w) red[lid.x] += red[lid.x + w];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lid.x == 0) x[i] -= red[0];
        GroupMemoryBarrierWithGroupSync();
    }
}
'''

NAMES = ('gemm', 'gemm_flat', 'trsm', 'copy', 'swap_gather', 'swap_scatter', 'solve_tri', 'solve_upd')


GT = 4                          # each thread's outputs: GT x GT (the tile 16 GT square)
# trsm's workgroup: TRSM_COLS columns, TRSM_LANES threads each. At N = 16k on an RX 7800 XT: 4 x 64 69 ms a
# factorisation, 16 x 16 271 ms, 16 x 8 377 ms (few rows a thread between barriers is what pays)
TRSM_COLS = int(os.environ.get('MOM25D_VK_TRSM_COLS') or 4)
TRSM_LANES = int(os.environ.get('MOM25D_VK_TRSM_LANES') or 64)


def source(gt=GT, nbmax=NB):
    return (f"#define TABLE {TABLE}\n#define GT {int(gt)}\n#define NBMAX {int(nbmax)}\n"
            f"#define TRSM_COLS {TRSM_COLS}\n#define TRSM_LANES {TRSM_LANES}\n" + SOURCE)


class _Mat:
    """A column-major complex64 matrix on the device: buffers of whole columns (MAX_BUFFER bytes at most)."""
    def __init__(self, be, rows, cols):
        self.rows, self.cols = int(rows), int(cols)
        per = max(1, MAX_BUFFER // (8 * max(self.rows, 1)))
        self.sh = int(np.floor(np.log2(per)))
        self.per = 1 << self.sh
        nbuf = -(-self.cols // self.per)
        if nbuf > TABLE:
            raise MemoryError(f"mom25d: a {rows} x {cols} matrix needs {nbuf} GPU buffers (at most {TABLE})")
        self.bufs = [be._buffer(8 * self.rows * min(self.per, self.cols - i * self.per)) for i in range(nbuf)]

    def desc(self, r0=0, c0=0):
        b = [x.device_address for x in self.bufs] + [0] * (TABLE - len(self.bufs))
        return {'b': b, 'sh': self.sh, 'mask': self.per - 1, 'ld': self.rows, 'r0': int(r0), 'c0': int(c0)}

    def upload_rows_of(self, Z):
        """Z (C-ordered, rows = this matrix's columns) into the buffers."""
        for i, b in enumerate(self.bufs):
            b.copy_from_numpy(np.ascontiguousarray(Z[i * self.per:(i + 1) * self.per], np.complex64).view(np.uint8).ravel())

    def download(self):
        """The matrix as (cols, rows) C-ordered: its transpose's memory."""
        return np.concatenate([b.to_numpy().view(np.complex64).reshape(-1, self.rows) for b in self.bufs])


class VulkanBackend:
    """mom25d.gpu's backend on a Vulkan GPU: factor (hybrid blocked LU), solve (on the GPU), download."""
    name, inplace, hip, api = 'vulkan', True, False, 'Vulkan'
    lu = piv = None

    def __init__(self):
        from emsdev import gpu as devices, vulkan
        self.info = devices.vulkan_device(need=('bda', 'int64'))  # raises Unavailable: no driver, GPU or slangpy
        try:
            self.dev = vulkan.slang_device()
            import slangpy as spy
        except Exception as e:
            raise devices.Unavailable(f"vulkan: {e}")
        self.spy, self.device = spy, self.info.name
        self.nb = int(os.environ.get('MOM25D_VK_NB') or NB)
        self.usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | \
            spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        mod = self.dev.load_module_from_source(f"mom25d_lu_{self.nb}_{GT}_{TRSM_COLS}_{TRSM_LANES}", source(GT, self.nb))
        self.k = {n: self.dev.create_compute_pipeline(self.dev.link_program([mod], [mod.entry_point(n)]))
                  for n in NAMES}
        self.enc, self._tmp, self._hosts = None, {}, {}
        from .fast import cap_blas_threads
        cap_blas_threads()                                        # the panels' LAPACK on the fast cores
        self.stats = {}

    # ---- the interface mom25d.gpu's backends share -----------------------------------------------------------
    def mem_free(self):
        f = self.info.free()
        return float(f) if f is not None else 0.9 * float(self.info.total or 0)

    def mem_total(self):
        return self.info.total

    def vram_needed(self, N):
        """The matrix, the panel staging, the swap and U12 work arrays, the diagonal blocks' inverses."""
        return 8.0 * N * N + 8.0 * N * self.nb * 5 + 0.4e9

    def fits(self, N, extra=0.0):
        """Whether an N-unknown LU fits in the free memory (+ extra: what kept factors would give back). vram_needed
        counts every device buffer factor() makes and 0.4 GB besides, so the margin is that and at least 0.5 GB
        (3% of the free): 90% of the free turned down antenna_cp_patch's 40.8k unknowns (14.15 GB of 15.7 GB),
        which factored there in 19 s with 1.95 GB left, against minutes on the CPU."""
        free = self.mem_free() + extra
        return self.vram_needed(N) <= free - max(0.5e9, 0.03 * free)

    def release(self):
        """Free the factor and the work arrays (the device memory; the mapped host buffers are kept)."""
        if self.enc is not None or self.lu is not None:
            self._submit(wait=True)
        self.lu = self.piv = None
        self._tmp = {}

    # ---- device plumbing -------------------------------------------------------------------------------------
    def _buffer(self, nbytes, data=None):
        try:
            b = self.dev.create_buffer(size=max(-(-int(nbytes) // 16) * 16, 16), usage=self.usage)
        except Exception as e:
            raise MemoryError(f"mom25d: a {nbytes / 1e9:.2f} GB GPU buffer: {e}")
        if data is not None:
            b.copy_from_numpy(np.ascontiguousarray(data).view(np.uint8).ravel())
        return b

    def _work(self, name, nbytes):
        """A work buffer kept across the panels (grown as needed)."""
        b = self._tmp.get(name)
        if b is None or b.size < nbytes:
            b = self._tmp[name] = self._buffer(nbytes)
        return b

    def _encoder(self):
        if self.enc is None:
            self.enc = self.dev.create_command_encoder()
        return self.enc

    def _run(self, name, groups, **vars):
        enc = self._encoder()
        p = enc.begin_compute_pass()
        cur = self.spy.ShaderCursor(p.bind_pipeline(self.k[name]))
        for key, v in vars.items():
            cur[key] = v
        p.dispatch_compute(list(groups))
        p.end()
        enc.global_barrier()

    @staticmethod
    def _flat(n, per=256):
        """Workgroups for n threads of `per` (x up to 65535, the rest in y)."""
        g = -(-int(n) // per)
        return (min(g, 65535), -(-g // 65535), 1)

    def _submit(self, wait=False):
        """Submit what is recorded; returns the submission's id (None: nothing recorded)."""
        sid = None
        if self.enc is not None:
            sid = self.dev.submit_command_buffer(self.enc.finish())
            self.enc = None
        if wait:
            self.dev.wait_for_idle()
        return sid

    def _gemm(self, A, B, C, m, n, k, alpha=-1.0, beta=1.0):
        """C = beta C + alpha A B (m x n, k deep); gemm_flat where A's k columns and B's n are each in one buffer."""
        if m <= 0 or n <= 0 or k <= 0:
            return
        tm = 16 * GT
        A, fa = self._one_buffer(A, k)
        B, fb = self._one_buffer(B, n)
        self._run('gemm_flat' if fa and fb else 'gemm', (-(-m // tm), -(-n // tm), 1), A_=A, B_=B, C_=C, m_=m,
                  n_=n, k_=k, alpha_=alpha, beta_=beta)

    @staticmethod
    def _one_buffer(d, ncols):
        """(descriptor, flat): d as a one-buffer descriptor (b[0] that buffer, c0 within it) when its ncols
        columns are in one buffer."""
        sh, c0 = d['sh'], d['c0']
        if (c0 >> sh) != ((c0 + max(ncols, 1) - 1) >> sh):
            return d, False
        b = list(d['b'])
        b[0] = b[c0 >> sh]
        return dict(d, b=b, c0=c0 & d['mask']), True

    def _copy(self, A, C, m, n):
        if m > 0 and n > 0:
            self._run('copy', self._flat(m * n), A_=A, C_=C, m_=m, n_=n)

    @staticmethod
    def _flatdesc(addr, rows, r0=0, c0=0):
        """A Mat descriptor of one contiguous column-major array at device address addr (device or mapped host)."""
        return {'b': [int(addr)] + [0] * (TABLE - 1), 'sh': 30, 'mask': (1 << 30) - 1, 'ld': int(rows),
                'r0': int(r0), 'c0': int(c0)}

    # ---- the factorisation -----------------------------------------------------------------------------------
    def _host(self, name, nbytes):
        """Mapped host memory kept across calls (emsdev.vulkan.HostBuffer: the kernels copy to and from it by
        address; slangpy's own read-back runs at a fraction of a GB/s on some drivers)."""
        from emsdev.vulkan import HostBuffer
        h = self._hosts.get(name)
        if h is None or h.nbytes < nbytes:
            if h is not None:
                h.free()
            h = self._hosts[name] = HostBuffer(self.dev, max(int(nbytes), 16))
        return h

    def _cpu_panel(self, hdn, k0, b, N):
        """Factor the panel the GPU left in hdn (M x b, column-major), in place: LAPACK's cgetrf. Returns the pivots
        (0-based, the panel's rows), (L11, U11) (copies: their inverses, the solves', are made while the CPU waits
        on the GPU), the rows it moves [(row, from)]."""
        M = N - k0
        P = hdn.view(np.complex64, M * b).reshape(b, M).T                     # Fortran (M, b), mapped memory
        lu, ip, info = lapack.cgetrf(P, overwrite_a=1)
        if not np.shares_memory(lu, P):
            P[...] = lu
        L11 = np.tril(P[:b, :b], -1) + np.eye(b, dtype=np.complex64)
        perm = {}
        for i in range(b):
            j = int(ip[i])
            if j != i:
                a, c = perm.get(k0 + i, k0 + i), perm.get(k0 + j, k0 + j)
                perm[k0 + i], perm[k0 + j] = c, a
        return ip, (L11, np.triu(P[:b, :b])), [(r, s) for r, s in perm.items() if r != s]

    def _upload(self, A, Z, chunk=1 << 28, c_start=0):
        """Z (host complex64, C-ordered: its rows are A's columns) into A: column blocks of 256 MB through two mapped
        host buffers, the copy kernel reading each across the bus while the host fills the other. 16 GB/s on an
        RX 7800 XT, where buffer.copy_from_numpy (_Mat.upload_rows_of) ran at 6.5: 0.6 s -> 0.25 s of each LU at
        22k unknowns."""
        N, ncols = A.rows, Z.shape[0]                          # Z's rows: A's columns c_start..
        cols = max(1, min(ncols, chunk // (8 * N)))
        hs = [self._host(f'up{i}', 8 * N * cols) for i in (0, 1)]
        subs = [None, None]
        for j, c0 in enumerate(range(0, ncols, cols)):
            cc, i = min(cols, ncols - c0), j % 2
            if subs[i] is not None:
                self.dev.wait_for_submit(subs[i])                 # that buffer's last copy is done
            hs[i].view(np.complex64, N * cc).reshape(cc, N)[...] = Z[c0:c0 + cc]
            self._copy(self._flatdesc(hs[i].address, N), A.desc(0, c_start + c0), N, cc)
            subs[i] = self._submit()
        for x in subs:
            if x is not None:
                self.dev.wait_for_submit(x)

    def factor(self, Z):
        """Z: host complex64, C-contiguous, equilibrated. Factors A = Z^T in place on the GPU (P L U, LAPACK's
        pivots), keeping the diagonal blocks' inverses for solve(). Look-ahead: the next panel's columns are updated
        first and go to the CPU, which factors them while the GPU runs the rest of the trailing update."""
        self.release()
        Z = np.asarray(Z, np.complex64)
        t0 = time.perf_counter()
        A = _Mat(self, Z.shape[0], Z.shape[0])
        self._upload(A, Z)
        self._factor(A, t0)

    def factor_rows(self, N, rows, block=2048):
        """factor() of the N x N Z whose rows rows(r0, r1) gives (host, complex64-able, C-ordered), block rows at a
        time: Z is never whole on the host (it goes up as it is filled), so the host needs a block where factor()'s
        caller needs 8 N^2 bytes (antenna_cp_patch's 40.8k unknowns: 13.3 GB)."""
        self.release()
        t0 = time.perf_counter()
        A = _Mat(self, N, N)
        for r0 in range(0, N, block):
            self._upload(A, np.asarray(rows(r0, min(N, r0 + block)), np.complex64), c_start=r0)
        self._factor(A, t0)

    def _factor(self, A, t0):
        """The blocked LU of the matrix uploaded into A (factor)."""
        N, nb = A.rows, self.nb
        hdn = self._host('panel', 8 * N * nb)                     # the panel, down to the CPU and back
        hrows = [self._host(f'rows{i}', 4 * 2 * nb) for i in (0, 1)]  # by step parity: S2 may read them while
        hsrc = [self._host(f'src{i}', 4 * 2 * nb) for i in (0, 1)]    # the host writes the next step's
        tmpS = self._work('swap', 8 * 2 * nb * N)
        U12 = self._work('u12', 8 * nb * N)
        dinv, pending, piv = [], [], np.zeros(N, np.int32)
        t_cpu = t_wait = 0.0

        def down(k0, b):                                          # record: panel A[k0:, k0:k0+b] -> hdn
            self._copy(A.desc(k0, k0), self._flatdesc(hdn.address, N - k0), N - k0, b)

        def cpu(k0, b):
            nonlocal t_cpu
            tc = time.perf_counter()
            ip, LU11, moved = self._cpu_panel(hdn, k0, b, N)
            pending.append(LU11)
            piv[k0:k0 + b] = ip + k0
            t_cpu += time.perf_counter() - tc
            return moved

        def inverses():                                           # the solves' diagonal blocks, off the critical path
            while pending:
                L11, U11 = pending.pop(0)
                Ui, _ = lapack.ctrtri(U11, lower=0, unitdiag=0)
                Lf, _ = lapack.ctrtri(L11, lower=1, unitdiag=1)
                dinv.append((np.asfortranarray(Ui, np.complex64), np.asfortranarray(Lf, np.complex64)))

        def swaps(step, moved, k0, b, ranges):                     # record: the interchanges on these column ranges
            (ca, na), (cb, nbb) = ranges
            n = na + nbb
            if moved and n > 0:
                kw = dict(C_=A.desc(), rows_=hrows[step % 2].address, src_=hsrc[step % 2].address,
                          tmp_=tmpS.device_address, cnt_=len(moved), ca_=ca, na_=na, cb_=cb, n_=n)
                self._run('swap_gather', self._flat(len(moved) * n), **kw)
                self._run('swap_scatter', self._flat(len(moved) * n), **kw)

        def trsm(k0, b, c, nc):                                   # record: U12[:, c:c+nc] = L11^-1 A12, in place
            if nc > 0:
                self._run('trsm', (-(-nc // TRSM_COLS), 1, 1), A_=A.desc(k0, k0), B_=A.desc(k0, k0 + b + c),
                          C_=self._flatdesc(U12.device_address, b, 0, c), k_=b, n_=nc)

        b0 = min(nb, N)
        down(0, b0)
        self.dev.wait_for_submit(self._submit())
        moved = cpu(0, b0)
        for step, k0 in enumerate(range(0, N, nb)):
            b, M = min(nb, N - k0), N - k0
            n2 = N - k0 - b
            nb1 = min(nb, n2)
            if moved:
                hrows[step % 2].view(np.int32)[:len(moved)] = [x[0] for x in moved]
                hsrc[step % 2].view(np.int32)[:len(moved)] = [x[1] for x in moved]
            # S1 (the CPU waits on it): the factored panel into place, the next panel's columns interchanged,
            # solved (U12) and updated, that panel down
            self._copy(self._flatdesc(hdn.address, M), A.desc(k0, k0), M, b)
            if n2 > 0:
                swaps(step, moved, k0, b, ((k0 + b, nb1), (0, 0)))
                trsm(k0, b, 0, nb1)
                L21 = A.desc(k0 + b, k0)
                self._gemm(L21, self._flatdesc(U12.device_address, b), A.desc(k0 + b, k0 + b), n2, nb1, b)
                down(k0 + b, nb1)
            s1 = self._submit()
            # S2 (while the CPU factors the next panel): the interchanges on the other columns, U12 and the
            # trailing update past the next panel
            swaps(step, moved, k0, b, ((0, k0), (k0 + b + nb1, n2 - nb1)))
            if n2 > nb1:
                trsm(k0, b, nb1, n2 - nb1)
                self._gemm(L21, self._flatdesc(U12.device_address, b, 0, nb1), A.desc(k0 + b, k0 + b + nb1),
                           n2, n2 - nb1, b)
            self._submit()
            if n2 <= 0:
                break
            inverses()
            tw = time.perf_counter()
            self.dev.wait_for_submit(s1)
            t_wait += time.perf_counter() - tw
            moved = cpu(k0 + b, nb1)
        inverses()
        Dbuf = []                                                 # the inverses on the device: block bi at bi NB^2
        for which in (0, 1):
            h = np.zeros(len(dinv) * nb * nb, np.complex64)
            for bi, d in enumerate(dinv):
                m = d[which]
                h[bi * nb * nb:bi * nb * nb + m.size] = m.ravel(order='F')
            Dbuf.append(self._buffer(h.nbytes, h))
        order = np.arange(N)                                      # x = P w: w[order]
        for i in range(N - 1, -1, -1):
            j = piv[i]
            if j != i:
                order[i], order[j] = order[j], order[i]
        self.dev.wait_for_idle()
        self.lu = (A, Dbuf, order)
        self.piv = piv
        self.stats = dict(seconds=time.perf_counter() - t0, cpu=t_cpu, wait=t_wait)

    # ---- solves and the factor on the host -------------------------------------------------------------------
    def download(self, chunk=1 << 28):
        """The factor on the host, as scipy.linalg.lu_factor's of A = Z^T: (lu, piv), Fortran order, 0-based."""
        A = self.lu[0]
        N = A.rows
        out = np.empty((N, N), np.complex64)                      # rows = A's columns: out.T is A, Fortran
        cols = max(1, min(N, chunk // (8 * N)))
        h = self._host('down', 8 * N * cols)
        for c0 in range(0, N, cols):
            cc = min(cols, N - c0)
            self._copy(A.desc(0, c0), self._flatdesc(h.address, N), N, cc)
            self.dev.wait_for_submit(self._submit())
            out[c0:c0 + cc] = h.view(np.complex64, N * cc).reshape(cc, N)
        return out.T, self.piv.copy()

    def solve(self, R):
        """Z^-1 R (R: host, N x k or N), on the GPU: U^T y = R, L^T w = y, x = P w. complex64 inside, complex128
        out."""
        A, Dbuf, order = self.lu
        N, nb = A.rows, self.nb
        R = np.asarray(R)
        one = R.ndim == 1
        R2 = np.asarray(R, np.complex64).reshape(N, -1)
        k = R2.shape[1]
        X = self._work('x', 8 * N * k)
        hx = self._host('x', 8 * N * k)
        hx.view(np.complex64, N * k).reshape(k, N)[...] = R2.T
        self._copy(self._flatdesc(hx.address, N * k), self._flatdesc(X.device_address, N * k), N * k, 1)
        xa = X.device_address
        nblk = -(-N // nb)
        for fwd in (1, 0):
            D = Dbuf[0 if fwd else 1].device_address              # U_bb^-1 (forward), L_bb^-1 (backward)
            for bi in (range(nblk) if fwd else range(nblk - 1, -1, -1)):
                b0 = bi * nb
                bn = min(nb, N - b0)
                self._run('solve_tri', (1, k, 1), X_=xa, D_=D, nrhs_=k, blk_=b0, bn_=bn, k_=N)
                rest = N - b0 - bn if fwd else b0
                if rest > 0:
                    self._run('solve_upd', (min(rest, 65535), -(-rest // 65535), 1), C_=A.desc(), X_=xa, nrhs_=k,
                              blk_=b0, bn_=bn, k_=N, fwd_=fwd)
        self._copy(self._flatdesc(X.device_address, N * k), self._flatdesc(hx.address, N * k), N * k, 1)
        self.dev.wait_for_submit(self._submit())
        W = hx.view(np.complex64, N * k).reshape(k, N).T.astype(complex)[order]
        return W[:, 0] if one else W
