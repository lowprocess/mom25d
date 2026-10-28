"""
mom25d.gpu - GPU dense LU for the mixed-precision direct solver (AMD ROCm or NVIDIA CUDA)

The matrix is still filled on the CPU (the C++ core takes ~2 s); the expensive O(N^3) part, the
complex64 LU factorisation, and the triangular solves used as the GMRES preconditioner run on the
GPU. Refinement products stay on the CPU in double precision, so results match the CPU solver.

Backends (picked in this order by gpu='auto'; force with MOM25D_GPU=cupy|torch|off):
  cupy   CuPy built for ROCm (hipSOLVER/rocSOLVER) or CUDA. LU in place: VRAM ~ 8 N^2 bytes.
  torch  PyTorch ROCm or CUDA build (torch.cuda.* also covers HIP). VRAM ~ 16 N^2 bytes
         (torch keeps input and factors).
  emulate  NumPy stand-in with the same interface (tests / development without a GPU).

Typical capacity with 16 GB of VRAM: ~41k unknowns with cupy, ~29k with torch.
"""
import numpy as np
import scipy.linalg as sla
from emsdev import gpu as devices                                 # the device and its memory: emsdev, shared with gfdtd


GPUUnavailable = devices.Unavailable


class _Backend:
    name, device, hip = '?', 'GPU', False
    inplace = False          # LU overwrites the uploaded matrix (VRAM = 1 x N^2 complex64)
    lu = piv = None

    def vram_needed(self, N):
        return (1 if self.inplace else 2) * 8.0 * N * N + 0.4e9

    def fits(self, N):
        return self.vram_needed(N) <= 0.9 * self.mem_free()

    def release(self):
        self.lu = self.piv = None


class CupyBackend(_Backend):
    name, inplace = 'cupy', True

    def __init__(self):
        self.dev = devices.cupy_device()                         # raises Unavailable: no CuPy, no driver, no GPU
        try:
            import cupy as cp
            import cupyx.scipy.linalg as cla
        except Exception as e:
            raise GPUUnavailable(f"cupy: {e}")
        self.cp, self.cla, self.device, self.hip = cp, cla, self.dev.name, self.dev.hip

    def mem_free(self):
        return self.dev.free()

    def mem_total(self):
        return self.dev.total

    def factor(self, Z):
        """Z: host complex64, C-contiguous, equilibrated. Factors Z^T (Fortran-contiguous view) in place."""
        self.release()
        A = self.cp.array(Z.T, order='F')                        # direct copy of the same buffer
        self.lu, self.piv = self.cla.lu_factor(A, overwrite_a=True, check_finite=False)
        self.cp.cuda.Device().synchronize()

    def solve(self, R):
        """Z^-1 R (R: host, N x k or N). Computed in complex64; returned as complex128."""
        x = self.cla.lu_solve((self.lu, self.piv), self.cp.asarray(np.asarray(R, np.complex64)), trans=1,
                              check_finite=False)                # (Z^T)^T x = R
        return self.cp.asnumpy(x).astype(complex)

    def release(self):
        super().release()
        self.cp.get_default_memory_pool().free_all_blocks()


class TorchBackend(_Backend):
    name = 'torch'

    def __init__(self):
        d = devices.torch_device()                               # ROCm builds report through torch.cuda
        import torch
        self.torch, self.dev, self.info = torch, torch.device('cuda'), d
        self.device, self.hip = d.name, d.hip

    def mem_free(self):
        return self.info.free()

    def mem_total(self):
        return self.info.total

    def factor(self, Z):
        """Upload Z (row-major), factor A = Z^T (a column-major view: no extra transpose copy)."""
        self.release()
        T = self.torch.from_numpy(Z).to(self.dev)
        self.lu, self.piv = self.torch.linalg.lu_factor(T.mT)
        del T
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()

    def solve(self, R):
        """Z x = R with LU(A = Z^T):  A^T x = R  <=>  A^H conj(x) = conj(R)."""
        R = np.asarray(R, np.complex64)
        b = self.torch.from_numpy(np.ascontiguousarray(np.conj(R.reshape(len(R), -1)))).to(self.dev)
        out = np.conj(self.torch.linalg.lu_solve(self.lu, self.piv, b, adjoint=True).cpu().numpy()).astype(complex)
        return out[:, 0] if R.ndim == 1 else out

    def release(self):
        super().release()
        self.torch.cuda.empty_cache()


class EmulatedBackend(_Backend):
    """NumPy/SciPy implementation of the backend interface (no GPU; for tests)."""
    name, inplace, device = 'emulate', True, 'CPU emulation'

    def __init__(self, mem=64e9):
        self._mem = mem

    def mem_free(self):
        return float(self._mem)

    mem_total = mem_free

    def factor(self, Z):
        self.lu, self.piv = sla.lu_factor(np.array(Z.T, order='F'), overwrite_a=True, check_finite=False)

    def solve(self, R):
        return sla.lu_solve((self.lu, self.piv), np.asarray(R, np.complex64), trans=1,
                            check_finite=False).astype(complex)


_BACKENDS = {'cupy': CupyBackend, 'torch': TorchBackend, 'emulate': EmulatedBackend}
_cache = {}


def get_backend(pref='auto'):
    """Return a backend instance or None. pref: 'auto' | 'cupy' | 'torch' | 'emulate' | 'off'.
    The environment variable MOM25D_GPU overrides pref. Reasons for failures are kept in
    get_backend.why for logging."""
    pref = devices.preference(pref, 'MOM25D_GPU')
    get_backend.why = []
    if pref == 'off':
        return None
    names = ['cupy', 'torch'] if pref == 'auto' else [pref]
    for n in names:
        if n not in _BACKENDS:
            get_backend.why.append(f"unknown GPU backend {n!r}")
            continue
        try:
            if n not in _cache:
                _cache[n] = _BACKENDS[n]()
            return _cache[n]
        except GPUUnavailable as e:
            get_backend.why.append(str(e))
    return None


get_backend.why = []


def describe():
    """One-line status for logs / --gpu-info."""
    b = get_backend()
    if b is None:
        return "GPU: none (" + "; ".join(get_backend.why or ['disabled']) + ")"
    api = 'ROCm/HIP' if b.hip else ('CUDA' if b.name != 'emulate' else 'none')
    return f"GPU: {b.device} via {b.name} ({api}), {b.mem_free() / 1e9:.1f} GB free"
