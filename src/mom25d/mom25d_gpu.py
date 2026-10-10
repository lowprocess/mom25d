"""
mom25d_gpu.py - GPU dense LU for the mixed-precision direct solver (AMD ROCm or NVIDIA CUDA)

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
import os
import numpy as np


class GPUUnavailable(RuntimeError):
    pass


class _Backend:
    name = '?'
    inplace = False          # LU overwrites the uploaded matrix (VRAM = 1 x N^2 complex64)

    def vram_needed(self, N):
        return (1 if self.inplace else 2) * 8.0 * N * N + 0.4e9

    def fits(self, N):
        return self.vram_needed(N) <= 0.9 * self.mem_free()


class CupyBackend(_Backend):
    name = 'cupy'
    inplace = True

    def __init__(self):
        try:
            import cupy as cp
            import cupyx.scipy.linalg as cla
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise GPUUnavailable("no GPU device")
        except Exception as e:                                   # ImportError, CUDARuntimeError, ...
            raise GPUUnavailable(f"cupy: {e}")
        self.cp, self.cla = cp, cla
        dev = cp.cuda.Device()
        try:
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            nm = props.get('name', b'GPU')
            self.device = nm.decode() if isinstance(nm, bytes) else str(nm)
        except Exception:
            self.device = 'GPU'
        self.hip = bool(getattr(cp.cuda.runtime, 'is_hip', False))
        self.lu = self.piv = None

    def mem_free(self):
        self.cp.get_default_memory_pool().free_all_blocks()
        return float(self.cp.cuda.Device().mem_info[0])

    def factor(self, Z):
        """Z: host complex64, C-contiguous, equilibrated. Factors Z^T (Fortran-contiguous view) in place."""
        cp = self.cp
        self.release()
        A = cp.array(Z.T, order='F')                             # direct copy of the same buffer
        self.lu, self.piv = self.cla.lu_factor(A, overwrite_a=True, check_finite=False)
        cp.cuda.Device().synchronize()

    def solve(self, R):
        """Z^-1 R (R: host, N x k or N). Computed in complex64; returned as complex128."""
        cp = self.cp
        b = cp.asarray(np.asarray(R, np.complex64))
        x = self.cla.lu_solve((self.lu, self.piv), b, trans=1, check_finite=False)   # (Z^T)^T x = R
        return cp.asnumpy(x).astype(complex)

    def release(self):
        self.lu = self.piv = None
        self.cp.get_default_memory_pool().free_all_blocks()


class TorchBackend(_Backend):
    name = 'torch'
    inplace = False

    def __init__(self):
        try:
            import torch
            if not torch.cuda.is_available():                    # ROCm builds report through torch.cuda
                raise GPUUnavailable("torch.cuda.is_available() is False")
        except GPUUnavailable:
            raise
        except Exception as e:
            raise GPUUnavailable(f"torch: {e}")
        self.torch = torch
        self.dev = torch.device('cuda')
        self.device = torch.cuda.get_device_name(0)
        self.hip = getattr(torch.version, 'hip', None) is not None
        self.lu = self.piv = None

    def mem_free(self):
        self.torch.cuda.empty_cache()
        return float(self.torch.cuda.mem_get_info()[0])

    def factor(self, Z):
        """Upload Z (row-major), factor A = Z^T (a column-major view: no extra transpose copy)."""
        torch = self.torch
        self.release()
        T = torch.from_numpy(Z).to(self.dev)
        self.lu, self.piv = torch.linalg.lu_factor(T.mT)
        del T
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def solve(self, R):
        """Z x = R with LU(A = Z^T):  A^T x = R  <=>  A^H conj(x) = conj(R)."""
        torch = self.torch
        R = np.asarray(R, np.complex64)
        vec = R.ndim == 1
        b = torch.from_numpy(np.ascontiguousarray(np.conj(R.reshape(len(R), -1)))).to(self.dev)
        x = torch.linalg.lu_solve(self.lu, self.piv, b, adjoint=True)
        out = np.conj(x.cpu().numpy()).astype(complex)
        return out[:, 0] if vec else out

    def release(self):
        self.lu = self.piv = None
        self.torch.cuda.empty_cache()


class EmulatedBackend(_Backend):
    """NumPy/SciPy implementation of the backend interface (no GPU; for tests)."""
    name = 'emulate'
    inplace = True
    device = 'CPU emulation'
    hip = False

    def __init__(self, mem=64e9):
        self._mem = mem
        self.lu = self.piv = None

    def mem_free(self):
        return float(self._mem)

    def factor(self, Z):
        import scipy.linalg as sla
        self.lu, self.piv = sla.lu_factor(np.array(Z.T, order='F'), overwrite_a=True, check_finite=False)

    def solve(self, R):
        import scipy.linalg as sla
        return sla.lu_solve((self.lu, self.piv), np.asarray(R, np.complex64), trans=1,
                            check_finite=False).astype(complex)

    def release(self):
        self.lu = self.piv = None


_BACKENDS = {'cupy': CupyBackend, 'torch': TorchBackend, 'emulate': EmulatedBackend}
_cache = {}


def get_backend(pref='auto'):
    """Return a backend instance or None. pref: 'auto' | 'cupy' | 'torch' | 'emulate' | 'off'.
    The environment variable MOM25D_GPU overrides pref. Reasons for failures are kept in
    get_backend.why for logging."""
    pref = os.environ.get('MOM25D_GPU', pref) or 'auto'
    pref = {'0': 'off', 'false': 'off', 'no': 'off', '1': 'auto', 'on': 'auto', 'true': 'auto'}.get(str(pref).lower(), str(pref).lower())
    get_backend.why = []
    if pref == 'off':
        return None
    names = ['cupy', 'torch'] if pref == 'auto' else [pref]
    for n in names:
        if n not in _BACKENDS:
            get_backend.why.append(f"unknown GPU backend {n!r}")
            continue
        if n in _cache:
            return _cache[n]
        try:
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
