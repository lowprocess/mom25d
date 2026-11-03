"""
mom25d.gpu - GPU dense LU for the mixed-precision direct solver (AMD ROCm, NVIDIA CUDA, or any GPU through Vulkan)

The matrix is still filled on the CPU (the C++ core takes ~2 s); the expensive O(N^3) part, the
complex64 LU factorisation, runs on the GPU. The direct solver downloads the factor (download) and does the
triangular solves of the GMRES preconditioner on the CPU: ROCm's complex64 triangular solve lost the accuracy of
ill-conditioned factors (demo_transition at 1 GHz, smallest pivot 2e-7: a forward error of 6e3 against 0.13 for the
same factor solved on the CPU), and the factor on the host is also kept for the next frequencies. Refinement
products stay on the CPU in double precision, so results match the CPU solver.

Backends (force one with MOM25D_GPU=vulkan|cupy|off; gpu='auto' tries them in emsdev.gpu.auto_order's order:
Vulkan first, CuPy first only where it runs on CUDA):
  vulkan slangpy on any Vulkan GPU (gpu_vk.py): a hybrid LU, the panels on the CPU (LAPACK), the updates on the
         GPU. LU in place: VRAM ~ 8 N^2 bytes. On an RX 7800 XT it factors faster than rocSOLVER from N = 4096.
  cupy   CuPy built for CUDA (cuSOLVER) or ROCm (hipSOLVER/rocSOLVER). LU in place: VRAM ~ 8 N^2 bytes.
  emulate  NumPy stand-in with the same interface (tests / development without a GPU).

Typical capacity with 16 GB of VRAM: ~41k unknowns.
"""
import numpy as np
import scipy.linalg as sla
from emsdev import gpu as devices                                 # the device and its memory: emsdev, shared with gfdtd


GPUUnavailable = devices.Unavailable


class _Backend:
    name, device, hip = '?', 'GPU', False

    @property
    def api(self):
        return 'ROCm/HIP' if self.hip else ('CUDA' if self.name == 'cupy' else 'none')
    inplace = False          # LU overwrites the uploaded matrix (VRAM = 1 x N^2 complex64)
    lu = piv = None

    def vram_needed(self, N):
        return (1 if self.inplace else 2) * 8.0 * N * N + 0.4e9

    def fits(self, N):
        return self.vram_needed(N) <= 0.9 * self.mem_free()

    def release(self):
        self.lu = self.piv = None

    def download(self):
        """The factor on the host, as scipy.linalg.lu_factor's of A = Z^T: (lu, piv), Fortran order, 0-based."""
        raise NotImplementedError


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

    def download(self):
        return self.cp.asnumpy(self.lu, order='F'), self.cp.asnumpy(self.piv)

    def release(self):
        super().release()
        self.cp.get_default_memory_pool().free_all_blocks()


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

    def download(self):
        return self.lu, self.piv


def _vulkan():
    from .gpu_vk import VulkanBackend
    return VulkanBackend()


_BACKENDS = {'cupy': CupyBackend, 'vulkan': _vulkan, 'emulate': EmulatedBackend}
_cache = {}


def get_backend(pref='auto'):
    """Return a backend instance or None. pref: 'auto' | 'vulkan' | 'cupy' | 'emulate' | 'off'.
    The environment variable MOM25D_GPU overrides pref. Reasons for failures are kept in
    get_backend.why for logging."""
    pref = devices.preference(pref, 'MOM25D_GPU')
    get_backend.why = []
    if pref == 'off':
        return None
    names = list(devices.auto_order()) if pref == 'auto' else [pref]
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
        except ImportError as e:                                # vulkan: no slangpy
            get_backend.why.append(f"{n}: {e}")
    return None


get_backend.why = []


def describe():
    """One-line status for logs / --gpu-info."""
    b = get_backend()
    if b is None:
        return "GPU: none (" + "; ".join(get_backend.why or ['disabled']) + ")"
    return f"GPU: {b.device} via {b.name} ({b.api}), {b.mem_free() / 1e9:.1f} GB free"
