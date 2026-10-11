"""
GPU self-check for mom25d:  python tools/gpu_selfcheck.py [--backend cupy|torch] [--n 8192]

1. reports the detected GPU / backend and free VRAM
2. complex64 LU + solve on the GPU vs SciPy on the CPU (accuracy and speed)
3. a small mom25d model solved with and without the GPU (S-parameters must agree)
"""
import argparse, os, sys, time
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import mom25d.gpu as mom25d_gpu
ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='auto')
ap.add_argument('--n', type=int, default=8192, help='matrix size for the LU benchmark')
ap.add_argument('--skip-sim', action='store_true')
a = ap.parse_args()
if a.backend != 'auto':
    os.environ['MOM25D_GPU'] = a.backend

print(mom25d_gpu.describe())
gpu = mom25d_gpu.get_backend(a.backend)
if gpu is None:
    print("\nNo usable GPU backend. Install one of:\n"
          "  PyTorch for ROCm:  pip install torch --index-url https://download.pytorch.org/whl/rocm<ver>\n"
          "  CuPy for ROCm:     see https://docs.cupy.dev/en/stable/install.html (AMD ROCm section)\n"
          "and check that the GPU is visible (rocminfo / rocm-smi). Consumer RDNA3 cards other than the\n"
          "7900 series may need HSA_OVERRIDE_GFX_VERSION=11.0.0 (RDNA2: 10.3.0).")
    sys.exit(1)

free = gpu.mem_free()
nmax = int(np.sqrt(max(0.0, 0.9 * free - 0.4e9) / ((1 if gpu.inplace else 2) * 8.0)))
print(f"largest model for the GPU LU with this much free VRAM: ~{nmax:,} unknowns")

# ---- 2. LU accuracy and speed ----------------------------------------------------------------
rng = np.random.default_rng(1)
for n in sorted({min(a.n, nmax), min(a.n // 2, nmax)}):
    if n < 256:
        continue
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex64) / np.sqrt(n)
    A[np.diag_indices(n)] += 3
    b = rng.standard_normal((n, 2)).astype(complex)
    Z = A.copy()
    gpu.factor(Z[:16, :16].copy())                          # warm-up (kernel load / JIT)
    t = time.perf_counter(); gpu.factor(Z); tg = time.perf_counter() - t
    t = time.perf_counter(); x = gpu.solve(b); ts = time.perf_counter() - t
    gpu.release()
    res = np.linalg.norm(A.astype(complex) @ x - b) / np.linalg.norm(b)
    t = time.perf_counter(); sla.lu_factor(A.T.copy(order='F'), overwrite_a=True, check_finite=False); tc = time.perf_counter() - t
    gfl = 8 / 3 * n ** 3 / tg / 1e9
    print(f"N={n:6d}: GPU LU {tg:7.3f} s ({gfl:6.0f} GFLOP/s), solve {ts * 1e3:6.1f} ms, residual {res:.1e} | "
          f"CPU LU {tc:7.2f} s  -> {tc / tg:5.1f}x")
    if res > 1e-4:
        print("  WARNING: residual too large - the GPU LU result looks wrong")

# ---- 3. a small simulation, GPU vs CPU -------------------------------------------------------
if not a.skip_sim:
    from shapely.geometry import box
    from mom25d.multilayer import Layer, LayerStack
    from mom25d.geo import LinePort
    from mom25d.sim import Simulation
    st = LayerStack([Layer(0.2e-3, 4.3, 0.02)])
    w, L = 0.35e-3, 30e-3
    shapes = {1: [box(-L / 2, -w / 2, L / 2, w / 2), box(-0.2e-3, -4e-3, 0.2e-3, 0)]}       # line + open stub
    ports = [LinePort(1, (-L / 2 + 0.3e-3, 0), (1, 0), w, -0.3e-3, s0=1e-3, s1=8e-3, step=0.7e-3, zc=50.0),
             LinePort(1, (L / 2 - 0.3e-3, 0), (-1, 0), w, -0.3e-3, s0=1e-3, s1=8e-3, step=0.7e-3, zc=50.0)]
    S = {}
    for g in ('off', a.backend):
        sim = Simulation(st, shapes, ports, dmax=0.15e-3, w_ref={1: w}, ncross=6, solver='direct',
                         precision='mixed', gpu=g, verbose=False)
        t = time.perf_counter(); S[g] = sim.sparams(5e9); dt = time.perf_counter() - t
        lu = {s: v for s, v in sim.log.totals().items() if s.startswith('LU')}
        print(f"simulation gpu={g:5s}: N={sim.Ne + sim.Nv}, {dt:.1f} s, " + ", ".join(f"{k} {v:.2f} s" for k, v in lu.items()))
    d = np.abs(S['off'] - S[a.backend]).max()
    print(f"max |S_gpu - S_cpu| = {d:.1e}  " + ("OK" if d < 1e-6 else "MISMATCH"))
