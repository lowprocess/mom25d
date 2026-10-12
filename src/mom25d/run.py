"""
mom25d.run - solve a Simulation over a band and return S-parameters at the output reference impedances.

    f, S, info = solve_band(sim, 1e9, 6e9, points=401, zref=[50, 50])

Adaptive by default (rational fit through as few EM solves as the tolerance allows, sampled on `points`
frequencies); linear=N solves at N equally spaced frequencies instead. S is renormalised from each port's
characterised line impedance (wave ports) or its own impedance (lumped ports) to zref.
"""
import time
import numpy as np


def renorm(S, zc, zref):
    """Renormalise S from reference impedances zc to zref (scalars or per-port, real)."""
    zc = np.asarray(zc, float)
    n = S.shape[0]
    zref = np.broadcast_to(np.asarray(zref, float), (n,))
    if np.allclose(zc, zref):
        return S
    I = np.eye(n)
    Z = np.diag(np.sqrt(zc)) @ (I + S) @ np.linalg.inv(I - S) @ np.diag(np.sqrt(zc))
    zn = np.diag(1 / np.sqrt(zref)) @ Z @ np.diag(1 / np.sqrt(zref))
    return (zn - I) @ np.linalg.inv(zn + I)


def out_refs(sim, z0):
    """Output reference per (modal) port: z0 (a number, or {port name: ohms}) single-ended, 2 z0 differential,
    z0 / 2 common."""
    out = []
    for p in sim.ports:
        mode = getattr(p, 'mode', None)
        base = getattr(p, 'parent', p).name if mode else p.name
        z = z0.get(base, z0.get(p.name, 50.0)) if isinstance(z0, dict) else z0
        out.append(2 * z if mode == 'd' else (z / 2 if mode == 'c' else z))
    return np.array(out, float)


def _zc(sim, f):
    return np.array([p.zc_fn(f) if getattr(p, 'zc_fn', None) else p.zc for p in sim.ports])


def solve_band(sim, fmin, fmax, points=401, zref=50.0, linear=None, afs_tol=2e-3, max_solves=40, log=print,
               keep_mid=False, progress=None):
    """S-parameters of sim over fmin..fmax (Hz): returns (f, S (nf, n, n), info dict with solves, seconds,
    zref, the solved frequencies, and afs / afs_tol: the adaptive sweep's history (afs.adaptive_sweep) and
    tolerance). max_solves: cap of the adaptive sweep. keep_mid: one extra solve at
    mid-band keeping the currents (sim.last). progress(solves, at most, f): after each EM solve."""
    t0 = time.time()
    zr = out_refs(sim, zref)
    done = []

    def solve(fr):
        log(f"  solving {fr / 1e9:.4f} GHz")
        S = renorm(sim.sparams(fr), _zc(sim, fr), zr)
        done.append(fr)
        if progress:
            progress(len(done), int(linear) if linear else max_solves, fr)
        return S
    if linear:
        fs = f = np.linspace(fmin, fmax, int(linear))
        S = np.array([solve(fi) for fi in f])
    else:
        from .afs import adaptive_sweep
        f, S, fs, _, hist = adaptive_sweep(solve, fmin, fmax, tol=afs_tol, ngrid=points, max_points=max_solves,
                                           log=log)
    if keep_mid:
        sim.sparams(fs[len(fs) // 2], keep=True)
    return f, S, dict(solves=len(fs), seconds=time.time() - t0, zref=zr.tolist(), solved=list(map(float, fs)),
                      names=[p.name for p in sim.ports], afs=[] if linear else hist, afs_tol=afs_tol)
