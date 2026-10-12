"""
mom25d.afs - adaptive frequency sweep with rational (AAA) interpolation

The EM solve per frequency is the expensive step; S(f) of a passive structure is a smooth
rational function between its resonances. Fit one barycentric rational model to all S entries
at once (set-valued AAA), and only solve at the frequency where two successive models disagree
most, until they agree everywhere on the band.
"""
import numpy as np


def aaa(z, F, tol=1e-10, mmax=80):
    """Set-valued AAA (Nakatsukasa, Sete, Trefethen 2018; Lietaert et al. 2022).
    z: (n,) sample points, F: (n, m) values.  Returns r(zz) -> (len(zz), m)."""
    z = np.asarray(z, complex)
    F = np.asarray(F, complex)
    if F.ndim == 1:
        F = F[:, None]
    n, m = F.shape
    scale = np.maximum(np.abs(F).max(0), 1e-300)
    Fs = F / scale
    J = np.ones(n, bool)
    sup = []
    R = np.tile(Fs.mean(0), (n, 1))
    w = None
    for _ in range(min(mmax, n)):
        err = np.abs(Fs - R).max(1)
        err[~J] = -1
        j = int(np.argmax(err))
        if err[j] <= tol and sup:
            break
        sup.append(j)
        J[j] = False
        zs, Fsup = z[sup], Fs[sup]
        C = 1.0 / (z[J][:, None] - zs[None, :])
        A = np.vstack([C * (Fs[J, k][:, None] - Fsup[None, :, k]) for k in range(m)])
        if A.shape[0] < A.shape[1]:
            A = np.vstack([A, np.zeros((A.shape[1] - A.shape[0], A.shape[1]))])
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        w = Vh[-1].conj()
        D = C @ w
        R = Fs.copy()
        R[J] = (C @ (w[:, None] * Fsup)) / D[:, None]
        if not J.any():
            break
    zs, Fsup = z[sup], F[sup]

    def r(zz):
        zz = np.atleast_1d(np.asarray(zz, complex))
        out = np.empty((len(zz), m), complex)
        with np.errstate(divide='ignore', invalid='ignore'):
            Cz = 1.0 / (zz[:, None] - zs[None, :])
            out[:] = (Cz @ (w[:, None] * Fsup)) / (Cz @ w)[:, None]
        hit = np.isinf(Cz).any(1) | np.isnan(out).any(1)
        for i in np.nonzero(hit)[0]:
            out[i] = Fsup[np.argmin(np.abs(zs - zz[i]))]
        return out
    r.support = zs
    r.weights = w
    return r


def adaptive_sweep(solve, fmin, fmax, n0=5, tol=2e-3, max_points=40, ngrid=2001, log=print):
    """
    solve(f) -> S (P x P).  Returns (f_grid, S_grid (ngrid, P, P), f_samples, S_samples, history): one dict per
    model after the first, solves (samples so far), change (max |dS| from the previous model, on the grid) at
    f_change (Hz), pred (how far the previous model was from the newest sample) and f_new (Hz: the next
    frequency solved, None when the sweep stopped).
    Converged when, twice in a row, the models from the last two iterations agree to `tol` (max |dS|) on the
    grid and the model predicted the newest sample to `tol` before it was solved. (The model change alone can
    be zero while the model is still wrong: a sparse fit that the new sample does not alter.)
    """
    fs = list(np.linspace(fmin, fmax, n0))
    Ss = [solve(f) for f in fs]
    P = Ss[0].shape[0]
    grid = np.linspace(fmin, fmax, ngrid)
    sc = 1e9
    prev, hist, ok, pred_err = None, [], 0, np.inf
    while True:
        o = np.argsort(fs)
        fz = np.array(fs)[o] / sc
        Fz = np.array([Ss[i].ravel() for i in o])
        model = aaa(fz, Fz)
        Mg = model(grid / sc)
        if prev is not None:
            dif = np.abs(Mg - prev).max(1)
            k = int(np.argmax(dif))
            hist.append(dict(solves=len(fs), change=float(dif[k]), f_change=float(grid[k]), pred=float(pred_err),
                             f_new=None))
            log(f"  AFS: {len(fs)} solves, max model change {dif[k]:.1e} at {grid[k] / 1e9:.3f} GHz, "
                f"prediction error {pred_err:.1e}")
            ok = ok + 1 if dif[k] < tol and pred_err < tol else 0
            if ok >= 2 or len(fs) >= max_points:
                break
            fnew = grid[k]
        else:
            gaps = np.diff(np.sort(fs))
            i = int(np.argmax(gaps))
            fnew = np.sort(fs)[i] + gaps[i] / 2
        # keep new samples away from existing ones
        if min(abs(np.array(fs) - fnew)) < (fmax - fmin) / (4 * ngrid):
            gaps = np.diff(np.sort(fs)); i = int(np.argmax(gaps)); fnew = np.sort(fs)[i] + gaps[i] / 2
        prev = Mg
        if hist:
            hist[-1]['f_new'] = float(fnew)
        Snew = solve(fnew)
        pred_err = float(np.abs(model(np.array([fnew / sc]))[0] - Snew.ravel()).max())
        fs.append(fnew); Ss.append(Snew)
    Sg = Mg.reshape(ngrid, P, P)
    o = np.argsort(fs)
    return grid, Sg, np.array(fs)[o], np.array(Ss)[o], hist
