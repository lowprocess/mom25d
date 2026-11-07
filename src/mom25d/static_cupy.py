"""Optional CuPy float64 pair integrator. No GPU import/probe until selected.

Geometry is uploaded once per fill. Only bounded pair batches and their
results cross the device boundary; dense matrices remain on the host.
"""
import numpy as np

from .static_backends import _far


def _analytic(xp, geometry, p, q, dz, scalar_only):
    V, length, tau, un, area, points, weights = geometry
    r = points[p]
    dv = V[q, None]-r[:, :, None]
    sm = (dv*tau[q, None]).sum(-1)
    sp = sm+length[q, None]
    t0 = (dv*un[q, None]).sum(-1)
    d = dz[:, None, None]
    ad = xp.abs(d)
    r02 = t0*t0+d*d
    r0 = xp.sqrt(r02)
    rm, rp = xp.sqrt(r02+sm*sm), xp.sqrt(r02+sp*sp)
    ok = r0 > 1e-14*length[q, None]
    safe = xp.where(ok, r0, 1.)
    lg = xp.where(ok, xp.arcsinh(sp/safe)-xp.arcsinh(sm/safe), 0.)
    beta = xp.arctan2(t0*sp, r02+ad*rp)-xp.arctan2(t0*sm, r02+ad*rm)
    inner = (t0*lg-ad*beta).sum(-1)
    w = weights[None]*area[p, None]
    m0 = (w*inner).sum(1)
    if scalar_only:
        return m0[:, None]
    edge = .5*(r02*lg+sp*rp-sm*rm)
    ir = (edge[..., None]*un[q, None]).sum(-2)
    source = ir+r*inner[..., None]
    ro = (w[..., None]*r*inner[..., None]).sum(1)
    rs = (w[..., None]*source).sum(1)
    rr = (w*(r*source).sum(-1)).sum(1)
    return xp.column_stack((m0, ro, rs, rr))


class CupyBackend:
    parallel_tiles = False

    def __init__(self, geometry, workers, chunk, device):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError('CuPy fill unavailable: install a mom25d GPU CuPy extra matching your driver') from exc
        try:
            self.device = cp.cuda.Device(0 if device is None else device)
            self.device_id = self.device.id
            with self.device:
                cp.zeros(1, dtype=cp.float64).sum().item()
                self.levels = {k: tuple(cp.asarray(a) for a in g) for k, g in geometry.levels.items()}
                self.points, self.area = cp.asarray(geometry.points), cp.asarray(geometry.area)
        except Exception as exc:
            raise RuntimeError(f'CuPy fill could not initialize device {device}: {exc}') from exc
        self.cp, self.chunk = cp, chunk

    def evaluate(self, p, q, d, kind, scalar_only):
        cp = self.cp
        with self.device:
            out = cp.empty((len(p), 1 if scalar_only else 6), dtype=cp.float64)
            for level in np.unique(kind):
                indices = np.flatnonzero(kind == level)
                step = min(self.chunk, 4096 if level < 0 else max(1, 4096//4**int(level)))
                for start in range(0, len(indices), step):
                    ids = indices[start:start+step]
                    pp, qq, dz = cp.asarray(p[ids]), cp.asarray(q[ids]), cp.asarray(d[ids])
                    result = (_far(cp, self.points, self.area, pp, qq, dz, scalar_only) if level < 0 else
                              _analytic(cp, self.levels[int(level)], pp, qq, dz, scalar_only))
                    out[cp.asarray(ids)] = result
            return cp.asnumpy(out)
