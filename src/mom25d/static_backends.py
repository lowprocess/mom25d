"""Execution backends for quasi-static triangle-pair integration.

Factories receive (FillGeometry, workers, chunk, device), returning an object
with `evaluate(p, q, d, kind, scalar_only)` and `parallel_tiles` (whether calls
may run concurrently). kind=-1 selects the smooth three-point product rule;
0/1/2 select analytic source integration at that observation refinement.
Results must be host float64 arrays: (npairs, 1) for scalar-only, otherwise
(npairs, 6), ordered M0, Mro_x, Mro_y, Mrs_x, Mrs_y, Mrr. Device backends own
upload/synchronization; the scheduler owns tiling, reciprocity and output.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module, metadata
import os
from pathlib import Path

import numpy as np

from .base import _BAR3
from .sing import _offplane_geometry, offplane_pairs


def default_workers():
    """Respect affinity and Linux cgroup CPU quotas; never change global OMP/BLAS settings."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    # Walk ancestors too: a nested cgroup can inherit a tighter parent quota.
    paths = []
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                current = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                paths = [current, *current.parents]
                break
    except OSError:
        pass
    paths.append(Path('/sys/fs/cgroup'))
    for path in paths:
        try:
            quota, period = (path / 'cpu.max').read_text().split()
            if quota != 'max':
                n = min(n, max(1, int(quota) // int(period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return max(1, n)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def bounded_map(function, items, workers):
    """Ordered results with at most `workers` tasks/results in flight, including on failure."""
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mom25d-fill') as pool:
        pending = deque()
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is None:
                    break
                pending.append(pool.submit(function, item))
            while pending:
                yield pending.popleft().result()
                item = next(iterator, None)
                if item is not None:
                    pending.append(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()


@dataclass(frozen=True)
class FillGeometry:
    nodes: np.ndarray
    tris: np.ndarray
    z: np.ndarray
    cent: np.ndarray
    radius: np.ndarray
    points: np.ndarray
    levels: dict

    @classmethod
    def prepare(cls, nodes, tris, z, outer_levels):
        nodes, tris, z = np.asarray(nodes, float), np.asarray(tris), np.asarray(z, float)
        if (nodes.ndim != 2 or nodes.shape[1] != 2 or not len(nodes) or
                tris.ndim != 2 or tris.shape[1] != 3 or not len(tris) or tris.dtype.kind not in 'iu' or
                z.shape != (len(tris),) or not np.isfinite(nodes).all() or not np.isfinite(z).all() or
                tris.min() < 0 or tris.max() >= len(nodes)):
            raise ValueError('fill requires finite planar nodes, integer triangles and one z per triangle')
        nodes = np.ascontiguousarray(nodes - nodes.mean(0))
        tris = np.array(tris, dtype=np.int32, order='C', copy=True)
        z = np.array(z, order='C', copy=True)
        V = nodes[tris]
        edge = V[:, 1:]-V[:, :1]
        if (edge[:, 0, 0]*edge[:, 1, 1]-edge[:, 0, 1]*edge[:, 1, 0] <= 0).any():
            raise ValueError('fill triangles must be nondegenerate and counter-clockwise')
        cent = V.mean(1)
        radius = np.linalg.norm(V-cent[:, None], axis=2).max(1)
        points = np.einsum('qk,tkc->tqc', _BAR3, V)
        levels = {}
        for level in {0, outer_levels}:
            g = _offplane_geometry(nodes, tris, level)
            # The unrefined weights are a shared module constant; freeze an
            # owned copy rather than changing its global writeability.
            levels[level] = (*g[:-1], g[-1].copy())
        for array in (nodes, tris, z, cent, radius, points, *(a for g in levels.values() for a in g)):
            array.setflags(write=False)
        return cls(nodes, tris, z, cent, radius, points, levels)

    @property
    def area(self):
        return self.levels[0][4]


def _far(xp, points, area, p, q, d, scalar_only):
    pa, pb = points[p], points[q]
    a, b = pa[:, :, None], pb[:, None, :]
    g = (area[p]*area[q]/9)[:, None, None] / xp.sqrt(((a-b)**2).sum(-1)+d[:, None, None]**2)
    m0 = g.sum((1, 2))
    if scalar_only:
        return m0[:, None]
    ro = xp.einsum('mq,mqc->mc', g.sum(2), pa)
    rs = xp.einsum('mq,mqc->mc', g.sum(1), pb)
    rr = xp.einsum('mab,mac,mbc->m', g, pa, pb)
    return xp.column_stack((m0, ro, rs, rr))


class NumpyBackend:
    parallel_tiles = True

    def __init__(self, geometry, workers, chunk, device):
        if device is not None:
            raise ValueError('device is only applicable to a device backend')
        self.g, self.chunk = geometry, chunk

    def evaluate(self, p, q, d, kind, scalar_only):
        out = np.empty((len(p), 1 if scalar_only else 6))
        for level in np.unique(kind):
            mask = kind == level
            if level == -1:
                out[mask] = _far(np, self.g.points, self.g.area, p[mask], q[mask], d[mask], scalar_only)
            else:
                m = offplane_pairs(self.g.nodes, self.g.tris, p[mask], q[mask], d[mask],
                                   _geometry=self.g.levels[int(level)], scalar_only=scalar_only,
                                   chunk=max(1, min(self.chunk, 4096//4**int(level))))
                out[mask] = m[:, None] if scalar_only else np.column_stack(m)
        return out


def _compiled():
    try:
        core = import_module('mom25d._core')
    except ImportError:
        return None
    return core if hasattr(core, 'static_pairs') and getattr(core, 'static_openmp', False) else None


class OpenMPBackend:
    def __init__(self, geometry, workers, chunk, device):
        if device is not None:
            raise ValueError('device is only applicable to a device backend')
        self.core = _compiled()
        if self.core is None:
            raise RuntimeError('OpenMP fill unavailable: rebuild with python -m mom25d.core.build')
        self.workers = workers
        self.observed_workers = self.core.static_thread_count(workers)
        self.parallel_tiles = workers == 1
        V, L, tau, un, area, _, _ = geometry.levels[0]
        self.data = dict(V=V, L=L, tau=tau, un=un, area=area, P3=geometry.points)
        for level in range(3):
            g = geometry.levels.get(level, geometry.levels[0])
            self.data[f'P{level}'], self.data[f'W{level}'] = g[5], g[6]

    def evaluate(self, p, q, d, kind, scalar_only):
        return self.core.static_pairs(self.data, p, q, d, kind, scalar_only, self.workers)


def _cupy_factory(geometry, workers, chunk, device):
    from .static_cupy import CupyBackend
    return CupyBackend(geometry, workers, chunk, device)


_FACTORIES = {'numpy': NumpyBackend, 'threaded': NumpyBackend, 'openmp': OpenMPBackend, 'cupy': _cupy_factory}


def register_fill_backend(name, factory):
    """Register a factory implementing the module's pair-integration protocol.
    Installed extensions may instead publish a `mom25d.static_backends` entry point.
    Existing names are protected, and providers are loaded only when selected.
    """
    if not isinstance(name, str) or not name or name == 'auto' or name in _FACTORIES:
        raise ValueError(f'invalid or already registered fill backend {name!r}')
    if not callable(factory):
        raise TypeError('fill backend factory must be callable')
    _FACTORIES[name] = factory


class FillEngine:
    def __init__(self, nodes, tris, z, near=3., outer_levels=1, backend='auto', workers=None,
                 chunk=16384, device=None, prefer_outer=False):
        if np.ndim(near) != 0 or near < 0 or np.isnan(near):
            raise ValueError('near must be nonnegative (or infinity)')
        if not isinstance(outer_levels, (int, np.integer)) or outer_levels not in (0, 1, 2):
            raise ValueError('outer_levels must be 0, 1 or 2')
        requested = backend or os.environ.get('MOM25D_FILL_BACKEND', 'auto')
        if requested == 'auto':
            requested = os.environ.get('MOM25D_FILL_BACKEND', 'auto')
        if workers is None:
            key = next((k for k in ('MOM25D_FILL_THREADS', 'MOM25D_OMP_THREADS', 'OMP_NUM_THREADS')
                        if k in os.environ), None)
            env = os.environ.get(key) if key else None
            try:
                workers = int(env.split(',')[0] if key == 'OMP_NUM_THREADS' else env) if env is not None else default_workers()
            except ValueError as exc:
                raise ValueError(f'{key} must be a positive integer') from exc
        workers = _positive_integer(workers, 'workers')
        self.chunk = _positive_integer(chunk, 'chunk')
        self.near, self.outer_levels = near, outer_levels
        name = ('openmp' if _compiled() is not None and not os.environ.get('MOM25D_NO_CORE')
                else 'threaded') if requested == 'auto' else requested
        if name not in _FACTORIES:
            providers = [ep for ep in metadata.entry_points(group='mom25d.static_backends') if ep.name == name]
            if len(providers) != 1:
                raise ValueError(f'unknown or ambiguous fill backend {name!r}')
            register_fill_backend(name, providers[0].load())
        if name in ('numpy', 'cupy'):
            workers = 1
        self.geometry = FillGeometry.prepare(nodes, tris, z, outer_levels)
        kernel_workers = 1 if name in ('numpy', 'threaded', 'cupy') or (prefer_outer and name == 'openmp') else workers
        self.backend = _FACTORIES[name](self.geometry, kernel_workers, self.chunk, device)
        self.tile_workers = workers if self.backend.parallel_tiles else 1
        self.info = dict(backend=name, workers=workers, kernel_workers=1 if self.backend.parallel_tiles else kernel_workers,
                         tile_workers=self.tile_workers, chunk=self.chunk,
                         device=getattr(self.backend, 'device_id', device), precision='float64',
                         triangles=len(self.geometry.tris), outer_levels=int(outer_levels),
                         near='all' if np.isinf(near) else float(near))
        if hasattr(self.backend, 'observed_workers'):
            self.info['observed_kernel_workers'] = self.backend.observed_workers

    def kinds(self, p, q, d):
        g = self.geometry
        distance = np.sqrt(((g.cent[p]-g.cent[q])**2).sum(-1)+d*d)
        size = g.radius[p]+g.radius[q]
        close = np.ones(len(p), bool) if np.isinf(self.near) else distance <= self.near*size
        kind = np.full(len(p), -1, np.int32)
        kind[close] = 0
        kind[close & (distance <= size)] = self.outer_levels
        return kind

    def evaluate(self, p, q, d, kind=None, scalar_only=False):
        p, q = np.asarray(p), np.asarray(q)
        n = len(self.geometry.tris)
        if (p.ndim != 1 or q.shape != p.shape or p.dtype.kind not in 'iu' or q.dtype.kind not in 'iu' or
                (p < 0).any() or (q < 0).any() or (p >= n).any() or (q >= n).any()):
            raise ValueError('pair indices must be matching integer vectors within the mesh')
        p, q = np.ascontiguousarray(p, np.int32), np.ascontiguousarray(q, np.int32)
        d = np.ascontiguousarray(np.broadcast_to(np.asarray(d, float), p.shape))
        if not np.isfinite(d).all():
            raise ValueError('pair offsets must be finite')
        if kind is None:
            kind = self.kinds(p, q, d)
        kind = np.asarray(kind)
        if (kind.shape != p.shape or kind.dtype.kind not in 'iu' or
                not np.isin(kind, [-1, *self.geometry.levels]).all()):
            raise ValueError('pair kinds must select far (-1) or a prepared analytic level')
        kind = np.ascontiguousarray(kind, np.int32)
        out = self.backend.evaluate(p, q, d, kind, scalar_only)
        if not isinstance(out, np.ndarray) or out.dtype != np.float64 or out.shape != (len(p), 1 if scalar_only else 6):
            raise TypeError('fill backend must return a host float64 array of pair moments')
        if not np.isfinite(out).all():
            raise FloatingPointError('non-finite triangle-pair moments from fill backend')
        return out
