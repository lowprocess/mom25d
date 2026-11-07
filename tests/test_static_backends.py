"""Backend equivalence, real host concurrency, bounded scheduling and device contracts."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mom25d.base import RWG, combine
from mom25d.static import static_moments
from mom25d.static_backends import FillEngine, NumpyBackend, bounded_map, register_fill_backend
import mom25d.static_backends as backends


@pytest.fixture
def mesh():
    # Two separated islands on two layers: self, adjacent, off-plane and far pairs.
    nodes, tris, z = [], [], []
    for h in (0., .00013):
        for x in (0., .025):
            k = len(nodes)
            nodes.extend([[x, 0.], [x+.002, 0.], [x+.002, .001], [x, .001]])
            tris.extend([[k, k+1, k+2], [k, k+2, k+3]])
            z.extend([h, h])
    return np.array(nodes), np.array(tris), np.array(z)


@pytest.mark.parametrize('backend,workers', [('threaded', 1), ('threaded', 2), ('openmp', 1), ('openmp', 2)])
@pytest.mark.parametrize('level', [0, 1, 2])
@pytest.mark.parametrize('scalar_only', [False, True])
def test_backends_match_serial_reference(mesh, backend, workers, level, scalar_only):
    if backend == 'openmp' and backends._compiled() is None:
        pytest.skip('OpenMP core not built')
    reference = static_moments(*mesh, backend='numpy', outer_levels=level, scalar_only=scalar_only, chunk=16)
    result, info = static_moments(*mesh, backend=backend, workers=workers, outer_levels=level,
                                  scalar_only=scalar_only, chunk=9, return_info=True)
    for expected, actual in zip(reference, result):
        assert actual == pytest.approx(expected, rel=2e-11, abs=1e-27)
    assert info['pairs'] == len(mesh[1])*(len(mesh[1])+1)//2
    assert 0 < info['analytic_pairs'] < info['pairs']
    if backend == 'openmp':
        assert info['observed_kernel_workers'] >= 1
    # Every pair has one owner: scheduling/thread count cannot change reductions.
    again = static_moments(*mesh, backend=backend, workers=workers, outer_levels=level,
                            scalar_only=scalar_only, chunk=4)
    for a, b in zip(result, again):
        assert np.array_equal(a, b)


def test_native_combined_inductance_and_all_analytic_path(mesh):
    if backends._compiled() is None:
        pytest.skip('OpenMP core not built')
    results = [static_moments(*mesh, backend=b, workers=2, near=np.inf) for b in ('numpy', 'openmp')]
    inductances = []
    for nodes, m0, mx, my, rr in results:
        inductances.append(combine(RWG(nodes, mesh[1]), rr, mx, my, m0))
    assert np.linalg.norm(inductances[0]-inductances[1])/np.linalg.norm(inductances[0]) < 1e-11
    assert np.linalg.eigvalsh(inductances[1]).min() > 0


def test_backend_extension_really_runs_concurrently(mesh, monkeypatch):
    monkeypatch.setattr(backends, '_FACTORIES', dict(backends._FACTORIES))
    barrier, lock, threads, calls = threading.Barrier(2), threading.Lock(), set(), [0]

    class RecordingBackend(NumpyBackend):
        def evaluate(self, *args):
            with lock:
                threads.add(threading.get_ident())
                calls[0] += 1
                first = calls[0] <= 2
            if first:
                barrier.wait(timeout=10)
            return super().evaluate(*args)

    register_fill_backend('recording', RecordingBackend)
    result = static_moments(*mesh, backend='recording', workers=2, chunk=4)
    reference = static_moments(*mesh, backend='numpy')
    assert len(threads) == 2
    for a, b in zip(result, reference):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match='registered'):
        register_fill_backend('recording', RecordingBackend)


def test_scheduler_does_not_queue_the_entire_fill():
    produced = []
    def tasks():
        for i in range(20):
            produced.append(i)
            yield i
    results = bounded_map(lambda i: i*i, tasks(), 2)
    assert next(results) == 0
    assert len(produced) == 2
    assert list(results) == [i*i for i in range(1, 20)]


def test_backend_failure_is_not_silently_retried(mesh, monkeypatch):
    monkeypatch.setattr(backends, '_FACTORIES', dict(backends._FACTORIES))
    class Broken(NumpyBackend):
        def evaluate(self, *args):
            raise RuntimeError('device failed')
    register_fill_backend('broken', Broken)
    with pytest.raises(RuntimeError, match='device failed'):
        static_moments(*mesh, backend='broken', workers=2, chunk=4)


def test_backend_output_contract(mesh, monkeypatch):
    monkeypatch.setattr(backends, '_FACTORIES', dict(backends._FACTORIES))
    class Bad(NumpyBackend):
        def evaluate(self, p, q, d, kind, scalar_only):
            return np.full((len(p), 1 if scalar_only else 6), np.nan)
    register_fill_backend('nonfinite', Bad)
    with pytest.raises(FloatingPointError, match='non-finite'):
        static_moments(*mesh, backend='nonfinite')


def test_environment_and_missing_backend_policy(mesh, monkeypatch):
    monkeypatch.setattr(backends, '_compiled', lambda: None)
    monkeypatch.delenv('MOM25D_FILL_BACKEND', raising=False)
    monkeypatch.setenv('MOM25D_FILL_THREADS', '2')
    e = FillEngine(*mesh)
    assert e.info['backend'] == 'threaded' and e.info['tile_workers'] == 2
    with pytest.raises(RuntimeError, match='rebuild'):
        FillEngine(*mesh, backend='openmp')
    monkeypatch.setenv('MOM25D_FILL_BACKEND', 'numpy')
    assert FillEngine(*mesh).info['workers'] == 1
    monkeypatch.setenv('MOM25D_FILL_THREADS', '0')
    with pytest.raises(ValueError, match='workers'):
        FillEngine(*mesh)


def test_existing_openmp_thread_preferences_are_respected(mesh, monkeypatch):
    monkeypatch.delenv('MOM25D_FILL_THREADS', raising=False)
    monkeypatch.delenv('MOM25D_OMP_THREADS', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '2,1')
    assert FillEngine(*mesh, backend='threaded').info['workers'] == 2
    monkeypatch.setenv('MOM25D_OMP_THREADS', '3')
    assert FillEngine(*mesh, backend='threaded').info['workers'] == 3
    monkeypatch.setenv('MOM25D_FILL_THREADS', '4')
    assert FillEngine(*mesh, backend='threaded', workers=1).info['workers'] == 1


def test_installed_backend_is_loaded_only_when_selected(mesh, monkeypatch):
    monkeypatch.setattr(backends, '_FACTORIES', dict(backends._FACTORIES))
    loaded = []
    def load():
        loaded.append(True)
        return NumpyBackend
    monkeypatch.setattr(backends.metadata, 'entry_points', lambda **kw: [SimpleNamespace(name='external', load=load)])
    FillEngine(*mesh, backend='numpy')
    assert loaded == []
    out = static_moments(*mesh, backend='external', workers=2, chunk=4)
    assert loaded == [True]
    reference = static_moments(*mesh, backend='numpy')
    assert all(np.array_equal(a, b) for a, b in zip(out, reference))


def test_missing_gpu_does_not_fall_back_to_cpu(mesh, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'cupy', None)
    with pytest.raises(RuntimeError, match='CuPy fill unavailable'):
        static_moments(*mesh, backend='cupy')


@pytest.mark.parametrize('kwargs', [dict(workers=0), dict(workers=True), dict(chunk=0),
                                   dict(chunk=2.5), dict(near=-1), dict(outer_levels=3)])
def test_invalid_configuration(mesh, kwargs):
    with pytest.raises(ValueError):
        static_moments(*mesh, **kwargs)


@pytest.mark.parametrize('level', [0, 1, 2])
@pytest.mark.parametrize('scalar_only', [False, True])
def test_portable_gpu_formula_on_host_namespace(mesh, level, scalar_only):
    # Exercise device formula independently of the CPU reference even without GPU hardware.
    from mom25d.static_cupy import _analytic
    e = FillEngine(*mesh, backend='numpy', outer_levels=level)
    p, q = np.array([0, 1, 2, 3, 4]), np.array([0, 0, 5, 7, 2])
    d = np.array([0., .0001, -.0002, 0., .0003])
    kind = np.full(len(p), level, np.int32)
    actual = _analytic(np, e.geometry.levels[level], p, q, d, scalar_only)
    assert actual == pytest.approx(e.evaluate(p, q, d, kind, scalar_only), rel=1e-12, abs=1e-27)


def test_actual_cupy_backend_when_device_available(mesh):
    pytest.importorskip('cupy')
    import cupy as cp
    try:
        cp.zeros(1).sum().item()
    except Exception as exc:
        pytest.skip(f'CuPy device unavailable: {exc}')
    for scalar in (False, True):
        expected = static_moments(*mesh, backend='numpy', scalar_only=scalar)
        actual = static_moments(*mesh, backend='cupy', device=0, scalar_only=scalar)
        for a, b in zip(expected, actual):
            assert b == pytest.approx(a, rel=2e-10, abs=1e-26)
