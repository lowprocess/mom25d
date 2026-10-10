"""GPU LU path: every backend's code runs end to end against NumPy stand-ins for the exact CuPy /
PyTorch calls it makes, and must reproduce the CPU mixed-precision solver. (Real hardware: run
tools/gpu_selfcheck.py on the GPU machine.)"""
import os, sys, types
import numpy as np
import pytest
import scipy.linalg as sla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from mom25d.multilayer import Layer, LayerStack
from mom25d.sim import Simulation, Via, ViaPort
import mom25d.gpu as mom25d_gpu
pytest.importorskip('mom25d._core')


# ---- NumPy stand-ins with the semantics of the real libraries ----------------------------------
def fake_cupy(mem=8e9):
    cp = types.ModuleType('cupy')
    pool = types.SimpleNamespace(free_all_blocks=lambda: None)
    cp.get_default_memory_pool = lambda: pool
    cp.array = lambda a, order='K': np.array(a, order=order)
    cp.asarray = np.asarray
    cp.asnumpy = np.asarray

    class Device:
        id = 0
        mem_info = (mem, 16e9)

        def synchronize(self):
            pass
    rt = types.SimpleNamespace(getDeviceCount=lambda: 1, is_hip=True,
                               getDeviceProperties=lambda i: {'name': b'Fake Radeon'})
    cp.cuda = types.SimpleNamespace(runtime=rt, Device=Device)
    calls = {'inplace': None}

    def lu_factor(a, overwrite_a=False, check_finite=True):
        lu, piv = sla.lu_factor(a, overwrite_a=overwrite_a, check_finite=False)
        calls['inplace'] = np.shares_memory(lu, a)
        return lu, piv
    cla = types.ModuleType('cupyx.scipy.linalg')
    cla.lu_factor = lu_factor
    cla.lu_solve = lambda lp, b, trans=0, check_finite=True: sla.lu_solve(lp, b, trans=trans, check_finite=False)
    cupyx = types.ModuleType('cupyx'); scipy_ = types.ModuleType('cupyx.scipy')
    cupyx.scipy = scipy_; scipy_.linalg = cla
    return {'cupy': cp, 'cupyx': cupyx, 'cupyx.scipy': scipy_, 'cupyx.scipy.linalg': cla}, calls


def fake_torch(mem=8e9):
    class T:                                            # minimal tensor: wraps an ndarray
        def __init__(self, a): self.a = a
        def to(self, dev): return T(np.array(self.a))   # "upload" = copy
        @property
        def mT(self): return T(self.a.T)
        def cpu(self): return self
        def numpy(self): return self.a
    torch = types.ModuleType('torch')
    torch.from_numpy = lambda a: T(a)
    torch.device = lambda s: s
    torch.version = types.SimpleNamespace(hip='6.2')
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: 'Fake Radeon',
                                       mem_get_info=lambda: (mem, 16e9), synchronize=lambda: None,
                                       empty_cache=lambda: None)

    def lu_factor(A):                                   # factors A exactly as given (torch semantics)
        lu, piv = sla.lu_factor(np.array(A.a), check_finite=False)
        return T(lu), T(piv)

    def lu_solve(LU, piv, B, adjoint=False):            # A X = B, or A^H X = B
        return T(sla.lu_solve((LU.a, piv.a), B.a, trans=2 if adjoint else 0, check_finite=False))
    torch.linalg = types.SimpleNamespace(lu_factor=lu_factor, lu_solve=lu_solve)
    return {'torch': torch}


# ---- small two-layer model with vias (same as the core/solver agreement test) --------------------
def make_sim(gpu):
    from shapely.geometry import box, Point
    st = LayerStack([Layer(0.2e-3, 4.3, 0.02), Layer(0.2e-3, 4.3, 0.02)])
    w, L = 0.35e-3, 8e-3
    geo2 = [box(-L / 2, -w / 2, L / 2, w / 2), Point(L / 2, 0).buffer(0.25e-3)]
    geo1 = [box(-L / 2, -2e-3, L / 2 + 1e-3, 2e-3).difference(Point(L / 2, 0).buffer(0.5e-3))]
    return Simulation(st, {1: geo1, 2: geo2}, [ViaPort(Via(-L / 2, 0, 1, 2, 0.1e-3)), ViaPort(Via(L / 2, 0, 0, 2, 0.1e-3))],
                      dmax=0.4e-3, dmax_layer={1: 0.6e-3}, w_ref={2: w, 1: 1e-3}, solver='direct',
                      precision='mixed', gpu=gpu, verbose=False)


@pytest.fixture(scope='module')
def S_cpu():
    return make_sim('off').sparams(4e9)


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv('MOM25D_GPU', raising=False)
    mom25d_gpu._cache.clear()
    yield monkeypatch
    mom25d_gpu._cache.clear()


@pytest.mark.parametrize('backend', ['cupy', 'torch', 'emulate'])
def test_gpu_backends_match_cpu(clean, S_cpu, backend):
    calls = {}
    if backend == 'cupy':
        mods, calls = fake_cupy()
    elif backend == 'torch':
        mods = fake_torch()
    else:
        mods = {}
    for k, v in mods.items():
        clean.setitem(sys.modules, k, v)
    sim = make_sim(backend)
    S = sim.sparams(4e9)
    assert any(r[0] == 'LU factor (GPU)' for r in sim.log.records), "GPU path not taken"
    assert np.abs(S - S_cpu).max() < 1e-7
    if backend == 'cupy':
        assert calls['inplace'], "CuPy LU must factor the uploaded matrix in place"


def test_gpu_too_small_falls_back_to_cpu(clean, S_cpu):
    mods, _ = fake_cupy(mem=1e6)                        # 1 MB free
    for k, v in mods.items():
        clean.setitem(sys.modules, k, v)
    sim = make_sim('cupy')
    S = sim.sparams(4e9)
    assert not any(r[0] == 'LU factor (GPU)' for r in sim.log.records)
    assert np.abs(S - S_cpu).max() < 1e-12


def test_auto_without_gpu_is_silent_cpu(clean, S_cpu):
    clean.setitem(sys.modules, 'cupy', None)            # import fails
    clean.setitem(sys.modules, 'torch', None)
    assert mom25d_gpu.get_backend('auto') is None
    S = make_sim('auto').sparams(4e9)
    assert np.abs(S - S_cpu).max() < 1e-12
