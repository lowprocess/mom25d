"""
The loop/tree basis (looptree.py, fast.FastOperator._solve_loop_tree): its loops carry no charge, its transforms
are the basis's matrices, and the direct solver in it is the RWG basis's answer where that holds and stays right
below, where that does not.
"""
import numpy as np
import pytest

from mom25d.multilayer import Layer, LayerStack
from mom25d.sim import Simulation, Via, ViaPort, Metal
from mom25d.looptree import LoopTree, transpose


def _sim(precision='double', loop_tree='auto', metal=None, dmax=0.4e-3):
    """A strip on M2 over a plane on M1 (an antipad round a via to the ground), a via port at each end: one between
    the strip's pad and the plane (two fans), one down to the ground (a fan and the ground); a via from the plane to
    the ground by the first, so its loop closes (at DC: the copper's resistance)."""
    from shapely.geometry import box, Point
    st = LayerStack([Layer(0.2e-3, 4.3, 0.02), Layer(0.2e-3, 4.3, 0.02)])
    w, L = 0.35e-3, 8e-3
    v1 = Via(L / 2, 0, 0, 2, 0.1e-3)
    geo2 = [box(-L / 2, -w / 2, L / 2, w / 2), Point(L / 2, 0).buffer(0.25e-3)]
    geo1 = [box(-L / 2, -2e-3, L / 2 + 1e-3, 2e-3).difference(Point(L / 2, 0).buffer(0.5e-3))]
    stitch = Via(-L / 2, 1.2e-3, 0, 1, 0.1e-3)
    sim = Simulation(st, {1: geo1, 2: geo2}, [ViaPort(Via(-L / 2, 0, 1, 2, 0.1e-3)), ViaPort(v1)], vias=[stitch], dmax=dmax,
                     dmax_layer={1: 0.6e-3}, w_ref={2: w, 1: 1e-3}, solver='direct', precision=precision,
                     metal=metal, verbose=False)
    sim.loop_tree, sim.gpu = loop_tree, 'off'
    return sim


def test_the_loops_carry_no_charge_and_the_transforms_are_the_basis():
    sim = _sim()
    lt = LoopTree(sim)
    N = lt.N
    assert lt.G in lt.bu or lt.G in lt.bv                   # the via to the ground ends on its node
    assert len(lt.tree) + len(lt.cot) == N and len(lt.roots) >= 1
    I = np.eye(N)
    Kt, Mm = lt.from_tree(I), lt.from_ideal(I)
    K = Mm @ Kt
    assert np.linalg.matrix_rank(K) == N
    QK = lt.charge_operator(sim) @ K
    assert np.abs(QK[:, lt.cot]).max() < 1e-12             # loops: no charge on any triangle
    tq = np.abs(QK[:, lt.tree])
    assert np.allclose(np.sort(tq.max(0)), 1.0)            # each tree branch +1 / -1 on its ends
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Z = Z + Z.T
    ref = K.T @ Z @ K
    try:
        import mom25d._core as core
    except ImportError:
        core = None
    for c in {None, core}:
        X = Z.copy()
        lt.to_ideal(X)
        lt.cols(X, c)
        lt.rows(X, c)
        assert np.abs(X - ref).max() < 1e-12 * np.abs(ref).max(), c
    E = rng.standard_normal((N, 2)) + 0j
    assert np.allclose(lt.rows(lt.rhs(E), core), K.T @ E, rtol=0, atol=1e-12 * np.abs(K.T @ E).max())
    A = rng.standard_normal((7, 7))
    assert np.array_equal(transpose(A.copy(), block=3), A.T)


def test_the_charge_block_is_the_tree_s_elastance():
    sim = _sim()
    lt = LoopTree(sim)
    rng = np.random.default_rng(2)
    Mq = rng.standard_normal((lt.Nt, lt.Nt)) + 0j
    Mq = Mq + Mq.T
    a = sim.rwg.area
    A = (lt.charge_operator(sim) @ lt.from_ideal(lt.from_tree(np.eye(lt.N))))[:, lt.tree]
    Phi = A.T @ (Mq / a[:, None] / a[None, :]) @ A
    try:
        import mom25d._core as core
    except ImportError:
        core = None
    for c in {None, core}:
        Z = np.zeros((lt.N, lt.N), complex)
        lt.charge_into(Z, Mq, a, c)
        assert np.abs(Z[np.ix_(lt.tree, lt.tree)] - Phi).max() < 1e-12 * np.abs(Phi).max(), c
        assert not Z[lt.cot].any() and not Z[:, lt.cot].any()


def test_the_loop_tree_solve_is_the_rwg_basis_s_where_that_holds():
    pytest.importorskip('mom25d._core')
    ref = _sim('double', 'off').sparams(1e9)
    for prec in ('double', 'mixed'):
        sim = _sim(prec, 'on')
        S = sim.sparams(1e9)
        assert any('loop/tree basis:' in n for n in sim.log.notes)
        assert np.abs(S - ref).max() < (1e-9 if prec == 'double' else 1e-6), prec


def test_below_the_rwg_basis_s_reach_it_stays_right():
    """Copper loss, 10 Hz: the loop/tree basis's S11 is the copper's DC resistance's, as at 10 kHz, and mixed
    precision converges in it (no double fallback). (The RWG basis there: its double LU 14 % off on this board,
    its complex64 LU no preconditioner.)"""
    pytest.importorskip('mom25d._core')
    cu = Metal(5.8e7, 35e-6)
    S = {}
    for f in (1e4, 10.0):
        for prec in ('double', 'mixed'):
            sim = _sim(prec, 'auto', metal=cu)
            S[(f, prec)] = sim.sparams(f)
            assert sim.fast._basis_name == 'loop/tree'
            assert not any('did not converge' in n for n in sim.log.notes), sim.log.notes
    for k, v in S.items():
        assert abs(v[0, 0].real - S[(1e4, 'double')][0, 0].real) < 1e-3 * abs(S[(1e4, 'double')][0, 0]), k
        assert np.linalg.svd(v, compute_uv=False).max() <= 1 + 1e-9, k


def test_auto_takes_the_loop_tree_basis_below_its_threshold_and_keeps_each_basis_lus():
    pytest.importorskip('mom25d._core')
    from mom25d import fast
    sim = _sim('mixed', 'auto')
    fo = sim.fast
    lo = 0.5 * np.sqrt(fast.LOOP_TREE_BELOW / fo.kh2(1.0))          # (k h)^2 grows as f^2
    hi = 4 * lo
    assert fo._loop_tree_at(lo) and not fo._loop_tree_at(hi)
    sim.sparams(lo)
    kept_lt = list(fo._kept)
    sim.sparams(hi)
    assert fo._basis_name == 'rwg' and all(k not in fo._kept for k in kept_lt)
    S_again = sim.sparams(1.05 * lo)
    assert fo._basis_name == 'loop/tree'
    assert any('reused' in n for n in sim.log.notes)                   # the loop/tree LU kept, used again
    S_fresh = _sim('double', 'on').sparams(1.05 * lo)
    assert np.abs(S_again - S_fresh).max() < 1e-5
