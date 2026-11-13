"""
The compiled block low-rank LU (core BLRFactor, fast._CompiledBLR): the preconditioner of solve_blr, against
blr.py's and the direct solver.
"""
import types

import numpy as np
import pytest

from mom25d.multilayer import Layer, LayerStack
from mom25d.sim import Simulation, Via, ViaPort


def _sim(solver='blr', dmax=0.25e-3):
    """A strip on M2 over a plane on M1 (an antipad round a via to the ground), a via port at each end."""
    from shapely.geometry import box, Point
    st = LayerStack([Layer(0.2e-3, 4.3, 0.02), Layer(0.2e-3, 4.3, 0.02)])
    w, L = 0.35e-3, 8e-3
    v1 = Via(L / 2, 0, 0, 2, 0.1e-3)
    geo2 = [box(-L / 2, -w / 2, L / 2, w / 2), Point(L / 2, 0).buffer(0.25e-3)]
    geo1 = [box(-L / 2, -2e-3, L / 2 + 1e-3, 2e-3).difference(Point(L / 2, 0).buffer(0.5e-3))]
    sim = Simulation(st, {1: geo1, 2: geo2}, [ViaPort(Via(-L / 2, 0, 1, 2, 0.1e-3)), ViaPort(v1)], dmax=dmax,
                     dmax_layer={1: 0.6e-3}, w_ref={2: w, 1: 1e-3}, solver=solver, precision='double',
                     verbose=False)
    sim.gpu, sim.loop_tree = 'off', 'off'
    return sim


def _without_compiled_blr(sim):
    core = sim.fast.core
    sim.fast.core = types.SimpleNamespace(**{k: getattr(core, k) for k in dir(core)
                                             if not k.startswith('_') and k != 'BLRFactor'})


def test_the_compiled_blr_solves_as_blr_py_and_the_direct_solver():
    core = pytest.importorskip('mom25d._core')
    if not hasattr(core, 'BLRFactor'):
        pytest.skip('a core without BLRFactor')
    f = 4e9
    ref = _sim('direct').sparams(f)
    S, stats = {}, {}
    for compiled in (True, False):
        sim = _sim()
        sim.blr_tile, sim.blr_tol, sim.blr_eta = 128, 1e-3, 0.7
        if not compiled:
            _without_compiled_blr(sim)
        S[compiled] = sim.sparams(f)
        stats[compiled] = sim.fast.blr.stats
        assert not any('did not converge' in n for n in sim.log.notes), sim.log.notes
    assert np.abs(S[True] - ref).max() < 1e-6 and np.abs(S[False] - ref).max() < 1e-6
    st, N = stats[True], ref.shape and _sim().Ne + _sim().Nv
    assert st['lowrank'] > 0 and st['singular'] == 0
    assert st['lowrank'] == stats[False]['lowrank']                 # the same tiles compressed
    assert st['bytes'] < 8.0 * N * N                                # less than a complex64 matrix


def test_a_smaller_eta_compresses_more_tiles_to_the_same_answer():
    """The admissibility (gap > eta x the larger cluster's diameter): a smaller eta compresses tiles nearer each
    other, the answer the same (GMRES's). (Compressing the tiles beside the diagonal too, eta < 0, leaves no
    preconditioner on tiles of 128: GMRES did not converge in 500 iterations.)"""
    core = pytest.importorskip('mom25d._core')
    if not hasattr(core, 'BLRFactor'):
        pytest.skip('a core without BLRFactor')
    f = 4e9
    ref = _sim('direct').sparams(f)
    low = {}
    for eta in (0.7, 0.3):
        sim = _sim()
        sim.blr_tile, sim.blr_tol, sim.blr_eta = 128, 1e-3, eta
        S = sim.sparams(f)
        assert np.abs(S - ref).max() < 1e-6, eta
        assert not any('did not converge' in n for n in sim.log.notes), sim.log.notes
        low[eta] = sim.fast.blr.stats['lowrank']
    assert low[0.3] > low[0.7]


def test_the_factor_inverts_the_matrix_to_its_tolerance():
    """M Z ~ I: the compiled factor applied to Z x gives x back, as near as the compression and complex64 allow
    (this Z's condition number is ~2e6: 1e-2 at any tolerance below 1e-6)."""
    core = pytest.importorskip('mom25d._core')
    if not hasattr(core, 'BLRFactor'):
        pytest.skip('a core without BLRFactor')
    from mom25d.blr import cluster_tiles
    from mom25d.fast import _CompiledBLR
    sim = _sim()
    fo = sim.fast
    f = 3e9
    fo.setup(f, mode='sparse')
    Sx = fo.extra_blocks(f)
    Ne = sim.Ne
    dZx = (fo.dZ + Sx[:Ne, :Ne]).tocsr()
    tiles, boxes = cluster_tiles(fo._points(), 128)
    raw = fo._raw_entry(dZx)
    N = Ne + sim.Nv
    allu = np.arange(N)
    Z = raw(allu, allu)
    dsc = 1.0 / np.sqrt(np.abs(np.diagonal(Z)))
    Zs = Z * dsc[:, None] * dsc[None, :]
    for tol, err in ((1e-3, 0.5), (1e-6, 0.05)):
        blr = _CompiledBLR(fo, tiles, boxes, dZx, dsc, tol, 0.7)
        x = np.random.default_rng(3).standard_normal((N, 2)) + 0j
        y = blr.solve(Zs @ x)
        assert np.linalg.norm(y - x) / np.linalg.norm(x) < err, tol


def test_a_factor_too_loose_to_precondition_is_made_again_tighter(monkeypatch):
    """A factor GMRES stalls with (fast.BLR_STALL) is factored again at a hundredth of the tolerance, kept for the
    frequencies after; the answer the direct solver's."""
    core = pytest.importorskip('mom25d._core')
    if not hasattr(core, 'BLRFactor'):
        pytest.skip('a core without BLRFactor')
    from mom25d import fast
    monkeypatch.setattr(fast, 'BLR_STALL', (4, 1e-3))
    ref = _sim('direct').sparams(4e9)
    sim = _sim()
    sim.blr_tile, sim.blr_tol = 128, 0.3
    S = sim.sparams(4e9)
    assert np.abs(S - ref).max() < 1e-6
    assert any('factored again at 0.003' in n for n in sim.log.notes)
    assert sim.fast._blr_tol < 1e-3
