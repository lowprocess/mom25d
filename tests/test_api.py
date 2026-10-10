"""mom25d.api.solve_csx: the one call a program (kicad2openems) makes - the same answer as the steps it wraps."""
import numpy as np
import pytest
from test_csx import write_csx


def test_solve_csx(tmp_path):
    from mom25d.api import solve_csx
    from mom25d.csx import CSXModel
    from mom25d.run import solve_band
    path = write_csx(tmp_path / 'g.xml', hints=True, line=16.0, sheet=True)
    freq = np.linspace(2e9, 6e9, 21)
    res = solve_csx(path, freq, dmax=0.8e-3, max_solves=8, log=lambda *a: None)
    assert res.S.shape == (21, 2, 2) and res.names == ['IN1', 'IN2'] and res.zref == [50.0, 50.0]
    assert [p['kind'] for p in res.ports] == ['wave', 'wave'] and all(p['zc'] > 0 for p in res.ports)
    assert res.unknowns > 50 and res.triangles > 30 and res.solves <= 8 and len(res.solved) == res.solves
    assert res.sim.metal is not None and res.sim.metal.sigma == 5.8e7      # the file's conducting sheet
    assert (np.linalg.svd(res.S, compute_uv=False).max(1) < 1 + 1e-6).all()
    m = CSXModel(path, f_loss=4e9, log=lambda *a: None)                     # the same by hand
    sim = m.simulation(dmax=0.8e-3, fmin=2e9, fmax=6e9, verbose=False, wave_down_to=2e9, solver='auto')
    f, S, _ = solve_band(sim, 2e9, 6e9, points=21, zref=m.port_z0, max_solves=8, afs_tol=5e-3, log=lambda *a: None)
    assert np.allclose(f, freq) and np.abs(S - res.S).max() < 1e-9
    lossless = solve_csx(path, freq, dmax=0.8e-3, max_solves=8, metal=None, log=lambda *a: None)
    assert lossless.sim.metal is None
    assert np.abs(lossless.S[:, 1, 0]).mean() > np.abs(res.S[:, 1, 0]).mean()   # copper loss costs transmission
