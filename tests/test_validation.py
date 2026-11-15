"""mom25d against references, and against itself (R-C9): a plane cavity's resonance against the cavity model, the
10 dB coupler (kicad2ems's rf_coupler example) against openEMS's stored answer, and stored S-parameters of two
boards (a change that moves them is seen). On the CPU: the GPU's LU differs from run to run near a port's gap
(its S does not). Stored S again: python tests/test_validation.py --update."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
GOLDEN = HERE / 'data' / 'golden_s.json'
FREQS = [1e9, 3e9, 6e9]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('MOM25D_GPU', 'off')


def _solved(name, freqs, fmin, fmax):
    """S of a board in tests/boards, renormalised to its ports' reference impedances, as solve_board gives it."""
    from mom25d.api import build_board
    from mom25d.run import renorm, _zc, out_refs
    d = json.loads((HERE / 'boards' / f'{name}.json').read_text())
    model, sim = build_board(d, fmin, fmax, log=lambda *a, **k: None)
    return np.array([renorm(sim.sparams(f), _zc(sim, f), out_refs(sim, model.port_z0)) for f in freqs])


def test_plane_cavity_resonance_against_the_cavity_model():
    """A 30 x 20 mm plane 0.2 mm over its ground (er 4.3) fed by a via: its first resonance, the peak of Re Zin,
    against c / (2 (a + 2 dl) sqrt(er)) with Hammerstad's open-end extension dl. +0.71 % at 1 mm cells, +0.90 %
    at the 1.5 mm here (9 s)."""
    from shapely.geometry import box
    from mom25d.base import C0
    from mom25d.multilayer import Layer, LayerStack
    from mom25d.sim import Simulation, Via, ViaPort
    a, b, h, er = 30e-3, 20e-3, 0.2e-3, 4.3
    sim = Simulation(LayerStack([Layer(h, er)]), {1: [box(-a / 2, -b / 2, a / 2, b / 2)]},
                     [ViaPort(Via(-a / 2 + 4e-3, -b / 2 + 4e-3, 0, 1, 0.15e-3), 50.0)], dmax=1.5e-3, verbose=False)
    r = lambda f: (lambda S: (50 * (1 + S) / (1 - S)).real)(sim.sparams(f)[0, 0])
    dl = 0.412 * h * (er + 0.3) * (b / h + 0.264) / ((er - 0.258) * (b / h + 0.8))
    model = C0 / (2 * (a + 2 * dl) * np.sqrt(er))
    lo, hi = 0.995 * model, 1.02 * model
    g = (np.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = r(c), r(d)
    for _ in range(10):                                             # golden-section search for the peak
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo); fc = r(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo); fd = r(d)
    f = (lo + hi) / 2
    assert abs(f / model - 1) < 0.01 and r(f) > 200                  # a resonance, within 1 % of the model


def test_coupler_in_db_against_openems():
    """kicad2ems's 10 dB edge-coupled coupler (tests/boards/rf/coupler.json, its rf_coupler example's board) at
    2.4 GHz against openEMS's answer stored in kicad2ems's examples/reference: S21 -0.75 dB, S31 -10.54 dB, S41
    -27.99 dB. mom25d here: -0.70, -10.62, -28.38."""
    S = _solved('rf/coupler', [2.4e9], 2e9, 3e9)[0]
    db = lambda x: 20 * np.log10(abs(x))
    assert abs(db(S[1, 0]) + 0.75) < 0.15 and abs(db(S[2, 0]) + 10.54) < 0.15 and abs(db(S[3, 0]) + 27.99) < 1.5


@pytest.mark.parametrize('name', ['thru', 'demo'])
def test_stored_s_parameters(name):
    """S of two test boards at 1, 3 and 6 GHz as stored (tests/data/golden_s.json): within 5e-4, which holds over
    BLAS and platforms and catches what moved answers before (A6's log sweep: 1.8; a mesh change: 1e-2)."""
    want = json.loads(GOLDEN.read_text())[name]
    got = _solved(name, FREQS, FREQS[0], FREQS[-1])
    np.testing.assert_allclose(got.real, np.array(want['re']), atol=5e-4)
    np.testing.assert_allclose(got.imag, np.array(want['im']), atol=5e-4)


if __name__ == '__main__' and '--update' in sys.argv:
    os.environ['MOM25D_GPU'] = 'off'
    out = {}
    for name in ('thru', 'demo'):
        S = _solved(name, FREQS, FREQS[0], FREQS[-1])
        out[name] = dict(f_Hz=FREQS, re=S.real.tolist(), im=S.imag.tolist())
        print(name, 'stored')
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(out, indent=1) + '\n')
