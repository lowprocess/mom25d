"""
mom25d's fast validation suite (about a minute).
Each test checks the solver against a closed-form result or a conservation law.
"""
import numpy as np
import pytest

from mom25d.base import C0, MU0, EPS0
from mom25d.multilayer import Layer, LayerStack, GreenTableML, stripline_z0_cohn
from mom25d.sim import line_char_xs
from mom25d_reference import GreenTable, Stackup                    # the first solver's Sommerfeld integrals

RHO = np.array([1e-5, 1e-4, 5e-4, 2e-3, 8e-3, 15e-3])


def test_single_layer_green_matches_sommerfeld():
    """One layer: the spectral network's kernels against the Sommerfeld integrals of the grounded slab."""
    f = 5e9
    g1 = GreenTable(f, Stackup(0.2e-3, 4.3, 0.02), 20e-3)
    g2 = GreenTableML(f, LayerStack([Layer(0.2e-3, 4.3, 0.02)]), 1, 1, 20e-3)
    A2, Q2 = g2.total(RHO)
    A1, Q1 = g1.cA / RHO + g1.A(RHO), g1.cq / RHO + g1.Q(RHO)
    assert np.max(abs(A2 - A1) / abs(A1)) < 1e-4
    assert np.max(abs(Q2 - Q1) / abs(Q1)) < 1e-4


@pytest.mark.parametrize("o,s", [(1, 1), (1, 2), (3, 2), (1, 3)])
def test_air_stack_matches_exact_images(o, s):
    """er = 1 everywhere: kernels are free-space + one ground image, on and between interfaces."""
    f = 10e9
    k0 = 2 * np.pi * f / C0
    st = LayerStack([Layer(0.1e-3, 1.0), Layer(0.15e-3, 1.0), Layer(0.2e-3, 1.0)])
    g = GreenTableML(f, st, o, s, 20e-3)
    zo, zs = st.z[o], st.z[s]
    R, Ri = np.sqrt(RHO ** 2 + (zo - zs) ** 2), np.sqrt(RHO ** 2 + (zo + zs) ** 2)
    img = np.exp(-1j * k0 * R) / R - np.exp(-1j * k0 * Ri) / Ri
    A, Q = g.total(RHO)
    assert np.max(abs(A - MU0 / (4 * np.pi) * img) / abs(MU0 / (4 * np.pi) * img)) < 2e-4
    assert np.max(abs(Q - img / (4 * np.pi * EPS0)) / abs(img / (4 * np.pi * EPS0))) < 2e-4


def test_stripline_vs_cohn():
    b, er, w = 0.4e-3, 4.3, 0.15e-3
    st = LayerStack([Layer(b / 2, er), Layer(b / 2, er)], top='pec')
    g, z = line_char_xs(st, 1, w, [5e9], ncross=6)
    eps_eff = (g[0].imag / (2 * np.pi * 5e9 / C0)) ** 2
    assert abs(z[0].real / stripline_z0_cohn(w, b, er) - 1) < 0.01      # within 1 %
    assert abs(eps_eff / er - 1) < 0.002                                # TEM: eps_eff = er
    assert abs(g[0].real) < 1e-6                                        # lossless


def test_split_substrate_equals_single_layer():
    w, h = 0.35e-3, 0.2e-3
    st = LayerStack([Layer(h / 2, 4.3, 0.02), Layer(h / 2, 4.3, 0.02)])
    g2, z2 = line_char_xs(st, 2, w, [5e9])                          # the substrate in two layers of one
    g1, z1 = line_char_xs(LayerStack([Layer(h, 4.3, 0.02)]), 1, w, [5e9])
    assert abs(z2[0].real / z1[0].real - 1) < 0.002
    assert abs(g2[0].imag / g1[0].imag - 1) < 0.002
