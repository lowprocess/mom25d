"""Finite-thickness slab impedance, exp(+jwt), including internal inductance.

`faces=1` assumes current enters from one face (field zero at the other);
`faces=2` assumes equal current on both faces. These are slab approximations,
not a solution for arbitrary through-thickness proximity currents or barrels.
"""
import numpy as np

from .base import MU0


def surface_impedance(f, thickness, sigma=5.8e7, faces=2):
    """Ohms per square; DC limit 1/(sigma*t), high-frequency skin impedance.
    f is Hz, thickness m. Returns complex values, including internal L.
    """
    if faces not in (1, 2) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('faces must be 1 or 2 and sigma finite and positive')
    f, thickness = np.asarray(f, float), np.asarray(thickness, float)
    if not np.isfinite(f).all() or (f < 0).any() or not np.isfinite(thickness).all() or (thickness <= 0).any():
        raise ValueError('frequency must be nonnegative and thickness finite and positive')
    u = np.sqrt(2j * np.pi * f * MU0 * sigma) * thickness / faces
    small = np.abs(u) < 1e-3
    safe = np.where(small, 1.0, u)
    ratio = np.where(small, 1 + u*u/3 - u**4/45, safe / np.tanh(safe))
    return ratio / (sigma * thickness)
