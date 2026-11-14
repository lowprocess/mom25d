"""
mom25d.afs - the adaptive frequency sweep with rational (AAA) interpolation: emsdev.afs, which pcbqs's
S-parameters share; here with what to try when an EM solve gives non-finite S-parameters.
"""
from emsdev import afs as _afs
from emsdev.afs import SMALL, aaa, _is_log                          # noqa: F401 (mom25d's names, as they were)


def adaptive_sweep(solve, fmin, fmax, **kw):
    """emsdev.afs.adaptive_sweep, its hint the GPU's (a non-finite solve has come from it)."""
    kw.setdefault('hint', 'on the GPU: MOM25D_GPU=off solves on the CPU')
    return _afs.adaptive_sweep(solve, fmin, fmax, **kw)
