"""Build the optional compiled core mom25d._core:  python -m mom25d.core.build   (needs g++ with OpenMP and pybind11;
also: python src/mom25d/core/build.py in a checkout).
The solver falls back to NumPy when the module is absent."""
from emsdev.build import build_core

build_core(__file__, 'mom25d_core.cpp', arch_env='MOM25D_ARCH')
