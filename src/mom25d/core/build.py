"""Build the optional compiled core:  python core/build.py   (needs g++ with OpenMP and pybind11).
The solver falls back to NumPy when the module is absent."""
import os, subprocess, sys, sysconfig
import pybind11
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, '..', 'mom25d_core' + sysconfig.get_config_var('EXT_SUFFIX'))
arch = os.environ.get('MOM25D_ARCH', '-march=native')
cmd = ['g++', '-O3', arch, '-fopenmp', '-shared', '-fPIC', '-std=c++17', '-Wno-attributes',
       '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'],
       os.path.join(here, 'mom25d_core.cpp'), '-o', out]
print(' '.join(cmd)); subprocess.check_call(cmd); print('built', os.path.abspath(out))
