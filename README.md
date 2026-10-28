# mom25d

A 2.5D method-of-moments solver for circuit boards: copper as zero-thickness sheets on the interfaces of a
layered dielectric over an infinite ground plane, vias as vertical wires, lumped parts as series elements.
It takes a board file and gives its S-parameters, and runs on the CPU (NumPy, or a compiled C++ / OpenMP
core) with an optional GPU LU (CuPy or PyTorch, on NVIDIA CUDA or AMD ROCm).

It is the planar solver of [kicad2ems](https://github.com/lowprocess/kicad2ems), which reads KiCad and
Gerber files and writes the board file (`<name>_board.json`) every solver gets. mom25d reads that file
without kicad2ems; kicad2ems's [mom25d page](https://github.com/lowprocess/kicad2ems/blob/main/docs/mom.md)
covers what it models, its ports, the mesh, antenna mode, the GPU paths and its speed against openEMS.

## Install

```bash
pip install .                       # numpy, scipy, shapely, triangle, emsdev (the GPU and memory probes)
pip install '.[core]'               # pybind11, then build the compiled core (g++ with OpenMP):
python -m mom25d.core.build         # mom25d/_core*.so, about five times faster
pip install '.[gpu-rocm7]'          # the GPU LU: the CuPy wheel for your stack (gpu-cuda12, gpu-cuda13, gpu-rocm7)
```

PyPI has no triangle 20230923 or later for every Python (none for 3.14). Where `pip install .` cannot find one,
build it from its repository first (needs a C compiler; kicad2ems's `setup.sh` does this for you):

```bash
pip install "triangle @ git+https://github.com/drufat/triangle.git@595b43eb6682992a0b1012b9671bf860c0e6ae56"
```

`MOM25D_GPU=off` turns the GPU off. Consumer AMD cards outside ROCm's support list may need
`HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3) or `10.3.0` (RDNA2).

## Use

```python
import numpy as np
from mom25d.api import solve_board

res = solve_board('coupler_board.json', np.linspace(1e9, 6e9, 101), dmax=0.3e-3)
res.f, res.S, res.names, res.zref                  # Hz, (nf, n, n), port names, reference impedances
```

`solve_board` builds the model, meshes it and runs an adaptive frequency sweep. Its options (cell size,
port kind, mixed-mode pairs, conductor loss, the GPU's tiled solver, the compiled core) are in its
docstring. The steps it wraps are public too:

```python
from mom25d.board import BoardModel                # the board file as a planar model
from mom25d.run import solve_band                  # the adaptive sweep over a Simulation
m = BoardModel(board_dict)
sim = m.simulation(dmax=0.3e-3, fmin=1e9, fmax=6e9)
f, S, info = solve_band(sim, 1e9, 6e9, points=101, zref=m.port_z0)
```

## Inside mom25d

| Module | Contents |
|---|---|
| `api.py` | `solve_board`: a board file in, S-parameters out |
| `board.py` | the board file as a planar model |
| `model.py` | stack, metal, vias, ports, port characterisation |
| `sim.py` | `Simulation`: mesh, ports, parts, solver choice (`memory_plan`) |
| `multilayer.py` | layered-media Green's functions |
| `base.py` | RWG basis functions, the 1/R moment integrals, the port wave fits |
| `via.py` | via kernels |
| `geo.py` | the mesher (Triangle, with offset rings along the copper edges) |
| `line.py`, `quads.py` | line basis functions on straight runs; quad cells |
| `fast.py` | the matrix fill and the solvers |
| `gpu.py`, `ldlt.py` | the GPU backends and the tiled solver |
| `blr.py` | the block low-rank solver |
| `afs.py`, `run.py` | the adaptive frequency sweep |
| `farfield.py` | far and near fields from the currents |
| `core/` | the compiled C++ kernels |
| `kicad2ems/` | kicad2ems's solver plugins, `mom25d` and `mixed` (the `kicad2ems.solvers` entry points; only kicad2ems imports them) |

The formulation is a mixed-potential integral equation with RWG basis functions. Layered-media kernels
come from a spectral transmission-line network. The singular terms are integrated analytically.

The tests check it against exact solutions and textbook formulas: layered Green's functions to about 1e-5,
stripline impedance to 0.6 %, coupled-line couplers to 0.1 dB, plane cavity resonances to 0.7 %. The board
files they read (`tests/boards`) are kicad2ems's example boards. `tools/` holds the benchmarks and the mesh
studies; some of them read boards through kicad2ems.

**Limits**: copper is zero-thickness, there is no surface-roughness model, and wave ports need a straight
feed line.

## Licence

GPL-3.0-or-later.
