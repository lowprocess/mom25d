# mom25d — a 2.5D Method-of-Moments solver for PCB structures

A planar (2.5D) full-wave MoM electromagnetic solver in Python (NumPy/SciPy, shapely, triangle),
aimed at signal-integrity and RF structures on multilayer boards: traces, couplers, via
transitions, planes, CPWG. It lives in this repository as the package `mom25d/`, next to kicad2ems,
and is one of kicad2ems's solver plugins ([solvers](usage.md#solvers)).

mom25d reads kicad2ems's board file (`<name>_board.json`, format `kicad2ems-board`) and imports nothing
from kicad2ems, so it also works on its own given a board file. KiCad and Gerber files reach it through
kicad2ems, which converts the board once for every solver; it needs no openEMS.

![Coupler dashboard](img/coupler_dashboard.png)

## Quick start

```bash
pip install -e '.[core]'                                    # kicad2ems and mom25d, one install
python -m mom25d.core.build                                 # optional C++/OpenMP core (needs g++, pybind11): ~5x faster
python -m pytest tests -q -k mom25d                         # mom25d's own tests

python examples/make_demo_board.py examples/demo_transition # an example board (no KiCad needed)
kicad2ems examples/demo_transition.toml --solvers mom       # the board solved by mom25d alone
```

Without the compiled core the solver runs on NumPy and says so in the log (`note: mom25d ... no compiled
core`). In kicad2ems, mom25d's settings are `--mom-dmax`, `--mom-port`, `--mom-mesher`, `--mom-solves` and
`--mom-tiled`, or `dmax`, `port`, `mesher`, `solves` and `tiled` under `[solver.mom25d]` in a config ([options](usage.md#options)).

![KiCad via transition](img/kicad_transition_dashboard.png)

*Example: the demo transition board: F.Cu microstrip → blind via → In2.Cu stripline, In1.Cu GND plane
with antipad, GND via fence; 8.7k unknowns, 19 EM solves for a 401-point sweep (adaptive), `direct`
solver. Wave-port separation is reliable above ~1.4 GHz for this feed length (the solver warns below).*

## From a board file

```python
import numpy as np
from mom25d.api import solve_board
res = solve_board('out/demo_transition_board.json', np.linspace(0.5e9, 10e9, 401), dmax=0.3e-3)
res.S, res.names, res.zref, res.ports, res.notes        # the stable fields; res.model / res.sim behind them
```

The same by steps, for more control:

```python
import json
from mom25d.board import BoardModel
from mom25d.run import solve_band
m = BoardModel(json.load(open('out/demo_transition_board.json')))
sim = m.simulation(dmax=0.3e-3, fmin=0.5e9, fmax=10e9, wave_down_to=0.5e9)
f, S, info = solve_band(sim, 0.5e9, 10e9, points=401, zref=m.port_z0)
```

What the planar model takes from the board (`mom25d.board`):

* **Ground.** The board's ground layer (kicad2ems `--ground`), else the lowest copper layer covering half
  the outline: the solver's infinite plane at z = 0 (not meshed). Copper below it is left out, with a note.
* **Stack and copper.** The dielectric layers above the ground, copper per interface (planes and pours
  are meshed metal). The board outline does not exist for MoM: its dielectrics are laterally infinite.
* **Vias** between the interfaces they span that have metal at the via (the ground counts).
* **Edge walls** as rows of vias every 0.5 mm: a planar solver has no vertical sheets.
* **Parts** as one series R / L / C element across the gap between the pads, the gap bridged by a strip.
* **Ports.** The board's vertical ports: de-embedded wave ports where a straight feed allows (`port =
  'auto'`), else lumped ports to the reference copper. A differential port (`P1+` / `P1-` in KiCad) is one
  mixed-mode port with differential (`P1d`, Z = 2·Z_odd) and common (`P1c`, Z = Z_even/2) modes, both from
  a coupled-line modal fit. A gap port (a dipole or slot feed) has no planar equivalent: kicad2ems does
  not hand such a board to mom25d.

Each wave port's reference impedance is computed from its cross-section (trace plus the metal covering
it) and the result is renormalised to the port's impedance.

## Meshers

`Simulation(mesher=...)` / `--mesher`: `polygon` (Triangle: constrained Delaunay with inward offset rings at
the conductor edges; the default, `tensor` for Manhattan layers), `gmsh` / `gmsh-rings` / `gmsh-graded`
(`pip install gmsh`). A study on a coupled-line filter
(`tools/mom25d/mesher_study.py`, [results](../tools/mom25d/results/), plot in `docs/img/mesher_study.png`): the edge
resolution (`ncross`) sets the accuracy, anisotropic edge cells (Triangle's rings) beat isotropic meshes
(gmsh-graded, and DistMesh, since removed) 2-4x in error at a third of the unknowns, and gmsh brings no gain per
unknown.
`tools/mom25d/mesher_gallery.py` (`docs/img/mesher_gallery.png`) shows the meshes, the current and S21 side by side.
Details: [mom.md](mom.md#meshers).

## GPU acceleration (AMD ROCm or NVIDIA CUDA)

The mixed-precision direct solver can run its LU factorisation and preconditioner solves on a GPU
in complex64. The matrix fill and the double-precision refinement stay on the CPU, so results
match the CPU solver (|ΔS| < 1e-7 in the tests). Two backends, tried in this order:

| Backend | Install | VRAM for N unknowns | 16 GB card |
|---|---|---|---|
| CuPy (hipSOLVER / cuSOLVER) | `pip install cupy-rocm-7-0` (AMD, ROCm 7), `cupy-cuda12x` or `cupy-cuda13x` (NVIDIA): one of them (or this package's extras `gpu-rocm7`, `gpu-cuda12`, `gpu-cuda13`) | 8 N² B (LU in place); tiled: 4 N² B | ~41k; tiled ~60k (more streamed from the host) |
| PyTorch (ROCm or CUDA build) | the ROCm wheel command from pytorch.org | 16 N² B | ~29k |

```bash
python -c "from mom25d.gpu import describe; print(describe())"     # what was detected
python tools/mom25d/gpu_selfcheck.py                                # accuracy + speed check on your card
```

The GPU is used automatically when a backend is installed and the LU fits in free
VRAM; otherwise the CPU LU runs with a note in the log.

**Which path** (`mom25d.sim.memory_plan`, logged as `memory: direct solver needs …`):

| path | when | host memory | GPU memory |
|---|---|---|---|
| double | fits in half the free RAM, no GPU (or N ≤ 3.5k) | 18 N² B | – |
| hybrid | a GPU where the complex64 LU fits, and 24 N² B on the host | 24 N² B (the double Z for the refinement) | 8 N² B |
| tiled | CuPy, where the hybrid does not fit (`tiled='on'`: always; `'off'`: never) | a block column, the sparse near field (+ the panels past the GPU) | 4 N² B (or what there is: the rest is streamed) |
| gpu-lu / cpu-mixed | the older complex64 paths, refined matrix-free | 10 N² B | 8 N² B / – |

The tiled path (`ldlt.py`) never forms Z: the matrix is symmetric, so the GPU keeps the lower half as
complex64 L Lᵀ panels, filled on the host one block column at a time (left-looking: each column updated by
the earlier panels, its diagonal block factored on the host in double), panels past the free VRAM held on the
host and streamed in. No pivoting: the equilibrated MoM matrix factors without (smallest pivots 2e-3 to 6e-3
on the examples). The near-field correction is not exactly symmetric (its singular parts are integrated
analytically over one triangle of a pair, by quadrature over the other: up to 0.07 of the unit diagonal
between a plane and a strip triangle), so the tiled path solves the symmetric part (Z + Zᵀ) / 2: |ΔS| ≤
1.2e-3 against Z on the SMA launch (1-6 GHz), 2.7e-5 on the coupler; preconditioning Z itself with the factor
of its symmetric part did not converge. Where the hybrid fits it is faster (SMA launch, 16.1k unknowns: 9.9 s
against 14.3 s a solve; coupler at 0.15 mm, 20.3k: 9.4 s against 12.4 s), so `tiled='auto'` takes the tiled
path only where the hybrid does not fit. What it is for: the coupler at `dmax` 0.1 mm (45,850 unknowns,
21 GB of RAM as a dense double Z) solves in 62 s on a 16 GB card (factor 44 s, 8.6 GB of panels; refinement 6-7
iterations, 13.5 s) where the CPU took over an hour.

Two vias in one place make the via block singular: edge walls that meet at a corner used to put a via at the
shared end twice (4 exact null vectors on the SMA launch), which a complex64 LU cannot factor (the mixed
refinement stalled and the sweep fell back to double). `board.wall_points` leaves out a wall via within 0.1 mm
of one already placed; the SMA launch sweep with the GPU now takes 159 s (14 solves, refinement 2-11
iterations) against 448 s in double, |ΔS| ≤ 3e-5 at the solved frequencies. `MOM25D_GPU=cupy|torch|off` forces a backend or disables it. AMD notes:
check the card with `rocminfo`; consumer RDNA3 cards other than the 7900 series usually need
`HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA2: `10.3.0`).

## What's inside

The package is `mom25d/` at the repository root (`import mom25d.sim`, …); its tests are `tests/test_mom25d_*.py`,
its tools `tools/mom25d/`, its examples `examples/mom25d/`.

| Module | Contents |
|---|---|
| `base.py` | Stage 1: microstrip MPIE/RWG, analytic singular integrals, port extraction |
| `multilayer.py` | Layer stacks (open / PEC cover), TL-network layered Green's functions, singular and nearest-image extraction, tensor mesher |
| `geo.py` | Polygon mesher (constrained Delaunay, edge-graded rings, port lines, vias, regional refinement), `LinePort` |
| `via.py` | Vertical-current kernels in layered media (via–strip, via–via) |
| `sing.py` | Analytic off-plane 1/R integrals (thin layers, two-sheet conductors) |
| `sim.py` | `Simulation`: polygons, vias, wave/gap/via ports, lumped R-L-C, metal loss, thick conductors, solvers |
| `fast.py` | Fast operator: far/near split, `direct` / `mixed` / `blr` / `aca` solvers |
| `gpu.py` | GPU LU backends (CuPy / PyTorch, ROCm or CUDA) for the mixed direct solver |
| `ldlt.py` | Tiled complex-symmetric L Lᵀ (complex64) on the GPU, panels streamed from the host past its memory |
| `blr.py` | Block low-rank LU (tile LU, SVD compression) used as a GMRES preconditioner |
| `core/mom25d_core.cpp` (`mom25d._core`) | C++/OpenMP kernels via pybind11: matrix fill, near correction, matrix-free matvec |
| `afs.py` | Adaptive frequency sweep (set-valued AAA rational fitting) |
| `model.py` | `PlanarModel`: stack, metal, vias, port definitions → wave / lumped ports, port characterisation, `Simulation` |
| `board.py` | kicad2ems's board (`<name>_board.json`) → `BoardModel` (a `PlanarModel`) |
| `run.py` | `solve_band`: adaptive / linear sweep, renormalisation to the output references |
| `api.py` | `solve_board(board, freq)`: a board file (or its content) in, a `Result` out (S, port names and kinds, unknowns, notes) - the call kicad2ems makes |
| `mesh_gmsh.py` | the gmsh polygon mesher, same contract as `geo.mesh_polygons` |
| `viz.py` | Dashboard: stackup, 3D view, mesh/ports/vias, currents, S-params, numerical health, kernels, timing |

**Formulation.** Mixed-potential integral equation with RWG basis functions; layered-media
kernels from a spectral transmission-line network (TE/TM), Sommerfeld integrals on a deformed
contour. Singular (1/ρ), near-singular cross-layer (1/√(ρ²+d²)) and nearest-image terms are
extracted and integrated analytically. Vias are thin wires with constant current per layer span
("chains"); their end charges are spread over the pad's vertex fan so they cancel the pad's own
charge exactly. Lumped elements are rank-one updates across a cut line or a via.

**Solvers.** `dense` (reference), `direct` (dense LU of the fast operator: centroid far field +
exact sparse near field; memory ≈ Z only), `mixed` (Z and LU in complex64, refined to double
accuracy by GMRES on the exact matrix-free operator; half the memory, |ΔS| ~ 1e-10), `blr`
(block low-rank LU preconditioner + exact matrix-free GMRES; lowest memory), `aca` (H-matrix +
GMRES). `auto` picks by size: dense ≤ 3.5k, mixed ≤ 14k, then blr.

**Performance** (cropped demo board, 8.3k unknowns, one frequency, 4 threads):

| | time | peak memory |
|---|---|---|
| pure NumPy, double | 150 s | 4.05 GB |
| C++ core, double (`direct`) | 28 s | 2.35 GB |
| C++ core, `mixed` | 18.6 s | 1.51 GB |
| C++ core, `blr` | 56–78 s | 1.00 GB |

Kernels moved to C++: far-field fill 62 s → 1.0 s, near correction 21 s → 2.3 s. The remaining time
is the LU (LAPACK, already threaded). Set `MOM25D_NO_CORE=1` to force the NumPy path.

## Validation (all in `tests/`)

| Check | Result |
|---|---|
| Layered Green's functions vs exact images (on/between interfaces) | ~1e-5 |
| Via kernels vs exact image solution (air, homogeneous dielectric) | ~1e-6 |
| Stripline Z0 vs Cohn / ε_eff | 0.6 % / 0.01 % |
| Broadside coupler vs coupled-line theory; power balance (lossless) | ~0.1 dB; 1 − 1e-9 |
| Series R / C in a line vs circuit theory | < 0.5 % |
| Via to ground vs Goldfarb–Pucel | +10 % (current crowding from the wide line, expected) |
| Two-sheet thick conductor vs 2D finite-thickness solver | 85–95 % of the thickness effect |
| Power plane cavity resonances vs cavity model (ACA + AFS, 17 solves) | within 0.7 % |
| Gerber round trip vs native KiCad import (kicad2ems's importers) | the same board |
| Coupled stripline Z_even / Z_odd vs Cohn | < 2.5 % |
| Straight differential line: \|Sdd21\|, mode conversion | 1 ± 2e-3; Scd < 1e-3 |
| C++ core vs NumPy; mixed / BLR vs double | 1e-10 / ~1e-9 |

## Limits and next steps

* Port de-embedding needs a straight feed of ≳ λ/8 at the lowest frequency; cuts must stay
  clear of discontinuities (the solver warns when the wave fit degrades).
* Thick conductors are an optional two-sheet model (≥ ~18 µm validated); default is zero thickness.
* Iterative (ACA) convergence slows near plane-pair resonances; `mixed` is the robust choice up
  to ~15k unknowns (≈ 3 GB at 12k); `blr` goes further on memory. Beyond ~50k unknowns a
  hierarchical LU (or STRUMPACK/MUMPS BLR) would be the next step.
* Not yet: dielectric/copper roughness models, KiCad custom-pad primitives
  beyond polygons/circles/lines, Calderón-type preconditioning.
