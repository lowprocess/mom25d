# Low-frequency copper currents, capacitance and fields

`mom25d.mqs.MQS` meshes explicitly selected copper, vias and terminals. The
default magnetic model solves divergence-free loop currents with sheet DC
resistance and partial inductance. No infinite ground plane or image is added:
include the return copper in `nets`. Geometry inputs and mesh sizes use mm;
frequencies use Hz; field observation coordinates use m.

```python
from mom25d.mqs import MQS

q = MQS(board, nets={'VIN', 'GND', 'SW'}, terminals=terminals,
        region=(xmin, ymin, xmax, ymax), dmax=0.5)
q.source('hot', 'VIN_pad', 'PGND_pad', part_h=0.5)
q.element('Cin', 'Cin_1', 'Cin_2', C=10e-6, ESR=3e-3, ESL=0.4e-9)
r = q.solve([1e5, 1e6, 1e7])
Z = r.z['hot']                    # includes lumped part impedances
Zc = r.zc['hot']                  # copper contribution in the magnetic model
H = q.h_field(r, 0, points_m)     # exact filaments, refined triangle quadrature
```

`r.i` holds branch currents in A, in the order RWG edges, via segments, parts.
`amps={'hot': 2.0}` sets the source current; every source defaults to 1 A.
Sources act simultaneously; their reported impedances include coupling from
the other driven sources. A source set to zero has an undefined impedance
(`nan`) but still participates as an open current-source branch.
The AC solvers require finite, strictly positive frequencies; use `pdn` for DC.

## Integration and speed

The static kernel classifies tiled triangle pairs within `near` times
the sum of their bounding radii (default 3). These pairs use analytic source
integration and a six-point observation rule; very close pairs subdivide the
observation triangle into four (`outer_levels=1`). The remaining pairs use a
three-point product rule. Each unordered pair is evaluated once, retaining
both observation and source moments to enforce reciprocity. Coordinates are
shifted locally before combining RWG moments to reduce cancellation.

Use `near=float('inf')` for the all-analytic reference path. Use
`outer_levels=2` to check close-pair quadrature convergence, and reduce `dmax`
to check mesh convergence. Analytic refers to the inner integral; the outer
integral remains numerical. Resistance assembly and loop projection use sparse
matrices. Resistance stays sparse through branch assembly. Inductance remains
in its copper blocks, without dense padding or duplicate branch copies.
Copper matrices are reused when sources or lumped parts are added. Dense
inductance and triangle-moment storage are still quadratic in mesh size.

Filling defaults to the compiled **OpenMP** backend when available, with a
multithreaded NumPy fallback. `fill_backend='numpy'` selects the serial
reference; `fill_workers` controls CPU threads. The optional CuPy backend and
third-party backends share the same double-precision pair interface. See
[backend configuration and extension protocol](static-backends.md) for thread
defaults, bounded scratch memory, GPU limitations and scaling benchmarks.

The reproducible CPU benchmark is:

```sh
OPENBLAS_NUM_THREADS=2 python tools/benchmark_mqs.py --dmax 0.8
```

On the development environment, a 974-triangle example took 1.20 s for the
hybrid fill versus 2.08 s for the refined all-analytic reference, with a
0.005% relative Frobenius error in inductance. A separate 2,474-triangle
sheet comparison against revision `6be2bc3` took 6.12 s versus 19.26 s
(3.1 times faster, including the more accurate close terms). These are
small synthetic examples, not whole-board benchmarks.

## Fast field planes and volumes

```python
H = q.h_grid(r, [0, 1, 2], x_m, y_m, z_m)
# shape: (nx, ny, nz, 3, 3 frequencies)
H0 = q.h_grid(r, 0, x_m, y_m, z_m)  # (nx, ny, nz, 3)
H, info = q.h_grid(r, [0, 1, 2], x_m, y_m, z_m,
                   rtol=0.02, max_refine=8, return_info=True)
```

`x_m` and `y_m` must be increasing uniform grids; `z_m` contains absolute
heights, including heights below or between the copper layers. Currents are
conservatively deposited, and FFT linear convolution applies the free-space
Biot–Savart kernel without periodic images. The internal source grid expands
to include all copper even when the requested window is small. Filaments
remain analytic. Current spectra and field kernels serve all requested
frequencies/heights. No implicit ground image is used.

Grid spacing should be well below the distance to the nearest sheet; refine
until the field converges. `refine=2` halves the internal deposition spacing
while preserving output coordinates. With `rtol`, successive refinements
must agree within that fraction of the peak component at **each height and
frequency**, or the call raises at `max_refine`. This checks grid convergence,
not mesh or model accuracy. `info` records the refinement and relative changes.
Spectra are batched to bound working memory. The near-field demo uses these
checks for its planes and volume by default.
Evaluation exactly on a sheet is rejected. For
isolated points and checks use `h_field`. In the benchmark above, 3,813 points
at three frequencies took 0.078 s by FFT versus 2.75 s directly (35 times
faster), with 0.38% maximum error relative to the peak reference field.

## Distributed capacitance (PEEC)

```python
q = MQS(board, nets=nets, terminals=terminals, dmax=0.5,
        capacitance=True, er=4.0)
q.source('hot', 'VIN_pad', 'PGND_pad')
r = q.solve([1e3, 1e6])
r.charge                         # (nf, n_nodes), C
r.potential                      # (nf, n_nodes), V relative to infinity
E = q.e_field(r, 0, points_m)     # Coulomb E, V/m
C, node_labels, terminal_labels = q.capacitance_matrix()
```

The default electrostatic model uses a **homogeneous dielectric**, with
explicitly specified real `er` (default 1). The reciprocal 1/R kernel supplies the
Galerkin coefficient-of-potential matrix P. Charge density is constant per
triangle; the triangles grouped into one terminal or via fan share one
charge density. Keep terminal regions local to their physical pads so that
the remaining copper can resolve edge charge and current distribution.

With branch incidence A positive at the starting node and exp(+jwt):

    Z i = A.T P q,    A i + jw q = 0.

The sparse loop/tree formulation separates each passive component's
electrostatic equilibrium and scales the tree-current correction by jw.
This avoids recovering tiny conduction charges by subtraction of nearly
equal currents at low frequency. A source between disconnected pieces of
copper can drive displacement current. Series RLC parts remain available.
`r.zc` is `None` for PEEC; use the full source impedance `r.z`.

PEEC caches its topology, electrostatic preparation and projected R/L matrices
until parts change. Frequency LU factors are reused across excitation changes
within an LRU budget (`factor_cache_bytes`, default 64 MiB; 0 disables storage).
This budget covers factors, not the model's dense matrices. Construct a new
model when changing geometry or material settings. `capacitance_matrix()` has
a scalar-only integration path and does not build magnetic matrices.

`tools/benchmark_peec.py` measured 0.23 s for capacitance-only extraction on
432 triangles, 0.57 s for the first three-frequency solve after extraction, and 0.015 s on a
changed excitation, with no additional factorizations and 17.2 MiB cached.

`capacitance_matrix()` returns the Maxwell capacitance matrix in F, with one
row/column per connected component of copper and vias (lumped parts do not
join components). The two label outputs map mesh nodes and terminal names
to those conductors. Potentials are relative to infinity: row sums need not
vanish. A floating two-conductor capacitance is
`1 / (e @ solve(C, e))`, with `e=[1,-1]`, rather than `C[0,0]`.

`e_field` uses exact edge integrals and solid angles for the constant triangle
charges. It returns only the Coulomb field, excluding inductive `-jw A`.
Observe away from charged surfaces, which have a discontinuous normal field.

## Layered dielectric electrostatics and loss

```python
q = MQS(board, nets=nets, terminals=terminals, capacitance=True,
        dielectric='stack', dielectric_tol=1e-8)
```

`dielectric='stack'` uses the board's contiguous dielectric slabs (`z0`, `z1`
in mm, `er`, `tand`) as laterally infinite materials. Copper must lie on slab
interfaces. Exterior half spaces are air, overridable with `stack.top_er` and
`stack.bottom_er`. Specify `er` only for the homogeneous model. There is no
implicit PEC ground. Image charges account for reflections and transmission
at every dielectric interface, with complex `er*(1-j*tand)` for constant
dielectric loss. Maxwell C is therefore complex in a lossy stack.

Close images retain analytic source integration; smooth image sums are
tabulated and integrated by the product rule with analytic close-pair
corrections. `dielectric_tol` cuts off individual ray amplitudes, not a bound
on the final field error: tighten it and refine the mesh for convergence.
Excessive image growth raises explicitly. High-contrast or many-layer stacks
can be expensive. Layered `e_field` evaluates the transmitted/reflected images
and requires observation points away from dielectric interfaces, where the
normal E is discontinuous. Material frequency dispersion is not yet included.

## Conductor loss

```python
q = MQS(board, nets=nets, terminals=terminals,
        conductor_loss='slab', conductor_faces=1)
```

The default remains `conductor_loss='dc'`. The slab option uses the complex
finite-thickness impedance `u*coth(u)/(sigma*t)`, with
`u=sqrt(j*w*mu0*sigma)*t/conductor_faces`. It includes internal inductance and
the skin-resistance increase, tending to `1/(sigma*t)` at DC. One face assumes
current enters from one side; two faces assume equal currents on both sides.
These are selectable approximations, not a resolved thickness/proximity
solution. Via barrel resistance remains DC. Both MQS and PEEC use this option.

## Validation and limits

Tests cover Wheeler's inductance formula, image-theory magnetic fields and
return currents, hybrid/reference agreement, reciprocity, positive magnetic
energy, coordinate translation, close-integral convergence, FFT/direct
agreement above and below copper, parallel-plate capacitance and E,
permittivity scaling, Maxwell capacitance, lumped/distributed capacitance
in parallel, charge continuity, and the PEEC limit approaching MQS down to
1 mHz. Layered images are checked against the independent spectral network;
tests also cover series dielectrics, normal displacement continuity, positive
dielectric loss and the slab's DC/internal-inductance/skin limits.

Both models are quasi-static: no retardation or radiation. Via
charges and part-body charges are omitted. Via inductance uses parallel
filaments; part/copper mutual inductance is omitted, and raised part fields
use a single horizontal filament. PEEC currently uses a dense branch solve,
so its cost is higher than the loop-only magnetic model. Resolved conductor
proximity effects, compressed whole-board solves, and validation of the complete
model near wavelength/10 remain future work. See the
[CM5 U702 benchmark](benchmarks/cm5-u702.md) for a real cropped converter case.

All three synthetic converter layouts also complete with converged FFT dumps:
the good/far/slot cases took approximately 1.3/4.7/3.4 s for planes and volume,
with 4,140/5,326/5,588 triangles respectively. These runs use the demo's
unchanged meshes, RLC assumptions and source waveform.
