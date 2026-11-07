# Quasi-static matrix-fill backends

The magnetic and homogeneous-capacitance fills use bounded tiles of unordered
triangle pairs. Layered capacitance uses the same pair integrators for direct
and image terms, with independent observation rows. All paths use double
precision, the same near/far criterion, and analytic source integration for
close pairs. The solver equations and reciprocal matrix assembly are common
to CPU and device backends.

```python
q = MQS(board, nets=nets, terminals=terminals,
        fill_backend='openmp', fill_workers=8, fill_chunk=16384)
r = q.solve(freqs)
print(q.fill_info)  # backend, worker counts, precision, timing, pair counts
```

| Backend | Execution |
|---|---|
| `auto` | OpenMP if the updated core is built; otherwise threaded NumPy |
| `openmp` | GIL-free C++ pair batches with explicit OpenMP team sizes |
| `threaded` | NumPy integration in a bounded `ThreadPoolExecutor`; ufuncs release the GIL |
| `numpy` | Serial NumPy reference; always one worker |
| `cupy` | Optional CuPy GPU integration; one geometry upload per fill, host matrices |

Build the OpenMP backend with:

```sh
pip install '.[core]'
python -m mom25d.core.build
```

An older core without the static kernels is treated as unavailable for this
fill, while remaining usable by the full-wave solver. Explicit selection of
an unavailable backend raises; execution failures never silently retry on a
different backend. Auto does not probe or initialize GPUs. `MOM25D_NO_CORE`
disables automatic compiled-core selection; explicit `openmp` still requests
it, matching the full-wave solver's explicit compiled selection semantics.

## Threads, memory and determinism

`fill_workers` sets the CPU parallelism. Without it, precedence is
`MOM25D_FILL_THREADS`, `MOM25D_OMP_THREADS`, `OMP_NUM_THREADS` (the first team
size), then CPU affinity constrained by Linux cgroup v2 quotas, or CPU count
on platforms without affinity. Explicit counts can exceed that default.
`MOM25D_FILL_BACKEND` selects a backend when the API requests auto.

```sh
MOM25D_FILL_BACKEND=openmp MOM25D_FILL_THREADS=8 python tools/nearfield_demo.py out/
```

The free-space OpenMP fill runs one host coordinator and a native OpenMP
team per pair batch. Layered electrostatics instead parallelizes independent
rows, with **one native worker per row**, avoiding nested thread pools.
Threaded NumPy parallelizes tiles/rows. No global OpenMP or BLAS thread limits
are changed. BLAS solves and field FFTs retain their separate controls.

`fill_chunk` bounds pairs per tile, not triangle count. The scheduler keeps
at most one task/result per host worker in flight. Integration scratch is
bounded per worker; refined NumPy/CuPy batches are subdivided further to bound
quadrature temporaries. Dense host matrices still require O(Nt²) storage.

Each pair has exactly one owner. Independent pair reductions retain their
summation order; one host writer fills both reciprocal entries. Tests check
bitwise repeatability across tile sizes and thread counts within a backend,
and numerical agreement between implementations. OpenMP/libm and GPU math
can differ from NumPy at floating-point roundoff, so cross-backend bitwise
identity is not required.

`q.fill_info` distinguishes total requested workers, native `kernel_workers`,
and host `tile_workers`. OpenMP also reports the team size observed in a
runtime probe; OpenMP dynamic limits can constrain teams. Capacitance metadata
notes when magnetic moments were reused. Timings cover fill preparation and
moment assembly, excluding RWG projection and the linear solve.

## Optional GPU fill

Install the CuPy extra matching your driver, then select it explicitly:

```python
q = MQS(board, nets=nets, terminals=terminals,
        fill_backend='cupy', fill_device=0, fill_chunk=16384)
```

The backend uploads geometry once and evaluates both near analytic integrals
and far product quadrature on the selected device. A fused CUDA/HIP kernel
assigns one thread per pair, keeping observation reductions local and avoiding
intermediate quadrature arrays and chains of small CuPy operations. Only bounded pair inputs
and float64 results cross the device boundary; there is no dense GPU matrix.
Each batch synchronizes before host assembly. CPU worker counts do not create
multiple GPU submitters. Initialization, allocation and kernel errors propagate.

This is an **experimental optional backend**: its formulas are independently
checked on a host array namespace, and hardware tests run when CuPy and a
working device are present. Local validation on an RX 7800 XT with CuPy 14.2.0/ROCm now covers the
real device path, all close-quadrature levels, magnetic/scalar fills and MQS,
PEEC and electrostatic solver tests. Other device/driver combinations and
consumer-device FP64 throughput still need target-hardware measurements.
Selecting a GPU fill does not move RWG projection or the linear solve to GPU.

## Add a backend

Register a factory, or expose it through the `mom25d.static_backends` Python
entry-point group. Providers are imported only when selected. Names cannot
replace built-ins or already registered providers.

```python
from mom25d.static_backends import register_fill_backend
register_fill_backend('my-device', make_backend)
```

The factory signature is `make_backend(geometry, workers, chunk, device)`.
`geometry` is a `FillGeometry` containing read-only local nodes, topology,
heights, centroids, radii, triangle areas and prepared quadrature/edge frames.
Keep it alive while using its arrays; device providers may upload them once.

The returned object implements:

```python
parallel_tiles = False  # True only for safe concurrent, internally serial calls

def evaluate(p, q, d, kind, scalar_only):
    # p, q: int32 triangle indices; d: vertical offsets (m)
    # kind: -1 = far three-point product rule; 0/1/2 = analytic
    #       source integration, with that observation refinement level
    # return host numpy float64 array, fully synchronized:
    # scalar_only: (npairs, 1), M0
    # otherwise:   (npairs, 6), M0, Mro_x, Mro_y, Mrs_x, Mrs_y, Mrr
    ...
```

Providers own their execution resources and device synchronization. They must
not mutate geometry or launch uncontrolled nested CPU teams. The engine
validates output shape, dtype and finiteness. Backend exceptions propagate;
the scheduler cancels queued work and waits for running tasks before returning
the error. A provider can implement CUDA, ROCm, Vulkan or another execution
system without changing MQS/PEEC's equations or matrix assembly.

## Reproduce scaling and accuracy checks

```sh
OPENBLAS_NUM_THREADS=1 python tools/benchmark_static_fill.py --dmax .5 --repeat 3 --out fill.json
OPENBLAS_NUM_THREADS=2 python tools/benchmark_converter.py CM5_MINIMA_3.kicad_pcb \
  --fill-backend openmp --fill-workers 2 --out cm5-openmp.json
```

The static benchmark reports median/sample times and moment errors against
the serial NumPy reference. `--scalar` benchmarks capacitance-only filling.
Device measurements can use `--backends numpy,cupy --workers 1 --device 0`.
The converter benchmark records fill metadata alongside solve and scan times.

On this two-CPU-quota environment, the 2,838-triangle benchmark gave these
medians over three runs (16,384-pair tiles):

| Backend | CPU workers | Fill time |
|---|---:|---:|
| Serial NumPy | 1 | 3.98 s |
| Threaded NumPy | 2 | 2.33 s |
| OpenMP | 1 | 0.75 s |
| OpenMP | 2 | 0.59 s |

The maximum relative moment-matrix difference from NumPy was 4.5e-16 for
OpenMP, and zero for threaded NumPy. The gain includes compiled integration
and tiling; the pure OpenMP thread-scaling gain here is about 1.27x. Host
classification, matrix writes and preparation remain partly serial, and this
does not predict scaling on larger machines. Raw samples are in
[static-fill.json](benchmarks/static-fill.json).

For the same 9,736-triangle CM5 U702 input loop, OpenMP moment filling took
12.65 s with one thread and 9.56 s with two. The complete assembly/sweep fell
from the previous 66.8 s to 22.4 s, with unchanged geometry and copper-matrix
storage. R/L differed by less than 1.4e-14 relative. These are individual
real-board runs, not controlled medians; see the
[CM5 benchmark report](benchmarks/cm5-u702.md).
