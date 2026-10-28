"""
mom25d.sim - the general simulation object (stage 3)

    sim = Simulation(stack, shapes, ports=[...], vias=[...], lumped=[...], dmax=...)
    res = sim.sweep(freqs)

shapes  : {interface: [rect (x0,y0,x1,y1) | shapely Polygon/MultiPolygon, ...]}
ports   : LinePort (wave port, de-embedded to a reference plane, any direction)
          GapPort  (lumped port across a strip, reference plane at the gap)
          ViaPort  (lumped port in a via, e.g. pad-to-ground)
vias    : Via(x, y, k_bot, k_top, radius)   k = 0 the bottom ground (an open bottom: metal there), k = N a PEC cover
lumped  : SeriesLumped (element across a strip); Via(..., Z=...) puts an element inside a via
metal   : Metal(sigma, t) -> surface impedance on strips, wall loss on vias (optional)

Unknowns: RWG edges on all metal layers, then one per via chain. Via-related matrix blocks use
the kernels from mom25d.via (validated against exact image solutions); the via end charges are
fan-distributed and enter through the same G_q interaction as the strip charges.
"""
from types import SimpleNamespace

import os
import numpy as np
import shapely
from emsdev.memory import available as available_memory      # noqa: F401  (bytes; mom25d's name for it)

from .model import KNOWN_SPAN
from .base import MU0, EPS0, RWG, qform, combine, singular_moments, quad_pts, gram_matrix, \
    _BAR3, _W3, _BAR6, _W6, fit_gamma, fit_waves, moments7, quad_moments
from .multilayer import RunLog, GreenTableML, make_mesh_ml
from .geo import LinePort, to_geometry, is_manhattan_rects, mesh_polygons, ngon
from .via import ViaKernels
from .sing import offplane_pairs, offplane_moments

FIT_SPAN = 0.4             # rad: a wave port's cuts must span this much to fit gamma (a port without a characterised one)


# ---- Components -------------------------------------------------------------
def _ls(layer):
    return list(layer) if isinstance(layer, tuple) else [layer]


def _sym(M):
    return 0.5 * (M + M.T)


def _zfun(Z):
    return Z if callable(Z) else (lambda f, Z=Z: complex(Z))


def rlc(R=0.0, L=0.0, C=None, parallel=False):
    """Impedance function of a series (default) or parallel R-L-C element."""
    def Z(f):
        w = 2 * np.pi * f
        if not parallel:
            return R + 1j * w * L + (1 / (1j * w * C) if C else 0)
        return 1 / ((1 / R if R else 0) + (1 / (1j * w * L) if L else 0) + (1j * w * C if C else 0))
    return Z


MIXED_MIN_N = 3500       # above this, 'auto' precision is mixed whenever the compiled core is there
# above this many unknowns in the LU (after quads), mixed precision on the GPU when there is one: 1.5-1.9x a solve
# from 1.4k to 3.4k unknowns against double on 8 P-cores (RX 7800 XT), but its start-up costs ~1.5 s a run - the
# break-even of a 15-20 solve sweep (rf_gcpw, 3.4k: 7.0 s against 8.7 s; rf_stubnotch coarsened, 1.5k: 4.5 against 3.4)
GPU_MIN_N = int(os.environ.get('MOM25D_GPU_MIN_N', 3000))


def memory_plan(n, has_core=True, avail=None, gpu=None, precision='auto', solver='auto', tiled='auto', block=1024,
                n_lu=None):
    """How the solver of n unknowns runs and the memory it needs: dict(solver, precision, path, host, gpu)
    (bytes). gpu: None, or (backend name, free bytes). path: 'dense' (NumPy, small n), 'double' (LU on the
    CPU), 'hybrid' (complex64 LU on the GPU, the double Z on the host for the refinement), 'tiled' (the
    symmetric part as L L^T tiles on the GPU, no dense Z: fast.FastOperator._solve_tiled), 'gpu-lu' (complex64
    Z on the host and the GPU), 'cpu-mixed' (complex64 LU on the CPU), 'blr' / 'aca' (compressed, no dense Z).
    tiled: 'auto' (where the hybrid does not fit), 'on' (whenever a CuPy GPU is there) or 'off'. n_lu: the unknowns
    the LU factors (fewer with quads; default n), which the thresholds GPU_MIN_N and MIXED_MIN_N compare."""
    avail = available_memory() if avail is None else avail
    name, free = gpu if gpu else (None, 0.0)
    gpu_lu = (1 if name in ('cupy', 'emulate') else 2) * 8.0 * n * n + 0.4e9      # gpu._Backend.vram_needed
    gpu_fits = name is not None and gpu_lu <= 0.9 * free
    work = 3 * 8.0 * n * block + 0.5e9                                          # fast.FastOperator._tiled_work
    can_tile = tiled != 'off' and has_core and name in ('cupy', 'emulate') and work <= 0.9 * free
    if precision == 'auto':
        # mixed where double does not fit, and above MIXED_MIN_N (GPU_MIN_N with a GPU): complex64 LU + GMRES refinement,
        # half the LU flops and 10 N^2 bytes instead of 18 (demo transition cropped, 8.4k unknowns, 2 cores: 14.0 s
        # a solve against 18.0 s, |dS| <= 7e-6 over 1-8 GHz); where the refinement stalls, fast.solve_direct redoes
        # it in double and stays in double up to MIXED_RETRY x that frequency
        m = n_lu or n
        precision = 'mixed' if m > (GPU_MIN_N if gpu_fits else MIXED_MIN_N) or 18.0 * n * n + 0.9e9 > 0.5 * avail \
            else 'double'
    if precision == 'mixed' and not has_core:
        precision = 'double'
    hybrid = precision == 'mixed' and gpu_fits and 24.0 * n * n + 0.9e9 <= 0.8 * avail
    if precision == 'double':
        path, host, dev = 'double', 18.0 * n * n + 0.9e9, 0.0
    elif can_tile and (tiled == 'on' or not hybrid):
        budget = max(0.0, 0.9 * free - work)
        path, dev = 'tiled', min(4.0 * n * n, budget) + work
        host = max(0.0, 4.0 * n * n - budget) + 2 * 16.0 * n * block + 1e4 * n + 0.9e9
    elif hybrid:
        path, host, dev = 'hybrid', 24.0 * n * n + 0.9e9, gpu_lu
    elif gpu_fits:
        path, host, dev = 'gpu-lu', 10.0 * n * n + 0.9e9, gpu_lu
    else:
        path, host, dev = 'cpu-mixed', 10.0 * n * n + 0.9e9, 0.0
    fits = host <= 0.8 * avail
    if solver == 'auto':
        # NumPy's dense solver only without the core: the core's direct solver is 3-8x faster on small boards too
        # (rf_stubnotch coarsened, 2.3k unknowns: 25.4 s against 7.3 s a sweep; docs/mom.md#core)
        solver = 'dense' if n <= 3500 and fits and not has_core else \
            ('direct' if fits else ('blr' if has_core else 'aca'))
    if solver == 'dense':
        path, host, dev = 'dense', 18.0 * n * n + 0.9e9, 0.0
    elif solver in ('blr', 'aca'):
        path, host, dev = solver, 2e4 * n + 0.9e9, 0.0
    return dict(solver=solver, precision=precision, path=path, host=host, gpu=dev, fits=fits)


class Via:
    """Plated via: axis (x, y), from interface k_bot to k_top, barrel radius. Z: optional series
    element (callable f -> ohms) placed in the lowest chain (e.g. a chip part to ground)."""
    def __init__(self, x, y, k_bot, k_top, radius=0.15e-3, Z=None, name='', ring=True):
        self.x, self.y, self.k_bot, self.k_top, self.a, self.name = x, y, k_bot, k_top, radius, name
        self.ring = ring          # refine the mesh with an octagon of the via radius around it
        self.Z = None if Z is None else _zfun(Z)


class GapPort:
    """Lumped port across a strip (delta gap). Reference plane = the gap."""
    kind = 'gap'

    def __init__(self, layer, center, direction, width, z0=50.0, name=''):
        self.lp = LinePort(layer, center, direction, width, 0.0)
        self.layer, self.zc, self.name = layer, z0, name

    c = property(lambda self: self.lp.c)
    d = property(lambda self: self.lp.d)
    w = property(lambda self: self.lp.w)

    def line(self, s=0.0): return self.lp.line(s)


class ViaPort:
    """Lumped port inside a via (EMF in its lowest chain, + toward the top; sign=-1: + toward the bottom, a port
    whose reference plane is above it, so the signal is + against its reference either way)."""
    kind = 'via'

    def __init__(self, via, z0=50.0, name='', sign=1):
        self.via, self.zc, self.name, self.layer, self.sign = via, z0, name, None, float(sign)


class DiffPort:
    """
    Differential wave port on a pair of parallel feeds (positive / negative line), e.g. an
    edge-coupled pair. Contributes two modal ports to the S-matrix: '<name>d' (differential) and
    '<name>c' (common). Each mode is fitted with its own propagation constant (odd/even modes of a
    symmetric pair), so mode conversion and modal dispersion are captured.
      pos, neg : LinePort for each line (same direction, layer, sampling and reference plane)
      zdiff, zcomm : modal reference impedances (2 Z_odd and Z_even / 2 of the pair; see line_char_pair)
    """
    def __init__(self, pos, neg, zdiff=100.0, zcomm=25.0, name='P'):
        assert np.allclose(pos.d, neg.d), "differential lines must run in the same direction"
        self.pos, self.neg, self.zdiff, self.zcomm, self.name = pos, neg, zdiff, zcomm, name
        self.zd_fn = self.zc_fn = None

    def modes(self):
        return [ModePort(self, 'd'), ModePort(self, 'c')]


class ModePort(LinePort):
    """One mode of a DiffPort (internal): geometry of the positive line plus the negative line."""
    def __init__(self, parent, mode):
        p = parent.pos
        super().__init__(p.layer, p.c, p.d, p.w, p.ref, s0=p.s0, step=p.step, s1=p.s1,
                         zc=parent.zdiff if mode == 'd' else parent.zcomm, name=parent.name + mode)
        self.parent, self.mode, self.lp_n = parent, mode, parent.neg
        self.sign_n, self.cscale = (-1.0, 1.0) if mode == 'd' else (1.0, 2.0)   # Id = (I+ - I-)/2,  Ic = I+ + I-
        if (fn := parent.zd_fn if mode == 'd' else parent.zc_fn) is not None:
            self.zc_fn = fn


class SeriesLumped:
    """Series element across a strip (cut line of the given width at `center`)."""
    def __init__(self, layer, center, direction, width, Z, name=''):
        self.lp = LinePort(layer, center, direction, width, 0.0)
        self.layer, self.Z, self.name = layer, _zfun(Z), name


class Metal:
    """Conductor loss: Zs = (gamma/sigma) coth(gamma t), gamma = (1+j)/delta  (-> 1/(sigma t) at DC)."""
    def __init__(self, sigma=5.8e7, t=35e-6):
        self.sigma, self.t = sigma, t

    def zs(self, f):
        g = (1 + 1j) * np.sqrt(2 * np.pi * f * MU0 * self.sigma / 2)
        return g / self.sigma / np.tanh(g * self.t)


def _same_area(x, y, area, sides):
    """The regular `sides`-gon of the given area about (x, y)."""
    return shapely.Polygon(ngon(x, y, np.sqrt(2 * area / (sides * np.sin(2 * np.pi / sides))), sides))


def _simplify_outside(geom, zone, max_area=1.0e-6, sides=4):
    """Small pieces of copper wholly outside `zone` (pads of stitching vias, a few hundred um across) as regular
    `sides`-gons of the same area about their centroid: their outline carries next to no current, and a round pad
    costs one unknown per side (the spokes of the via's fan)."""
    out = []
    for p in getattr(geom, 'geoms', [geom]):
        if p.area < max_area and not p.intersects(zone) and not p.interiors:
            q = _same_area(p.centroid.x, p.centroid.y, p.area, sides)
            p = q if q.is_valid and q.buffer(1e-9).contains(p.centroid) else p
        out.append(p)
    return shapely.unary_union(out)


PAD_MAX = 0.75e-3       # _square_vias: copper around a via within this radius is a pad (or an antipad, for a hole)


def _square_vias(geom, pts, sides=4):
    """The pads and antipads of the vias at `pts` as squares, for a coarse mesh around them: a round pad has ~30
    edges and its via an octagon, each a spoke of the fan; a square, 4. A pad on its own becomes a square of the
    same area; one joined to a trace becomes the square around it (it keeps the joint), and a round hole (an
    antipad) around a via that does not attach here a square hole of the same area."""
    for x, y in pts:
        P = shapely.Point(x, y)
        parts = list(getattr(geom, 'geoms', [geom]))
        hit = [i for i, p in enumerate(parts) if p.buffer(1e-9).contains(P)]
        if hit:
            p = parts[hit[0]]
            r = min([p.exterior.distance(P)] + [h.distance(P) for h in p.interiors])
            ring_ = P.buffer(1.6 * r).difference(P.buffer(r))
            if r <= 0 or r > PAD_MAX or p.intersection(ring_).area > 0.4 * ring_.area:
                continue        # a via in a pour (near its edge or a hole) or on a trace: the vertex is enough
            if p.area < 1.2 * np.pi * r * r:              # a pad by itself
                q = _same_area(x, y, p.area, sides)
            else:                                         # a pad on a trace: the square that holds the round pad
                q = p.union(shapely.Polygon(ngon(x, y, r / np.cos(np.pi / sides), sides)))
            if q.is_valid and q.buffer(1e-9).contains(P):
                parts[hit[0]] = q
        else:
            for i, p in enumerate(parts):
                ins = list(p.interiors)
                j = next((j for j, h in enumerate(ins) if shapely.Polygon(h).contains(P)), None)
                if j is None or shapely.Polygon(ins[j]).area > np.pi * PAD_MAX ** 2:
                    continue
                new = shapely.Polygon(p.exterior, ins[:j] + ins[j + 1:] +
                                      [_same_area(x, y, shapely.Polygon(ins[j]).area, sides).exterior])
                if new.is_valid:
                    parts[i] = new
                break
        geom = shapely.unary_union(parts)
    return geom


# ---- Simulation -------------------------------------------------------------
class Simulation:
    def __init__(self, stack, shapes, ports=(), vias=(), lumped=(), dmax=0.3e-3, dmax_layer=None,
                 w_ref=None, ncross=6, grade=1.0, near_fac=5.0, mesher='auto', metal=None,
                 solver='dense', aca_tol=1e-4, refine=1.0e-3, precision='auto', gpu='auto', verbose=True, title='',
                 mesh_only=False, tiled='auto', core='auto', quads=False, quad_tol=35.0, zone=None, dmax_out=None,
                 signal_vias='round', stitching_vias='auto', signal=None, via_barrel=True, line_basis=None,
                 line_w_max=2.0e-3, line_step=2.0, coarsen=None, grow=0.5, structured='auto'):
        # mesh_only: stop once the mesh and the unknowns (Ne, Nv) are known (a cost estimate; nothing to solve with)
        # coarsen (experimental): cells grow from dmax at the features (corners, vias, ports) to coarsen x dmax away
        # from them, by grow x the distance (geo.mesh_polygons): straight runs and plane interiors in larger cells.
        # structured: straight runs of trace as a grid of rectangles (geo.mesh_polygons); 'auto': with quads
        self.st, self.title, self.metal, self.solver, self.log = stack, title, metal, solver, RunLog(verbose)
        self.gpu = gpu                      # 'auto' | 'cupy' | 'torch' | 'off' (mixed direct solver LU)
        self.precision = precision          # 'auto' | 'double' | 'mixed' (direct: complex64 Z + refinement)
        self.tiled = tiled                  # 'auto' | 'on' | 'off': mixed precision as L L^T tiles on the GPU (memory_plan)
        self.core_mode = core               # 'compiled': the compiled core's direct solver for small boards too
        self.blr_tol, self.blr_tile, self.blr_eta = 1e-3, 512, 0.7   # solver='blr' preconditioner settings
        self.dmax, self.ncross, self.grade, self.near_fac = dmax, ncross, grade, near_fac
        self.user_ports = list(ports)
        self.ports = [m for p in ports for m in (p.modes() if isinstance(p, DiffPort) else [p])]
        self.vias, self.lumped = list(vias), list(lumped)
        self.vias += [p.via for p in self.ports if isinstance(p, ViaPort) and p.via not in self.vias]
        self.layers = sorted(shapes)
        for k in self.layers:
            assert k in stack.metal_interfaces, f"interface {k} cannot carry metal in this stack"
        geoms = {k: to_geometry(shapes[k]) for k in self.layers}
        self.zone = zone
        if zone is not None and stitching_vias == 'auto':
            # (experimental) zone: where the current crowds (the signal copper and around it, BoardModel). Outside
            # it the copper is meshed plainly: no edge rings, no via octagons, small pads (stitching vias) as
            # coarse polygons, and no refinement of the layers nearby under it
            geoms = {k: _simplify_outside(g, zone) for k, g in geoms.items()}
        # how each via is meshed: 'round' (its pad and antipad as drawn, an octagon of the via radius) or n sides
        # ('square' = 4): pad and antipad as n-gons, the barrel an n-gon of the via radius (via_barrel) or a single
        # vertex; stitching 'auto': square outside the zone
        signal_vias, stitching_vias = (m if m in ('round', 'auto') else 4 if m == 'square' else int(m)
                                       for m in (signal_vias, stitching_vias))
        assert signal_vias != 'auto' and all(m in ('round', 'auto') or m >= 3 for m in (signal_vias, stitching_vias)), \
            "a via is 'round', 'square' or n >= 3 sides (stitching vias also 'auto')"

        def via_mode(v):
            kind = getattr(v, 'kind', None)                   # None: port vias, vias of hand-built models
            if kind == 'stitching' and stitching_vias == 'auto':
                return 4 if zone is not None and not zone.contains(shapely.Point(v.x, v.y)) else 'round'
            return {'signal': signal_vias, 'stitching': stitching_vias}.get(kind, 'round')
        self.via_mode, self.via_barrel = {id(v): via_mode(v) for v in self.vias}, via_barrel
        for n in sorted({m for m in self.via_mode.values() if m != 'round'}):
            vs = [v for v in self.vias if self.via_mode[id(v)] == n]
            geoms = {k: _square_vias(g, [(v.x, v.y) for v in vs if v.k_bot <= k <= v.k_top], sides=n)
                     for k, g in geoms.items()}
        self.geoms = geoms
        dmax_layer = dmax_layer or {}

        # ---- via chains: split where the via touches metal ------------------------------
        self.chains = []                  # (via, k_bot, k_top)
        self.via_attach = []              # per via: list of metal interfaces it attaches to
        for v in self.vias:
            on = lambda k: k in geoms and geoms[k].buffer(1e-9).contains(shapely.Point(v.x, v.y))
            for k in (v.k_bot, v.k_top):
                assert stack.is_pec(k) or on(k), \
                    f"via {v.name} ends on M{k} where there is no metal"
            att = [v.k_bot] + [k for k in range(v.k_bot + 1, v.k_top) if on(k)] + [v.k_top]
            self.via_attach.append([k for k in att if k in geoms])
            self.chains += [(v, a, b) for a, b in zip(att[:-1], att[1:])]

        # ---- port sample positions and constraint lines -----------------------------------
        self.samples, lines = [], {k: [] for k in self.layers}
        add = lambda lp, xs: [lines[k].append((lp, x)) for k in _ls(lp.layer) for x in xs]
        for p in self.ports:
            if isinstance(p, LinePort):
                step = p.step or dmax
                s0, s1 = p.s0 if p.s0 is not None else 2 * step, p.s1 if p.s1 is not None else p.ref
                s = s1 - step * np.arange(int(np.floor((s1 - s0) / step + 1e-9)) + 1)[::-1]
                self.samples.append(s)
                for lp in [p] + ([p.lp_n] if isinstance(p, ModePort) else []):
                    add(lp, np.append(s, 0.0))
            else:
                self.samples.append(None)
                if isinstance(p, GapPort):
                    add(p.lp, [0.0])
        for el in self.lumped:
            add(el.lp, [0.0])

        # ---- mesh ---------------------------------------------------------------------------
        with self.log.timer('mesh'):
            nodes, tris, tl, off, self.mesher_used = [], [], [], 0, {}
            via_vertex = {}
            for k in self.layers:
                dm, lp_here = dmax_layer.get(k, dmax), [lp for lp, _ in lines[k]]
                vias_here = [(i, v) for i, v in enumerate(self.vias) if k in self.via_attach[i]]
                use_tensor = (mesher == 'tensor' or (mesher == 'auto' and is_manhattan_rects(shapes[k])
                              and not vias_here and all(lp.axis_aligned for lp in lp_here)))
                wr = w_ref.get(k) if isinstance(w_ref, dict) else w_ref
                if use_tensor:
                    xl, yl = [], []
                    for lp, s in lines[k]:
                        x = abs(lp.d[0]) > 0.5
                        (xl if x else yl).append((lp.c + lp.d * s)[0 if x else 1])
                    nd, tr = make_mesh_ml(shapes[k], dm, ncross, 1.01 * max(wr or 0, 1.5 * dm), xl, yl, grade=grade)
                    vids = []
                else:
                    wr = wr or self._guess_wref(geoms[k], lp_here)
                    seg = [lp.line(s) for lp, s in lines[k]]
                    struct = bool(quads) if structured == 'auto' else structured

                    def vpt(v):
                        m = self.via_mode[id(v)]
                        if m == 'round':
                            return (v.x, v.y, v.a if v.ring else 0.0)
                        # an n-sided via: its barrel as an n-gon (signal vias, via_barrel; square ones on a structured
                        # mesh, which aligns its grid with them: geo._run_grid) or a single vertex
                        stitch = struct and m == 4 and getattr(v, 'kind', None) == 'stitching'
                        if stitch and v.ring:                  # may snap to a grid cell (its fan: that cell)
                            return (v.x, v.y, v.a, 4, True)
                        barrel = v.ring and (getattr(v, 'kind', None) == 'signal' and via_barrel or struct and m == 4)
                        return (v.x, v.y, v.a, m) if barrel else (v.x, v.y, 0.0)
                    vpts, fine = [vpt(v) for _, v in vias_here], None
                    if refine and dm > 1.2 * dmax:
                        # resolve return currents under/over metal on nearby layers
                        near = [(geoms[j] if zone is None else geoms[j].intersection(zone))
                                .buffer(3 * abs(stack.z[j] - stack.z[k]) + 2 * dmax)
                                for j in self.layers if j != k and abs(stack.z[j] - stack.z[k]) < refine]
                        fine = shapely.unary_union(near) if near else None
                    zkw = dict(edge_zone=zone, split_chains=True) if zone is not None else {}
                    dm_mesh, dm_fine = dm, (2 * dmax if fine is not None else None)
                    if zone is not None and dmax_out and dmax_out > dm:
                        # outside the zone the copper carries little current: cells up to dmax_out there,
                        # dm (or the refinement's 2 dmax) inside it
                        zf = geoms[k].intersection(zone)
                        if not zf.is_empty and zf.area < 0.99 * geoms[k].area:      # else: all of it is signal
                            fine = zf if fine is None else shapely.unary_union([fine, zf])
                            dm_mesh, dm_fine = dmax_out, min(dm, dm_fine or dm)
                    nd, tr, vids = mesh_polygons(geoms[k], dm_mesh, wr, ncross, grade, lines=seg, vias=vpts,
                                                 fine=fine, dmax_fine=dm_fine, coarsen=coarsen, grow=grow,
                                                 structured=struct, **zkw)
                for (i, v), vid in zip(vias_here, vids):
                    assert vid >= 0, f"via {v.name}: not on the M{k} mesh"
                    via_vertex[(i, k)] = vid + off
                self.mesher_used[k] = 'tensor' if use_tensor else 'polygon'
                nodes.append(nd); tris.append(tr + off); tl.append(np.full(len(tr), k)); off += len(nd)
            self.rwg, self.tri_layer = RWG(np.vstack(nodes), np.vstack(tris)), np.concatenate(tl)
            self.edge_layer = self.tri_layer[self.rwg.tp]
        r = self.rwg
        self.Ne, self.Nv = r.Ne, len(self.chains)
        if mesh_only:
            return
        self.tri_idx = {k: np.nonzero(self.tri_layer == k)[0] for k in self.layers}
        Nt = len(r.tris)

        # ---- via fans and extended divergence ---------------------------------------------
        self.fan = {key: np.nonzero((r.tris == vid).any(1))[0] for key, vid in via_vertex.items()}
        self.F = np.array(sorted({t for v in self.fan.values() for t in v}), int)
        fpos = {t: j for j, t in enumerate(self.F)}
        self.dF = np.zeros((len(self.F), self.Nv))
        for n, (v, kb, kt) in enumerate(self.chains):
            i = self.vias.index(v)
            for k, div in ((kb, +1.0), (kt, -1.0)):
                if (i, k) in self.fan:
                    ts = self.fan[(i, k)]
                    self.dF[[fpos[t] for t in ts], n] += div / r.area[ts].sum()
        self.log.note(f"mesh: {Nt} triangles, {r.Ne} edge + {self.Nv} via unknowns  ("
                      + ", ".join(f"M{k}: {len(self.tri_idx[k])} tri ({self.mesher_used[k]})"
                                  for k in self.layers) + ")")

        V = r.nodes[r.tris]
        self.tri_size = np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2).max(1)
        self.P3, self.W3 = quad_pts(r, _BAR3), r.area[:, None] * _W3[None, :]
        self.P6, self.W6 = quad_pts(r, _BAR6), r.area[:, None] * _W6[None, :]
        self.rho_max = 1.05 * np.hypot(*(r.nodes.max(0) - r.nodes.min(0))) + 1e-4
        self._gram, self._xcache, self.qT = None, {}, None
        self._edge_sets()
        # port gaps, sample cuts and lumped elements keep their edges as unknowns (line functions, quad cells)
        keep_e = [i for g in self.gap if g is not None for i in g[0]]
        keep_e += [i for c in self.cuts if c is not None for idx, _ in c for i in idx]
        keep_e += [i for e in self.lumped_edges for i in e[0]]
        if line_basis and line_basis != 'off':
            # (experimental) line basis functions on straight runs of trace (line.py): per column along a run, a
            # few functions with the edge-peaked shape across it, in place of the RWG unknowns inside the run
            from .line import find_runs, line_subspace
            prof = {'even': ('maxwell',), 'on': ('maxwell', 'odd'), 'full': ('maxwell', 'uniform', 'odd')}[line_basis]
            # the columns' length along the runs: at least 2 cells - tents a cell long interpolate onto the triangles
            # badly (their kinks fall inside cells: the GCPW at 2 GHz 0.17 off, the differential via all wrong)
            h_of = {k: max(line_step, 2.0) * dmax for k in self.layers}
            with self.log.timer('line functions'):
                self.line_runs = find_runs(self.geoms, h_of, signal=signal, w_max=line_w_max)
                self.qT, qi = line_subspace(r, r.Ne + self.Nv, self.tri_layer, self.line_runs, h_of, profiles=prof,
                                            exclude_tris=self.F, exclude_edges=keep_e)
            self.quad_info = qi
            self.log.note(f"lines: {qi['runs']} straight runs, {qi['edges_replaced']} edge unknowns -> "
                          f"{qi['functions']} line functions ({'+'.join(prof)}), {qi['full']} -> {qi['reduced']} unknowns")
            if quads:
                self.log.note("quads: not with line functions yet - ignored")
                quads = False
        if quads:
            # (experimental) near-rectangular triangle pairs as quad cells: their diagonal is no longer an
            # unknown (quads.py); the matrix is filled on the triangles and projected before the LU
            from .quads import quad_subspace
            excl = self.F
            if quads != 'all':
                # no triangle with a corner on the copper outline: the edge strip, where the current crowds, stays
                # triangles (quads there: rf_gcpw 0.015 from a fine reference at 9 GHz, hairpin 0.033 at 3.35 GHz;
                # kept: 0.002 and 0.009, the triangles' 0.0015 and 0.007). 'centre': also none off the signal copper
                # where it is known (BoardModel): no pours or pads. 'all': quads wherever triangles pair up
                edges, count = np.unique(np.sort(r.tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1), axis=0,
                                         return_counts=True)
                bnode = np.zeros(len(r.nodes), bool)
                bnode[edges[count == 1].ravel()] = True                     # the nodes on the copper outline
                off_ = bnode[r.tris].any(1)
                if quads == 'centre' and signal is not None:
                    off_ |= ~shapely.contains_xy(signal, r.cent[:, 0], r.cent[:, 1])
                excl = np.union1d(self.F, np.nonzero(off_)[0])
            with self.log.timer('quad cells'):
                self.qT, qi = quad_subspace(r, r.Ne + self.Nv, exclude_tris=excl, exclude_edges=keep_e,
                                            tol_deg=quad_tol)
            self.quad_info = qi
            self.log.note(f"quads: {qi['quads']} quad cells ({qi['tris_merged']} of {qi['tris']} triangles), "
                          f"{qi['full']} -> {qi['reduced']} unknowns")
        if self.qT is not None and solver not in ('dense', 'direct', 'auto'):
            self.log.note(f"{'lines' if line_basis and line_basis != 'off' else 'quads'}: not with solver {solver} "
                          f"yet - ignored")
            self.qT = None
        self.tables, self.last = {}, None
        for p in self.ports:
            if isinstance(p, LinePort) and not isinstance(p, ModePort) and getattr(p, 'zc_fn', None) is None and p.zc == 50.0:
                self.log.note(f"note: wave port {p.name} uses zc = 50 ohm. Set it to the feed line's Zc "
                              f"(line_char / line_char_xs, or BoardModel.characterize) - a wrong zc makes S "
                              f"non-reciprocal when ports differ")
        n = r.Ne + self.Nv
        from .fast import _core, cap_blas_threads      # None without the build, or with MOM25D_NO_CORE set
        cap_blas_threads()
        has_core = _core is not None
        if not has_core:
            self.log.note("note: no compiled core (python -m mom25d.core.build, or MOM25D_NO_CORE is set): "
                          "the NumPy kernels are several times slower")
        # mixed above MIXED_MIN_N (memory_plan): the complex64 LU on the GPU (5.7 s a solve against 14.9 s on
        # the via transition, 13.5k unknowns, RX 7800 XT) or on the CPU (about 1.3x faster than double). The
        # complex64 preconditioner can stall (fine meshes, low frequency, many via chains; docs/mom.md): that
        # solve is redone in double, and the frequencies up to MIXED_RETRY x it stay in double
        from .gpu import get_backend
        n_lu = self.qT.shape[1] if self.qT is not None else n
        want_gpu = has_core and gpu != 'off' and (precision == 'mixed' or (precision == 'auto' and n_lu > GPU_MIN_N))
        b = get_backend(gpu) if want_gpu else None
        g = (b.name, b.mem_free()) if b is not None else None
        avail = available_memory()
        plan = memory_plan(n, has_core, avail, g, precision, solver, self.tiled, n_lu=n_lu)
        self.plan, self.precision, precision = plan, plan['precision'], plan['precision']
        if solver in ('auto', 'direct'):
            self.log.note(f"memory: direct solver needs ~{plan['host'] / 1e9:.1f} GB ({precision} precision, "
                          f"{plan['path']}" + (f", {plan['gpu'] / 1e9:.1f} GB on the GPU" if plan['gpu'] else '')
                          + f"), {avail / 1e9:.1f} GB available")
            if solver == 'auto':
                solver = plan['solver']
                if solver == 'dense' and self.core_mode == 'compiled' and has_core:   # the core, not NumPy's
                    plan = memory_plan(n, has_core, avail, g, precision, 'direct', self.tiled, n_lu=n_lu)
                    self.precision, self.plan, solver = plan['precision'], plan, plan['solver']
            elif not plan['fits']:
                solver = 'blr' if has_core else 'aca'
                self.log.note(f"WARNING: direct solver would need ~{plan['host'] / 1e9:.1f} GB but only "
                              f"{avail / 1e9:.1f} GB is available - switching to {solver}. Coarser dmax / "
                              f"dmax_layer, a smaller region, or an inner-layer ground reduce N.")
            self.solver = solver
            if self.qT is not None and solver not in ('dense', 'direct'):
                self.log.note(f"quads: not with solver {solver} yet - ignored")
                self.qT = None
            self.log.note(f"solver: {solver} (N = {n})")
        if solver in ('aca', 'direct', 'blr'):
            from .fast import FastOperator
            self.fast = FastOperator(self, tol=aca_tol)
            return

        # ---- static singular parts ------------------------------------------------------------
        with self.log.timer('singular integrals (static, once)'):
            M0, Mrr, Mr, Qs = np.zeros((Nt, Nt)), np.zeros((Nt, Nt)), np.zeros((Nt, Nt, 2)), np.zeros((Nt, Nt), complex)
            for k in self.layers:
                idx = self.tri_idx[k]
                m0, mr, mrr = singular_moments(SimpleNamespace(nodes=r.nodes, tris=r.tris[idx], area=r.area[idx]))
                ix = np.ix_(idx, idx)
                M0[ix], Mrr[ix], Mr[ix] = _sym(m0), _sym(mrr), mr
                Qs[ix] = M0[ix] / (2 * np.pi * EPS0 * stack.eps_sum(k))
            self.SA = _sym(MU0 / (4 * np.pi) * combine(r, Mrr, Mr[..., 0], Mr[..., 1], M0))
            self.SQ, self.Qs_F = _sym(qform(r.D, Qs, r.D)), Qs[:, self.F].copy()

    @staticmethod
    def _guess_wref(g, lps):
        ws = [lp.w for lp in lps if lp.w > 0]
        if ws:
            return min(ws)
        # no line ports here: the narrowest conductor, 2 A / P per piece (= w for a strip, larger for pads/planes)
        x0, y0, x1, y1 = g.bounds
        return min([0.25 * min(x1 - x0, y1 - y0)] + [2 * q.area / q.length for q in getattr(g, 'geoms', [g])
                                                      if q.geom_type == 'Polygon' and q.length > 0])

    # ---- edges on lines --------------------------------------------------------------------
    def _line_edges(self, lp, s, tol=1e-8):
        r = self.rwg
        e0, e1 = r.ends[:, 0] - (lp.c + lp.d * s), r.ends[:, 1] - (lp.c + lp.d * s)
        on = (np.abs(e0 @ lp.d) < tol) & (np.abs(e1 @ lp.d) < tol) & np.isin(self.edge_layer, _ls(lp.layer))
        on &= (np.abs(e0 @ lp.t) <= lp.w / 2 + tol) & (np.abs(e1 @ lp.t) <= lp.w / 2 + tol)
        idx = np.nonzero(on)[0]
        return idx, np.sign(r.nrm[idx] @ lp.d)

    def _edge_sets(self):
        self.gap, self.cuts = [], []
        for p, s in zip(self.ports, self.samples):
            if isinstance(p, ModePort):
                def cut(x, k=0.5):
                    (ip, sp), (i_n, sn) = self._line_edges(p, x), self._line_edges(p.lp_n, x)
                    return np.concatenate([ip, i_n]), np.concatenate([k * sp, k * p.sign_n * sn])
                gap = cut(0.0, 1.0)
            elif isinstance(p, LinePort):
                cut = lambda x: self._line_edges(p, x)
                gap = cut(0.0)
            else:
                gap, s = (self._line_edges(p.lp, 0.0) if isinstance(p, GapPort) else None), None
            self.gap.append(gap)
            self.cuts.append(None if s is None else [cut(x) for x in s])
            assert gap is None or len(gap[0]), f"port {p.name}: no mesh edges on the gap line"
            assert s is None or all(len(c[0]) for c in self.cuts[-1]), f"port {p.name}: sample cut off the conductor"
        self.lumped_edges = [self._line_edges(el.lp, 0.0) for el in self.lumped]
        for el, e in zip(self.lumped, self.lumped_edges):
            assert len(e[0]) > 0, f"lumped element {el.name}: no mesh edges on its line"

    def _port_chain(self, p):
        """Index of the lowest chain of the via of a ViaPort."""
        return next(n for n, (v, _, _) in enumerate(self.chains) if v is p.via)

    # ---- matrix --------------------------------------------------------------------------------
    def green_tables(self, f):
        """The layered-media tables of every pair of metal interfaces at f: {(a, b): GreenTableML}, a <= b."""
        self.tables = {}
        for i, a in enumerate(self.layers):
            for b in self.layers[i:]:
                xs = a != b and abs(self.st.z[a] - self.st.z[b]) < 3 * self.dmax
                im = a == b and self.st.nearest_discontinuity(a) < 1.5 * self.dmax
                with self.log.timer('Green tables', f"M{a}-M{b} @ {f / 1e9:.2f} GHz"):
                    self.tables[(a, b)] = GreenTableML(f, self.st, a, b, self.rho_max, cross_sing=xs, image_sing=im)
        return self.tables

    def impedance(self, f):
        w = 2 * np.pi * f
        r = self.rwg
        Nt = len(r.tris)
        A0, Axo, Ayo, Arr, Q0 = (np.zeros((Nt, Nt), complex) for _ in range(5))
        for (a, b), gt in self.green_tables(f).items():
            ia, ib = self.tri_idx[a], self.tri_idx[b]
            with self.log.timer('smooth-kernel fill'):
                a0, axo, ayo, axs, ays, arr, q0 = self._block(ia, ib, a, b, gt)
            ix = np.ix_(ia, ib)
            A0[ix], Axo[ix], Ayo[ix], Arr[ix], Q0[ix] = a0, axo, ayo, arr, q0
            if a != b:
                jx = np.ix_(ib, ia)
                A0[jx], Axo[jx], Ayo[jx], Arr[jx], Q0[jx] = a0.T, axs.T, ays.T, arr.T, q0.T
        N = self.Ne + self.Nv
        Z = np.zeros((N, N), complex)
        with self.log.timer('assemble'):
            Q0 = _sym(Q0)
            Z[:self.Ne, :self.Ne] = (1j * w * (self.SA + combine(r, _sym(Arr), Axo, Ayo, _sym(A0)))
                                     + (self.SQ + qform(r.D, Q0, r.D)) / (1j * w))
        if self.Nv:
            with self.log.timer('via kernels', f"{self.Nv} chains"):
                B, C = self._via_blocks(f, Q0[:, self.F] + self.Qs_F)
            Z[:self.Ne, self.Ne:] = B
            Z[self.Ne:, :self.Ne] = B.T
            Z[self.Ne:, self.Ne:] = C
        for el, (idx, sig) in zip(self.lumped, self.lumped_edges):
            u = np.zeros(N); u[idx] = sig * r.l[idx]
            Z += el.Z(f) * np.outer(u, u)
        if self.metal is not None:
            self._gram = gram_matrix(r) if self._gram is None else self._gram
            Z[:self.Ne, :self.Ne] += self.metal.zs(f) * self._gram
        return _sym(Z)

    VIA_TABLE_PAIRS = 64                 # via pairs of one span pair above which R comes from a table

    def _via_blocks(self, f, QF):
        """Via rows/columns: B (Ne x Nv) strip-via coupling, C (Nv x Nv) via-via (incl. elements in vias)."""
        w, r = 2 * np.pi * f, self.rwg
        B, C = np.zeros((self.Ne, self.Nv), complex), np.zeros((self.Nv, self.Nv), complex)
        vk = ViaKernels(f, self.st, [(kb, kt) for _, kb, kt in self.chains], self.layers, self.rho_max)
        # fan-distributed end charges (G_q between triangles, same machinery as strips)
        if len(self.F):
            T = QF @ self.dF                                   # (Nt, Nv) potential of fan charges
            B += (r.D.T @ T) / (1j * w)
            C += (self.dF.T @ T[self.F]) / (1j * w)
        # smooth via -> strip kernel D(rho), Galerkin-tested with the RWG divergence
        for n, (v, kb, kt) in enumerate(self.chains):
            c = np.zeros(len(r.tris), complex)
            for k in self.layers:
                idx = self.tri_idx[k]
                dist = np.hypot(self.P6[idx, :, 0] - v.x, self.P6[idx, :, 1] - v.y)
                c[idx] = (vk.D[(k, (kb, kt))](dist) * self.W6[idx]).sum(1)
            B[:, n] += r.D.T @ c
        # via-via: group all pairs by (span_m, span_n) and evaluate each group's Hankel transform once
        groups = {}
        for m, (vm, am, bm) in enumerate(self.chains):
            for n, (vn, an, bn) in enumerate(self.chains[m:], m):
                d = vm.a if vm is vn else max(np.hypot(vm.x - vn.x, vm.y - vn.y), 0.5 * (vm.a + vn.a))
                groups.setdefault(((am, bm), (an, bn)), []).append((m, n, d))
        for (sm, sn), lst in groups.items():
            m, n, dd = (np.array(a) for a in zip(*lst))
            # many pairs: the smooth remainder from a table (the SMA launch's 143k pairs took 24.6 s a solve of
            # 26.8 for its via blocks); a few: exactly
            vals = (vk.R_table(sm, sn)(dd) if len(dd) > self.VIA_TABLE_PAIRS else vk.R(sm, sn, dd)) + vk.homog(sm, sn, dd)
            C[m, n] += vals                                            # each pair once, in one group
            C[n[m != n], m[m != n]] += vals[m != n]
        seen, zs = set(), self.metal.zs(f) if self.metal is not None else 0.0
        for n, (v, kb, kt) in enumerate(self.chains):
            if v.Z is not None and id(v) not in seen:             # an element in a via: in its lowest chain
                C[n, n] += v.Z(f)
            seen.add(id(v))
            C[n, n] += zs * (self.st.z[kt] - self.st.z[kb]) / (2 * np.pi * v.a)     # wall loss
        return B, C

    def _block(self, ia, ib, a, b, gt):
        r, st = self.rwg, self.st
        ca, cb = r.cent[ia], r.cent[ib]
        rho = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
        GA, GQ = gt.AQ(rho)
        xs = a != b and gt.cA != 0                       # near-singular cross-layer part extracted
        im = a == b and gt.dimg > 0                      # nearest image of a thin layer extracted
        if xs:
            R3 = np.sqrt(rho ** 2 + gt.d ** 2)
            GA, GQ = GA + gt.cA / R3, GQ + gt.cq / R3
        # (the extracted image is integrated analytically for ALL pairs below, consistent with the
        #  exact 1/R part: far from a source over ground they nearly cancel - a dipole field)
        wgt = r.area[ia][:, None] * r.area[ib][None, :]
        out = moments7(GA * wgt, GQ * wgt, ca[:, None], cb[None])
        Ls = abs(st.z[a] - st.z[b]) if a != b else 2 * min(
            st.layers[a - 1].h, (st.layers[a].h if a < st.N else np.inf))
        size = np.maximum(self.tri_size[ia][:, None], self.tri_size[ib][None, :])
        p, q = np.nonzero(rho ** 2 + Ls ** 2 < (self.near_fac * size) ** 2)
        for M, v in zip(out, quad_moments(*((self.P3, self.W3) if a == b else (self.P6, self.W6)), ia[p], ib[q], gt)):
            M[p, q] = v
        for flag, dd, cA, cq, allp in ((xs, gt.d, gt.cA, gt.cq, False), (im, gt.dimg, gt.cAi, gt.cqi, True)):
            if not flag:
                continue
            pp, qq = (np.repeat(np.arange(len(ia)), len(ib)), np.tile(np.arange(len(ib)), len(ia))) \
                if allp else (p, q)
            if (a, b, dd) not in self._xcache:             # geometry only: cache across frequencies
                self._xcache[(a, b, dd)] = offplane_pairs(r.nodes, r.tris, ia[pp], ib[qq], dd)
            for M, v in zip(out, offplane_moments(cA, cq, self._xcache[(a, b, dd)])):
                M[pp, qq] += v
        return out

    # ---- excitation and port quantities -------------------------------------------------
    def excitation(self):
        E = np.zeros((self.Ne + self.Nv, len(self.ports)), complex)
        for j, p in enumerate(self.ports):
            if isinstance(p, ViaPort):
                E[self.Ne + self._port_chain(p), j] = p.sign
            else:
                idx, sig = self.gap[j]
                E[idx, j] = sig * self.rwg.l[idx]
        return E

    def port_current(self, I, j):
        p = self.ports[j]
        if isinstance(p, ViaPort):
            return p.sign * I[self.Ne + self._port_chain(p)]
        idx, sig = self.gap[j]
        return ((sig * self.rwg.l[idx])[:, None] * I[idx]).sum(0)

    def cut_current(self, I, i):
        l = self.rwg.l
        return np.array([((sig * l[idx])[:, None] * I[idx]).sum(0) for idx, sig in self.cuts[i]])

    def solve(self, Z, E):
        with self.log.timer('LU solve', f"N={Z.shape[0]}"):
            d = 1.0 / np.sqrt(np.abs(np.diagonal(Z)) + 1e-300)     # symmetric equilibration
            return d[:, None] * np.linalg.solve(Z * d[:, None] * d[None, :], d[:, None] * E)

    def currents(self, f, E):
        """Basis currents for excitation columns E, with the configured solver. A result that isn't finite (the
        GPU solve once returned NaN on a 22.7k-unknown board) is solved again on the CPU, and if that isn't finite
        either, FloatingPointError: never NaN S-parameters."""
        I = self._currents(f, E)
        if np.isfinite(I).all():
            return I
        if self.gpu != 'off' and self.solver == 'direct':
            self.log.note(f"WARNING the GPU solve at {f / 1e9:.4g} GHz was not finite - solving it again on the CPU")
            gpu, self.gpu = self.gpu, 'off'
            try:
                I = self._currents(f, E)
            finally:
                self.gpu = gpu
            if np.isfinite(I).all():
                return I
        raise FloatingPointError(f"mom25d: the {self.solver} solve at {f / 1e9:.4g} GHz is not finite (N = "
                                 f"{len(I)}): a degenerate mesh, or a port with no return path")

    def _currents(self, f, E):
        if self.solver == 'aca':
            return self.fast.solve(f, E)
        if self.solver == 'direct':
            return self.fast.solve_direct(f, E, precision=self.precision)
        if self.solver == 'blr':
            return self.fast.solve_blr(f, E, tol=self.blr_tol, tile=self.blr_tile, eta=self.blr_eta)
        if self.qT is not None:
            T = self.qT
            return T @ self.solve(np.asarray(T.T @ (T.T @ self.impedance(f).T).T), np.asarray(T.T @ E))
        return self.solve(self.impedance(f), E)

    def sparams(self, f, keep=False):
        for p in self.ports:                          # frequency-dependent reference impedance
            if getattr(p, 'zc_fn', None) is not None:
                p.zc = float(p.zc_fn(f))
        I, n = self.currents(f, self.excitation()), len(self.ports)
        Ap, Bp = np.zeros((n, n), complex), np.zeros((n, n), complex)
        gam, resid = np.full(n, np.nan + 0j), np.zeros(n)
        self.low_span = []
        for i, p in enumerate(self.ports):
            if isinstance(p, LinePort):
                s, Ik = self.samples[i], self.cut_current(I, i)
                # gamma from the line characterisation where there is one: fitted on a short feed with a strong
                # standing wave it came out 2-10x low (radial stub, |S11| 1.19), and switching between the fit and
                # the characterised gamma at a span made S jump, which the adaptive sweep chased (38 solves)
                gk = getattr(p, 'gamma_fn', None)
                p.known_gamma = gk is not None
                g = gam[i] = gk(f) if p.known_gamma else fit_gamma(Ik, abs(s[1] - s[0]))
                a, b = fit_waves(Ik, s - p.ref, g)
                M = np.stack([np.exp(-g * (s - p.ref)), np.exp(g * (s - p.ref))], 1)
                # residual relative to the largest response of this port over all excitations (a mode
                # that is barely excited by some column, e.g. differential under common drive, would
                # otherwise give noise / ~0)
                resid[i] = np.max(np.linalg.norm(Ik - M @ np.vstack([a, b]), axis=0)) / np.max(np.linalg.norm(Ik, axis=0))
                cs = getattr(p, 'cscale', 1.0)                 # modal current definition (diff ports)
                Ap[i], Bp[i] = np.sqrt(p.zc) * cs * a, -np.sqrt(p.zc) * cs * b
                span = abs(gam[i].imag) * abs(s[-1] - s[0])
                if span < (KNOWN_SPAN if p.known_gamma else FIT_SPAN):
                    self.low_span.append(p.name)
                    self.log.note(f"WARNING {f / 1e9:.2f} GHz port {p.name}: feed samples span only {span:.2f} rad - "
                                  f"wave separation unreliable (lengthen the straight feed or use a via/gap port)")
            else:
                Vp, Ip = np.eye(n)[i], self.port_current(I, i)          # port voltage = applied EMF
                Ap[i], Bp[i] = (Vp + p.zc * Ip) / (2 * np.sqrt(p.zc)), (Vp - p.zc * Ip) / (2 * np.sqrt(p.zc))
        S = Bp @ np.linalg.inv(Ap)
        for i in np.nonzero(resid > 1e-2)[0]:
            self.log.note(f"WARNING {f / 1e9:.2f} GHz port {self.ports[i].name}: wave-fit residual "
                          f"{resid[i]:.1e} - sample cuts too close to a discontinuity (use s1) "
                          f"or too short a feed for this frequency")
        self.last = dict(f=f, I=I if keep else None, gamma=gam, resid=resid, A=Ap, B=Bp)   # A, B: the waves of each
        return S                                                                          # excitation column

    def sweep(self, freqs, zc=None, currents_at=None):
        out = SimpleNamespace(f=np.asarray(freqs), S=[], gamma=[], resid=[], recip=[], passiv=[], I=None, fI=None)
        fI = None if currents_at is None else freqs[np.argmin(abs(np.asarray(freqs) - currents_at))]
        for f in freqs:
            if zc is not None:
                for p, z in zip(self.ports, zc(f)):
                    p.zc = z
            S = self.sparams(f, keep=(f == fI))
            out.S.append(S); out.gamma.append(self.last['gamma']); out.resid.append(self.last['resid'])
            out.recip.append(np.max(np.abs(S - S.T)))
            out.passiv.append(np.max(np.linalg.svd(S, compute_uv=False)))
            if f == fI:
                out.I, out.fI = self.last['I'], f
            if self.log.verbose:
                print(f"    {f / 1e9:6.2f} GHz  fit residual {np.max(out.resid[-1]):.1e}  max sv(S) {out.passiv[-1]:.4f}")
        for k in ('S', 'gamma', 'resid', 'recip', 'passiv'):
            setattr(out, k, np.array(getattr(out, k)))
        return out

    def current_density(self, I):
        r, I = self.rwg, I[:self.Ne]
        a = r.Aal @ I
        return r.cent[:, 0] * a - r.Bx @ I, r.cent[:, 1] * a - r.By @ I


def _line_fit(sim, IR, IL, step):
    """gamma and Zc of a centre-fed line from the currents (n x 1) on its two halves' sample cuts (ports 0, 1)."""
    g = fit_gamma(np.hstack([IR, IL]), step)
    (A, B), (C, D) = fit_waves(IR, sim.samples[0], g), fit_waves(IL, sim.samples[1], g)
    return g, 1.0 / ((A[0] - B[0]) - (C[0] - D[0]))


def line_char_xs(stack, layer, w, freqs, planes=None, coplanar=None, via_pitch=1e-3, via_r=0.15e-3,
                 length=16e-3, dmax=0.4e-3, ncross=4, verbose=False):
    """
    Zc and gamma of a strip of width w on `layer` with extra reference metal: planes = {interface:
    width} adds a centred wide strip on that interface (e.g. a meshed GND plane above or below);
    coplanar = [(y0, y1), ...] adds same-layer ground strips (grounded CPW), stitched to ground by
    a via row every via_pitch near their inner edge. Centre-fed long line, travelling-wave fit.
    """
    L2, margin = length / 2, 2e-3
    shapes = {layer: [(-L2, -w / 2, L2, w / 2)], **{k: [(-L2, -W / 2, L2, W / 2)] for k, W in (planes or {}).items()}}
    vias = []
    for y0, y1 in (coplanar or []):
        shapes[layer].append((-L2, y0, L2, y1))
        inner = y0 if abs(y0) < abs(y1) else y1
        yv = inner + np.sign(y1 + y0) * min(0.5 * abs(y1 - y0), max(2 * via_r, 0.4e-3))
        vias += [Via(x, yv, 0, layer, radius=via_r, name='fence')
                 for x in np.arange(-L2 + via_pitch / 2, L2, via_pitch)]
    step = max(dmax, (L2 - margin - 1e-3) / 8)          # well-conditioned fit at low frequency
    ports = [LinePort(layer, (0, 0), (d, 0), w, L2 - margin, s0=1e-3, step=step, name=n) for d, n in ((1, 'R'), (-1, 'L'))]
    wr = {layer: w, **{k: min(W, 2e-3) for k, W in (planes or {}).items() if k != layer}}
    sim = Simulation(stack, shapes, ports, vias=vias, dmax=dmax, w_ref=wr, ncross=ncross, verbose=verbose,
                     refine=0, solver='auto')
    out = []
    for f in freqs:
        I = sim.currents(f, sim.excitation()[:, [0]])
        out.append(_line_fit(sim, sim.cut_current(I, 0), sim.cut_current(I, 1), step))
    return tuple(np.array(out).T)                                   # gamma, Zc


def line_char_pair(stack, layer, w, gap, freqs, planes=None, length=16e-3, dmax=0.4e-3, ncross=4, verbose=False):
    """
    Even/odd modes of a symmetric edge-coupled pair (strip width w, edge gap `gap`) on `layer`, with
    optional reference metal (planes = {interface: width}). Returns (gamma_e, Z_even, gamma_o, Z_odd);
    the DiffPort reference impedances are Z_diff = 2 Z_odd and Z_comm = Z_even / 2.
    """
    L2, margin = length / 2, 2e-3
    yc = (w + gap) / 2
    shapes = {layer: [(-L2, yc - w / 2, L2, yc + w / 2), (-L2, -yc - w / 2, L2, -yc + w / 2)],
              **{k: [(-L2, -W / 2, L2, W / 2)] for k, W in (planes or {}).items()}}
    step = max(dmax, (L2 - margin - 1e-3) / 8)
    ps = [LinePort(layer, (0, s * yc), (d, 0), w, L2 - margin, s0=1e-3, step=step, name=f"{s}{d}")
          for s in (1, -1) for d in (1, -1)]                       # (+,R) (+,L) (-,R) (-,L)
    wr = {layer: w, **{k: min(W, 2e-3) for k, W in (planes or {}).items()}}
    sim = Simulation(stack, shapes, ps, dmax=dmax, w_ref=wr, ncross=ncross, verbose=verbose, refine=0,
                     solver='auto')
    out = []
    for f in freqs:
        E = sim.excitation()
        I = sim.currents(f, np.stack([E[:, 0] + E[:, 2], E[:, 0] - E[:, 2]], 1))      # even / odd drive at x = 0
        IR, IL = sim.cut_current(I, 0), sim.cut_current(I, 1)                          # line + only
        out.append([*_line_fit(sim, IR[:, [0]], IL[:, [0]], step), *_line_fit(sim, IR[:, [1]], IL[:, [1]], step)])
    return tuple(np.array(out).T)                                   # gamma_e, Z_even, gamma_o, Z_odd


def coupled_stripline_cohn(w, s, b, er):
    """Exact Z_even, Z_odd of zero-thickness edge-coupled strips centred between grounds spaced b."""
    from scipy.special import ellipk
    a = np.pi * w / (2 * b); c = np.pi * (w + s) / (2 * b)
    ke = np.tanh(a) * np.tanh(c); ko = np.tanh(a) / np.tanh(c)
    Z = lambda k: 30 * np.pi / np.sqrt(er) * ellipk(1 - k ** 2) / ellipk(k ** 2)
    return Z(ke), Z(ko)
