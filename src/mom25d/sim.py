"""
mom25d.sim - the general simulation object (stage 3)

    sim = Simulation(stack, shapes, ports=[...], vias=[...], lumped=[...], dmax=...)
    res = sim.sweep(freqs)

shapes  : {interface: [rect (x0,y0,x1,y1) | shapely Polygon/MultiPolygon, ...]}
ports   : LinePort (wave port, de-embedded to a reference plane, any direction)
          GapPort  (lumped port across a strip, reference plane at the gap)
          ViaPort  (lumped port in a via, e.g. pad-to-ground)
          legacy MLPort objects are accepted and converted
vias    : Via(x, y, k_bot, k_top, radius)   k = 0 is the bottom ground, k = N a PEC cover
lumped  : SeriesLumped (element across a strip); Via(..., Z=...) puts an element inside a via
metal   : Metal(sigma, t) -> surface impedance on strips, wall loss on vias (optional)

Unknowns: RWG edges on all metal layers, then one per via chain. Via-related matrix blocks use
the kernels from mom25d.via (validated against exact image solutions); the via end charges are
fan-distributed and enter through the same G_q interaction as the strip charges.
"""
from types import SimpleNamespace

import numpy as np
import shapely

from .base import MU0, C0, EPS0, RWG, qform, combine, singular_moments, quad_pts, gram_matrix, \
    _BAR3, _W3, _BAR6, _W6, PlanarMoM
from .multilayer import RunLog, GreenTableML, make_mesh_ml, MLPort
from .geo import LinePort, port_from_axis, to_geometry, is_manhattan_rects, mesh_polygons
from .via import ViaKernels
from .sing import offplane_pairs


# =============================================================================
# Components
# =============================================================================
def _ls(layer):
    return list(layer) if isinstance(layer, tuple) else [layer]


def _zfun(Z):
    return Z if callable(Z) else (lambda f, Z=Z: complex(Z))


def rlc(R=0.0, L=0.0, C=None, parallel=False):
    """Impedance function of a series (default) or parallel R-L-C element."""
    def Z(f):
        w = 2 * np.pi * f
        if not parallel:
            z = R + 1j * w * L
            return z + (1 / (1j * w * C) if C else 0)
        y = (1 / R if R else 0) + (1 / (1j * w * L) if L else 0) + (1j * w * C if C else 0)
        return 1 / y
    return Z


def available_memory():
    """Bytes of memory available to this process: MemAvailable, capped by a cgroup limit."""
    avail = None
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    for lim, use in (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                     ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        try:
            L = open(lim).read().strip()
            if L != 'max' and int(L) < 1 << 60:
                free = int(L) - int(open(use).read())
                avail = free if avail is None else min(avail, free)
            break
        except (OSError, ValueError):
            continue
    if avail is None:
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except ImportError:
            avail = 8e9
    return float(avail)


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

    @property
    def c(self): return self.lp.c

    @property
    def d(self): return self.lp.d

    @property
    def w(self): return self.lp.w

    def line(self, s=0.0): return self.lp.line(s)


class ViaPort:
    """Lumped port inside a via (EMF in its lowest chain, + toward the top)."""
    kind = 'via'

    def __init__(self, via, z0=50.0, name=''):
        self.via, self.zc, self.name = via, z0, name
        self.layer = None


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
        self.parent, self.mode = parent, mode
        self.lp_n = parent.neg
        self.sign_n = -1.0 if mode == 'd' else 1.0
        self.cscale = 1.0 if mode == 'd' else 2.0          # Id = (I+ - I-)/2,  Ic = I+ + I-
        fn = parent.zd_fn if mode == 'd' else parent.zc_fn
        if fn is not None:
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
        w = 2 * np.pi * f
        g = (1 + 1j) * np.sqrt(w * MU0 * self.sigma / 2)
        return g / self.sigma / np.tanh(g * self.t)


# =============================================================================
# Simulation
# =============================================================================
class Simulation:
    def __init__(self, stack, shapes, ports=(), vias=(), lumped=(), dmax=0.3e-3, dmax_layer=None,
                 w_ref=None, ncross=6, grade=1.0, near_fac=5.0, mesher='auto', metal=None,
                 solver='dense', aca_tol=1e-4, refine=1.0e-3, precision='auto', gpu='auto', verbose=True, title=''):
        self.st, self.title, self.metal = stack, title, metal
        self.solver = solver
        self.gpu = gpu                      # 'auto' | 'cupy' | 'torch' | 'off' (mixed direct solver LU)
        self.precision = precision          # 'auto' | 'double' | 'mixed' (direct: complex64 Z + refinement)
        self.blr_tol, self.blr_tile, self.blr_eta = 1e-3, 512, 0.7   # solver='blr' preconditioner settings
        self.log = RunLog(verbose)
        self.dmax, self.ncross, self.grade, self.near_fac = dmax, ncross, grade, near_fac
        self.user_ports = list(ports)
        self.ports = []
        for p in ports:
            p = port_from_axis(p) if isinstance(p, MLPort) else p
            self.ports += p.modes() if isinstance(p, DiffPort) else [p]
        self.vias, self.lumped = list(vias), list(lumped)
        for p in self.ports:
            if isinstance(p, ViaPort) and p.via not in self.vias:
                self.vias.append(p.via)
        self.layers = sorted(shapes)
        for k in self.layers:
            assert k in stack.metal_interfaces, f"interface {k} cannot carry metal in this stack"
        geoms = {k: to_geometry(shapes[k]) for k in self.layers}
        self.geoms = geoms
        dmax_layer = dmax_layer or {}

        # ---- via chains: split where the via touches metal ------------------------------
        self.chains = []                  # (via, k_bot, k_top)
        self.via_attach = []              # per via: list of metal interfaces it attaches to
        for v in self.vias:
            att = [v.k_bot]
            for k in range(v.k_bot + 1, v.k_top):
                if k in geoms and geoms[k].buffer(1e-9).contains(shapely.Point(v.x, v.y)):
                    att.append(k)
            att.append(v.k_top)
            for k in (v.k_bot, v.k_top):
                pec = k == 0 or (stack.top == 'pec' and k == stack.N)
                if not pec:
                    assert k in geoms and geoms[k].buffer(1e-9).contains(shapely.Point(v.x, v.y)), \
                        f"via {v.name} ends on M{k} where there is no metal"
            self.via_attach.append([k for k in att if k in geoms])
            for a, b in zip(att[:-1], att[1:]):
                self.chains.append((v, a, b))

        # ---- port sample positions and constraint lines -----------------------------------
        self.samples = []
        lines = {k: [] for k in self.layers}
        for p in self.ports:
            if isinstance(p, LinePort):
                step = p.step or dmax
                s0 = p.s0 if p.s0 is not None else 2 * step
                s1 = p.s1 if p.s1 is not None else p.ref
                n = int(np.floor((s1 - s0) / step + 1e-9)) + 1
                s = s1 - step * np.arange(n)[::-1]
                self.samples.append(s)
                for L_ in _ls(p.layer):
                    lines[L_] += [(p, x) for x in np.append(s, 0.0)]
                if isinstance(p, ModePort):
                    for L_ in _ls(p.lp_n.layer):
                        lines[L_] += [(p.lp_n, x) for x in np.append(s, 0.0)]
            else:
                self.samples.append(None)
                if isinstance(p, GapPort):
                    for L_ in _ls(p.layer):
                        lines[L_].append((p.lp, 0.0))
        for el in self.lumped:
            for L_ in _ls(el.layer):
                lines[L_].append((el.lp, 0.0))

        # ---- mesh ---------------------------------------------------------------------------
        with self.log.timer('mesh'):
            nodes, tris, tl, off, self.mesher_used = [], [], [], 0, {}
            via_vertex = {}
            for k in self.layers:
                dm = dmax_layer.get(k, dmax)
                lp_here = [lp for lp, _ in lines[k]]
                vias_here = [(i, v) for i, v in enumerate(self.vias) if k in self.via_attach[i]]
                use_tensor = (mesher == 'tensor' or (mesher == 'auto' and is_manhattan_rects(shapes[k])
                              and not vias_here and all(lp.axis_aligned for lp in lp_here)))
                if use_tensor:
                    xl, yl = [], []
                    for lp, s in lines[k]:
                        pt = lp.c + lp.d * s
                        (xl if abs(lp.d[0]) > 0.5 else yl).append(pt[0] if abs(lp.d[0]) > 0.5 else pt[1])
                    wr = w_ref.get(k) if isinstance(w_ref, dict) else w_ref
                    narrow = 1.01 * max(wr or 0, 1.5 * dm)
                    nd, tr = make_mesh_ml(shapes[k], dm, ncross, narrow, xl, yl, grade=grade)
                    vids = []
                else:
                    wr = (w_ref.get(k) if isinstance(w_ref, dict) else w_ref) or self._guess_wref(geoms[k], lp_here)
                    seg = [lp.line(s) for lp, s in lines[k]]
                    vpts = [(v.x, v.y, v.a if v.ring else 0.0) for _, v in vias_here]
                    fine = None
                    if refine and dm > 1.2 * dmax:
                        # resolve return currents under/over metal on nearby layers
                        near = [geoms[j].buffer(3 * abs(stack.z[j] - stack.z[k]) + 2 * dmax)
                                for j in self.layers if j != k and abs(stack.z[j] - stack.z[k]) < refine]
                        fine = shapely.unary_union(near) if near else None
                    mesh_fn = mesh_polygons
                    if mesher.startswith('gmsh'):                    # gmsh | gmsh-rings | gmsh-graded
                        from functools import partial
                        from .mesh_gmsh import mesh_polygons_gmsh
                        mesh_fn = partial(mesh_polygons_gmsh, mode=mesher.split('-')[1] if '-' in mesher else 'layers')
                    nd, tr, vids = mesh_fn(geoms[k], dm, wr, ncross, grade, lines=seg, vias=vpts,
                                           fine=fine, dmax_fine=2 * dmax if fine is not None else None)
                for (i, v), vid in zip(vias_here, vids):
                    assert vid >= 0, f"via {v.name}: not on the M{k} mesh"
                    via_vertex[(i, k)] = vid + off
                self.mesher_used[k] = 'tensor' if use_tensor else (mesher if mesher.startswith('gmsh') else 'polygon')
                nodes.append(nd); tris.append(tr + off); tl.append(np.full(len(tr), k)); off += len(nd)
            self.rwg = RWG(np.vstack(nodes), np.vstack(tris))
            self.tri_layer = np.concatenate(tl)
            self.edge_layer = self.tri_layer[self.rwg.tp]
        r = self.rwg
        self.Ne, self.Nv = r.Ne, len(self.chains)
        self.tri_idx = {k: np.nonzero(self.tri_layer == k)[0] for k in self.layers}
        Nt = len(r.tris)

        # ---- via fans and extended divergence ---------------------------------------------
        self.fan = {}
        for (i, k), vid in via_vertex.items():
            self.fan[(i, k)] = np.nonzero((r.tris == vid).any(1))[0]
        F = sorted({t for v in self.fan.values() for t in v})
        self.F = np.array(F, int)
        fpos = {t: j for j, t in enumerate(F)}
        self.dF = np.zeros((len(F), self.Nv))
        for n, (v, kb, kt) in enumerate(self.chains):
            i = self.vias.index(v)
            for k, div in ((kb, +1.0), (kt, -1.0)):
                if (i, k) in self.fan:
                    ts = self.fan[(i, k)]
                    A = r.area[ts].sum()
                    for t in ts:
                        self.dF[fpos[t], n] += div / A
        self.log.note(f"mesh: {Nt} triangles, {r.Ne} edge + {self.Nv} via unknowns  ("
                      + ", ".join(f"M{k}: {len(self.tri_idx[k])} tri ({self.mesher_used[k]})"
                                  for k in self.layers) + ")")

        V = r.nodes[r.tris]
        self.tri_size = np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2).max(1)
        self.P3, self.W3 = quad_pts(r, _BAR3), r.area[:, None] * _W3[None, :]
        self.P6, self.W6 = quad_pts(r, _BAR6), r.area[:, None] * _W6[None, :]
        ext = r.nodes.max(0) - r.nodes.min(0)
        self.rho_max = 1.05 * np.hypot(*ext) + 1e-4
        self._gram = None
        self._xcache = {}
        self._edge_sets()
        self.tables, self.last = {}, None
        for p in self.ports:
            if isinstance(p, LinePort) and not isinstance(p, ModePort) and getattr(p, 'zc_fn', None) is None and p.zc == 50.0:
                self.log.note(f"note: wave port {p.name} uses zc = 50 ohm. Set it to the feed line's Zc "
                              f"(line_char / line_char_xs, or BoardModel.characterize) - a wrong zc makes S "
                              f"non-reciprocal when ports differ")
        n = r.Ne + self.Nv
        from .fast import _core                        # None without the build, or with MOM25D_NO_CORE set
        has_core = _core is not None
        if not has_core:
            self.log.note("note: no compiled core (python -m mom25d.core.build, or MOM25D_NO_CORE is set): "
                          "the NumPy kernels are several times slower")
        if precision == 'auto':
            # double whenever it fits comfortably: the complex64 LU preconditioner can stall (fine meshes, low
            # frequency, many via chains) and the refinement then costs more than a double LU (study in
            # kicad2ems docs/mom.md); mixed only where it saves the memory that makes the direct solver fit
            precision = 'double' if (not has_core or 18.0 * n * n + 0.9e9 <= 0.5 * available_memory()) else 'mixed'
        if precision == 'mixed' and not has_core:
            precision = 'double'
        self.precision = precision
        if solver in ('auto', 'direct'):
            # peak memory of the direct solver (Z + in-place LU + sparse near field + workspace),
            # checked against what is available now (cgroup aware) so a big model never swaps
            need = (10.0 if precision == "mixed" else 18.0) * n * n + 0.9e9
            avail = available_memory()
            self.log.note(f"memory: direct solver needs ~{need / 1e9:.1f} GB ({precision} precision), "
                          f"{avail / 1e9:.1f} GB available")
            fits = need <= 0.8 * avail
            if solver == 'auto':
                solver = 'dense' if n <= 3500 and fits else ('direct' if fits else ('blr' if has_core else 'aca'))
            elif not fits:
                alt = 'blr' if has_core else 'aca'
                self.log.note(f"WARNING: direct solver would need ~{need / 1e9:.1f} GB but only "
                              f"{avail / 1e9:.1f} GB is available - switching to {alt}. Coarser dmax / "
                              f"dmax_layer, a smaller region, or an inner-layer ground reduce N.")
                solver = alt
            self.solver = solver
            self.log.note(f"solver: {solver} (N = {n})")
        if solver in ('aca', 'direct', 'blr'):
            from .fast import FastOperator
            self.fast = FastOperator(self, tol=aca_tol)
            return

        # ---- static singular parts ------------------------------------------------------------
        with self.log.timer('singular integrals (static, once)'):
            M0, Mrr = np.zeros((Nt, Nt)), np.zeros((Nt, Nt))
            Mr = np.zeros((Nt, Nt, 2)); Qs = np.zeros((Nt, Nt), complex)
            for k in self.layers:
                idx = self.tri_idx[k]
                m0, mr, mrr = singular_moments(SimpleNamespace(nodes=r.nodes, tris=r.tris[idx], area=r.area[idx]))
                ix = np.ix_(idx, idx)
                M0[ix] = 0.5 * (m0 + m0.T); Mrr[ix] = 0.5 * (mrr + mrr.T); Mr[ix] = mr
                Qs[ix] = M0[ix] / (2 * np.pi * EPS0 * stack.eps_sum(k))
            SA = MU0 / (4 * np.pi) * combine(r, Mrr, Mr[..., 0], Mr[..., 1], M0)
            self.SA = 0.5 * (SA + SA.T)
            SQ = qform(r.D, Qs, r.D)
            self.SQ = 0.5 * (SQ + SQ.T)
            self.Qs_F = Qs[:, self.F].copy()
            del M0, Mrr, Mr, Qs

    @staticmethod
    def _guess_wref(g, lps):
        ws = [lp.w for lp in lps if lp.w > 0]
        if ws:
            return min(ws)
        b = g.bounds
        cap = 0.25 * min(b[2] - b[0], b[3] - b[1])
        # no line ports here: the narrowest conductor, 2 A / P per piece (= w for a strip, larger for pads/planes)
        pieces = [q for q in getattr(g, 'geoms', [g]) if q.geom_type == 'Polygon' and q.length > 0]
        return min([cap] + [2 * q.area / q.length for q in pieces])

    # ---- edges on lines --------------------------------------------------------------------
    def _line_edges(self, lp, s, tol=1e-8):
        r = self.rwg
        c = lp.c + lp.d * s
        e0, e1 = r.ends[:, 0] - c, r.ends[:, 1] - c
        on = (np.abs(e0 @ lp.d) < tol) & (np.abs(e1 @ lp.d) < tol)
        on &= (np.abs(e0 @ lp.t) <= lp.w / 2 + tol) & (np.abs(e1 @ lp.t) <= lp.w / 2 + tol)
        on &= np.isin(self.edge_layer, list(lp.layer) if isinstance(lp.layer, tuple) else [lp.layer])
        idx = np.nonzero(on)[0]
        return idx, np.sign(r.nrm[idx] @ lp.d)

    def _edge_sets(self):
        self.gap, self.cuts = [], []
        for p, s in zip(self.ports, self.samples):
            if isinstance(p, ModePort):
                def both(x, half):
                    ip, sp = self._line_edges(p, x)
                    i_n, sn = self._line_edges(p.lp_n, x)
                    k = 0.5 if half else 1.0
                    return np.concatenate([ip, i_n]), np.concatenate([k * sp, k * p.sign_n * sn])
                self.gap.append(both(0.0, False))
                self.cuts.append([both(x, True) for x in s])
                assert all(len(c[0]) for c in self.cuts[-1]), f"port {p.name}: sample cut off the conductor"
            elif isinstance(p, LinePort):
                self.gap.append(self._line_edges(p, 0.0))
                self.cuts.append([self._line_edges(p, x) for x in s])
                assert all(len(c[0]) for c in self.cuts[-1]), f"port {p.name}: sample cut off the conductor"
            elif isinstance(p, GapPort):
                self.gap.append(self._line_edges(p.lp, 0.0)); self.cuts.append(None)
            else:
                self.gap.append(None); self.cuts.append(None)
            if self.gap[-1] is not None:
                assert len(self.gap[-1][0]) > 0, f"port {p.name}: no mesh edges on the gap line"
        self.lumped_edges = []
        for el in self.lumped:
            e = self._line_edges(el.lp, 0.0)
            assert len(e[0]) > 0, f"lumped element {el.name}: no mesh edges on its line"
            self.lumped_edges.append(e)

    def _port_chain(self, p):
        """Index of the lowest chain of the via of a ViaPort."""
        return next(n for n, (v, _, _) in enumerate(self.chains) if v is p.via)

    # ---- matrix --------------------------------------------------------------------------------
    def impedance(self, f):
        w = 2 * np.pi * f
        r = self.rwg
        Nt = len(r.tris)
        A0, Axo, Ayo, Arr, Q0 = (np.zeros((Nt, Nt), complex) for _ in range(5))
        self.tables = {}
        for i, a in enumerate(self.layers):
            for b in self.layers[i:]:
                xs = a != b and abs(self.st.z[a] - self.st.z[b]) < 3 * self.dmax
                im = a == b and self._near_layer(a) < 1.5 * self.dmax
                with self.log.timer('Green tables', f"M{a}-M{b} @ {f / 1e9:.2f} GHz"):
                    gt = GreenTableML(f, self.st, a, b, self.rho_max, cross_sing=xs, image_sing=im)
                self.tables[(a, b)] = gt
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
            sym = lambda M: 0.5 * (M + M.T)
            ZA = self.SA + combine(r, sym(Arr), Axo, Ayo, sym(A0))
            Q0 = sym(Q0)
            ZQ = self.SQ + qform(r.D, Q0, r.D)
            Z[:self.Ne, :self.Ne] = 1j * w * ZA + ZQ / (1j * w)
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
            if self._gram is None:
                self._gram = gram_matrix(r)
            zs = self.metal.zs(f)
            Z[:self.Ne, :self.Ne] += zs * self._gram
        return 0.5 * (Z + Z.T)

    def _via_blocks(self, f, QF):
        """Via rows/columns: B (Ne x Nv) strip-via coupling, C (Nv x Nv) via-via (incl. elements in vias)."""
        w = 2 * np.pi * f
        r = self.rwg
        B = np.zeros((self.Ne, self.Nv), complex)
        C = np.zeros((self.Nv, self.Nv), complex)
        spans = [(kb, kt) for _, kb, kt in self.chains]
        vk = ViaKernels(f, self.st, spans, self.layers, self.rho_max)
        self.via_kernels = vk
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
            for n, (vn, an, bn) in enumerate(self.chains):
                if n < m:
                    continue
                d = vm.a if vm is vn else max(np.hypot(vm.x - vn.x, vm.y - vn.y), 0.5 * (vm.a + vn.a))
                groups.setdefault(((am, bm), (an, bn)), []).append((m, n, d))
        for (sm, sn), lst in groups.items():
            dd = np.array([d for _, _, d in lst])
            vals = vk.R(sm, sn, dd) + vk.homog(sm, sn, dd)
            for (m, n, _), val in zip(lst, vals):
                C[m, n] += val
                if n != m:
                    C[n, m] += val
        for n, (v, kb, kt) in enumerate(self.chains):
            if v.Z is not None and n == self.chains.index(next(c for c in self.chains if c[0] is v)):
                C[n, n] += v.Z(f)
        if self.metal is not None:
            zs = self.metal.zs(f)
            for n, (v, kb, kt) in enumerate(self.chains):
                C[n, n] += zs * (self.st.z[kt] - self.st.z[kb]) / (2 * np.pi * v.a)
        return B, C

    def _block(self, ia, ib, a, b, gt):
        r = self.rwg
        ca, cb, aa, ab = r.cent[ia], r.cent[ib], r.area[ia], r.area[ib]
        rho = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
        GA, GQ = gt.AQ(rho)
        xs = a != b and gt.cA != 0                       # near-singular cross-layer part extracted
        im = a == b and gt.dimg > 0                      # nearest image of a thin layer extracted
        if xs:
            R3 = np.sqrt(rho ** 2 + gt.d ** 2)
            GA, GQ = GA + gt.cA / R3, GQ + gt.cq / R3
        # (the extracted image is integrated analytically for ALL pairs below, consistent with the
        #  exact 1/R part: far from a source over ground they nearly cancel - a dipole field)
        wgt = aa[:, None] * ab[None, :]
        A0 = GA * wgt
        out = [A0, A0 * ca[:, None, 0], A0 * ca[:, None, 1], A0 * cb[None, :, 0], A0 * cb[None, :, 1],
               A0 * (ca @ cb.T), GQ * wgt]
        st = self.st
        Ls = abs(st.z[a] - st.z[b]) if a != b else 2 * min(
            st.layers[a - 1].h, (st.layers[a].h if a < st.N else np.inf))
        size = np.maximum(self.tri_size[ia][:, None], self.tri_size[ib][None, :])
        p, q = np.nonzero(rho ** 2 + Ls ** 2 < (self.near_fac * size) ** 2)
        P, W = (self.P3, self.W3) if a == b else (self.P6, self.W6)
        for c0 in range(0, len(p), 20000):
            pp, qq = ia[p[c0:c0 + 20000]], ib[q[c0:c0 + 20000]]
            pa, pb = P[pp][:, :, None, :], P[qq][:, None, :, :]
            Wt = W[pp][:, :, None] * W[qq][:, None, :]
            ga, gq = gt.AQ(np.sqrt(((pa - pb) ** 2).sum(-1)))
            ga, gq = ga * Wt, gq * Wt
            vals = [ga.sum((1, 2)), (ga * pa[..., 0]).sum((1, 2)), (ga * pa[..., 1]).sum((1, 2)),
                    (ga * pb[..., 0]).sum((1, 2)), (ga * pb[..., 1]).sum((1, 2)),
                    (ga * (pa * pb).sum(-1)).sum((1, 2)), gq.sum((1, 2))]
            for M, v in zip(out, vals):
                M[p[c0:c0 + 20000], q[c0:c0 + 20000]] = v
        for flag, dd, cA, cq, allp in ((xs, gt.d, gt.cA, gt.cq, False), (im, gt.dimg, gt.cAi, gt.cqi, True)):
            if not flag:
                continue
            pp, qq = (np.repeat(np.arange(len(ia)), len(ib)), np.tile(np.arange(len(ib)), len(ia))) \
                if allp else (p, q)
            key = (a, b, dd)
            if key not in self._xcache:                    # geometry only: cache across frequencies
                self._xcache[key] = offplane_pairs(r.nodes, r.tris, ia[pp], ib[qq], dd)
            m0, mro, mrs, mrr = self._xcache[key]
            for M, v in zip(out, [cA * m0, cA * mro[:, 0], cA * mro[:, 1], cA * mrs[:, 0],
                                  cA * mrs[:, 1], cA * mrr, cq * m0]):
                M[pp, qq] += v
        return out

    def _near_layer(self, k):
        return self.st.nearest_discontinuity(k)

    # ---- excitation and port quantities -------------------------------------------------
    def excitation(self):
        r = self.rwg
        E = np.zeros((self.Ne + self.Nv, len(self.ports)), complex)
        for j, p in enumerate(self.ports):
            if isinstance(p, ViaPort):
                E[self.Ne + self._port_chain(p), j] = 1.0
            else:
                idx, sig = self.gap[j]
                E[idx, j] = sig * r.l[idx]
        return E

    def port_current(self, I, j):
        p = self.ports[j]
        if isinstance(p, ViaPort):
            return I[self.Ne + self._port_chain(p)]
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
        """Basis currents for excitation columns E, with the configured solver."""
        if self.solver == 'aca':
            return self.fast.solve(f, E)
        if self.solver == 'direct':
            return self.fast.solve_direct(f, E, precision=self.precision)
        if self.solver == 'blr':
            return self.fast.solve_blr(f, E, tol=self.blr_tol, tile=self.blr_tile, eta=self.blr_eta)
        return self.solve(self.impedance(f), E)

    def sparams(self, f, keep=False):
        for p in self.ports:                          # frequency-dependent reference impedance
            if getattr(p, 'zc_fn', None) is not None:
                p.zc = float(p.zc_fn(f))
        I = self.currents(f, self.excitation())
        n = len(self.ports)
        Ap, Bp = np.zeros((n, n), complex), np.zeros((n, n), complex)
        gam, resid = np.full(n, np.nan + 0j), np.zeros(n)
        for i, p in enumerate(self.ports):
            if isinstance(p, LinePort):
                s = self.samples[i]
                Ik = self.cut_current(I, i)
                g = PlanarMoM.fit_gamma(Ik, abs(s[1] - s[0]))
                a, b = PlanarMoM.fit_waves(Ik, s - p.ref, g)
                M = np.stack([np.exp(-g * (s - p.ref)), np.exp(g * (s - p.ref))], 1)
                # residual relative to the largest response of this port over all excitations (a mode
                # that is barely excited by some column, e.g. differential under common drive, would
                # otherwise give noise / ~0)
                resid[i] = np.max(np.linalg.norm(Ik - M @ np.vstack([a, b]), axis=0)) / np.max(np.linalg.norm(Ik, axis=0))
                cs = getattr(p, 'cscale', 1.0)                 # modal current definition (diff ports)
                Ap[i], Bp[i] = np.sqrt(p.zc) * cs * a, -np.sqrt(p.zc) * cs * b
                gam[i] = g
            else:
                Vp = np.eye(n)[i]                     # port voltage = applied EMF
                Ip = self.port_current(I, i)
                Ap[i] = (Vp + p.zc * Ip) / (2 * np.sqrt(p.zc))
                Bp[i] = (Vp - p.zc * Ip) / (2 * np.sqrt(p.zc))
        S = Bp @ np.linalg.inv(Ap)
        self.low_span = []
        for i, p in enumerate(self.ports):
            if isinstance(p, LinePort):
                span = abs(self.samples[i][-1] - self.samples[i][0])
                if abs(gam[i].imag) * span < 0.4:
                    self.low_span.append(p.name)
                    self.log.note(f"WARNING {f / 1e9:.2f} GHz port {p.name}: feed samples span only "
                                  f"{abs(gam[i].imag) * span:.2f} rad - wave separation unreliable "
                                  f"(lengthen the straight feed or use a via/gap port)")
        for i in np.nonzero(resid > 1e-2)[0]:
            self.log.note(f"WARNING {f / 1e9:.2f} GHz port {self.ports[i].name}: wave-fit residual "
                          f"{resid[i]:.1e} - sample cuts too close to a discontinuity (use s1) "
                          f"or too short a feed for this frequency")
        self.last = dict(f=f, I=I if keep else None, gamma=gam, resid=resid)
        return S

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
        r = self.rwg
        I = I[:self.Ne]
        a = r.Aal @ I
        return r.cent[:, 0] * a - r.Bx @ I, r.cent[:, 1] * a - r.By @ I


# =============================================================================
# Thick conductors: two sheets + sidewall connectors
# =============================================================================
class Sheets(tuple):
    """A metal 'layer' made of several interfaces (the two sheets of a thick conductor)."""
    pass


def thicken(stack, shapes, thickness, pitch=0.25e-3, radius=None):
    """
    Two-sheet model of thick conductors.
      thickness : {interface k: t}   metal on k becomes sheets at k (bottom) and k+1 (top), a new
                  layer of thickness t (permittivity of the layer above, or air on top) between them
      pitch     : spacing of the sidewall connectors along every polygon boundary
    Returns (stack2, shapes2, kmap, side_vias). kmap maps old interface -> new bottom sheet index;
    use sheets(k) for ports / lumped elements that must act on both sheets.
    """
    from .multilayer import Layer, LayerStack
    layers = [Layer(l.h, l.er, l.tand, l.name) for l in stack.layers]
    top_er = stack.top_er
    shapes2 = {k: list(v) for k, v in shapes.items()}
    kmap = {k: k for k in range(stack.N + 1)}
    sheets = {}
    for k in sorted(thickness, reverse=True):
        t = thickness[k]
        if k == len(layers) and stack.top == 'open':
            layers.append(Layer(t, top_er, 0.0, f"t{k}"))
        else:
            above = layers[k]
            assert above.h > t, f"layer above M{k} thinner than the metal"
            layers[k] = Layer(above.h - t, above.er, above.tand, above.name)
            layers.insert(k, Layer(t, above.er, above.tand, f"t{k}"))
        new = {}
        for kk, v in shapes2.items():
            new[kk + 1 if kk > k else kk] = v
        new[k + 1] = list(shapes2[k])
        shapes2 = new
        for kk in list(kmap):
            if kmap[kk] > k:
                kmap[kk] += 1
    st2 = LayerStack(layers, top=stack.top, top_er=top_er)
    side = []
    for k, t in thickness.items():
        kb = kmap[k]
        rad = radius or min(pitch / (2 * np.pi), t / 2)     # thin-wire model needs radius < length
        g = to_geometry(shapes[k])
        polys = [g] if g.geom_type == 'Polygon' else list(g.geoms)
        for p in polys:
            for ring in [p.exterior] + list(p.interiors):
                n = max(4, int(np.ceil(ring.length / pitch)))
                for s in np.arange(n) / n * ring.length:
                    pt = ring.interpolate(s)
                    side.append(Via(pt.x, pt.y, kb, kb + 1, radius=rad, name=f"side{k}", ring=False))
        sheets[k] = Sheets((kb, kb + 1))
    return st2, shapes2, kmap, side, sheets


def line_char(stack, layer, w, freqs, length=20e-3, dmax=0.5e-3, ncross=6, metal=None, thick=None,
              pitch=0.25e-3, verbose=False):
    """Zc and gamma of a uniform strip (centre-fed long line); optional thickness and metal loss."""
    L2, margin = length / 2, 2e-3
    shapes = {layer: [(-L2, -w / 2, L2, w / 2)]}
    st, vias, lay = stack, [], layer
    if thick:
        st, shapes, kmap, vias, sh = thicken(stack, shapes, {layer: thick}, pitch=pitch)
        lay = sh[layer]
    pR = LinePort(lay, (0, 0), (1, 0), w, L2 - margin, s0=1e-3, step=dmax, name='R')
    pL = LinePort(lay, (0, 0), (-1, 0), w, L2 - margin, s0=1e-3, step=dmax, name='L')
    sim = Simulation(st, shapes, [pR, pL], vias=vias, dmax=dmax, w_ref=w, ncross=ncross, metal=metal,
                     verbose=verbose)
    gam, zc = [], []
    for f in freqs:
        I = sim.currents(f, sim.excitation()[:, [0]])
        IR, IL = sim.cut_current(I, 0), sim.cut_current(I, 1)
        g = PlanarMoM.fit_gamma(np.hstack([IR, IL]), dmax)
        A, B = PlanarMoM.fit_waves(IR, sim.samples[0], g)
        Cp, Dp = PlanarMoM.fit_waves(IL, sim.samples[1], g)
        gam.append(g); zc.append(1.0 / ((A[0] - B[0]) - (Cp[0] - Dp[0])))
    return np.array(gam), np.array(zc), sim


def line_char_xs(stack, layer, w, freqs, planes=None, coplanar=None, via_pitch=1e-3, via_r=0.15e-3,
                 length=16e-3, dmax=0.4e-3, ncross=4, verbose=False):
    """
    Zc and gamma of a strip of width w on `layer` with extra reference metal: planes = {interface:
    width} adds a centred wide strip on that interface (e.g. a meshed GND plane above or below);
    coplanar = [(y0, y1), ...] adds same-layer ground strips (grounded CPW), stitched to ground by
    a via row every via_pitch near their inner edge. Centre-fed long line, travelling-wave fit.
    """
    L2, margin = length / 2, 2e-3
    shapes = {layer: [(-L2, -w / 2, L2, w / 2)]}
    for k, W in (planes or {}).items():
        shapes[k] = [(-L2, -W / 2, L2, W / 2)]
    vias = []
    for y0, y1 in (coplanar or []):
        shapes[layer].append((-L2, y0, L2, y1))
        inner = y0 if abs(y0) < abs(y1) else y1
        yv = inner + np.sign(y1 + y0) * min(0.5 * abs(y1 - y0), max(2 * via_r, 0.4e-3))
        vias += [Via(x, yv, 0, layer, radius=via_r, name='fence')
                 for x in np.arange(-L2 + via_pitch / 2, L2, via_pitch)]
    step = max(dmax, (L2 - margin - 1e-3) / 8)          # well-conditioned fit at low frequency
    pR = LinePort(layer, (0, 0), (1, 0), w, L2 - margin, s0=1e-3, step=step, name='R')
    pL = LinePort(layer, (0, 0), (-1, 0), w, L2 - margin, s0=1e-3, step=step, name='L')
    wr = {k: (w if k == layer else min(W, 2e-3)) for k, W in [(layer, w)] + list((planes or {}).items())}
    sim = Simulation(stack, shapes, [pR, pL], vias=vias, dmax=dmax, w_ref=wr, ncross=ncross, verbose=verbose,
                     refine=0, solver='auto')
    gam, zc = [], []
    for f in freqs:
        I = sim.currents(f, sim.excitation()[:, [0]])
        IR, IL = sim.cut_current(I, 0), sim.cut_current(I, 1)
        g = PlanarMoM.fit_gamma(np.hstack([IR, IL]), step)
        A, B = PlanarMoM.fit_waves(IR, sim.samples[0], g)
        Cp, Dp = PlanarMoM.fit_waves(IL, sim.samples[1], g)
        gam.append(g); zc.append(1.0 / ((A[0] - B[0]) - (Cp[0] - Dp[0])))
    return np.array(gam), np.array(zc)


def line_char_pair(stack, layer, w, gap, freqs, planes=None, length=16e-3, dmax=0.4e-3, ncross=4, verbose=False):
    """
    Even/odd modes of a symmetric edge-coupled pair (strip width w, edge gap `gap`) on `layer`, with
    optional reference metal (planes = {interface: width}). Returns (gamma_e, Z_even, gamma_o, Z_odd);
    the DiffPort reference impedances are Z_diff = 2 Z_odd and Z_comm = Z_even / 2.
    """
    L2, margin = length / 2, 2e-3
    yc = (w + gap) / 2
    shapes = {layer: [(-L2, yc - w / 2, L2, yc + w / 2), (-L2, -yc - w / 2, L2, -yc + w / 2)]}
    for k, W in (planes or {}).items():
        shapes[k] = [(-L2, -W / 2, L2, W / 2)]
    step = max(dmax, (L2 - margin - 1e-3) / 8)
    ps = [LinePort(layer, (0, s * yc), (d, 0), w, L2 - margin, s0=1e-3, step=step, name=f"{s}{d}")
          for s in (1, -1) for d in (1, -1)]                       # (+,R) (+,L) (-,R) (-,L)
    wr = {layer: w, **{k: min(W, 2e-3) for k, W in (planes or {}).items()}}
    sim = Simulation(stack, shapes, ps, dmax=dmax, w_ref=wr, ncross=ncross, verbose=verbose, refine=0,
                     solver='auto')
    out = {'e': ([], []), 'o': ([], [])}
    for f in freqs:
        E = sim.excitation()
        V = np.stack([E[:, 0] + E[:, 2], E[:, 0] - E[:, 2]], 1)       # even / odd drive at x = 0
        I = sim.currents(f, V)
        for c, m in ((0, 'e'), (1, 'o')):
            IR, IL = sim.cut_current(I, 0)[:, c], sim.cut_current(I, 1)[:, c]     # line + only
            g = PlanarMoM.fit_gamma(np.stack([IR, IL], 1), step)
            A, B = PlanarMoM.fit_waves(IR[:, None], sim.samples[0], g)
            Cp, Dp = PlanarMoM.fit_waves(IL[:, None], sim.samples[1], g)
            out[m][0].append(g); out[m][1].append(1.0 / ((A[0] - B[0]) - (Cp[0] - Dp[0])))
    return (np.array(out['e'][0]), np.array(out['e'][1]), np.array(out['o'][0]), np.array(out['o'][1]))


def coupled_stripline_cohn(w, s, b, er):
    """Exact Z_even, Z_odd of zero-thickness edge-coupled strips centred between grounds spaced b."""
    from scipy.special import ellipk
    a = np.pi * w / (2 * b); c = np.pi * (w + s) / (2 * b)
    ke = np.tanh(a) * np.tanh(c); ko = np.tanh(a) / np.tanh(c)
    Z = lambda k: 30 * np.pi / np.sqrt(er) * ellipk(1 - k ** 2) / ellipk(k ** 2)
    return Z(ke), Z(ko)
