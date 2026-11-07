"""
mom25d.mqs - the magneto-quasistatic currents of a board's copper (the board much smaller than a wavelength: a
DC-DC converter's 0.1-30 MHz): where an AC current flows (its return pulled under its path as the frequency rises,
or around a slot), the loop's R and L, and the near magnetic field it makes (Biot-Savart: mom25d.biot).

    q = MQS(board, nets={'VIN', 'GND', 'SW'}, terminals={'U1.VIN': [('F.Cu', pad)], ...}, region=box(...), dmax=0.5)
    q.source('hot', 'U1.VIN', 'U1.PGND', part_h=0.5)          # a current source in the part between its pads
    q.element('C1', 'C1.1', 'C1.2', C=10e-6, ESR=3e-3, ESL=0.4e-9)
    r = q.solve([1e6, 3e6])                                    # per frequency: currents, the sources' Z = R + j w L
    H = q.h_field(r, 0, pts)                                   # (n, 3) A/m at points (m) for 1 A in the sources

The copper of the nets on every layer (cropped to `region`, mm) is meshed by mom25d's mesher (pdn.mesh, the pads'
outlines as mesh edges; `fine`, a shapely region in mm, meshed at dmax_fine: the loop's own area) and carries RWG
currents; a via is a filament between the layers it joins; a part (a source,
a capacitor) a branch between its terminals, drawn for the field as a filament `part_h` above its pads. The unknowns
are the loop currents of the branch graph (a spanning tree's cotree: Kirchhoff's current law holds by construction,
no node potentials), so a sweep is one small solve per frequency:
    Z_loops = C^T (R + j w L) C,   R: the sheets' Gram matrix / (sigma t) and the barrels',  L: mu0 / 4 pi of
    int int f_m . f_n / R (mom25d's 1/R moments, analytic inner integral, between layers at their offset), the vias'
    partial inductances (parallel filaments), and the parts' series R, L, C.
Horizontal and vertical currents do not couple (orthogonal); the parts' own fields count, their mutual inductance to
the copper does not (a part's body: a short filament well above it).

With capacitance=True, a homogeneous-dielectric surface-charge PEEC model
replaces the closed-loop constraint by charge continuity (mom25d.peec).
It also provides Maxwell capacitance and Coulomb E; see docs/quasistatic.md.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags, block_diag, csr_matrix
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from .base import MU0, RWG, combine, qform, _BAR3, _W3, quad_pts
from .static import static_moments
from .biot import rwg_currents, h_sheets, h_segments
from .pdn.mesh import copper_of, mesh_sheet, stack_z

SIGMA_CU = 5.8e7


def filament_mutual(z0a, z1a, z0b, z1b, s):
    """Partial mutual inductance (H) of two parallel filaments along z, a from z0a to z1a and b from z0b to z1b (m),
    s apart (the self term: s = the radius). Grover: mu0 / 4 pi [g(a1 - b0) + g(a0 - b1) - g(a1 - b1) - g(a0 - b0)],
    g(u) = u asinh(u / s) - sqrt(u^2 + s^2)."""
    g = lambda u: u * np.arcsinh(u / s) - np.sqrt(u * u + s * s)
    return MU0 / (4 * np.pi) * (g(z1a - z0b) + g(z0a - z1b) - g(z1a - z1b) - g(z0a - z0b))


@dataclass
class Result:
    f: np.ndarray
    i: np.ndarray                   # (nf, nbranch) branch currents (A): edges (total current across), vias, parts
    z: dict                         # {source: Z (nf,) complex}: the loop each source drives, its parts included
    names: list
    zc: dict = None                 # {source: Z (nf,)}: the copper of that loop alone (the parts' own Z left out)
    charge: np.ndarray = None      # PEEC only: (nf, n_nodes), C
    potential: np.ndarray = None   # PEEC only: (nf, n_nodes), V relative to infinity


class MQS:
    def __init__(self, board, nets, terminals=None, region=None, dmax=0.5, sigma=SIGMA_CU, plating=0.025, log=print,
                 fine=None, dmax_fine=None, near=3.0, outer_levels=1, capacitance=False, er=1.0,
                 factor_cache_bytes=64 * 1024**2, conductor_loss='dc', conductor_faces=2,
                 dielectric='homogeneous', dielectric_tol=1e-8, fill_backend='auto', fill_workers=None,
                 fill_chunk=16384, fill_device=None):
        """Mesh the selected copper. `near` controls analytic 1/R integration
        (infinity: all pairs); `outer_levels` refines its close-pair quadrature.
        `capacitance=True` enables surface-charge PEEC: homogeneous `er`, or
        `dielectric='stack'` for the board's planar slabs and dielectric loss.
        `conductor_loss='slab'` adds finite-thickness sheet impedance with
        `conductor_faces` one or two; barrels retain their DC resistance.
        PEEC's reusable frequency factors have a `factor_cache_bytes` budget.
        `fill_backend` chooses auto/openmp/threaded/numpy/cupy; `fill_workers`
        caps CPU parallelism and `fill_chunk` bounds pairs per tile. Device
        backends use `fill_device`; `fill_info` records actual execution.
        """
        if not np.isfinite(er) or er <= 0:
            raise ValueError('er must be finite and positive')
        self.near, self.outer_levels, self.capacitance, self.er = near, outer_levels, capacitance, er
        self._fill_options = dict(backend=fill_backend, workers=fill_workers, chunk=fill_chunk, device=fill_device)
        self.fill_info = {}
        if factor_cache_bytes < 0:
            raise ValueError('factor_cache_bytes must be nonnegative')
        self.factor_cache_bytes = factor_cache_bytes
        if conductor_loss not in ('dc', 'slab') or conductor_faces not in (1, 2):
            raise ValueError('conductor_loss must be dc or slab; conductor_faces must be 1 or 2')
        self.conductor_loss, self.conductor_faces = conductor_loss, conductor_faces
        if dielectric not in ('homogeneous', 'stack'):
            raise ValueError('dielectric must be homogeneous or stack')
        if dielectric == 'stack' and er != 1:
            raise ValueError('er applies to homogeneous dielectric; stack uses board permittivities')
        self.dielectric = dielectric
        self._dielectric_stack = None
        if dielectric == 'stack':
            from .electrostatic import DielectricStack
            self._dielectric_stack = DielectricStack.from_board(board, dielectric_tol)
        self.board, self.sigma, self.log = board, sigma, log
        self.nets = {nets} if isinstance(nets, str) else set(nets)
        zt = stack_z(board)
        self.terms = dict(terminals or {})
        reg = box(*region) if isinstance(region, (tuple, list)) else region
        vias = [v for v in board['vias'] if v.get('net') in self.nets and (reg is None or reg.contains(Point(v['x'], v['y'])))]
        pts = [(v['x'], v['y'], v['r']) for v in vias]
        nodes, tris, layer_of, z_of, t_of = [], [], [], [], []
        self.layer_tris, self.term_tris, ring_tris = {}, {n: [] for n in self.terms}, {}
        off = 0
        for lay in sorted(zt, key=lambda l: zt[l][0]):
            g = copper_of(board, self.nets, lay)
            if g is None:
                continue
            if reg is not None:
                g = g.intersection(reg).buffer(0)
                if g.is_empty:
                    continue
            polys = [(n, poly) for n, pads in self.terms.items() for l, poly in pads if l == lay]
            nd, tr, rings, regs = mesh_sheet(g, pts, dmax, regions=[p for _, p in polys], fine=fine,
                                             dmax_fine=dmax_fine if fine is not None else None)
            if not len(tr):
                continue
            nt0 = sum(len(t) for t in tris)
            cen = nd[tr].mean(1)
            for (n, poly), ids in zip(polys, regs):
                inside = np.flatnonzero(_inside(poly, cen))
                self.term_tris[n] += list(nt0 + inside)
            for k, ring in enumerate(rings):
                if ring is None:
                    continue
                fan = np.flatnonzero(np.isin(tr, ring[:1]).any(1))        # the triangles around its centre
                ring_tris[(k, lay)] = list(nt0 + fan)
            nodes.append(nd * 1e-3)
            tris.append(tr + off)
            off += len(nd)
            self.layer_tris[lay] = np.arange(nt0, nt0 + len(tr))
            layer_of += [lay] * len(tr); z_of += [zt[lay][0] * 1e-3] * len(tr); t_of += [zt[lay][1] * 1e-3] * len(tr)
        if not tris:
            raise ValueError(f"no copper of {', '.join(sorted(self.nets))} in the region")
        self.nodes, self.tris = np.vstack(nodes), np.vstack(tris)
        self.layer_of, self.z_of, self.t_of = np.array(layer_of), np.array(z_of), np.array(t_of)
        self.rwg = RWG(self.nodes, self.tris)
        missing = [n for n, ts in self.term_tris.items() if not ts]
        if missing:
            raise ValueError(f"terminal(s) {', '.join(missing)} on none of the meshed copper")
        # the graph's nodes: a triangle each, but every terminal's triangles one node, every via end's fan one node
        Nt = len(self.tris)
        parent = np.arange(Nt)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def merge(ts):
            r0 = find(ts[0])
            for t in ts[1:]:
                parent[find(t)] = r0
        for ts in self.term_tris.values():
            merge(ts)
        for ts in ring_tris.values():
            if ts:
                merge(ts)
        roots = np.array([find(t) for t in range(Nt)])
        _, self.node_of = np.unique(roots, return_inverse=True)
        self.n_nodes = self.node_of.max() + 1
        self.term_node = {n: int(self.node_of[ts[0]]) for n, ts in self.term_tris.items()}
        # branches: the RWG edges (their total current), then the via segments, then the parts (added later)
        r = self.rwg
        self.br = [(int(self.node_of[a]), int(self.node_of[b])) for a, b in zip(r.tp, r.tm)]
        self.vseg = []                                       # (x, y, r, z0, z1) m, per via segment
        for k, v in enumerate(vias):
            on = sorted([(zt[l][0], l) for l in self.layer_tris if (k, l) in ring_tris and ring_tris[(k, l)]
                         and v['z0'] - 1e-6 <= zt[l][0] <= v['z1'] + 1e-6])
            for (za, la), (zb, lb) in zip(on, on[1:]):
                na, nb = int(self.node_of[ring_tris[(k, la)][0]]), int(self.node_of[ring_tris[(k, lb)][0]])
                self.br.append((na, nb))
                self.vseg.append((v['x'] * 1e-3, v['y'] * 1e-3, v['r'] * 1e-3, za * 1e-3, zb * 1e-3))
        self.ne, self.nv = r.Ne, len(self.vseg)
        self.parts = []                                       # dict(name, a, b, kind, R, L, C, part_h)
        self.plating = plating * 1e-3
        self._mats = None
        self._copper = None
        self._potential = None
        self._peec = None
        log(f"mqs: {Nt:,} triangles on {len(self.layer_tris)} layer(s), {self.ne:,} edges, {self.nv} via segments, "
            f"{len(self.terms)} terminals")

    # ---- the parts ----
    def source(self, name, a, b, part_h=0.5):
        """A 1 A current source (scaled by solve's amps) through a part from terminal a to terminal b."""
        self.parts.append(dict(name=name, a=a, b=b, kind='I', part_h=part_h))
        self._mats = None
        self._peec = None

    def element(self, name, a, b, R=0.0, L=0.0, C=None, ESR=None, ESL=None, part_h=0.5):
        """A series R, L (and C) through a part between terminals a and b."""
        self.parts.append(dict(name=name, a=a, b=b, kind='Z', R=ESR if ESR is not None else R,
                               L=ESL if ESL is not None else L, C=C, part_h=part_h))
        self._mats = None
        self._peec = None

    # ---- the matrices ----
    def _edge_matrices(self):
        moments, info = static_moments(self.nodes, self.tris, self.z_of, self.near, self.outer_levels,
                                       **self._fill_options, return_info=True)
        nodes, M0, Mrx, Mry, Mrr = moments
        self.fill_info['magnetic'] = info
        self.log(f'mqs: {info["backend"]} matrix fill, {info["workers"]} worker(s), {info["seconds"]:.2f} s')
        r = RWG(nodes, self.tris)
        L = MU0 / (4 * np.pi) * combine(r, Mrr, Mrx, Mry, M0)
        L = 0.5 * (L + L.T)
        if self.capacitance and self._potential is None:
            if self.dielectric == 'homogeneous':
                self._set_potential(M0)
                self.fill_info['capacitance'] = dict(info, reused_magnetic_moments=True)
            else:
                self._ensure_potential()
        # R: the Gram matrix of the RWG functions, each triangle at its sheet resistance
        Ar, Pq = r.area, quad_pts(r, _BAR3)
        rs = 1.0 / (self.sigma * self.t_of)
        def gram(weights):
            m = lambda v: diags(Ar * weights * (_W3[None, :] * v).sum(1))
            return combine(r, m((Pq ** 2).sum(-1)), m(Pq[..., 0]), m(Pq[..., 1]), diags(Ar * weights)).tocsr()
        R = gram(rs)
        # in total currents across each edge (i = l I): Z / (l_m l_n)
        s = 1.0 / r.l
        scale = diags(s)
        self._sheet_terms = []
        if self.conductor_loss == 'slab':
            for idx in self.layer_tris.values():
                weights = np.zeros(len(self.tris)); weights[idx] = rs[idx]
                self._sheet_terms.append((self.t_of[idx[0]], (scale @ gram(weights) @ scale).tocsr()))
        return (scale @ R @ scale).tocsr(), L * np.outer(s, s)

    def _resistance_delta(self, f):
        from .conductor import surface_impedance
        nb = len(self.br) + len(self.parts)
        if self.conductor_loss == 'dc':
            return csr_matrix((nb, nb), dtype=complex)
        delta = csr_matrix((self.ne, self.ne), dtype=complex)
        for t, R in self._sheet_terms:
            ratio = surface_impedance(f, t, self.sigma, self.conductor_faces) * self.sigma * t
            delta = delta + (ratio - 1) * R
        return block_diag((delta, csr_matrix((nb-self.ne, nb-self.ne))), format='csr')

    def _set_potential(self, M0, er=None):
        from .base import EPS0
        area = np.bincount(self.node_of, weights=self.rwg.area, minlength=self.n_nodes)
        W = coo_matrix((1 / area[self.node_of], (np.arange(len(self.tris)), self.node_of)),
                       shape=(len(self.tris), self.n_nodes)).tocsr()
        self._potential = qform(W, M0, W) / (4 * np.pi * EPS0 * (self.er if er is None else er))

    def _ensure_potential(self):
        if self._potential is None:
            if self.dielectric == 'stack':
                from .electrostatic import potential_moments
                self._set_potential(potential_moments(self), er=1.)
            else:
                moments, info = static_moments(self.nodes, self.tris, self.z_of, self.near,
                                               self.outer_levels, scalar_only=True,
                                               **self._fill_options, return_info=True)
                _, M0 = moments
                self.fill_info['capacitance'] = info
                self._set_potential(M0)
        return self._potential

    def _via_matrices(self):
        n = self.nv
        R, L = np.zeros(n), np.zeros((n, n))
        for i, (x, y, rr, z0, z1) in enumerate(self.vseg):
            ri = max(rr - self.plating, 0.0)
            R[i] = (z1 - z0) / (self.sigma * np.pi * (rr ** 2 - ri ** 2))
            for j, (x2, y2, r2, z02, z12) in enumerate(self.vseg):
                s = np.hypot(x - x2, y - y2) if i != j else rr
                L[i, j] = filament_mutual(z0, z1, z02, z12, max(s, rr if i == j else 1e-7))
        return diags(R).tocsr(), L

    def _loops(self, allow_open=False, return_tree=False):
        """The loop basis: C (nbranch x nloop), the source loops' columns separate (sources' branches in no tree)."""
        nb = len(self.br) + len(self.parts)
        ends = list(self.br) + [(self.term_node[p['a']], self.term_node[p['b']]) for p in self.parts]
        is_src = np.array([False] * len(self.br) + [p['kind'] == 'I' for p in self.parts])
        parent = list(range(self.n_nodes))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        tree = np.zeros(nb, bool)
        order = [k for k in range(nb) if not is_src[k]] + [k for k in range(nb) if is_src[k]]
        for k in order:
            if is_src[k]:
                continue
            a, b = ends[k]
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                tree[k] = True
        adj = [[] for _ in range(self.n_nodes)]
        for k in np.flatnonzero(tree):
            a, b = ends[k]
            adj[a].append((b, k, +1)); adj[b].append((a, k, -1))
        up = np.full(self.n_nodes, -1); upb = np.full(self.n_nodes, -1); ups = np.zeros(self.n_nodes, int)
        depth = np.full(self.n_nodes, -1)
        for root in range(self.n_nodes):
            if depth[root] >= 0:
                continue
            depth[root] = 0
            stack = [root]
            while stack:
                u = stack.pop()
                for v, k, sgn in adj[u]:
                    if depth[v] < 0:
                        depth[v], up[v], upb[v], ups[v] = depth[u] + 1, u, k, sgn
                        stack.append(v)
        rows, cols, vals, src_cols, dead = [], [], [], {}, []
        col = 0
        for k in np.flatnonzero(~tree):
            a, b = ends[k]
            if find(a) != find(b):                                # its ends on copper joined only through it
                dead.append(k)
                continue
            # the loop: branch k from a to b, then the tree from b back to a
            path = []
            u, v = b, a
            pu, pv = [], []
            while depth[u] > depth[v]:
                pu.append((upb[u], -ups[u])); u = up[u]           # b towards the root: against the branch's sense
            while depth[v] > depth[u]:
                pv.append((upb[v], ups[v])); v = up[v]
            while u != v:
                pu.append((upb[u], -ups[u])); u = up[u]
                pv.append((upb[v], ups[v])); v = up[v]
            path = pu + pv[::-1]
            rows.append(k); cols.append(col); vals.append(1.0)
            for kb, sg in path:
                rows.append(kb); cols.append(col); vals.append(float(sg))
            if is_src[k]:
                src_cols[k] = col
            col += 1
        C = coo_matrix((vals, (rows, cols)), shape=(nb, col)).tocsc()
        bad = [k for k in dead if is_src[k]]
        if bad and not allow_open:
            names = [self.parts[k - len(self.br)]['name'] for k in bad]
            raise ValueError(f"source(s) {', '.join(names)}: their terminals are on copper joined only through them "
                             f"(an island: a pad with no via or path to the rest): no loop to drive")
        return (C, src_cols, is_src, tree) if return_tree else (C, src_cols, is_src)

    def _assemble(self):
        if self._mats is not None:
            return self._mats
        import time
        t = time.time()
        if self._copper is None:
            Re, Le = self._edge_matrices()
            Rv, Lv = self._via_matrices()
            self._copper = Re, Le, Rv, Lv
        Re, Le, Rv, Lv = self._copper
        ne, nv, npart = self.ne, self.nv, len(self.parts)
        nb = ne + nv + npart
        R = block_diag((Re, Rv, csr_matrix((npart, npart))), format='csr')
        # The magnetic coupling has support only on copper branches. Parts
        # carry their explicit RLC impedance separately, without padding L.
        L = (Le, Lv)
        C, src_cols, is_src, tree = self._loops(allow_open=self.capacitance, return_tree=True)
        Rl, Ll = (None, None) if self.capacitance else (
            (C.T @ R @ C).toarray(), self._project_inductance(C, Le, Lv))
        self._mats = dict(R=R, L=L, C=C, Rl=Rl, Ll=Ll, src=src_cols, is_src=is_src, nb=nb, tree=tree)
        self.log(f"mqs: {C.shape[1]:,} loops; matrices in {time.time() - t:.1f} s")
        return self._mats

    def _project_inductance(self, basis, Le, Lv):
        return qform(basis[:self.ne], Le, basis[:self.ne]) + qform(
            basis[self.ne:self.ne + self.nv], Lv, basis[self.ne:self.ne + self.nv])

    def _inductive_voltage(self, currents):
        Le, Lv = self._mats['L']
        out = np.zeros_like(currents, dtype=complex)
        out[:self.ne] = Le @ currents[:self.ne]
        out[self.ne:self.ne + self.nv] = Lv @ currents[self.ne:self.ne + self.nv]
        return out

    def solve(self, freqs, amps=None):
        """The currents at each frequency (Hz) for the sources' currents (amps: {source: A}, default 1 A each)."""
        if self.capacitance:
            from .peec import solve
            return solve(self, freqs, amps)
        f = np.atleast_1d(np.asarray(freqs, float))
        if f.ndim != 1 or not np.isfinite(f).all() or (f <= 0).any():
            raise ValueError('frequencies must be finite and positive')
        m = self._assemble()
        nb, ne, nv = m['nb'], self.ne, self.nv
        Cd = m['C']
        src = {self.parts[k - ne - nv]['name']: c for k, c in m['src'].items()}
        free = [c for c in range(Cd.shape[1]) if c not in src.values()]
        I = np.zeros((len(f), nb), complex)
        Z = {n: np.zeros(len(f), complex) for n in src}
        Zc = {n: np.zeros(len(f), complex) for n in src}
        cu = np.r_[np.ones(ne + nv), np.zeros(nb - ne - nv)]           # the copper's branches
        for i, fi in enumerate(f):
            w = 2 * np.pi * fi
            zp = np.zeros(nb, complex)                           # the parts' own impedances
            for k, p in enumerate(self.parts):
                if p['kind'] == 'Z':
                    zp[ne + nv + k] = p['R'] + 1j * w * p['L'] + (1 / (1j * w * p['C']) if p.get('C') else 0)
            Zl = m['Rl'] + 1j * w * m['Ll'] + (Cd.T @ diags(zp) @ Cd).toarray()
            dR = self._resistance_delta(fi)
            if dR.nnz:
                Zl += (Cd.T @ dR @ Cd).toarray()
            x = np.zeros(Cd.shape[1], complex)
            for n, c in src.items():
                x[c] = (amps or {}).get(n, 1.0)
            sc = list(src.values())
            rhs = -Zl[np.ix_(free, sc)] @ x[sc]
            x[free] = np.linalg.solve(Zl[np.ix_(free, free)], rhs)
            I[i] = Cd @ x
            v = (m['R'] + dR) @ I[i] + 1j * w * self._inductive_voltage(I[i]) + zp * I[i]
            for n, c in src.items():
                k = next(kk for kk, cc in m['src'].items() if cc == c)
                # Kirchhoff's voltage law round its loop: the drops in the rest of it are the copper loop's Z x I
                column = Cd[:, c].toarray().ravel()
                Z[n][i] = (column @ v - Cd[k, c] * v[k]) / x[c] if x[c] else np.nan
                Zc[n][i] = (column * cu) @ v / x[c] if x[c] else np.nan
        return Result(f, I, Z, [p['name'] for p in self.parts], Zc)

    def moment(self, res, fi):
        """The magnetic dipole moment (A m^2, (3,) complex) of the currents at frequency index fi: m = 1/2 int r x J
        dV over the sheets, the vias and the parts. It sets the field far from the loop (|H| ~ |m| / 2 pi r^3 on its
        axis): the layout's figure of merit for what it radiates at a distance, as the loop's L is for its noise."""
        from .base import _BAR6, _W6
        i = res.i[fi]
        r = self.rwg
        a, b = rwg_currents(r, i[:self.ne] / r.l)
        V = self.nodes[self.tris]
        area = r.area
        Q = np.einsum('qk,tkc->tqc', _BAR6, V)
        J = a[:, None, None] * Q + b[:, None, :]
        w = area[:, None] * _W6[None]
        z = self.z_of[:, None]
        # r x J with J horizontal: (y Jz - z Jy, z Jx - x Jz, x Jy - y Jx), Jz = 0
        mx = (-(z * J[..., 1]) * w).sum()
        my = ((z * J[..., 0]) * w).sum()
        mz = ((Q[..., 0] * J[..., 1] - Q[..., 1] * J[..., 0]) * w).sum()
        m = 0.5 * np.array([mx, my, mz])
        segs = []
        for (x, y, rr, z0, z1), ii in zip(self.vseg, i[self.ne:self.ne + self.nv]):
            segs.append((np.array([x, y, z0]), np.array([x, y, z1]), ii))
        for k, p in enumerate(self.parts):
            ca, cb = self._term_centre(p['a']), self._term_centre(p['b'])
            hgt = np.array([0, 0, p['part_h'] * 1e-3])
            segs.append((ca + hgt, cb + hgt, i[self.ne + self.nv + k]))
        for p0, p1, ii in segs:                                   # a filament: 1/2 I (r0 x r1)
            m = m + 0.5 * ii * np.cross(p0, p1)
        return m

    # ---- the field ----
    def capacitance_matrix(self):
        """Maxwell capacitance (F), node conductor labels, terminal labels.
        Requires capacitance=True; see mom25d.peec.capacitance_matrix.
        """
        if not self.capacitance:
            raise ValueError('capacitance_matrix requires capacitance=True')
        from .peec import capacitance_matrix
        return capacitance_matrix(self)

    def e_field(self, res, fi, pts):
        """Electrostatic E (V/m) of PEEC's surface charges at points (m).
        fi may be an index or a list, like h_field. Excludes inductive -jw A.
        """
        from .peec import electric_field
        return electric_field(self, res, fi, pts)

    def h_grid(self, res, fi, x, y, heights, refine=1, rtol=None, max_refine=8, return_info=False):
        """Fast open-boundary FFT H on uniform x/y grids and absolute z heights
        (all m). Shape (nx, ny, nh, 3), or (nx, ny, nh, 3, nf) for a list of fi.
        `refine` increases internal resolution without changing output points.
        `rtol` checks successive grids at each height/frequency, raising if
        max_refine is reached without agreement. return_info adds metadata.
        This is a grid check, not a mesh/model error bound. Use h_field for
        individual points.
        """
        from .biot import h_grid
        if rtol is not None and (not np.isfinite(rtol) or rtol <= 0):
            raise ValueError('rtol must be finite and positive')
        if not isinstance(max_refine, (int, np.integer)) or max_refine < refine:
            raise ValueError('max_refine must be an integer at least refine')
        H = h_grid(self, res, fi, x, y, heights, refine)
        info = dict(refine=refine, converged=None, relative_change=None)
        if rtol is not None:
            while refine < max_refine:
                refine = min(2 * refine, max_refine)
                finer = h_grid(self, res, fi, x, y, heights, refine)
                axes = (0, 1, 3)
                scale = np.maximum(np.max(np.abs(finer), axis=axes), np.finfo(float).tiny)
                error = np.max(np.abs(finer - H), axis=axes) / scale
                H = finer
                info = dict(refine=refine, converged=bool(np.all(error <= rtol)),
                            relative_change=np.asarray(error).tolist())
                if info['converged']:
                    break
            if not info['converged']:
                raise RuntimeError(f'FFT field did not converge to {rtol:g} at refinement {refine}; '
                                   f'relative change {info["relative_change"]}')
        return (H, info) if return_info else H

    def h_field(self, res, fi, pts):
        """H (n, 3) complex, A/m, at points (n, 3), m, at frequency index fi (or a list of them: H (n, 3, k), the
        geometry made once for all): the sheets' currents (Biot-Savart), the vias' and the parts' filaments."""
        multi = np.ndim(fi) > 0
        fis = list(np.atleast_1d(fi))
        i = res.i[fis].T                                        # (nbranch, k)
        r = self.rwg
        a, b = rwg_currents(r, i[:self.ne] / r.l[:, None])
        sheets = [(self.nodes, self.tris[idx], a[idx], b[idx], float(self.z_of[idx[0]])) for idx in self.layer_tris.values()]
        H = h_sheets(sheets, pts)
        if self.nv:
            v = np.array(self.vseg)
            H = H + h_segments(np.c_[v[:, 0], v[:, 1], v[:, 3]], np.c_[v[:, 0], v[:, 1], v[:, 4]],
                               i[self.ne:self.ne + self.nv], pts)
        p0, p1, ip = [], [], []
        for k, p in enumerate(self.parts):
            ca, cb = self._term_centre(p['a']), self._term_centre(p['b'])
            p0.append(ca + np.array([0, 0, p['part_h'] * 1e-3])); p1.append(cb + np.array([0, 0, p['part_h'] * 1e-3]))
            ip.append(i[self.ne + self.nv + k])
        if p0:
            H = H + h_segments(np.array(p0), np.array(p1), np.array(ip), pts)
        return H if multi else H[:, :, 0]

    def _term_centre(self, name):
        ts = self.term_tris[name]
        c = self.rwg.cent[ts].mean(0)
        return np.array([c[0], c[1], float(self.z_of[ts].mean())])

    def current_density(self, res, fi):
        """{layer: (centroids (m), J (A/m, (n, 2)))} per triangle at its centroid."""
        r = self.rwg
        a, b = rwg_currents(r, res.i[fi][:self.ne] / r.l)
        J = a[:, None] * r.cent + b
        return {lay: (r.cent[idx], J[idx]) for lay, idx in self.layer_tris.items()}


def _inside(poly, pts):
    import shapely
    return shapely.contains_xy(poly.buffer(1e-6), pts[:, 0], pts[:, 1])


if __name__ == '__main__':
    from mom25d.mqs_cli import main
    raise SystemExit(main())
