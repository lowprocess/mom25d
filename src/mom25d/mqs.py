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
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from .base import MU0, RWG, combine, _BAR3, _W3, quad_pts
from .sing import offplane_pairs
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


class MQS:
    def __init__(self, board, nets, terminals=None, region=None, dmax=0.5, sigma=SIGMA_CU, plating=0.025, log=print,
                 fine=None, dmax_fine=None):
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
        log(f"mqs: {Nt:,} triangles on {len(self.layer_tris)} layer(s), {self.ne:,} edges, {self.nv} via segments, "
            f"{len(self.terms)} terminals")

    # ---- the parts ----
    def source(self, name, a, b, part_h=0.5):
        """A 1 A current source (scaled by solve's amps) through a part from terminal a to terminal b."""
        self.parts.append(dict(name=name, a=a, b=b, kind='I', part_h=part_h))
        self._mats = None

    def element(self, name, a, b, R=0.0, L=0.0, C=None, ESR=None, ESL=None, part_h=0.5):
        """A series R, L (and C) through a part between terminals a and b."""
        self.parts.append(dict(name=name, a=a, b=b, kind='Z', R=ESR if ESR is not None else R,
                               L=ESL if ESL is not None else L, C=C, part_h=part_h))
        self._mats = None

    # ---- the matrices ----
    def _edge_matrices(self):
        r = self.rwg
        Nt = len(self.tris)
        M0 = np.zeros((Nt, Nt)); Mrx = np.zeros((Nt, Nt)); Mry = np.zeros((Nt, Nt)); Mrr = np.zeros((Nt, Nt))
        layers = list(self.layer_tris)
        for la in layers:
            for lb in layers:
                pa, pb = self.layer_tris[la], self.layer_tris[lb]
                d = float(self.z_of[pa[0]] - self.z_of[pb[0]])
                P, Q = np.meshgrid(pa, pb, indexing='ij')
                m0, mro, mrs, mrr = offplane_pairs(self.nodes, self.tris, P.ravel(), Q.ravel(), d, chunk=200_000)
                M0[np.ix_(pa, pb)] = m0.reshape(P.shape)
                Mrx[np.ix_(pa, pb)] = mro[:, 0].reshape(P.shape)
                Mry[np.ix_(pa, pb)] = mro[:, 1].reshape(P.shape)
                Mrr[np.ix_(pa, pb)] = mrr.reshape(P.shape)
        L = MU0 / (4 * np.pi) * combine(r, Mrr, Mrx, Mry, M0)
        L = 0.5 * (L + L.T)
        # R: the Gram matrix of the RWG functions, each triangle at its sheet resistance
        Ar, Pq = r.area, quad_pts(r, _BAR3)
        rs = 1.0 / (self.sigma * self.t_of)
        m = lambda v: np.diag(Ar * rs * (_W3[None, :] * v).sum(1))
        R = combine(r, m((Pq ** 2).sum(-1)), m(Pq[..., 0]), m(Pq[..., 1]), np.diag(Ar * rs))
        # in total currents across each edge (i = l I): Z / (l_m l_n)
        s = 1.0 / r.l
        return R * np.outer(s, s), L * np.outer(s, s)

    def _via_matrices(self):
        n = self.nv
        R, L = np.zeros(n), np.zeros((n, n))
        for i, (x, y, rr, z0, z1) in enumerate(self.vseg):
            ri = max(rr - self.plating, 0.0)
            R[i] = (z1 - z0) / (self.sigma * np.pi * (rr ** 2 - ri ** 2))
            for j, (x2, y2, r2, z02, z12) in enumerate(self.vseg):
                s = np.hypot(x - x2, y - y2) if i != j else rr
                L[i, j] = filament_mutual(z0, z1, z02, z12, max(s, rr if i == j else 1e-7))
        return np.diag(R), L

    def _loops(self):
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
        if bad:
            names = [self.parts[k - len(self.br)]['name'] for k in bad]
            raise ValueError(f"source(s) {', '.join(names)}: their terminals are on copper joined only through them "
                             f"(an island: a pad with no via or path to the rest): no loop to drive")
        return C, src_cols, is_src

    def _assemble(self):
        if self._mats is not None:
            return self._mats
        import time
        t = time.time()
        Re, Le = self._edge_matrices()
        Rv, Lv = self._via_matrices()
        ne, nv, npart = self.ne, self.nv, len(self.parts)
        nb = ne + nv + npart
        R = np.zeros((nb, nb)); L = np.zeros((nb, nb))
        R[:ne, :ne], L[:ne, :ne] = Re, Le
        R[ne:ne + nv, ne:ne + nv], L[ne:ne + nv, ne:ne + nv] = Rv, Lv
        C, src_cols, is_src = self._loops()
        Cd = C.toarray()
        Rl, Ll = Cd.T @ R @ Cd, Cd.T @ L @ Cd
        self._mats = dict(R=R, L=L, C=Cd, Rl=Rl, Ll=Ll, src=src_cols, is_src=is_src, nb=nb)
        self.log(f"mqs: {Cd.shape[1]:,} loops; matrices in {time.time() - t:.1f} s")
        return self._mats

    def solve(self, freqs, amps=None):
        """The currents at each frequency (Hz) for the sources' currents (amps: {source: A}, default 1 A each)."""
        m = self._assemble()
        f = np.atleast_1d(np.asarray(freqs, float))
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
            Zl = m['Rl'] + 1j * w * m['Ll'] + (Cd.T * zp) @ Cd
            x = np.zeros(Cd.shape[1], complex)
            for n, c in src.items():
                x[c] = (amps or {}).get(n, 1.0)
            sc = list(src.values())
            rhs = -Zl[np.ix_(free, sc)] @ x[sc]
            x[free] = np.linalg.solve(Zl[np.ix_(free, free)], rhs)
            I[i] = Cd @ x
            v = (m['R'] + 1j * w * m['L']) @ I[i] + zp * I[i]      # branch voltages
            for n, c in src.items():
                k = next(kk for kk, cc in m['src'].items() if cc == c)
                # Kirchhoff's voltage law round its loop: the drops in the rest of it are the copper loop's Z x I
                Z[n][i] = (Cd[:, c] @ v - Cd[k, c] * v[k]) / x[c]
                Zc[n][i] = (Cd[:, c] * cu) @ v / x[c]
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
