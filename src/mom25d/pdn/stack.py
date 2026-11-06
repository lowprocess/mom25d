"""
A rail's PDN impedance over all of its plane pairs: every layer where the rail tree (rail.Rail) is a plane, against
the nearest ground plane above and below it, each pair a cavity (cavity.py's equations on its own mesh), solved
together; the decaps, the regulator and the loads attached through their own copper; the rail's nets without a
plane (a filtered branch after a bead, a regulator's output before its sense resistor) as lumped nodes.

    rail = Rail(board, '+3V3', loads={'U21': 2.0})
    pdn = RailPDN(rail, vrm=dict(R=1e-3, L=30e-9))
    Z = pdn.z(f)                   # {load: |Z(f)| complex}, with every decap and the regulator attached
    pdn.parts                      # per part: its connection (cavity vias or lumped node), mount L, element

Joined:
  * two cavities sharing the rail plane (a plane between two grounds) at every ground via through both their
    grounds; two sharing a ground at every rail via through both rails; and both at every part's port (its vias);
  * a part on a plane net through its pad's copper to the vias there, which reach the plane: the path's
    inductance mu0 h N (h: the height of that copper over the nearest plane, N: the squares from the pad to the vias,
    from a conduction solve of that copper) on each side, and the via pair's mu0 h / pi ln(s / r) (s: the
    distance from its rail vias to its ground vias), shared by the vias of each side;
  * a series part (a sense resistor, a bead) between its two sides: R, a bead R_dc + (j w L || R_100) with
    |Z| = R_100 at 100 MHz, plus its package's ESL.
Each cavity's voltage is the rail plane's minus its ground's, the ground planes taken as joined at the ports.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from ..base import MU0, EPS0
from .cavity import fill_holes, surface_resistance, SIGMA_CU
from .mesh import copper_of, mesh_sheet, stiffness, mass, stack_z
from .rail import terminal, part_value, field_value

ESL_PKG = {'0201': 0.25e-9, '0402': 0.35e-9, '0603': 0.45e-9, '0805': 0.6e-9, '1206': 0.8e-9, '1210': 0.9e-9}
CLEAR = 0.3                 # mm beyond a via's radius: its antipad, filled in the cavity (a via through it is a port)


def package(lib):
    m = re.search(r'(?<!\d)(0201|0402|0603|0805|1206|1210)(?!\d)', lib or '')
    if m:
        return m.group(1)
    m = re.search(r'(\d{4})Metric', lib or '')
    return {'0603': '0201', '1005': '0402', '1608': '0603', '2012': '0805', '3216': '1206', '3225': '1210'}.get(
        m.group(1)) if m else None


@dataclass
class Cavity:
    rail_layer: str
    ground_layer: str
    geom: object
    d: float                # mm
    er: float
    tand: float
    t: float                # mm (the thinner plane)
    nodes: np.ndarray = None
    tris: np.ndarray = None
    K: object = None
    M: object = None


@dataclass
class Conn:
    """How one side of a part meets the PDN: the vias of its pad's copper that reach a plane (cavity), or a lumped
    node (a net with no plane)."""
    vias: list = field(default_factory=list)        # [(x, y, r)]
    node: str = None
    L_path: float = 0.0                               # H, its pad to its vias
    h: float = 0.0                                    # mm, its pad's layer to the planes
    why: str = ''


class RailPDN:
    def __init__(self, rail, dmax=1.0, plane_min=None, vrm=None, hole_fill=2.0, window=6.0, log=print):
        self.rail, self.log, self.vrm = rail, log, dict(vrm or dict(R=1e-3, L=30e-9))
        b = self.board = rail.board
        zt = self.zt = stack_z(b)
        self.layers = sorted(zt, key=lambda l: zt[l][0])
        nets, g = rail.nets, set(rail.ground)
        area = shape(b['outline']).area
        plane_min = plane_min or 0.1 * area
        rv = [v for v in b['vias'] if v['net'] in nets]
        gv = [v for v in b['vias'] if v['net'] in g]
        self.rail_vias, self.gnd_vias = rv, gv
        rcover = unary_union([Point(v['x'], v['y']).buffer(v['r'] + CLEAR, quad_segs=4) for v in rv]) if rv else None
        gcover = unary_union([Point(v['x'], v['y']).buffer(v['r'] + CLEAR, quad_segs=4) for v in gv]) if gv else None
        rail_planes = {l: copper_of(b, nets, l) for l in self.layers}
        rail_planes = {l: q for l, q in rail_planes.items() if q is not None and q.area > plane_min}
        gnd_planes = {l: copper_of(b, g, l) for l in self.layers}
        gnd_planes = {l: q for l, q in gnd_planes.items() if q is not None and q.area > plane_min}
        self.plane_nets = {n for n in nets if any(copper_of(b, n, l) is not None and copper_of(b, n, l).area > plane_min
                                                 for l in rail_planes)}
        self.cavities, self.notes = [], list(rail.notes)
        for rl, rg in rail_planes.items():
            zr = zt[rl][0]
            for side in (-1, 1):
                cand = [l for l in gnd_planes if (zt[l][0] - zr) * side > 0]
                if not cand:
                    continue
                gl = min(cand, key=lambda l: abs(zt[l][0] - zr))
                between = [l for l in self.layers if (zt[l][0] - zr) * side > 0 and abs(zt[l][0] - zr) < abs(zt[gl][0] - zr)]
                if any(l in rail_planes for l in between):
                    continue                            # another rail plane is nearer that way: its own pair
                geom = fill_holes(rg, hole_fill, gcover).intersection(fill_holes(gnd_planes[gl], hole_fill, rcover)).buffer(0)
                if geom.is_empty or geom.area < 0.05 * plane_min:
                    continue
                za, zb = sorted([zr, zt[gl][0]])
                slabs = [(max(l['z0'], za), min(l['z1'], zb), l['er'], l.get('tand', 0.0))
                         for l in b['stack']['layers'] if min(l['z1'], zb) > max(l['z0'], za)]
                d = sum(q - p for p, q, _, _ in slabs)
                er = d / sum((q - p) / e for p, q, e, _ in slabs)
                tand = sum((q - p) / e * tn for p, q, e, tn in slabs) / sum((q - p) / e for p, q, e, _ in slabs)
                self.cavities.append(Cavity(rl, gl, geom, d, er, tand, min(zt[rl][1], zt[gl][1])))
        if not self.cavities:
            raise ValueError(f"{rail.net}: no plane pair (rail and ground planes over {plane_min:.0f} mm2)")
        self.plane_layers = sorted({c.rail_layer for c in self.cavities} | {c.ground_layer for c in self.cavities},
                                   key=lambda l: zt[l][0])
        # the parts and how each side meets the PDN
        self.parts = {}
        for p in rail.decaps:
            C = part_value(p) if p.props.get('Val') or p.value else None
            if C is None or C <= 0 or C > 1:
                C = None
            if C is None:
                self.notes.append(f"{p.ref}: no capacitance in its value ({p.value!r}): left out")
                continue
            pkg = package(p.lib)
            esr, esl = field_value(p, 'ESR', 'esr'), field_value(p, 'ESL', 'esl')
            el = dict(C=C, ESR=esr if esr is not None else (0.02 if C < 1e-6 else 0.005),
                      ESL=esl if esl is not None else ESL_PKG.get(pkg, 0.5e-9), defaulted=esr is None or esl is None, pkg=pkg)
            self.parts[p.ref] = dict(role='decap', el=el, part=p)
        for p in rail.source_parts:
            self.parts[p.ref] = dict(role='source', el=dict(self.vrm), part=p)
        for ref in rail.loads:
            self.parts[ref] = dict(role='load', el=None, part=rail.parts[ref])
        for name, info in self.parts.items():
            p = info['part']
            pads = p.on(nets) if info['role'] != 'source' else rail.source_pads
            info['rail'] = self._connect(pads, nets, rv, window)
            info['gnd'] = self._connect(p.on(g), g, gv, window) if info['role'] != 'source' else \
                self._connect(rail.ground_source_pads, g, gv, window)
        self.series = []
        for s in rail.series:
            za = self._connect([q for q in s.part.pads if q.net == s.a], nets, rv, window)
            zb = self._connect([q for q in s.part.pads if q.net == s.b], nets, rv, window)
            self.series.append(dict(s=s, a=za, b=zb, esl=ESL_PKG.get(package(s.part.lib), 0.5e-9)))
        for info in self.parts.values():
            info['mount'] = self._mount(info['rail'], info['gnd'])
        self._assemble(dmax)

    # ---- a part's copper ----------------------------------------------------------------------------------------
    def _connect(self, pads, nets, vias, window):
        """Conn of a set of pads (one side of a part) on `nets`: their pad copper's vias that reach a plane, or the
        lumped node of a planeless net."""
        if not pads:
            return Conn(why='no pad')
        net = pads[0].net
        if net not in self.plane_nets and net in self.rail.nets:
            return Conn(node=net, why='planeless net')
        lay = next((l for l in pads[0].layers if l in self.zt), 'F.Cu' if pads[0].layers and '*' in pads[0].layers[0] else None)
        if lay is None:
            return Conn(why='pad on no copper layer')
        zl = self.zt[lay][0]
        if lay in self.plane_layers and any(c.rail_layer == lay or c.ground_layer == lay for c in self.cavities):
            x, y = pads[0].xy
            return Conn(vias=[(x, y, 0.1)], h=0.0, why='pad on the plane')
        cu = copper_of(self.board, nets, lay)
        if cu is None:
            return Conn(why='no copper')
        pg = unary_union([q.geom for q in pads])
        win = pg.buffer(window)
        isl = [q for q in getattr(cu.intersection(win), 'geoms', [cu.intersection(win)])
               if q.geom_type == 'Polygon' and q.buffer(1e-4).intersects(pg)]
        isl = unary_union(isl) if isl else None
        if isl is None or isl.is_empty:
            return Conn(why='pad not on its copper')
        planes = [self.zt[l][0] for l in self.plane_layers]
        reach = [v for v in vias if isl.buffer(1e-4).contains(Point(v['x'], v['y'])) and
                 any(v['z0'] - 1e-6 <= z <= v['z1'] + 1e-6 for z in planes) and
                 (v['z0'] - 1e-6 <= zl <= v['z1'] + 1e-6)]
        if not reach:
            return Conn(why=f'no via to a plane within {window} mm of its pad')
        h = min(abs(zl - z) for z in planes)
        # the squares from the pad to its vias: a conduction solve of that copper alone
        from .dc import Network
        terms = {'pad': [(lay, pg.intersection(isl))]}
        terms.update({f'v{k}': [(lay, Point(v['x'], v['y']).buffer(v['r'], quad_segs=4))] for k, v in enumerate(reach)})
        bd = dict(self.board, copper={lay: [dict(net='_isl', polygon=isl.__geo_interface__)]}, vias=[],
                  stack=dict(self.board['stack'], copper=[c for c in self.board['stack']['copper'] if c['name'] == lay]))
        try:
            nw = Network(bd, '_isl', terms, dmax=max(0.15, min(0.5, isl.area ** 0.5 / 20)), log=lambda *a: None)
            r = nw.solve({'pad': 1.0, **{f'v{k}': 0.0 for k in range(len(reach))}}, {})
            squares = (1.0 / max(r.loss, 1e-12)) * SIGMA_CU * self.zt[lay][1] * 1e-3
        except Exception as e:                          # a pad on its via (via in pad): no path
            squares = 0.0
        return Conn(vias=[(v['x'], v['y'], v['r']) for v in reach], L_path=MU0 * h * 1e-3 * squares, h=h,
                    why=f'{len(reach)} via(s), {squares:.2f} squares over {h:.2f} mm')

    def _mount(self, rc, gc):
        """The series inductance between a part's pads and the plane pair: its two paths and its via pair."""
        if not rc.vias:
            return 0.0
        L = rc.L_path + gc.L_path
        if gc.vias:
            s = min(np.hypot(a[0] - c[0], a[1] - c[1]) for a in rc.vias for c in gc.vias)
            r = min(a[2] for a in rc.vias)
            h = 0.5 * (rc.h + gc.h) + 0.5 * min(c.d for c in self.cavities)
            L += MU0 * h * 1e-3 / np.pi * np.log(max(s, 2.5 * r) / r) / max(1, min(len(rc.vias), len(gc.vias)))
        return L

    # ---- the cavities, the ports, the nodes -------------------------------------------------------------------
    def _assemble(self, dmax):
        # the points each cavity needs as vertices: every part's vias, and the vias that join cavities
        port_pts = {}
        for name, info in self.parts.items():
            if info['rail'].vias:
                port_pts[name] = info['rail'].vias
        for k, sr in enumerate(self.series):
            for side in ('a', 'b'):
                if sr[side].vias:
                    port_pts[f'~{k}{side}'] = sr[side].vias
        ties = []                                       # (cavity i, cavity j, (x, y, r))
        cs = self.cavities
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if cs[i].rail_layer == cs[j].rail_layer:
                    zs = [self.zt[cs[i].ground_layer][0], self.zt[cs[j].ground_layer][0]]
                    vs = self.gnd_vias
                elif cs[i].ground_layer == cs[j].ground_layer:
                    zs = [self.zt[cs[i].rail_layer][0], self.zt[cs[j].rail_layer][0]]
                    vs = self.rail_vias
                else:
                    continue
                for v in vs:
                    if v['z0'] - 1e-6 <= min(zs) and v['z1'] + 1e-6 >= max(zs):
                        ties.append((i, j, (v['x'], v['y'], v['r'])))
        names = list(port_pts)
        self.dofmap, parent = [], {}

        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        off = 0
        for ci, c in enumerate(cs):
            pts, owner = [], []
            for n in names:
                for v in port_pts[n]:
                    pts.append(v); owner.append(('port', n))
            for t in ties:
                if ci in t[:2]:
                    pts.append(t[2]); owner.append(('tie', t))
            ok = [k for k, (x, y, r) in enumerate(pts) if c.geom.buffer(1e-6).contains(Point(x, y))]
            nodes, tris, rings, _ = mesh_sheet(c.geom, [pts[k] for k in ok], dmax)
            c.nodes, c.tris, c.off = nodes, tris, off
            c.K = stiffness(nodes, tris).tocoo()
            c.M = mass(nodes, tris).tocoo()
            for k, ring in zip(ok, rings):
                if ring is None:
                    continue
                kind, who = owner[k]
                key = ('port', who) if kind == 'port' else ('tie', id(who))
                for q in ring:
                    union(key, ('v', off + q))
            off += len(nodes)
            self.log(f"cavity {c.rail_layer}/{c.ground_layer}: {c.geom.area:.0f} mm2, d {c.d:.3g} mm, er {c.er:.3g}, "
                     f"C {EPS0 * c.er * c.geom.area * 1e-6 / (c.d * 1e-3) * 1e9:.3g} nF; {len(nodes):,} vertices")
        nv = off
        roots = {}
        idx = np.empty(nv, int)
        for q in range(nv):
            r = find(('v', q))
            idx[q] = roots.setdefault(r, len(roots))
        self.port_dof = {n: roots[find(('port', n))] for n in names if find(('port', n)) in roots}
        missing = [n for n in names if n not in self.port_dof]
        for n in missing:
            self.notes.append(f"{n}: its vias are off every plane pair's copper: left out")
        nodes_l = sorted({info['rail'].node for info in self.parts.values() if info['rail'].node} |
                         {sr[s].node for sr in self.series for s in ('a', 'b') if sr[s].node})
        self.node_dof = {n: len(roots) + k for k, n in enumerate(nodes_l)}
        self.n = len(roots) + len(nodes_l)
        self.P = coo_matrix((np.ones(nv), (np.arange(nv), idx)), shape=(nv, self.n)).tocsr()
        self.nv = nv
        for c in cs:
            Pc = self.P[c.off:c.off + len(c.nodes)]
            c.Kg = (Pc.T @ c.K.tocsr() @ Pc).tocsc()
            c.Mg = (Pc.T @ c.M.tocsr() @ Pc).tocsc() * 1e-6
        self.log(f"rail PDN {self.rail.net}: {len(cs)} plane pair(s), {nv:,} vertices, {len(self.port_dof)} ports, "
                 f"{len(nodes_l)} lumped node(s), {len(ties)} ties between pairs")

    def dof(self, name):
        info = self.parts.get(name)
        if info is not None and info['rail'].node:
            return self.node_dof[info['rail'].node]
        return self.port_dof.get(name)

    def _series_z(self, sr, w):
        s = sr['s']
        if s.kind == 'bead':
            Lb = s.z100 / (2 * np.pi * 1e8)
            zl = 1j * w * Lb
            return s.R + zl * s.z100 / (zl + s.z100) + 1j * w * sr['esl']
        return s.R + 1j * w * sr['esl']

    def matrix(self, f, bare=False):
        w = 2 * np.pi * f
        Y = None
        for c in self.cavities:
            zs = 2 * surface_resistance(f, c.t * 1e-3) + 1j * w * MU0 * c.d * 1e-3
            ysh = 1j * w * EPS0 * c.er * (1 - 1j * c.tand) / (c.d * 1e-3)
            y = c.Kg / zs + c.Mg * ysh
            Y = y if Y is None else Y + y
        y = np.zeros(self.n, complex)
        rows, cols, vals = [], [], []
        if not bare:
            for name, info in self.parts.items():
                d = self.dof(name)
                el = info['el']
                if d is None or el is None:
                    continue
                m = info['mount'] + (0.5e-9 if info['rail'].node else 0.0)     # a lumped node's part: a nominal 0.5 nH
                if 'C' in el:
                    z = el['ESR'] + 1j * w * (el['ESL'] + m) + 1 / (1j * w * el['C'])
                else:
                    z = el.get('R', 0.0) + 1j * w * (el.get('L', 0.0) + m)
                y[d] += 1 / z
        for k, sr in enumerate(self.series):                    # the series parts join their two sides
            da = sr['a'].node and self.node_dof[sr['a'].node] if sr['a'].node else self.port_dof.get(f'~{k}a')
            db = sr['b'].node and self.node_dof[sr['b'].node] if sr['b'].node else self.port_dof.get(f'~{k}b')
            if da is None or db is None or da == db:
                continue
            g = 1 / self._series_z(sr, w)
            rows += [da, db, da, db]; cols += [da, db, db, da]; vals += [g, g, -g, -g]
        Y = Y + diags(y + 1e-12)
        if rows:
            Y = Y + coo_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return Y.tocsc()

    def z(self, f, observe=None, bare=False):
        """{name: Z(f) complex}: the impedance each load (or `observe`: part names) sees between its pads, the
        rest attached (bare: the planes alone, no part, no series path)."""
        names = [n for n in (observe or [n for n, i in self.parts.items() if i['role'] == 'load']) if self.dof(n) is not None]
        f = np.atleast_1d(np.asarray(f, float))
        out = {n: np.zeros(len(f), complex) for n in names}
        E = np.zeros((self.n, len(names)))
        for k, n in enumerate(names):
            E[self.dof(n), k] = 1.0
        for i, fi in enumerate(f):
            X = splu(self.matrix(fi, bare=bare)).solve(E.astype(complex))
            for k, n in enumerate(names):
                info = self.parts[n]
                extra = 1j * 2 * np.pi * fi * (info['mount'] if info['rail'].vias else 0.0)   # its own pads to the planes
                out[n][i] = X[self.dof(n), k] + extra
        return out

    def describe(self):
        lines = [f"{c.rail_layer} over {c.ground_layer}: {c.d:.3g} mm, er {c.er:.3g}, {c.geom.area:.0f} mm2"
                 for c in self.cavities]
        for name, info in sorted(self.parts.items()):
            r, g = info['rail'], info['gnd']
            where = f"node {r.node}" if r.node else (f"rail {r.why}; gnd {g.why}" if r.vias else f"not attached: {r.why}")
            lines.append(f"  {info['role']:6s} {name}: {where}; mount {info['mount'] * 1e9:.2f} nH")
        return lines + [f"note: {n}" for n in self.notes]
