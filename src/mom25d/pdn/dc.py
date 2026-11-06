"""
DC voltage drop (IR drop) of a power rail and its return: the steady conduction in the copper, each layer's copper a
resistive sheet (sigma x thickness, linear triangles on mom25d's mesh), the vias plated barrels between the layers
they join, the pads equipotential terminals: the source's held at its voltage, each load's drawing its current.

    rail = Rail(board, '+3V3', loads={'U21': 2.0, 'U14': 1.0})       # rail.py: the tree, the source, the parts
    res = solve_rail(rail, volts=3.3, dmax=1.0)
    res['loads']['U21']          # dict(V: what it sees, its supply pads' minus its ground pads', drop, power_drop,
                                 #      ground_rise)
    res['power'].sheets          # {layer: Sheet with V (node volts), J (A per mm of width per triangle)}
    res['power'].vias            # [dict(x, y, r, I, segments)]; res['series']: each series part's current and drop

The network (Network) is the copper of a set of nets (a rail tree, or the ground); a terminal is a set of pads
([(layer, polygon)]: a through-hole pad on every layer it is plated through), meshed with its outline as mesh edges
and held to one voltage (its vertices tied to one junction); a series part (a sense resistor, a bead, a jumper) is
a resistance between its two pads' terminals. A via joins the copper of the nets on every layer between its ends that
it lies on, through a barrel of plating thickness `plating` (mm): rho dz / area.

solve(board, net, source, loads) is the one-net form: source dict(x, y, r, layer, V), loads [dict(name, x, y, r,
layer, I)], each a disc pad; ground=True solves a return (the source at 0 V, the loads' currents put back).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely.geometry import Point
from shapely.ops import unary_union

from .mesh import net_sheets, stiffness, gradient, stack_z

SIGMA_CU = 5.8e7                # S/m
TIE_G = 1e6                     # S: a terminal or a via end to its vertices, in all (1 uOhm: equipotential)


def _barrel_g(r_mm, dz_mm, plating_mm, sigma):
    """A via barrel's conductance, S: sigma x its plating's cross-section / its length."""
    ri = max(r_mm - plating_mm, 0.0)
    return sigma * np.pi * (r_mm ** 2 - ri ** 2) * 1e-6 / (dz_mm * 1e-3)


class Network:
    """The conduction network of the copper of `nets` with its terminals {name: [(layer, polygon)]}. fine: the
    names of terminals meshed finer around (within `fine_r` mm, at dmax_fine), the current crowding into them."""

    def __init__(self, board, nets, terminals, dmax=1.0, sigma=SIGMA_CU, plating=0.025, fine=(), fine_r=2.0,
                 dmax_fine=None, log=print):
        self.board, self.nets = board, {nets} if isinstance(nets, str) else set(nets)
        self.sigma, self.terms = sigma, dict(terminals)
        layers = list(stack_z(board))
        self.vias = [v for v in board['vias'] if v.get('net') in self.nets]
        regions, self.tkey = {}, {}                 # tkey: terminal -> [(layer, index in that layer's regions)]
        for name, pads in self.terms.items():
            for lay, poly in pads:
                if lay not in layers:
                    continue
                regions.setdefault(lay, []).append(poly)
                self.tkey.setdefault(name, []).append((lay, len(regions[lay]) - 1))
        fz = [poly.buffer(fine_r) for n in fine for _, poly in self.terms.get(n, [])]
        fzone = unary_union(fz) if fz else None
        pts = [(v['x'], v['y'], v['r']) for v in self.vias]
        self.sheets = net_sheets(board, self.nets, pts, dmax=dmax, regions=regions, fine=fzone,
                                 dmax_fine=dmax_fine or (dmax / 4 if fzone is not None else None))
        if not self.sheets:
            raise ValueError(f"no copper of {', '.join(sorted(self.nets))}")
        off = 0
        for sh in self.sheets.values():
            sh.off = off
            off += len(sh.nodes)
        self.n_mesh = off
        rows, cols, vals = [], [], []

        def link(a, b, g):
            rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([g, g, -g, -g])
        self._link = link
        blocks = []
        for sh in self.sheets.values():
            K = stiffness(sh.nodes, sh.tris, sigma * sh.t * 1e-3).tocoo()
            blocks.append((K.row + sh.off, K.col + sh.off, K.data))
        self.blocks = blocks
        j = off
        self.via_segs = []                      # per via: [(layer_a, layer_b, junction_a, junction_b, g)]
        for k, v in enumerate(self.vias):
            on = sorted([(sh.z, lay) for lay, sh in self.sheets.items() if sh.at(k) and
                         v['z0'] - 1e-6 <= sh.z <= v['z1'] + 1e-6])
            js = []
            for z, lay in on:
                sh = self.sheets[lay]
                ring = sh.at(k)[1:] or sh.at(k)
                for q in ring:
                    link(j, sh.off + q, TIE_G / len(ring))
                js.append((j, lay, z))
                j += 1
            segs = []
            for (ja, la, za), (jb, lb, zb) in zip(js, js[1:]):
                g = _barrel_g(v['r'], zb - za, plating, sigma)
                link(ja, jb, g)
                segs.append((la, lb, ja, jb, g))
            self.via_segs.append(segs)
        self.tj, self.notes = {}, []            # terminal -> its junction
        for name, keys in self.tkey.items():
            ids = [self.sheets[lay].off + q for lay, i in keys if lay in self.sheets for q in self.sheets[lay].region(i)]
            if not ids:
                self.notes.append(f"terminal {name}: on none of the copper of {', '.join(sorted(self.nets))}")
                continue
            for q in ids:
                link(j, q, TIE_G / len(ids))
            self.tj[name] = j
            j += 1
        self.links = []                         # series parts: (name, terminal a, terminal b, R)
        self.n = j
        self._rows, self._cols, self._vals = rows, cols, vals
        log(f"dc {', '.join(sorted(self.nets))}: {self.n_mesh:,} vertices on {len(self.sheets)} layer(s), "
            f"{sum(1 for s in self.via_segs if s)} vias, {len(self.tj)} terminals")

    def series(self, name, a, b, R):
        """A series part (resistance R, ohms) between terminals a and b."""
        if a in self.tj and b in self.tj:
            self._link(self.tj[a], self.tj[b], 1.0 / max(R, 1e-9))
            self.links.append((name, a, b, R))
        else:
            self.notes.append(f"{name}: a pad not on the copper: left out")

    def matrix(self):
        r_ = np.concatenate([b[0] for b in self.blocks] + [np.asarray(self._rows, int)])
        c_ = np.concatenate([b[1] for b in self.blocks] + [np.asarray(self._cols, int)])
        d_ = np.concatenate([b[2] for b in self.blocks] + [np.asarray(self._vals, float)])
        return coo_matrix((d_, (r_, c_)), shape=(self.n, self.n)).tocsr()

    def solve(self, fixed, inject):
        """Terminal voltages fixed {name: V} and currents injected {name: A, + into the copper}: the voltages,
        currents and loss. Copper not connected to a fixed terminal is left out (NaN) and said."""
        G = self.matrix()
        fx = np.array([self.tj[t] for t in fixed])
        V = np.full(self.n, np.nan)
        V[fx] = [fixed[t] for t in fixed]
        inj = np.zeros(self.n)
        for t, i in inject.items():
            if t in self.tj:
                inj[self.tj[t]] += i
        _, lab = connected_components(G, directed=False)
        live = np.isin(lab, lab[fx])
        notes = list(self.notes)
        if not live[:self.n_mesh].all():
            notes.append(f"{(~live[:self.n_mesh]).sum()} mesh vertices on copper not connected to the source (left out)")
        lost = [t for t in inject if t in self.tj and not live[self.tj[t]]]
        if lost:
            raise ValueError(f"terminal(s) {', '.join(lost)} on copper not connected to the source")
        free = np.flatnonzero(live & ~np.isin(np.arange(self.n), fx))
        V[free] = spsolve(G[free][:, free].tocsc(), inj[free] - G[free][:, fx] @ V[fx])
        Vz = np.nan_to_num(V)
        out = DCResult(self.nets, self.sheets, [], {}, float(Vz @ (G @ Vz)), self.n_mesh, notes)
        for lay, sh in self.sheets.items():
            sh.V = V[sh.off:sh.off + len(sh.nodes)]
            gv = gradient(sh.nodes, sh.tris, np.nan_to_num(sh.V))      # V/mm
            sh.J = self.sigma * sh.t * 1e-3 * np.hypot(gv[:, 0], gv[:, 1])   # A per mm of width
        for v, segs in zip(self.vias, self.via_segs):
            ss = [(la, lb, float(g * (Vz[ja] - Vz[jb]))) for la, lb, ja, jb, g in segs]
            out.vias.append(dict(x=v['x'], y=v['y'], r=v['r'], segments=ss, I=max([abs(s[2]) for s in ss], default=0.0)))
        out.terminals = {t: float(V[j]) for t, j in self.tj.items()}
        out.series = {name: dict(I=abs(float((Vz[self.tj[a]] - Vz[self.tj[b]]) / R)),     # along its current
                                 drop=abs(float(Vz[self.tj[a]] - Vz[self.tj[b]])))
                      for name, a, b, R in self.links}
        return out


@dataclass
class DCResult:
    net: object
    sheets: dict                    # {layer: Sheet with .V (node volts), .J (A/mm per triangle), .off (dof offset)}
    vias: list                      # [dict(x, y, r, I, segments=[(layer_a, layer_b, I)])]
    loads: dict                     # {name: dict(V, I, layer)}
    loss: float                     # W in the copper
    nodes: int
    notes: list = field(default_factory=list)
    source: dict = None
    terminals: dict = None
    series: dict = None


def _disc(p):
    return [(p['layer'], Point(p['x'], p['y']).buffer(p['r'], quad_segs=4))]


def solve(board, net, source, loads, dmax=1.0, sigma=SIGMA_CU, plating=0.025, ground=False, log=print, coarsen=None,
          fine=True):
    """The one-net form (see the module docstring); pads are discs (x, y, r) on a layer."""
    terms = {'_source': _disc(source)}
    terms.update({l.get('name', f'L{j + 1}'): _disc(l) for j, l in enumerate(loads)})
    nw = Network(board, net, terms, dmax=dmax, sigma=sigma, plating=plating, log=log,
                 fine=list(terms) if fine else ())
    for t in terms:
        if t not in nw.tj:
            raise ValueError(f"pad {t} at ({(source if t == '_source' else next(l for l in loads if l.get('name') == t))['x']}, "
                             f"...) is not on {net} copper")
    s = 1 if ground else -1
    r = nw.solve({'_source': 0.0 if ground else float(source['V'])},
                 {l.get('name', f'L{j + 1}'): s * l['I'] for j, l in enumerate(loads)})
    r.loads = {l.get('name', f'L{j + 1}'): dict(V=r.terminals[l.get('name', f'L{j + 1}')], I=l['I'], layer=l['layer'])
               for j, l in enumerate(loads)}
    r.source = dict(source)
    log(f"dc {net}: {r.loss * 1e3:.2f} mW in the copper")
    return r


def supply(board, net, ground_net, source, loads, ground_source=None, ground_loads=None, **kw):
    """The one-net power and return solved, and what each load sees: ({name: dict(V, drop, power_drop, ground_rise, I)},
    power DCResult, ground DCResult)."""
    p = solve(board, net, source, loads, **kw)
    g = solve(board, ground_net, ground_source or source, ground_loads or loads, ground=True, **kw)
    seen = {}
    for name, lp in p.loads.items():
        lg = g.loads[name]
        seen[name] = dict(V=lp['V'] - lg['V'], drop=source['V'] - (lp['V'] - lg['V']),
                          power_drop=source['V'] - lp['V'], ground_rise=lg['V'], I=lp['I'])
    return seen, p, g


def solve_rail(rail, volts, dmax=1.0, sigma=SIGMA_CU, plating=0.025, log=print, fine=True):
    """A rail (rail.Rail) and its return at DC: the source's pads held at `volts`, each load drawing rail.loads[ref]
    amps over its pads on the tree and returning it over its ground pads. {'loads': {ref: dict(V, drop, power_drop,
    ground_rise, I)}, 'series': {ref: dict(I, drop)}, 'power': DCResult, 'ground': DCResult, 'notes'}."""
    from .rail import terminal
    layers = list(stack_z(rail.board))
    g = set(rail.ground)
    pt = {'source': terminal(rail.source_pads, layers)}
    for s in rail.series:
        pt[f'{s.part.ref}:a'] = terminal([q for q in s.part.pads if q.net == s.a], layers)
        pt[f'{s.part.ref}:b'] = terminal([q for q in s.part.pads if q.net == s.b], layers)
    for ref in rail.loads:
        pt[ref] = terminal(rail.parts[ref].on(rail.nets), layers)
    pw = Network(rail.board, rail.nets, pt, dmax=dmax, sigma=sigma, plating=plating, log=log,
                 fine=['source', *rail.loads] if fine else ())
    for s in rail.series:
        pw.series(s.part.ref, f'{s.part.ref}:a', f'{s.part.ref}:b', s.R)
    rp = pw.solve({'source': volts}, {ref: -i for ref, i in rail.loads.items()})
    gt = {'source': terminal(rail.ground_source_pads, layers)}
    gt.update({ref: terminal(rail.parts[ref].on(g), layers) for ref in rail.loads})
    no_gnd = [ref for ref in rail.loads if not gt[ref]]
    for ref in no_gnd:
        gt.pop(ref)
    gn = Network(rail.board, g, gt, dmax=dmax, sigma=sigma, plating=plating, log=log,
                 fine=['source', *rail.loads] if fine else ())
    rg = gn.solve({'source': 0.0}, {ref: i for ref, i in rail.loads.items() if ref in gt})
    loads = {}
    for ref, i in rail.loads.items():
        vp, vg = rp.terminals.get(ref, np.nan), rg.terminals.get(ref, 0.0)
        loads[ref] = dict(V=vp - vg, drop=volts - (vp - vg), power_drop=volts - vp, ground_rise=vg, I=i)
    notes = rp.notes + rg.notes + [f"{r}: no ground pad (its return not modelled)" for r in no_gnd]
    log(f"dc rail {rail.net}: {(rp.loss + rg.loss) * 1e3:.1f} mW in the copper; worst drop "
        f"{max(l['drop'] for l in loads.values()) * 1e3:.2f} mV" if loads else '')
    return dict(loads=loads, series=rp.series, power=rp, ground=rg, notes=notes)
