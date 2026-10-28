"""
mom25d.board - a board in kicad2ems's file format (<name>_board.json, format kicad2ems-board) as a mom25d
model: the canonical board kicad2ems gives every solver, read without kicad2ems.

    import json
    from mom25d.board import BoardModel
    model = BoardModel(json.load(open('out/filter_board.json')))        # or kicad2ems's Board.to_dict()
    sim = model.simulation(dmax=0.3e-3, fmin=1e9, fmax=6e9)
    res = mom25d.api.solve_board('out/filter_board.json', freqs)          # the same in one call

The board is in mm with y up; the model in metres. What a planar solver takes from it:

  ground   the board's ground layer, else the lowest copper layer covering half the outline: the solver's
           infinite plane at z = 0 (its copper is not meshed); copper below it is left out (with a note).
           ground='none': the whole stack instead, no infinite plane: every copper layer meshed (planes too),
           the lowest on interface 0, air above and below (LayerStack bottom='open'), ports referenced to the
           nearest plane under or over them
  stack    the dielectric layers above the ground (whole stack: between the lowest and highest copper), copper
           layers on their interfaces
  copper   per interface; the outline is not modelled: MoM's dielectrics are laterally infinite
  vias     between the interfaces they span that have metal at the via (the ground counts)
  walls    rows of vias every wall_pitch: a planar solver has no vertical sheets
  parts    a series R / L / C element across the gap between the pads, the gap bridged by a strip (a 0 ohm
           part: the strip alone)
  ports    the vertical ports: de-embedded wave ports where a straight feed allows, else lumped (via)
           ports; with pairs=True the legs of a differential port are one mixed-mode port. A gap port (a
           dipole or slot feed) has no planar equivalent.
"""
import numpy as np
from shapely import affinity
from shapely.geometry import shape, box, Point
from shapely.ops import unary_union

from .model import PlanarModel, MM

FORMAT, VERSION = 'kicad2ems-board', 1


WALL_R = 0.05                       # mm: the barrel radius of an edge-wall via
WALL_GAP = 0.1                      # mm: a wall via closer than this to a via already placed is left out


def wall_points(walls, pitch, taken=()):
    """[(wall index, x, y)] (mm) of the vias that stand for edge walls [(axis, at, t0, t1, z0, z1)], every
    `pitch` mm along each. A point within WALL_GAP of a via already there (`taken`, or an earlier wall's: walls
    meeting at a corner share their end) is left out: two vias in one place make the via block singular (the
    SMA launch: 4 duplicated wall ends, an exact null space a complex64 LU cannot factor)."""
    pts, out = [tuple(p) for p in taken], []
    for i, (axis, at, t0, t1, _, _) in enumerate(walls):
        n = max(2, int(round((t1 - t0) / pitch)) + 1)
        for t in np.linspace(t0, t1, n):
            x, y = (at, t) if axis == 'x' else (t, at)
            if pts and np.min(np.hypot(*(np.asarray(pts) - (x, y)).T)) < WALL_GAP:
                continue
            pts.append((x, y))
            out.append((i, x, y))
    return out


class BoardModel(PlanarModel):
    def __init__(self, data, ground=None, port='auto', pairs=False, wall_pitch=0.5e-3, log=print):
        """data: the board (board.json's content). ground: a copper layer name (default: the board's own,
        else the lowest solid one). port: 'auto' | 'wave' | 'via' (lumped). pairs: P<n>+ / P<n>- legs as
        one mixed-mode port. wall_pitch (m): via spacing along edge walls."""
        from .multilayer import Layer, LayerStack
        if data.get('format') != FORMAT:
            raise ValueError(f"not a {FORMAT} file (format {data.get('format')!r})")
        if int(data.get('version', 0)) > VERSION:
            raise ValueError(f"board version {data['version']} is newer than mom25d reads ({VERSION}): update mom25d")
        self.notes, self.log, self.cover = [], log, None
        st = data['stack']
        cu_z = {c['name']: c['z'] for c in st['copper']}
        copper = {l: [shape(p['polygon']) for p in ps] for l, ps in data.get('copper', {}).items()}
        outline = shape(data['outline'])
        self.outline_area = outline.area * MM * MM
        # ---- the ground: z = 0 of the solver (whole stack: the lowest copper layer, meshed) ----
        self.whole = isinstance(ground, str) and ground.lower() == 'none'     # not None: the board's own ground
        name = min(cu_z, key=cu_z.get) if self.whole else ground or data.get('ground')
        if name is None:
            solid = [l for l in sorted(cu_z, key=cu_z.get) if sum(p.area for p in copper.get(l, [])) >= 0.5 * outline.area]
            if not solid:
                raise ValueError("no solid ground plane (no copper layer covers half the board): name the ground layer")
            name = solid[0]
            self.notes.append(f"ground: {name} (the lowest solid copper layer)")
        if name not in cu_z:
            raise ValueError(f"no copper layer {name!r} (layers: {', '.join(cu_z)})")
        self.ground, zg = (None if self.whole else name), cu_z[name]
        # ---- the stack above it, copper on the interfaces ----
        ztop = max(cu_z.values())
        diels = sorted([l for l in st['layers'] if l['z0'] >= zg - 1e-9 and (not self.whole or l['z1'] <= ztop + 1e-9)],
                       key=lambda l: l['z0'])
        self.stack = LayerStack([Layer((l['z1'] - l['z0']) * MM, l['er'], l['tand'], l['name']) for l in diels],
                                bottom='open' if self.whole else 'pec')
        if self.whole:
            self.notes.append(f"whole stack: no infinite ground, {len(cu_z)} copper layers meshed (planes too), "
                              f"air above and below")
        zk = np.array([zg] + [l['z1'] for l in diels])                  # interface heights, mm
        self.iface, below = {}, []
        for lay, z in sorted(cu_z.items(), key=lambda kv: kv[1]):
            if z < zg - 1e-9:
                below += [lay] if lay in copper else []
            elif lay != name or self.whole:
                k = int(np.argmin(np.abs(zk - z)))
                if abs(zk[k] - z) > 1e-6:
                    self.notes.append(f"{lay} at z = {z:g} mm is not on a dielectric interface - left out")
                    continue
                self.iface[lay] = k
        if below:
            self.notes.append(f"copper below the ground ({', '.join(below)}) is left out")
        self._zk = zk
        S = lambda g: affinity.scale(g, MM, MM, origin=(0, 0))
        self.shapes = {k: [S(unary_union(copper[lay])).buffer(0)] for lay, k in self.iface.items() if copper.get(lay)}
        # the signal copper (every net but the ground nets), all layers projected: where the current crowds and
        # the mesh needs its detail (zone=, PlanarModel.simulation); None when the board names no ground net
        gnets = set(data.get('ground_nets') or [])
        sig = [shape(p['polygon']) for lay, ps in data.get('copper', {}).items() if lay in self.iface
               for p in ps if gnets and p.get('net') not in gnets]
        self.signal = S(unary_union(sig)).buffer(0) if sig else None
        # the signal vias (their return vias sit around them) and the thickest dielectric (how far the fields
        # reach from a trace), for the zone's extent
        self.signal_vias = [(v['x'] * MM, v['y'] * MM) for v in data.get('vias', [])
                            if gnets and v.get('net') not in gnets]
        self.h_max = max((l['z1'] - l['z0']) * MM for l in diels) if diels else 0.0
        # ---- vias, edge walls, parts, ports ----
        self.vias = []
        for v in data.get('vias', []):
            via = self._via(v['x'], v['y'], v['z0'], v['z1'], v['r'], f"{v.get('net', '')}@({v['x']:.2f},{-v['y']:.2f})")
            if via:
                if gnets:                           # Simulation(signal_vias=, stitching_vias=): how each is meshed
                    via.kind = 'stitching' if v.get('net') in gnets else 'signal'
                self.vias.append(via)
        nvia = len(self.vias)
        walls = [(w['axis'], w['at'], w['t0'], w['t1'], w['z0'], w['z1']) for w in data.get('walls', [])]
        for i, x, y in wall_points(walls, wall_pitch / MM, [(v['x'], v['y']) for v in data.get('vias', [])]):
            via = self._via(x, y, *walls[i][4:], WALL_R, f"wall@({x:.2f},{-y:.2f})", ring=False)
            if via:
                via.kind = 'stitching'
                self.vias.append(via)
        if walls:
            self.notes.append(f"edge walls: {len(self.vias) - nvia} vias every {wall_pitch * 1e3:g} mm")
        # parts: a series R / L / C element across the gap between the pads, the gap bridged by a strip
        from .sim import SeriesLumped, rlc
        self.lumped = []
        for p in data.get('parts', []):
            k = self.iface.get(p['layer'])
            if k is None:
                self.notes.append(f"{p['ref']}: on {p['layer']}, not a metal interface - left out")
                continue
            s, e = np.array(p['s'][:2]) * MM, np.array(p['e'][:2]) * MM
            self.shapes[k] = [unary_union(self.shapes.get(k, []) + [box(*s, *e)]).buffer(0)]     # the strip
            R, L, C = p.get('R'), p.get('L'), p.get('C')
            given = [v for v in (R, L, C) if v is not None]
            if given == [0.0] and C is None:                               # 0 ohm: the strip is the part
                continue
            Z = rlc(R=R or 0.0, L=L or 0.0, C=C) if len(given) > 1 else rlc(R=R, L=L, C=C, parallel=True)
            a = int(p['axis'])
            self.lumped.append(SeriesLumped(k, 0.5 * (s + e), np.eye(2)[a], e[1 - a] - s[1 - a], Z, name=p['ref']))
        # ports (PlanarModel.port_specs)
        self.port_specs, self.port_z0, legs = [], {}, {}
        for p in data.get('ports', []):
            if p.get('kind', 'z') != 'z':
                raise ValueError(f"{p['name']}: a gap port (a dipole or slot feed) has no planar equivalent")
            k = self.iface.get(p['layer'])
            if k is None:
                raise ValueError(f"{p['name']}: port layer {p['layer']} is not a metal interface above the ground "
                                 f"{self.ground}")
            d = np.zeros(2)
            d[int(p['axis'])] = p['sgn']
            if self.whole:
                kref, kind = self._reference(p, k), 'via'                 # wave ports assume the ground plane
            else:
                kref = 0 if p.get('ref_layer') in (None, '', self.ground) else self.iface.get(p['ref_layer'], 0)
                kind = port
            spec = dict(name=p['name'], c=np.asarray(p['c'], float) * MM, d=d, k=k, w=p['w'] * MM, z0=float(p['z0']),
                        kind=kind, kref=kref, radius=max(p['w'] / 4, 0.02) * MM)
            self.port_z0[p['name']] = float(p['z0'])
            if pairs and p.get('pair'):
                legs.setdefault(p['pair'], {})[p['leg']] = spec
                if len(legs[p['pair']]) == 2:
                    self.port_specs.append(dict(name=p['pair'], legs=legs[p['pair']]))
            else:
                self.port_specs.append(spec)

    def _reference(self, p, k):
        """A port's reference interface in the whole stack: its ref_layer, else the nearest other layer whose
        copper covers the port's centre (the plane under or over it). ValueError when there is none."""
        if p.get('ref_layer') in self.iface:
            return self.iface[p['ref_layer']]
        pt = Point(p['c'][0] * MM, p['c'][1] * MM)
        cover = [j for j, gs in self.shapes.items() if j != k and gs and gs[0].buffer(1e-9).contains(pt)]
        if not cover:
            raise ValueError(f"{p['name']}: no copper under or over the port on another layer to reference it to "
                             f"(whole stack: name its ref_layer)")
        j = min(cover, key=lambda j: abs(self.stack.z[j] - self.stack.z[k]))
        self.notes.append(f"{p['name']}: lumped port to {next(n for n, i in self.iface.items() if i == j)}")
        return j

    def _via(self, x, y, z0, z1, r, name, ring=True, Z=None):
        """A via (board mm) over the interfaces between z0 and z1 that have metal at (x, y) (the ground counts)."""
        from .sim import Via
        metal = lambda k: (k == 0 and not self.whole) or \
            (bool(g := self.shapes.get(k)) and g[0].buffer(1e-9).contains(Point(x * MM, y * MM)))
        ks = sorted(k for k in [0] + list(self.iface.values()) if z0 - 1e-9 <= self._zk[k] <= z1 + 1e-9 and metal(k))
        if len(set(ks)) < 2:
            return None
        return Via(x * MM, y * MM, ks[0], ks[-1], radius=r * MM, name=name, ring=ring, Z=Z)
