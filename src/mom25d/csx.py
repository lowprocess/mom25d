"""
mom25d.csx - an AppCSXCAD / openEMS geometry file (.xml, CSXCAD's ContinuousStructure) as a mom25d model,
so a MoM run and an openEMS (FDTD) run can solve the very same converted board.

    model = CSXModel('board.xml')                       # ground: from the hints, else the lowest solid plane
    sim = model.simulation(dmax=0.3e-3, fmin=1e9, fmax=6e9)
    mom25d board.xml --fmin 1 --fmax 6                  # the same from the command line

What is read (CSX drawing units scaled by the grid's DeltaUnit; the mesh itself is ignored)
  Material        dielectric slabs: LinPoly / Box primitives give z ranges, Epsilon and Kappa (a
                  conductivity, turned into tan(delta) at f_loss) the layer constants. Slabs must be
                  laterally uniform for a planar solver: at each height the slab covering the most area
                  wins. Debye materials use Epsilon plus their poles' deltas (static value).
  Metal,          planar copper: Polygon (NormDir 2) and zero-height Box primitives at their height;
  ConductingSheet the conductivity of a conducting sheet becomes the metal loss (sim.Metal)
  Cylinder        vertical cylinders (any metal) are vias between the metal interfaces they cross
  vertical Box    vertical metal sheets (e.g. edge walls) become rows of vias (pitch wall_pitch)
  LumpedElement   'port_resist_<n>' (openEMS lumped ports): port n, its R the reference impedance;
                  other horizontal ones: series R / L / C across a cut (a metal strip bridges their box);
                  vertical ones: a via with that impedance

The infinite ground (z = 0 of the solver) is the `ground` copper layer: named in the call or the hints,
else the lowest metal layer that covers at least half of the board. Metal on or below it is not meshed.

Ports. An openEMS lumped port is a vertical sheet across a trace, from its reference copper up to it. Its
feed direction is the horizontal axis the sheet has no extent along; the sign (into the circuit) is the
side with copper. port='auto' (default) makes a de-embedded wave port where a straight feed allows and a
lumped port (a via with the port in it) otherwise; 'wave' / 'via' force one kind.

Hints. A converter may add an element <mom25d version="1" ground="B.Cu"> to the root with children
    <Layer name="F.Cu" z="1.51"/>                                   copper layer names and heights
    <Dielectric name="core" z0="0" z1="1.51" er="4.4" tand="0.02"/>  exact constants (overrides Kappa)
    <Port number="1" name="P1" layer="F.Cu" ref_layer="B.Cu" x=".." y=".." dx="1" dy="0" width=".."/>
(coordinates in drawing units, direction (dx, dy) into the circuit). CSXCAD ignores the element when it
reads the file (AppCSXCAD drops it when it saves). Without hints everything comes from the geometry.
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union

from .model import PlanarModel, DiffLegs, MM

EPS0 = 8.8541878128e-12
_Z_TOL = 1e-6                       # relative to the drawing unit: heights closer than this are one height


def _num(el, key, default=None):
    v = el.get(key)
    if v is None:
        return default
    return float(v.split(',')[0])                    # anisotropic values: 'ex,ey,ez' -> ex


def _pt(el):
    return np.array([float(el.get(k)) for k in ('X', 'Y', 'Z')])


class _Prim:
    """One CSX primitive in drawing units: kind 'box' / 'poly' / 'linpoly' / 'cyl', with what it needs."""
    def __init__(self, el):
        self.el, self.kind, self.priority = el, None, int(float(el.get('Priority', 0)))
        tag = el.tag
        if tag == 'Box':
            a, b = _pt(el.find('P1')), _pt(el.find('P2'))
            self.kind, self.lo, self.hi = 'box', np.minimum(a, b), np.maximum(a, b)
        elif tag in ('Polygon', 'LinPoly'):
            self.normal = int(float(el.get('NormDir', 2)))
            self.elev = float(el.get('Elevation', 0))
            self.length = float(el.get('Length', 0)) if tag == 'LinPoly' else 0.0
            pts = [(float(v.get('X1')), float(v.get('X2'))) for v in el.findall('Vertex')]
            self.poly = Polygon(pts).buffer(0) if len(pts) >= 3 else None
            self.kind = 'linpoly' if tag == 'LinPoly' else 'poly'
        elif tag == 'Cylinder':
            self.p1, self.p2 = _pt(el.find('P1')), _pt(el.find('P2'))
            self.r = float(el.get('Radius'))
            self.kind = 'cyl'
        self.transformed = el.find('Transformation') is not None


class CSXModel(PlanarModel):
    def __init__(self, path, ground=None, cover=None, port='auto', f_loss=None, pairs=False, wall_pitch=0.5e-3,
                 log=print):
        """path: the CSX .xml. ground / cover: copper layer names (or heights in drawing units). port: 'auto' |
        'wave' | 'via'. f_loss (Hz): frequency at which a material's Kappa gives its tan(delta) (needed when
        the hints do not give tan(delta); default 1 GHz with a note). pairs: P<n>+ / P<n>- hint ports become
        one mixed-mode port (default: single-ended legs). wall_pitch (m): via spacing along vertical metal."""
        from .multilayer import Layer, LayerStack
        from .sim import Via
        self.path, self.notes, self.log = path, [], log
        root = ET.parse(path).getroot()
        cs = root if root.tag == 'ContinuousStructure' else root.find('.//ContinuousStructure')
        if cs is None:
            raise ValueError(f"{path}: no ContinuousStructure (not a CSXCAD / openEMS geometry file)")
        hints = root.find('mom25d') if root.find('mom25d') is not None else cs.find('mom25d')
        grid = cs.find('RectilinearGrid')
        self.unit = float(grid.get('DeltaUnit', 1.0)) if grid is not None else 1.0      # metres per drawing unit
        u = self.unit
        ztol = _Z_TOL * (1e-3 / u)                                                     # 1 nm
        self.hints = self._read_hints(hints)

        # ---- properties --------------------------------------------------------------------------------
        slabs, sheets, cyls, walls, lumped, skipped = [], [], [], [], [], set()
        self.sigma = self.sheet_t = None
        for prop in cs.find('Properties') if cs.find('Properties') is not None else []:
            name = prop.get('Name', '')
            prims = [_Prim(e) for e in (prop.find('Primitives') if prop.find('Primitives') is not None else [])]
            if any(p.transformed for p in prims):
                self.notes.append(f"{name}: primitive transformations are not supported (used as drawn)")
            tag = prop.tag
            if tag.endswith('Material'):
                er, kappa = self._material(prop)
                for p in prims:
                    if p.kind == 'linpoly' and p.normal == 2 and p.poly is not None:
                        slabs.append(dict(name=name, z0=p.elev, z1=p.elev + p.length, er=er, kappa=kappa,
                                          area=p.poly.area, prio=p.priority))
                    elif p.kind == 'box':
                        slabs.append(dict(name=name, z0=p.lo[2], z1=p.hi[2], er=er, kappa=kappa,
                                          area=(p.hi[0] - p.lo[0]) * (p.hi[1] - p.lo[1]), prio=p.priority))
                    elif p.kind is not None:
                        skipped.add(f"{name} ({p.el.tag})")
            elif tag in ('Metal', 'ConductingSheet'):
                if tag == 'ConductingSheet':
                    pe = prop.find('Property')
                    if pe is not None:
                        self.sigma, self.sheet_t = _num(pe, 'Conductivity'), _num(pe, 'Thickness')
                for p in prims:
                    if p.kind in ('poly', 'linpoly') and p.normal == 2 and p.poly is not None:
                        z = p.elev + (p.length if p.kind == 'linpoly' else 0.0)
                        if p.kind == 'linpoly' and abs(p.length) > ztol:
                            self.notes.append(f"{name}: thick metal drawn as a sheet at its top")
                        sheets.append((name, z, p.poly))
                    elif p.kind == 'box':
                        ext = p.hi - p.lo
                        if ext[2] <= ztol:
                            sheets.append((name, p.lo[2], box(p.lo[0], p.lo[1], p.hi[0], p.hi[1])))
                        elif min(ext[0], ext[1]) <= ztol:                    # a vertical sheet
                            walls.append((name, p.lo, p.hi))
                        else:
                            skipped.add(f"{name} (solid box)")
                    elif p.kind == 'cyl':
                        if np.hypot(*(p.p2 - p.p1)[:2]) > ztol:
                            skipped.add(f"{name} (slanted cylinder)")
                        else:
                            cyls.append((name, p.p1[0], p.p1[1], min(p.p1[2], p.p2[2]), max(p.p1[2], p.p2[2]), p.r))
                    elif p.kind is not None:
                        skipped.add(f"{name} ({p.el.tag})")
            elif tag == 'LumpedElement':
                vals = {k: _num(prop, k) for k in ('R', 'L', 'C')}
                series = int(_num(prop, 'LEtype', 0) or 0) == 1
                for p in prims:
                    if p.kind == 'box':
                        lumped.append(dict(name=name, lo=p.lo, hi=p.hi, dir=int(float(prop.get('Direction', 2))),
                                           series=series, **vals))
        if skipped:
            self.notes.append("not modelled (no planar equivalent): " + ", ".join(sorted(skipped)))
        if not slabs:
            raise ValueError(f"{path}: no dielectric (Material with LinPoly / Box primitives)")

        # ---- ground, interfaces and the layer stack -------------------------------------------------------
        outline = max(slabs, key=lambda s: s['area'])['area']
        zs = sorted({round(z / ztol) * ztol for _, z, _ in sheets})
        per_z = {z: unary_union([g for _, zz, g in sheets if abs(zz - z) <= ztol]) for z in zs}
        names = self._layer_names(sheets, zs, ztol)
        zg = self._height(ground if ground is not None else self.hints.get('ground'), names, ztol)
        if zg is None:
            solid = [z for z in zs if per_z[z].area >= 0.5 * outline]
            if not solid:
                raise ValueError(f"{path}: no solid ground plane (no metal layer covers half the board) - "
                                 f"name the ground layer (ground=...)")
            zg = solid[0]
            self.notes.append(f"ground: {names.get(zg, f'z = {zg:g}')} (lowest solid metal layer)")
        self.ground = names.get(zg, f"z={zg:g}")
        zc = self._height(cover if cover is not None else self.hints.get('cover'), names, ztol)
        self.cover = names.get(zc) if zc is not None else None
        top_slab = max(s['z1'] for s in slabs)
        bounds = sorted({zg} | {z for s in slabs for z in (s['z0'], s['z1']) if z > zg + ztol}
                        | {z for z in zs if z > zg + ztol})
        merged = [bounds[0]]
        for z in bounds[1:]:
            if z - merged[-1] > ztol:
                merged.append(z)
        layers, top = [], 'open'
        hd = self.hints.get('dielectrics', [])
        f_loss = f_loss or self.hints.get('f_loss')
        need_f = False
        for z0, z1 in zip(merged[:-1], merged[1:]):
            if z0 >= top_slab - ztol:
                if any(z > z0 + ztol for z in zs):                      # metal above the dielectric: air
                    layers.append(Layer((z1 - z0) * u, 1.0, 0.0, 'air'))
                    continue
                break
            if zc is not None and z0 >= zc - ztol:
                top = 'pec'
                break
            zm = 0.5 * (z0 + z1)
            h = next((d for d in hd if d['z0'] - ztol <= zm <= d['z1'] + ztol), None)
            if h is not None:
                layers.append(Layer((z1 - z0) * u, h['er'], h['tand'], h['name']))
                continue
            cover_ = [s for s in slabs if s['z0'] - ztol <= zm <= s['z1'] + ztol]
            if not cover_:
                layers.append(Layer((z1 - z0) * u, 1.0, 0.0, 'air'))
                continue
            s = max(cover_, key=lambda s: (s['area'], s['prio']))
            tand = 0.0
            if s['kappa']:
                need_f = need_f or not f_loss
                tand = s['kappa'] / (2 * math.pi * (f_loss or 1e9) * EPS0 * s['er'])
            layers.append(Layer((z1 - z0) * u, s['er'], tand, s['name']))
        if zc is not None and top != 'pec':
            top = 'pec'
        if need_f:
            self.notes.append("tan(delta) from the materials' Kappa at 1 GHz (give f_loss, or hints)")
        self.stack = LayerStack(layers, top=top)
        zk = np.concatenate([[zg], zg + np.cumsum([l.h for l in layers]) / u])        # interface heights
        self.iface, self.z_of = {}, {}
        below = []
        for z in zs:
            if z < zg - ztol:
                below.append(names.get(z, f"z={z:g}"))
                continue
            if abs(z - zg) <= ztol:
                continue                                                # the ground itself
            k = int(np.argmin(np.abs(zk - z)))
            if abs(zk[k] - z) > 10 * ztol:
                self.notes.append(f"metal at z = {z:g} is not on a layer interface - left out")
                continue
            if top == 'pec' and k == self.stack.N:
                continue                                                # the cover itself
            self.iface[names.get(z, f"z={z:g}")] = k
            self.z_of[k] = z
        if below:
            self.notes.append(f"metal below the ground ({', '.join(below)}) is left out")
        self._k_at = lambda z: next((k for k, zz in self.z_of.items() if abs(zz - z) <= 10 * ztol),
                                    0 if abs(z - zg) <= 10 * ztol else None)
        S = lambda g: affinity.scale(g, u, u, origin=(0, 0))
        self.shapes = {k: [S(per_z[z]).buffer(0)] for k, z in self.z_of.items() if not per_z[z].is_empty}
        self.cu_thickness = {}

        # ---- vias, walls, lumped elements, ports ------------------------------------------------------------
        self.vias, self.lumped, self._zg = [], [], zg
        for name, x, y, z0, z1, r in cyls:
            v = self._via(x, y, z0, z1, r, f"{name}@({x:.2f},{y:.2f})", Via)
            if v:
                self.vias.append(v)
        nwall = 0
        for name, lo, hi in walls:
            ax = 0 if hi[0] - lo[0] > hi[1] - lo[1] else 1
            L = (hi[ax] - lo[ax]) * u
            n = max(2, int(round(L / wall_pitch)) + 1)
            for t in np.linspace(lo[ax], hi[ax], n):
                x, y = (t, lo[1]) if ax == 0 else (lo[0], t)
                v = self._via(x, y, lo[2], hi[2], 0.05e-3 / u, f"{name}@({x:.2f},{y:.2f})", Via, ring=False)
                if v:
                    self.vias.append(v)
                    nwall += 1
        if walls:
            self.notes.append(f"vertical metal ({', '.join(sorted({w[0] for w in walls}))}): {nwall} vias "
                              f"every {wall_pitch * 1e3:g} mm")
        self.port_fps, self.port_z0 = {}, {}
        ports = [l for l in lumped if re.match(r'port_resist_(\d+)$', l['name'])]
        for l in lumped:
            if l not in ports:
                self._part(l, Via)
        self._ports(ports, port, pairs)
        self.region = None
        self.nets = None

    # ---- reading helpers ----------------------------------------------------------------------------------
    @staticmethod
    def _material(prop):
        pe = prop.find('Property')
        er = _num(pe, 'Epsilon', 1.0) if pe is not None else 1.0
        kappa = _num(pe, 'Kappa', 0.0) if pe is not None else 0.0
        if prop.tag != 'Material' and pe is not None:                   # dispersive: static permittivity
            er += sum(float(v.split(',')[0]) for k, v in pe.attrib.items() if re.match(r'EpsilonDelta_\d+$', k))
        return er, kappa or 0.0

    @staticmethod
    def _read_hints(h):
        if h is None:
            return {}
        out = dict(ground=h.get('ground'), cover=h.get('cover') or None,
                   f_loss=float(h.get('f_loss')) if h.get('f_loss') else None,
                   layers={c.get('name'): float(c.get('z')) for c in h.findall('Layer')},
                   dielectrics=[dict(name=c.get('name'), z0=float(c.get('z0')), z1=float(c.get('z1')),
                                     er=float(c.get('er')), tand=float(c.get('tand', 0))) for c in h.findall('Dielectric')],
                   ports={int(c.get('number')): {k: c.get(k) for k in c.keys()} for c in h.findall('Port')})
        return out

    def _layer_names(self, sheets, zs, ztol):
        names = {}
        for lay, z in self.hints.get('layers', {}).items():
            for zz in zs:
                if abs(zz - z) <= 10 * ztol:
                    names[zz] = lay
        for name, z, _ in sheets:
            zz = min(zs, key=lambda q: abs(q - z))
            if zz not in names and name.startswith('cu_'):
                names[zz] = name[3:]
        return names

    def _height(self, spec, names, ztol):
        if spec is None or spec == '':
            return None
        for z, n in names.items():
            if n == spec:
                return z
        try:
            return float(spec)
        except (TypeError, ValueError):
            raise ValueError(f"{self.path}: no copper layer {spec!r} (layers: {', '.join(names.values())})") from None

    def _metal_at(self, k, x, y):
        if k == 0:
            return True
        g = self.shapes.get(k)
        return bool(g) and g[0].buffer(1e-9).contains(Point(x * self.unit, y * self.unit))

    def _via(self, x, y, z0, z1, r, name, Via, ring=True, Z=None):
        """A via over the metal interfaces it crosses that have metal at (x, y) (the ground counts)."""
        ks = sorted({k for k, z in list(self.z_of.items()) + [(0, self._zg)]
                     if z0 - 1e-9 <= z <= z1 + 1e-9 and self._metal_at(k, x, y)})
        if len(ks) < 2:
            return None
        u = self.unit
        return Via(x * u, y * u, ks[0], ks[-1], radius=r * u, name=name, ring=ring, Z=Z)

    def _part(self, l, Via):
        from .sim import SeriesLumped, rlc
        R, L, C, series = l['R'], l['L'], l['C'], l['series']
        Z = rlc(R=R or 0.0, L=L or 0.0, C=C, parallel=False) if series else rlc(R=R, L=L, C=C, parallel=True)
        lo, hi, u = l['lo'], l['hi'], self.unit
        if l['dir'] == 2 or hi[2] - lo[2] > 1e-9:
            c = 0.5 * (lo + hi)
            v = self._via(c[0], c[1], lo[2], hi[2], 0.25 * max(hi[0] - lo[0], hi[1] - lo[1], 1e-6), l['name'], Via, Z=Z)
            if v:
                self.vias.append(v)
            else:
                self.notes.append(f"{l['name']}: vertical element without metal at both ends - left out")
            return
        k = self._k_at(lo[2])
        if not k:
            self.notes.append(f"{l['name']}: not on a metal interface - left out")
            return
        a = l['dir']
        strip = box(lo[0] * u, lo[1] * u, hi[0] * u, hi[1] * u)
        self.shapes[k] = [unary_union(self.shapes.get(k, []) + [strip]).buffer(0)]
        c = 0.5 * (lo[:2] + hi[:2]) * u
        d = np.eye(2)[a]
        w = (hi[1 - a] - lo[1 - a]) * u
        self.lumped.append(SeriesLumped(k, c, d, w, Z, name=l['name']))

    def _ports(self, ports, kind, pairs):
        u, hp = self.unit, self.hints.get('ports', {})
        legs = {}
        for l in sorted(ports, key=lambda l: int(l['name'].rsplit('_', 1)[1])):
            n = int(l['name'].rsplit('_', 1)[1])
            h = hp.get(n, {})
            lo, hi = l['lo'], l['hi']
            z0 = l['R'] or 50.0
            if 'x' in h:
                c = np.array([float(h['x']), float(h['y'])])
                d = np.array([float(h['dx']), float(h['dy'])])
                w = float(h['width'])
                ks = self.iface.get(h.get('layer'))
                kr = 0 if h.get('ref_layer') in (None, '', self.ground) else self.iface.get(h.get('ref_layer'), 0)
            else:
                c, d, w, ks, kr = self._infer_port(lo, hi)
            name = h.get('name') or f"P{n}"
            if ks is None:
                raise ValueError(f"{name}: no metal interface at the port's signal end")
            lay = next(nm for nm, k in self.iface.items() if k == ks)
            g = self.shapes[ks][0]
            p = Point(*(c * u))
            if not g.buffer(1e-9).contains(p):                          # step onto the copper along the feed
                for s in np.linspace(0, max(w, 1e-3 / u), 21)[1:]:
                    if g.buffer(1e-9).contains(Point(*((c + d * s) * u))):
                        c = c + d * s
                        break
            fp = dict(name=name, x=c[0] * u / MM, y=-c[1] * u / MM, rot=math.degrees(math.atan2(d[1], d[0])), side='F',
                      ref=name, props={'mom.layer': lay, 'mom.z0': z0, 'mom.port': kind, 'mom.kref': kr,
                                       'mom.radius': max(w * u / MM / 4, 0.02)})
            self.port_z0[name] = z0
            m = re.match(r'(.+)([+-])$', name)
            if pairs and m:
                legs.setdefault(m.group(1), DiffLegs())[m.group(2)] = fp
                if len(legs[m.group(1)]) == 2:
                    legs[m.group(1)].name = m.group(1)
                    self.port_fps[len(self.port_fps) + 1] = legs[m.group(1)]
                continue
            self.port_fps[len(self.port_fps) + 1] = fp

    def _infer_port(self, lo, hi):
        """(centre, direction into the circuit, width, signal and reference interface) of a lumped-port box."""
        ext = hi - lo
        a = 0 if ext[0] <= ext[1] else 1                                # the feed axis: no extent along it
        w = ext[1 - a]
        c = 0.5 * (lo[:2] + hi[:2])
        ka, kb = self._k_at(lo[2]), self._k_at(hi[2])
        if ka == 0 or kb == 0 or ka is None or kb is None:
            ks, kr = (kb, 0) if ka in (0, None) else (ka, 0)
        else:                                                           # the smaller copper there is the trace
            area = lambda k: next((q.area for q in getattr(self.shapes[k][0], 'geoms', [self.shapes[k][0]])
                                   if q.buffer(1e-9).contains(Point(*(c * self.unit)))), np.inf)
            ks, kr = (ka, kb) if area(ka) <= area(kb) else (kb, ka)
        best, dbest = -1.0, np.eye(2)[a]
        for sgn in (1, -1):
            d = np.eye(2)[a] * sgn
            ray = LineString([c * self.unit, (c + d * 50e-3 / self.unit) * self.unit])
            seg = ray.intersection(self.shapes[ks][0]) if ks in self.shapes else None
            parts = [] if seg is None or seg.is_empty else list(getattr(seg, 'geoms', [seg]))
            L = sum(q.length for q in parts if q.distance(Point(*(c * self.unit))) < max(w * self.unit, 1e-6))
            if L > best:
                best, dbest = L, d
        return c, dbest, w, ks, kr

    def describe(self):
        return "\n".join([f"CSX geometry {self.path} (1 unit = {self.unit * 1e3:g} mm)", super().describe()])
