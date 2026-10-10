"""
mom25d.kicad - KiCad board import: .kicad_pcb (native) -> mom25d Simulation

    brd = KiCadBoard('board.kicad_pcb')
    model = brd.build(nets=['RF_IN', 'RF_OUT', 'GND'], ground='B.Cu', region=(x0, y0, x1, y1),
                      pos_file='board-top.pos')
    sim = model.simulation(dmax=0.3e-3)

What is read
  stackup     (setup (stackup ...)): copper / dielectric order, thickness, epsilon_r, loss tangent,
              solder mask; dielectric sub-layers (addsublayer) are split
  copper      tracks (segment, arc), vias (through / blind / micro), filled zones (filled_polygon:
              the real copper, antipads included), footprint pads (rect, roundrect, circle, oval,
              trapezoid as rect, custom via its primitives), plated through-holes (as vias),
              gr_poly / gr_rect / gr_circle on copper layers
  ports       footprints marked as ports by text P<n> on a Fab layer (fp_text user / property on
              F.Fab or B.Fab), or by Reference / Value P<n> when listed in a pick-and-place file.
              The footprint position is the port location, its rotation the direction INTO the
              circuit (0 deg = +x). Footprint properties (fields) refine the port:
                  mom.port = wave | via | gap   (default: wave if a straight feed is found, else via)
                  mom.z0   = reference impedance (default 50)
                  mom.ref  = reference-plane shift along the direction, mm (default 0 = footprint)

Conventions
  KiCad coordinates are mm with y down; the solver uses metres with y up (y_sim = -y_kicad).
  The copper layer named by `ground` becomes the solver's infinite ground (z = 0) and is not
  meshed; copper layers above it are metal interfaces; an optional `cover` copper layer becomes a
  PEC cover (stripline-only cross sections). Copper below the ground layer is ignored (use
  flip=True to model the bottom side).
"""
import re
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

from .model import PlanarModel, DiffLegs, MM


# =============================================================================
# S-expressions
# =============================================================================
_TOK = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def parse_sexpr(text):
    stack, cur = [], []
    for m in _TOK.finditer(text):
        t = m.group(0)
        if t == '(':
            stack.append(cur); cur = []
        elif t == ')':
            done = cur; cur = stack.pop(); cur.append(done)
        elif t[0] == '"':
            cur.append(bytes(t[1:-1], 'utf-8').decode('unicode_escape') if '\\' in t else t[1:-1])
        else:
            cur.append(t)
    return cur[0]


def _items(node, name):
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == name]


def _item(node, name, default=None):
    for c in node[1:]:
        if isinstance(c, list) and c and c[0] == name:
            return c
    return default


def _val(node, name, default=None, cast=str):
    it = _item(node, name)
    return default if it is None or len(it) < 2 else cast(it[1])


def _xy(node, name):
    it = _item(node, name)
    return None if it is None else (float(it[1]), float(it[2]))


def _layers(node):
    out = []
    for key in ('layer', 'layers'):
        it = _item(node, key)
        if it:
            out += [str(x) for x in it[1:] if isinstance(x, str)]
    return out


# =============================================================================
# Geometry helpers (all in KiCad mm, y down; converted at the end)
# =============================================================================
def _rot(v, deg):
    """Rotate a KiCad-coordinate vector by a KiCad angle (CCW on screen, y down)."""
    a = np.radians(deg)
    x, y = v
    return np.array([x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)])


def _arc_points(p0, pm, p1, seg=0.1):
    """Points along the circular arc start-mid-end (KiCad mm)."""
    p0, pm, p1 = map(np.asarray, (p0, pm, p1))
    A = np.array([[p0[0] - pm[0], p0[1] - pm[1]], [p0[0] - p1[0], p0[1] - p1[1]]]) * 2
    b = np.array([p0 @ p0 - pm @ pm, p0 @ p0 - p1 @ p1])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.array([p0, p1])
    r = np.linalg.norm(p0 - c)
    a0, am, a1 = (np.arctan2(*(p - c)[::-1]) for p in (p0, pm, p1))
    # choose the sweep that passes through the mid point
    def norm(a): return (a + 2 * np.pi) % (2 * np.pi)
    s1 = norm(a1 - a0); sm = norm(am - a0)
    sweep = s1 if sm <= s1 else s1 - 2 * np.pi
    n = max(4, int(abs(sweep) * r / seg))
    t = a0 + sweep * np.linspace(0, 1, n + 1)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def _pad_shape(pad):
    """Pad outline in pad-local mm coordinates (before rotation/translation)."""
    shape = pad[3] if len(pad) > 3 else 'rect'
    sz = _xy(pad, 'size') or (0, 0)
    w, h = sz
    if shape == 'circle':
        return Point(0, 0).buffer(w / 2, quad_segs=8)
    if shape == 'oval':
        r = min(w, h) / 2
        L = LineString([(-(w / 2 - r), 0), (w / 2 - r, 0)]) if w >= h else LineString([(0, -(h / 2 - r)), (0, h / 2 - r)])
        return L.buffer(r, quad_segs=8) if L.length > 0 else Point(0, 0).buffer(r, quad_segs=8)
    if shape == 'roundrect':
        rr = _val(pad, 'roundrect_rratio', 0.25, float) * min(w, h)
        return box(-w / 2 + rr, -h / 2 + rr, w / 2 - rr, h / 2 - rr).buffer(rr, quad_segs=4) if rr > 0 else box(-w / 2, -h / 2, w / 2, h / 2)
    if shape == 'custom':
        g = [box(-w / 2, -h / 2, w / 2, h / 2)] if w > 0 else []
        prims = _item(pad, 'primitives')
        for pr in (prims[1:] if prims else []):
            if pr[0] in ('gr_poly', 'fp_poly'):
                pts = [(float(p[1]), float(p[2])) for p in _item(pr, 'pts')[1:] if p[0] == 'xy']
                wid = _val(pr, 'width', 0, float)
                poly = Polygon(pts)
                g.append(poly.buffer(wid / 2) if wid else poly)
            elif pr[0] in ('gr_circle',):
                c = np.array(_xy(pr, 'center')); e = np.array(_xy(pr, 'end'))
                g.append(Point(*c).buffer(np.linalg.norm(e - c), quad_segs=8))
            elif pr[0] in ('gr_line',):
                g.append(LineString([_xy(pr, 'start'), _xy(pr, 'end')]).buffer(_val(pr, 'width', 0.1, float) / 2))
        return unary_union(g) if g else Point(0, 0).buffer(0.05)
    return box(-w / 2, -h / 2, w / 2, h / 2)              # rect, trapezoid (approx.)


def _place(geom, origin, deg):
    """Rotate a local KiCad-mm geometry by a KiCad angle and translate to origin."""
    g = affinity.rotate(geom, -deg, origin=(0, 0))        # KiCad CCW-on-screen == CW in y-down maths
    return affinity.translate(g, *origin)


# =============================================================================
# Board
# =============================================================================
class KiCadBoard:
    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-8') as fh:
            self.root = parse_sexpr(fh.read())
        assert self.root[0] == 'kicad_pcb', "not a .kicad_pcb file"
        self._nets()
        self._stackup()
        self._copper()

    # ---- nets ---------------------------------------------------------------------------
    def _nets(self):
        self.net_names = {}
        for n in _items(self.root, 'net'):
            if len(n) >= 3:
                self.net_names[str(n[1])] = n[2]

    def _net_of(self, node):
        it = _item(node, 'net')
        if not it or len(it) < 2:
            return ''
        if len(it) >= 3:
            return it[2]
        v = str(it[1])
        return self.net_names.get(v, v)

    # ---- stackup ---------------------------------------------------------------------------
    def _stackup(self):
        setup = _item(self.root, 'setup')
        su = _item(setup, 'stackup') if setup else None
        self.stack_items = []
        self.copper_order = [str(l[1]) for l in (_item(self.root, 'layers') or [])[1:]
                             if isinstance(l, list) and len(l) > 1 and str(l[1]).endswith('.Cu')]
        if su is None:
            self.has_stackup = False
            return
        self.has_stackup = True
        for l in _items(su, 'layer'):
            name, typ = l[1], (_val(l, 'type', '') or '').lower()
            if typ == 'copper':
                self.stack_items.append(dict(kind='copper', name=name, t=_val(l, 'thickness', 0.035, float) * MM))
            elif 'mask' in typ:
                self.stack_items.append(dict(kind='mask', name=name, t=_val(l, 'thickness', 0.01, float) * MM,
                                             er=_val(l, 'epsilon_r', 3.3, float),
                                             tand=_val(l, 'loss_tangent', 0.0, float)))
            elif typ in ('core', 'prepreg') or 'dielectric' in str(name).lower():
                # split sub-layers: a flat list (thickness, material, epsilon_r, loss_tangent)*
                subs, cur = [], {}
                for c in l[2:]:
                    if not isinstance(c, list):
                        continue
                    if c[0] == 'addsublayer':
                        subs.append(cur); cur = {}
                    elif c[0] in ('thickness', 'epsilon_r', 'loss_tangent'):
                        cur[c[0]] = float(c[1])
                subs.append(cur)
                for s in subs:
                    self.stack_items.append(dict(kind='diel', name=name, t=s.get('thickness', 0.1) * MM,
                                                 er=s.get('epsilon_r', 4.5), tand=s.get('loss_tangent', 0.02)))

    def describe_stackup(self):
        out = []
        for it in self.stack_items:
            if it['kind'] == 'copper':
                out.append(f"  {it['name']:<14s} copper   {it['t'] * 1e6:6.1f} um")
            else:
                out.append(f"  {it['name']:<14s} {it['kind']:<8s} {it['t'] * 1e3:6.3f} mm  er {it['er']:.2f}  tand {it['tand']:.4f}")
        return "\n".join(out)

    # ---- copper -----------------------------------------------------------------------------
    def _copper(self):
        """Collect copper items as (layer, net, geometry-in-KiCad-mm) plus vias and port footprints."""
        self.items = []          # (layer, net, shapely geometry (mm, y down))
        self.vias = []           # dict(x, y, drill, size, layers, net)
        self.footprints = []
        cu = set(self.copper_order) or None
        all_cu = self.copper_order

        def expand(lys):
            out = []
            for l in lys:
                if l in ('*.Cu', 'F&B.Cu'):
                    out += all_cu if l == '*.Cu' else ['F.Cu', 'B.Cu']
                elif l.endswith('.Cu'):
                    out.append(l)
            return out

        for s in _items(self.root, 'segment'):
            g = LineString([_xy(s, 'start'), _xy(s, 'end')]).buffer(_val(s, 'width', 0.2, float) / 2, quad_segs=6)
            for l in expand(_layers(s)):
                self.items.append((l, self._net_of(s), g))
        for a in _items(self.root, 'arc'):
            pts = _arc_points(_xy(a, 'start'), _xy(a, 'mid'), _xy(a, 'end'))
            g = LineString(pts).buffer(_val(a, 'width', 0.2, float) / 2, quad_segs=6)
            for l in expand(_layers(a)):
                self.items.append((l, self._net_of(a), g))
        for v in _items(self.root, 'via'):
            x, y = _xy(v, 'at')
            lys = expand(_layers(v))
            kind = 'micro' if 'micro' in v else ('blind' if 'blind' in v else 'through')
            if kind == 'through' and len(lys) == 2 and all_cu:
                i0, i1 = sorted(all_cu.index(l) for l in lys)
                lys = all_cu[i0:i1 + 1]
            size = _val(v, 'size', 0.6, float); drill = _val(v, 'drill', 0.3, float)
            self.vias.append(dict(x=x, y=y, size=size, drill=drill, layers=lys, net=self._net_of(v)))
            for l in lys:
                self.items.append((l, self._net_of(v), Point(x, y).buffer(size / 2, quad_segs=6)))
        for z in _items(self.root, 'zone'):
            net = _val(z, 'net_name', '') or self._net_of(z)
            for fp in _items(z, 'filled_polygon'):
                lay = _val(fp, 'layer')
                pts = [(float(p[1]), float(p[2])) for p in _item(fp, 'pts')[1:] if p[0] == 'xy']
                if len(pts) >= 3:
                    self.items.append((lay, net, Polygon(pts).buffer(0)))
        for tag in ('gr_poly', 'gr_rect', 'gr_circle'):
            for g in _items(self.root, tag):
                lys = [l for l in _layers(g) if l.endswith('.Cu')]
                if not lys:
                    continue
                if tag == 'gr_poly':
                    geom = Polygon([(float(p[1]), float(p[2])) for p in _item(g, 'pts')[1:] if p[0] == 'xy'])
                elif tag == 'gr_rect':
                    (x0, y0), (x1, y1) = _xy(g, 'start'), _xy(g, 'end')
                    geom = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                else:
                    c, e = np.array(_xy(g, 'center')), np.array(_xy(g, 'end'))
                    geom = Point(*c).buffer(np.linalg.norm(e - c), quad_segs=12)
                for l in lys:
                    self.items.append((l, self._net_of(g), geom))
        for fp in _items(self.root, 'footprint') + _items(self.root, 'module'):
            self._footprint(fp, expand)

    def _footprint(self, fp, expand):
        at = _item(fp, 'at')
        org = (float(at[1]), float(at[2]))
        rot = float(at[3]) if len(at) > 3 else 0.0
        ref, val, fab_texts, props = '', '', [], {}
        for p in _items(fp, 'property'):
            props[p[1]] = p[2] if len(p) > 2 else ''
            if any(str(l).endswith('.Fab') for l in _layers(p)) and len(p) > 2:
                fab_texts.append(p[2])
        for t in _items(fp, 'fp_text'):
            kind, txt = t[1], t[2]
            if kind == 'reference':
                ref = txt
            elif kind == 'value':
                val = txt
            if any(str(l).endswith('.Fab') for l in _layers(t)):
                fab_texts.append(txt)
        ref = props.get('Reference', ref); val = props.get('Value', val)
        side = 'B' if _val(fp, 'layer', 'F.Cu').startswith('B') else 'F'
        pads = []
        for pad in _items(fp, 'pad'):
            pat = _item(pad, 'at')
            pxy = np.array([float(pat[1]), float(pat[2])])
            prot = float(pat[3]) if len(pat) > 3 else 0.0
            if fp[0] == 'module':                         # KiCad 5: pad angle relative
                prot += rot
            centre = np.array(org) + _rot(pxy, rot)
            geom = _place(_pad_shape(pad), centre, prot)
            lys = expand(_layers(pad))
            net = self._net_of(pad)
            for l in lys:
                self.items.append((l, net, geom))
            drill = _item(pad, 'drill')
            if pad[2] == 'thru_hole' and drill and len(lys) > 1:
                d = float(drill[1]) if len(drill) > 1 and drill[1] != 'oval' else float(drill[2])
                self.vias.append(dict(x=centre[0], y=centre[1], size=0, drill=d, layers=lys, net=net, pth=True))
            pads.append(dict(name=pad[1], xy=centre, layers=lys, net=net, geom=geom))
        self.footprints.append(dict(ref=ref, value=val, fab=fab_texts, props=props, x=org[0], y=org[1],
                                    rot=rot, side=side, lib=str(fp[1]), pads=pads))

    # ---- ports ----------------------------------------------------------------------------
    def port_footprints(self, pos_refs=None):
        """{n: footprint} for single-ended ports (Fab text P<n>), {n: {'+': fp, '-': fp}} for
        differential ports (Fab text P<n>+ / P<n>- or P<n>P / P<n>N). Reference/Value are used for
        refs listed in a pick-and-place file."""
        pat = re.compile(r'^P(\d+)([+\-PN])?$', re.I)
        out = {}

        def put(m, fp):
            n, leg = int(m.group(1)), (m.group(2) or '').upper()
            if leg:
                out.setdefault(n, DiffLegs())['+' if leg in ('+', 'P') else '-'] = fp
            else:
                out[n] = fp
        for fp in self.footprints:
            done = False
            for t in fp['fab']:
                m = pat.match(str(t).strip())
                if m:
                    put(m, fp); done = True; break
            if not done and pos_refs is not None and fp['ref'] in pos_refs:
                for t in (fp['value'], fp['ref'], *fp['fab']):
                    m = pat.match(str(t).strip())
                    if m:
                        put(m, fp); break
        for n, v in out.items():
            if isinstance(v, DiffLegs) and set(v) != {'+', '-'}:
                raise ValueError(f"differential port P{n} needs both P{n}+ and P{n}- footprints")
        return dict(sorted(out.items()))

    # ---- model -------------------------------------------------------------------------------
    def build(self, nets=None, ground='B.Cu', cover=None, region=None, pos_file=None, include_mask=True,
              flip=False, default_er=4.5, default_tand=0.02):
        return BoardModel(self, nets, ground, cover, region, pos_file, include_mask, flip,
                          default_er, default_tand)


def read_pos(path):
    """KiCad pick-and-place file (.pos ASCII or .csv). Returns list of dict(ref, val, package, x, y, rot, side)."""
    rows = []
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    if path.lower().endswith('.csv'):
        import csv
        for r in csv.DictReader(text.splitlines()):
            rows.append(dict(ref=r['Ref'], val=r['Val'], package=r['Package'], x=float(r['PosX']),
                             y=float(r['PosY']), rot=float(r['Rot']), side=r['Side']))
    else:
        for line in text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 7:
                rows.append(dict(ref=p[0], val=p[1], package=p[2], x=float(p[3]), y=float(p[4]),
                                 rot=float(p[5]), side=p[6]))
    return rows


def resolve_nets(wanted, available):
    """Match requested net names to board nets. Hierarchical KiCad nets ('/sheet/REF_SMA_N') also
    match their bare name ('REF_SMA_N'); matching is case-sensitive first, then case-insensitive."""
    available = {str(n) for n in available if n}
    out = set()
    for w in wanted:
        hits = ({w} & available) or {n for n in available if n.rsplit('/', 1)[-1] == w.lstrip('/')} \
            or {n for n in available if n.rsplit('/', 1)[-1].lower() == w.lstrip('/').lower()}
        if not hits:
            raise ValueError(f"net {w!r} not found on the board; nets with copper: "
                             + ", ".join(sorted(available, key=str.lower)))
        out |= hits
    return out




def pos_port_refs(rows, pattern=r'^P\d+[+\-PN]?$', package_hint='PORT'):
    """References of port components: Value or Reference P<n>, or package containing the hint."""
    pat = re.compile(pattern, re.I)
    return {r['ref'] for r in rows if pat.match(r['val']) or pat.match(r['ref'])
            or (package_hint and package_hint.lower() in r['package'].lower())}


# =============================================================================
# Board -> solver model
# =============================================================================
class BoardModel(PlanarModel):
    """A KiCad (or Gerber) board as a PlanarModel."""
    def __init__(self, board, nets, ground, cover, region, pos_file, include_mask, flip,
                 default_er, default_tand):
        from .multilayer import Layer, LayerStack
        self.board, self.ground, self.cover = board, ground, cover
        self.notes = []
        # ---- stack ----
        items = list(board.stack_items)
        if not items:
            raise ValueError("board has no stackup section: define one in Board Setup > Physical Stackup")
        if flip:
            items = items[::-1]
        names = [it['name'] for it in items]
        assert ground in names, f"ground layer {ground} not in stackup"
        gi = names.index(ground)
        above = items[:gi][::-1]                             # bottom -> top, starting just above ground
        layers, iface, z = [], {}, 0.0
        top = 'open'
        for it in above:
            if it['kind'] == 'diel' or (it['kind'] == 'mask' and include_mask):
                layers.append(Layer(it['t'], it['er'], it['tand'], it['name']))
            elif it['kind'] == 'copper':
                if it['name'] == cover:
                    top = 'pec'; break
                iface[it['name']] = len(layers)
        self.cu_thickness = {it['name']: it['t'] for it in items if it['kind'] == 'copper'}
        # drop dielectrics between the top copper and the mask boundary? (mask stays: it sits on top)
        self.stack = LayerStack(layers, top=top)
        self.iface = {n: k for n, k in iface.items() if k >= 1}
        bad = [n for n, k in iface.items() if k < 1]
        if bad:
            self.notes.append(f"copper {bad} sits on the ground interface and is ignored")
        # ---- geometry per interface (sim metres, y up) ----
        sel = resolve_nets(nets, {n for _, n, _ in board.items} | {v['net'] for v in board.vias}) if nets else None
        self.nets = sel
        reg = box(*region) if region else None
        per = {}
        for lay, net, g in board.items:
            if lay not in self.iface or (sel is not None and net not in sel):
                continue
            if reg is not None:
                g = g.intersection(reg)
            if not g.is_empty:
                per.setdefault(self.iface[lay], []).append(g)
        self.shapes = {}
        for k, gs in per.items():
            g = unary_union(gs).buffer(0)
            g = affinity.scale(g, MM, -MM, origin=(0, 0))
            self.shapes[k] = [g.simplify(1e-6)]
        self.region = reg
        # ---- vias ----
        from .sim import Via
        self.vias = []
        for v in board.vias:
            if sel is not None and v['net'] not in sel:
                continue
            if reg is not None and not reg.contains(Point(v['x'], v['y'])):
                continue
            ks = sorted({0 if l == ground else self.iface[l] for l in v['layers']
                         if l in self.iface or l == ground})
            if len(ks) < 2:
                continue
            k0, k1 = ks[0], ks[-1]
            self.vias.append(Via(v['x'] * MM, -v['y'] * MM, k0, k1, radius=v['drill'] / 2 * MM,
                                 name=f"{v['net']}@({v['x']:.2f},{v['y']:.2f})"))
        # ---- ports ----
        refs = None
        if pos_file:
            rows = read_pos(pos_file)
            refs = pos_port_refs(rows)
            self.notes.append(f"pick-and-place: port components {sorted(refs)}")
        self.port_fps = board.port_footprints(refs)
        if not self.port_fps:
            self.notes.append("no port footprints found (Fab text P<n>)")
