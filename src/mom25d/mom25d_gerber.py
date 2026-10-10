"""
mom25d_gerber.py - Gerber (RS-274X / X2) + Excellon + KiCad job file -> the same board model as
the native KiCad importer.

    brd = GerberBoard('fab/board-job.gbrjob')            # finds copper Gerbers and drill files
    model = brd.build(ground='B.Cu', region=(x0, y0, x1, y1), pos_file='fab/board-top.pos')
    sim = model.simulation(dmax=0.3e-3)

  * stackup     from the job file (MaterialStackup: copper / dielectric / solder mask)
  * copper      draws with any aperture (convex hull of start/end apertures), arcs (G02/G03, G75),
                flashes (C, R, O, P and aperture macros: primitives 1, 4, 5, 20, 21 with variables
                and arithmetic - covers KiCad's RoundRect / RotRect / Outline / FreePoly), regions
                (G36/G37), dark/clear polarity (LPD / LPC)
  * vias        plated drill files; the layer span is read from the X2 file attribute
                (TF.FileFunction,Plated,<from>,<to>,...) so blind/buried files are handled
  * ports       from the pick-and-place file (Value or Reference P<n>); Gerbers carry no nets, so
                every copper shape on the used layers is included - crop with `region`
Coordinates are converted to KiCad's convention (mm, y down) so BoardModel applies unchanged.
"""
import json
import os
import re

import numpy as np
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

from mom25d_kicad import _arc_points, read_pos, pos_port_refs, BoardModel, DiffLegs


# =============================================================================
# Aperture macros
# =============================================================================
def _eval(expr, var):
    e = expr.strip().replace('x', '*').replace('X', '*')
    e = re.sub(r'\$(\d+)', lambda m: repr(var.get(int(m.group(1)), 0.0)), e)
    if not re.fullmatch(r'[0-9eE+\-*/(). ]*', e):
        raise ValueError(f"bad macro expression {expr!r}")
    return float(eval(e, {"__builtins__": {}}, {}))


def _macro_shape(body, params):
    var = {i + 1: p for i, p in enumerate(params)}
    dark, clear = [], []
    for stmt in body:
        stmt = stmt.strip()
        if not stmt or stmt.startswith('0'):
            continue
        if stmt.startswith('$'):
            k, e = stmt[1:].split('=', 1)
            var[int(k)] = _eval(e, var)
            continue
        a = [_eval(x, var) for x in stmt.split(',')]
        code, exp = int(a[0]), a[1]
        if code == 1:                                         # circle: exp, dia, x, y [, rot]
            g = Point(a[3], a[4]).buffer(a[2] / 2, quad_segs=8)
            if len(a) > 5 and a[5]:
                g = affinity.rotate(g, a[5], origin=(0, 0))
        elif code == 20:                                      # vector line: exp, w, x1, y1, x2, y2, rot
            g = LineString([(a[3], a[4]), (a[5], a[6])]).buffer(a[2] / 2, cap_style='flat')
            g = affinity.rotate(g, a[7], origin=(0, 0))
        elif code == 21:                                      # centre line: exp, w, h, x, y, rot
            g = box(a[4] - a[2] / 2, a[5] - a[3] / 2, a[4] + a[2] / 2, a[5] + a[3] / 2)
            g = affinity.rotate(g, a[6], origin=(0, 0))
        elif code == 4:                                       # outline: exp, n, x0, y0, ..., rot
            n = int(a[2])
            pts = [(a[3 + 2 * i], a[4 + 2 * i]) for i in range(n + 1)]
            g = affinity.rotate(Polygon(pts).buffer(0), a[3 + 2 * (n + 1)], origin=(0, 0))
        elif code == 5:                                       # polygon: exp, n, x, y, dia, rot
            n = int(a[2]); r = a[5] / 2
            t = np.radians(a[6]) + 2 * np.pi * np.arange(n) / n
            g = Polygon(np.column_stack([a[3] + r * np.cos(t), a[4] + r * np.sin(t)]))
        else:
            continue                                          # moire / thermal: not used for copper
        (dark if exp else clear).append(g)
    g = unary_union(dark) if dark else Point(0, 0).buffer(1e-6)
    return g.difference(unary_union(clear)) if clear else g


# =============================================================================
# Gerber layer
# =============================================================================
def parse_gerber(path):
    """Returns (shapely geometry in mm, y up, X2 attributes dict)."""
    text = open(path, encoding='utf-8', errors='replace').read()
    attrs = {}
    macros, aps = {}, {}
    fmt = (4, 6); unit = 1.0
    x = y = 0.0
    interp, quadrant = 'G01', 'G75'
    cur_ap, region, region_pts = None, False, []
    dark, out = [], []            # accumulated geometry, with polarity handling
    polarity = 'D'

    def coord(v):
        if v is None:
            return None
        if '.' in v:
            return float(v) * unit
        neg = v.startswith('-')
        v = v.lstrip('+-')
        val = int(v) / 10 ** fmt[1]
        return (-val if neg else val) * unit

    def add(g):
        nonlocal dark
        if polarity == 'D':
            dark.append(g)
        else:
            out.append(('D', dark)); out.append(('C', [g])); dark = []

    # split into extended (%...%) and normal (...*) statements
    for m in re.finditer(r'%([^%]*)%|([^%*]+)\*', text, re.S):
        if m.group(1) is not None:
            blk = m.group(1).strip()
            if blk.startswith('FS'):
                mm = re.search(r'X(\d)(\d)Y(\d)(\d)', blk)
                fmt = (int(mm.group(1)), int(mm.group(2)))
            elif blk.startswith('MO'):
                unit = 25.4 if 'IN' in blk else 1.0
            elif blk.startswith('TF'):
                parts = blk[2:].rstrip('*').split(',')
                attrs[parts[0].lstrip('.')] = parts[1:]
            elif blk.startswith('AM'):
                name, *body = blk[2:].split('*')
                macros[name.strip()] = [b for b in body if b.strip()]
            elif blk.startswith('AD'):
                mm = re.match(r'ADD(\d+)([A-Za-z_.$][\w.$\-]*)(?:,(.*))?\*?$', blk.rstrip('*'), re.S)
                code, name, par = int(mm.group(1)), mm.group(2), mm.group(3)
                p = [float(v) * unit for v in par.split('X')] if par else []
                if name == 'C':
                    g = Point(0, 0).buffer(p[0] / 2, quad_segs=8)
                elif name == 'R':
                    g = box(-p[0] / 2, -p[1] / 2, p[0] / 2, p[1] / 2)
                elif name == 'O':
                    w, h = p[0], p[1]; r = min(w, h) / 2
                    L = LineString([(-(w / 2 - r), 0), (w / 2 - r, 0)] if w >= h else [(0, -(h / 2 - r)), (0, h / 2 - r)])
                    g = L.buffer(r, quad_segs=8) if L.length else Point(0, 0).buffer(r, quad_segs=8)
                elif name == 'P':
                    n = int(p[1] / unit); rot = p[2] / unit if len(p) > 2 else 0
                    t = np.radians(rot) + 2 * np.pi * np.arange(n) / n
                    g = Polygon(np.column_stack([p[0] / 2 * np.cos(t), p[0] / 2 * np.sin(t)]))
                else:
                    pr = [float(v) for v in par.split('X')] if par else []
                    g = _macro_shape(macros[name], pr)
                    if unit != 1.0:
                        g = affinity.scale(g, unit, unit, origin=(0, 0))
                aps[code] = g
            elif blk.startswith('LP'):
                polarity = blk[2]
            continue
        st = m.group(2).strip()
        if not st:
            continue
        for g in re.findall(r'G0?[123]\b|G7[45]|G3[67]', st):
            g = g if len(g) == 3 else 'G0' + g[1]
            if g in ('G01', 'G02', 'G03'):
                interp = g
            elif g in ('G74', 'G75'):
                quadrant = g
            elif g == 'G36':
                region, region_pts = True, []
            elif g == 'G37':
                if len(region_pts) >= 3:
                    add(Polygon(region_pts).buffer(0))
                region = False
        dsel = re.fullmatch(r'(?:G54)?D(\d+)', st)
        if dsel and int(dsel.group(1)) >= 10:
            cur_ap = int(dsel.group(1)); continue
        mm = re.search(r'(?:X([+-]?[\d.]+))?(?:Y([+-]?[\d.]+))?(?:I([+-]?[\d.]+))?(?:J([+-]?[\d.]+))?D0?([123])', st)
        if not mm:
            continue
        nx, ny = coord(mm.group(1)), coord(mm.group(2))
        nx = x if nx is None else nx
        ny = y if ny is None else ny
        i_, j_ = coord(mm.group(3)) or 0.0, coord(mm.group(4)) or 0.0
        d = mm.group(5)
        if d == '1':
            if interp == 'G01':
                pts = [(x, y), (nx, ny)]
            else:
                c = np.array([x + i_, y + j_])
                a0 = np.arctan2(y - c[1], x - c[0]); a1 = np.arctan2(ny - c[1], nx - c[0])
                if interp == 'G02':                          # clockwise
                    sw = (a1 - a0) % (2 * np.pi) - 2 * np.pi
                    if abs(sw) < 1e-12: sw = -2 * np.pi
                else:
                    sw = (a1 - a0) % (2 * np.pi)
                    if abs(sw) < 1e-12: sw = 2 * np.pi
                r = np.hypot(x - c[0], y - c[1])
                t = a0 + sw * np.linspace(0, 1, max(4, int(abs(sw) * r / 0.05)) + 1)
                pts = list(zip(c[0] + r * np.cos(t), c[1] + r * np.sin(t)))
            if region:
                if not region_pts:
                    region_pts = [pts[0]]
                region_pts += pts[1:]
            elif cur_ap is not None:
                ap = aps[cur_ap]
                segs = []
                for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                    segs.append(unary_union([affinity.translate(ap, xa, ya), affinity.translate(ap, xb, yb)]).convex_hull)
                add(unary_union(segs))
        elif d == '2':
            if region and len(region_pts) >= 3:
                add(Polygon(region_pts).buffer(0))
            region_pts = [(nx, ny)] if region else []
        elif d == '3' and cur_ap is not None:
            add(affinity.translate(aps[cur_ap], nx, ny))
        x, y = nx, ny
    out.append(('D', dark))
    geom = Polygon()
    for pol, gs in out:
        if not gs:
            continue
        u = unary_union(gs)
        geom = geom.union(u) if pol == 'D' else geom.difference(u)
    return geom.buffer(0), attrs


def parse_excellon(path):
    """Plated holes: list of (x, y, diameter) in mm (y up), and the X2 layer span if present."""
    text = open(path, encoding='utf-8', errors='replace').read()
    unit = 1.0
    span = None
    m = re.search(r'TF\.FileFunction,(?:Plated|NonPlated),(\d+),(\d+)', text)
    if m:
        span = (int(m.group(1)), int(m.group(2)))
    plated = 'NonPlated' not in text and 'NPTH' not in os.path.basename(path).upper()
    tools, holes, tool = {}, [], None
    lz = True
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('INCH'):
            unit = 25.4; lz = 'TZ' not in line
        elif line.startswith('METRIC'):
            unit = 1.0
        mt = re.match(r'T(\d+)C([\d.]+)', line)
        if mt:
            tools[int(mt.group(1))] = float(mt.group(2)) * unit; continue
        ms = re.fullmatch(r'T(\d+)', line)
        if ms:
            tool = int(ms.group(1)); continue
        mx = re.match(r'X([+-]?[\d.]+)Y([+-]?[\d.]+)', line)
        if mx and tool is not None:
            def num(v):
                return float(v) if '.' in v else int(v) / (1000.0 if unit == 1.0 else 10000.0)
            holes.append((num(mx.group(1)) * unit, num(mx.group(2)) * unit, tools.get(tool, 0.3)))
    return holes, span, plated


# =============================================================================
# Board from a job file
# =============================================================================
class GerberBoard:
    def __init__(self, job, drills=None, pos_rows=None):
        self.path = job
        d = os.path.dirname(job)
        J = json.load(open(job, encoding='utf-8'))
        self.stack_items, self.copper_order = [], []
        for it in J.get('MaterialStackup', []):
            t = it.get('Type', '')
            th = float(it.get('Thickness', 0)) * 1e-3
            if t == 'Copper':
                self.stack_items.append(dict(kind='copper', name=it['Name'], t=th))
                self.copper_order.append(it['Name'])
            elif t == 'Dielectric':
                self.stack_items.append(dict(kind='diel', name=it.get('Name', 'diel'), t=th,
                                             er=float(it.get('DielectricConstant', 4.5)),
                                             tand=float(it.get('LossTangent', 0.02))))
            elif t == 'SolderMask':
                self.stack_items.append(dict(kind='mask', name=it.get('Name', 'mask'), t=th,
                                             er=float(it.get('DielectricConstant', 3.3)),
                                             tand=float(it.get('LossTangent', 0.0))))
        self.has_stackup = bool(self.stack_items)
        n_cu = len(self.copper_order)
        self.items, self.vias, self.footprints = [], [], []
        for fa in J.get('FilesAttributes', []):
            ff = fa.get('FileFunction', '')
            if not ff.startswith('Copper'):
                continue
            m = re.match(r'Copper,L(\d+)', ff)
            li = int(m.group(1)) - 1
            name = self.copper_order[li] if li < n_cu else f"L{li + 1}"
            g, _ = parse_gerber(os.path.join(d, fa['Path']))
            g = affinity.scale(g, 1, -1, origin=(0, 0))          # -> KiCad convention (y down)
            self.items.append((name, '', g))
        drills = drills if drills is not None else [os.path.join(d, f) for f in os.listdir(d)
                                                      if f.lower().endswith(('.drl', '.xln', '.exc'))]
        for f in drills:
            holes, span, plated = parse_excellon(f)
            if not plated:
                continue
            lys = self.copper_order[(span[0] - 1):span[1]] if span else self.copper_order
            for (x, y, dia) in holes:
                self.vias.append(dict(x=x, y=-y, size=0, drill=dia, layers=lys, net=''))

    def describe_stackup(self):
        from mom25d_kicad import KiCadBoard
        return KiCadBoard.describe_stackup(self)

    def port_footprints(self, pos_refs=None):
        return self._ports

    def build(self, pos_file, ground='B.Cu', cover=None, region=None, include_mask=True, flip=False,
              port_props=None):
        rows = read_pos(pos_file)
        refs = pos_port_refs(rows)
        pat = re.compile(r'^P(\d+)([+\-PN])?$', re.I)
        self._ports = {}
        for r in rows:
            if r['ref'] not in refs:
                continue
            m = pat.match(r['val']) or pat.match(r['ref'])
            if not m:
                continue
            n, leg = int(m.group(1)), (m.group(2) or '').upper()
            props = dict((port_props or {}).get(f"P{n}", {}))
            fp = dict(ref=r['ref'], value=r['val'], fab=[r['val']], props=props, x=r['x'],
                      y=-r['y'], rot=r['rot'], side='B' if r['side'].lower().startswith('b') else 'F',
                      lib=r['package'], pads=[])
            if leg:
                self._ports.setdefault(n, DiffLegs())['+' if leg in ('+', 'P') else '-'] = fp
            else:
                self._ports[n] = fp
        for n, v in self._ports.items():
            if isinstance(v, DiffLegs) and set(v) != {'+', '-'}:
                raise ValueError(f"differential port P{n} needs both P{n}+ and P{n}- rows")
        self._ports = dict(sorted(self._ports.items()))
        return BoardModel(self, None, ground, cover, region, None, include_mask, flip, 4.5, 0.02)


# =============================================================================
# Writer (test fixtures; also handy for checking a model visually in a Gerber viewer)
# =============================================================================
def write_gerber(path, geom_mm_yup, function='Copper,L1,Top', tracks=(), flashes=()):
    """Regions for polygons; tracks as D01 draws with circular apertures; flashes of C / RoundRect."""
    L = ['%TF.GenerationSoftware,mom25d*%', f'%TF.FileFunction,{function}*%', '%FSLAX46Y46*%', '%MOMM*%',
         '%AMRoundRect*0 Rectangle with rounded corners*0 $1 Rounding radius*0 $2 $3 $4 $5 $6 $7 $8 $9 X,Y pos of 4 corners*'
         '4,1,4,$2,$3,$4,$5,$6,$7,$8,$9,$2,$3,0*1,1,$1+$1,$2,$3*1,1,$1+$1,$4,$5*1,1,$1+$1,$6,$7*1,1,$1+$1,$8,$9*'
         '20,1,$1+$1,$2,$3,$4,$5,0*20,1,$1+$1,$4,$5,$6,$7,0*20,1,$1+$1,$6,$7,$8,$9,0*20,1,$1+$1,$8,$9,$2,$3,0*%',
         '%LPD*%']
    c = lambda v: f"{int(round(v * 1e6))}"
    code = 10
    for (x0, y0, x1, y1, w) in tracks:
        L.append(f'%ADD{code}C,{w:.6f}*%'); L.append(f'D{code}*')
        L.append(f'X{c(x0)}Y{c(y0)}D02*'); L.append(f'X{c(x1)}Y{c(y1)}D01*'); code += 1
    for (kind, x, y, a, b) in flashes:
        if kind == 'C':
            L.append(f'%ADD{code}C,{a:.6f}*%')
        else:                                              # RoundRect: a = half size, b = radius
            q = a - b
            L.append(f'%ADD{code}RoundRect,{b:.6f}X{-q:.6f}X{-q:.6f}X{q:.6f}X{-q:.6f}X{q:.6f}X{q:.6f}X{-q:.6f}X{q:.6f}*%')
        L.append(f'D{code}*'); L.append(f'X{c(x)}Y{c(y)}D03*'); code += 1
    polys = [geom_mm_yup] if geom_mm_yup.geom_type == 'Polygon' else list(getattr(geom_mm_yup, 'geoms', []))
    for p in polys:
        if p.is_empty:
            continue
        for ring, pol in [(p.exterior, 'D')] + [(r, 'C') for r in p.interiors]:
            L.append(f'%LP{pol}*%'); L.append('G36*')
            pts = list(ring.coords)
            L.append(f'X{c(pts[0][0])}Y{c(pts[0][1])}D02*')
            L += [f'G01X{c(x)}Y{c(y)}D01*' for x, y in pts[1:]]
            L.append('G37*')
        L.append('%LPD*%')
    L.append('M02*')
    open(path, 'w').write("\n".join(L) + "\n")
