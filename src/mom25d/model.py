"""
mom25d.model - a planar model ready to solve: the stack, metal per interface, vias and port definitions,
whatever it was read from (KiCad, Gerber, an AppCSXCAD / openEMS XML). The subclasses fill in

    stack       multilayer.LayerStack (ground at z = 0, interfaces 1..N)
    iface       {layer name: interface index}
    shapes      {interface: [shapely geometry, metres, y up]}
    vias        [sim.Via]
    port_fps    {port number: dict(x, y (mm, KiCad orientation: y down), rot (deg, into the circuit),
                 props {'mom.port', 'mom.z0', 'mom.layer', 'mom.ref', ...}, side 'F' | 'B', ref)} or a
                 DiffLegs pair; lumped: [sim.SeriesLumped]
    ground, cover, notes

and the methods here turn the port definitions into solver ports (wave ports where a straight feed
allows, lumped via ports otherwise), characterise the wave ports' line impedance and build the
Simulation.
"""
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

MM = 1e-3


class DiffLegs(dict):
    """{'+': footprint, '-': footprint} of a differential port (footprints are dicts themselves)."""


class PlanarModel:
    board = None
    lumped = ()

    def describe(self):
        L = ["stack (from ground up):"]
        for j, l in enumerate(self.stack.layers):
            cu = [n for n, k in self.iface.items() if k == j + 1]
            L.append(f"  D{j + 1} {l.name:<14s} h={l.h * 1e3:.3f} mm er={l.er:.2f} tand={l.tand:.4f}"
                     + (f"   -> metal {cu[0]} on interface {j + 1}" if cu else ''))
        L.append(f"  ground: {self.ground} (z=0)" + (f", PEC cover: {self.cover}" if self.cover else ''))
        L.append(f"metal interfaces: " + ", ".join(f"{n}->M{k}" for n, k in self.iface.items() if k in self.shapes))
        L.append(f"vias: {len(self.vias)}   ports: " + ", ".join(
            f"P{n}=" + (f"{fp['+']['ref']}/{fp['-']['ref']} (diff)" if isinstance(fp, DiffLegs) else fp['ref'])
            for n, fp in self.port_fps.items()))
        return "\n".join(L + self.notes)

    # ---- ports from footprints -----------------------------------------------------------
    def _copper_at(self, k):
        return unary_union(self.shapes[k]) if k in self.shapes else None

    def _feed(self, g, c, d, step):
        """Width of the copper strip through c across direction d, and straight length along d."""
        t = np.array([-d[1], d[0]])

        def width(s):
            p = c + d * s
            seg = LineString([p - t * 5e-3, p + t * 5e-3]).intersection(g)
            parts = [seg] if seg.geom_type == 'LineString' else list(getattr(seg, 'geoms', []))
            for q in parts:
                if q.distance(Point(*p)) < 1e-9:
                    return q.length, q
            return 0.0, None
        w0, _ = width(step / 2)
        if w0 == 0:
            return 0.0, 0.0
        s = step / 2
        while True:
            w, _ = width(s + step)
            if abs(w - w0) > 0.02 * w0 or s > 0.2:
                break
            s += step
        return w0, s

    def _port_geom(self, name, fp):
        c = np.array([fp['x'], -fp['y']]) * MM
        a = np.radians(fp['rot'])
        d = np.array([np.cos(a), np.sin(a)])
        props = fp['props']
        lay = props.get('mom.layer') or ('B.Cu' if fp['side'] == 'B' else 'F.Cu')
        if lay not in self.iface:
            raise ValueError(f"{name}: port layer {lay} is not a metal interface of the model")
        k = self.iface[lay]
        g = self._copper_at(k)
        if g is None or not g.buffer(1e-6).contains(Point(*c)):
            here = sorted({str(n) for l, n, gg in getattr(self.board, 'items', ())
                           if l == lay and gg.buffer(1e-6).contains(Point(fp['x'], fp['y']))})
            side = 'B.Cu' if fp['side'] == 'B' else 'F.Cu'
            on_side = props.get('mom.layer') and lay != side and any(
                l == side and gg.buffer(1e-6).contains(Point(fp['x'], fp['y'])) for l, n, gg in getattr(self.board, 'items', ()))
            hint = (f"; the footprint field mom.layer={lay} points away from the {side} copper under it - "
                    f"delete the field or set it to {side}" if on_side else
                    f"; copper of net(s) {here} is there - add it to nets" if here else
                    "; no copper of any net there - check the footprint position and mom.layer")
            raise ValueError(f"{name} ({fp['ref']}) at ({fp['x']:.3f}, {fp['y']:.3f}) mm is not on {lay} copper "
                             f"of the selected nets/region{hint}")
        return c, d, k, g, props

    def _wave_window(self, w, L, step):
        """(gap offset, first and last cut, cut spacing) for a straight feed of width w, length L."""
        clear = max(2 * step, 4 * w, 3 * self.stack.z[-1], 1.0e-3)   # stay out of the next discontinuity's near field
        need = 6 * step + clear
        if w <= 0 or L < need:
            return None, need
        g0 = step / 2                                                # gap just inside the trace end
        s0, s1 = g0 + 2 * step, L - clear
        return (g0, s0, s1, max(step, (s1 - s0) / 10)), need          # ~10 intervals: well-conditioned fit

    def ports(self, step=None, dmax=0.3e-3, fmin=None):
        from .geo import LinePort
        from .sim import ViaPort, GapPort, Via, DiffPort
        step = step or dmax
        out = []
        for n, fp in self.port_fps.items():
            name = (getattr(fp, 'name', None) if isinstance(fp, DiffLegs) else fp.get('name')) or f"P{n}"
            if isinstance(fp, DiffLegs):                                  # differential pair
                legs = {}
                for sgn in ('+', '-'):
                    c, d, k, g, props = self._port_geom(name + sgn, fp[sgn])
                    w, L = self._feed(g, c, d, step)
                    legs[sgn] = (c, d, k, w, L, props)
                (cp, dp_, kp, wp, Lp, pp), (cn, dn, kn, wn, Ln, pn) = legs['+'], legs['-']
                if kp != kn or not np.allclose(dp_, dn, atol=1e-6):
                    raise ValueError(f"{name}: + and - footprints must be on the same layer with the same rotation")
                win, need = self._wave_window(max(wp, wn), min(Lp, Ln), step)
                if win is None:
                    raise ValueError(f"{name}: straight coupled feed too short for a differential wave port "
                                     f"({min(Lp, Ln) * 1e3:.2f} mm, need {need * 1e3:.2f} mm)")
                g0, s0, s1, pstep = win
                ref_shift = float(pp.get('mom.ref', 0)) * MM
                mk = lambda c, w: LinePort(kp, c + dp_ * g0, dp_, w, ref_shift - g0, s0=s0 - g0, s1=s1 - g0,
                                           step=pstep)
                out.append(DiffPort(mk(cp, wp), mk(cn, wn), float(pp.get('mom.zdiff', 100)),
                                    float(pp.get('mom.zcomm', 25)), name=name))
                continue
            c, d, k, g, props = self._port_geom(name, fp)
            z0 = float(props.get('mom.z0', 50))
            kind = props.get('mom.port', 'auto')
            ref_shift = float(props.get('mom.ref', 0)) * MM
            if kind in ('auto', 'wave'):
                w, L = self._feed(g, c, d, step)
                win, need = self._wave_window(w, L, step)
                if win is not None and kind == 'auto' and fmin:
                    # the wave fit needs the sample cuts to span >= 0.4 rad at the lowest frequency
                    er = self.stack.layers[k - 1].er
                    beta = 2 * np.pi * fmin * np.sqrt(0.5 * (er + 1)) / 299792458.0      # a lower bound
                    if (win[2] - win[1]) * beta < 0.4:
                        need = max(need, L - (win[2] - win[1]) + 0.4 / beta)
                        win = None
                if win is not None:
                    g0, s0, s1, pstep = win
                    out.append(LinePort(k, c + d * g0, d, w, ref_shift - g0, s0=s0 - g0, s1=s1 - g0, step=pstep,
                                        zc=z0, name=name))
                    continue
                if kind == 'wave':
                    raise ValueError(f"{name}: straight feed too short for a wave port ({L * 1e3:.2f} mm, need {need * 1e3:.2f} mm)")
            if kind in ('auto', 'via'):
                if kind == 'auto':
                    self.notes.append(f"{name}: no straight feed of {need * 1e3:.1f} mm for a wave port"
                                      + (f" down to {fmin / 1e9:g} GHz" if fmin else '')
                                      + f" (found {L * 1e3:.2f} mm) - lumped port")
                kref = int(props.get('mom.kref', 0))                     # reference: ground or a plane below
                v = Via(c[0], c[1], kref, k, radius=float(props.get('mom.radius', 0.1)) * MM, name=f"{name}_via")
                out.append(ViaPort(v, z0=z0, name=name)); continue
            if kind == 'gap':
                w = float(props.get('mom.width', 0.3)) * MM
                out.append(GapPort(k, c, d, w, z0=z0, name=name)); continue
            raise ValueError(f"{name}: unknown port kind {kind}")
        return out

    def _planes_at(self, layer, pt, d, w):
        """Metal on other layers covering point pt (reference planes), with its width across d (capped)."""
        planes = {}
        t = np.array([-d[1], d[0]])
        for k, gs in self.shapes.items():
            if k == layer:
                continue
            g = unary_union(gs)
            if g.buffer(1e-9).contains(pt):
                p0 = np.array(pt.coords[0])
                seg = LineString([p0 - t * 0.05, p0 + t * 0.05]).intersection(g)
                dz = abs(self.stack.z[layer] - self.stack.z[k])
                W = min(getattr(seg, 'length', 0.0), max(16 * dz + 2 * w, 20 * w, 2e-3))
                if W > 2 * w:
                    planes[k] = W
        return planes

    def _coplanar_at(self, layer, pt, d, w, max_gap=None):
        """Same-layer metal beside the trace at pt (grounded-CPW side grounds): [(y0, y1)] offsets
        across the line (trace centred at 0), widths capped; gaps wider than max_gap are ignored."""
        h = self.stack.z[layer] if layer < len(self.stack.z) else self.stack.z[-1]
        max_gap = max_gap or max(5 * h, 3 * w)
        t = np.array([-d[1], d[0]])
        p0 = np.array(pt.coords[0])
        g = unary_union(self.shapes.get(layer, []))
        span = 0.05
        seg = LineString([p0 - t * span, p0 + t * span]).intersection(g)
        parts = [seg] if seg.geom_type == 'LineString' else list(getattr(seg, 'geoms', []))
        iv = sorted(tuple(sorted(((np.array(c) - p0) @ t) for c in (q.coords[0], q.coords[-1]))) for q in parts
                    if q.length > 0)
        own = [i for i, (a, b) in enumerate(iv) if a <= 1e-9 <= b or a <= -1e-9 <= b]
        if not own:
            return []
        i = own[0]
        out = []
        if i + 1 < len(iv) and iv[i + 1][0] - iv[i][1] <= max_gap:            # right side
            a, b = iv[i + 1]
            cap = min(max(2 * (w + 2 * (a - iv[i][1])), 1.0e-3), 2.0e-3)     # the field sees a few gaps
            out.append((a, min(b, a + cap)))
        if i - 1 >= 0 and iv[i][0] - iv[i - 1][1] <= max_gap:                  # left side
            a, b = iv[i - 1]
            cap = min(max(2 * (w + 2 * (iv[i][0] - b)), 1.0e-3), 2.0e-3)
            out.append((max(a, b - cap), b))
        return out

    def characterize(self, ports, fmin, fmax, n=3, dmax=0.4e-3, log=print):
        """Reference impedance of each wave port's line cross-section (trace + metal on other layers
        covering the port), solved as a short centre-fed line at n frequencies and interpolated."""
        from .geo import LinePort
        from .sim import line_char_xs
        fs = np.linspace(max(fmin, fmax / 3), fmax, n)    # the fit is well conditioned here; Zc ~ flat
        cache = {}
        from .sim import DiffPort, line_char_pair
        for p in ports:
            if isinstance(p, DiffPort):
                a, b = p.pos, p.neg
                mid = 0.5 * (a.s0 + a.s1)
                pt = Point(*(0.5 * (a.c + b.c) + a.d * mid))
                planes = self._planes_at(a.layer, pt, a.d, a.w)
                gap = abs((a.c - b.c) @ a.t) - 0.5 * (a.w + b.w)
                ge, ze, go, zo = line_char_pair(self.stack, a.layer, 0.5 * (a.w + b.w), gap, fs, planes=planes, dmax=dmax)
                zd, zcm = 2 * zo.real, ze.real / 2
                p.zd_fn = (lambda f, fs=fs, z=zd: np.interp(f, fs, z))
                p.zc_fn = (lambda f, fs=fs, z=zcm: np.interp(f, fs, z))
                p.zdiff, p.zcomm = float(zd[len(zd) // 2]), float(zcm[len(zcm) // 2])
                log(f"  {p.name}: coupled pair on M{a.layer}, w={a.w * 1e3:.3f} mm, gap={gap * 1e3:.3f} mm: "
                    f"Zdiff = " + ", ".join(f"{v:.1f}" for v in zd) + ", Zcomm = " + ", ".join(f"{v:.1f}" for v in zcm) + " ohm")
                continue
            if not isinstance(p, LinePort):
                continue
            pt = Point(*(p.c + p.d * 0.5 * (p.s0 + p.s1)))
            planes = {}
            for k, gs in self.shapes.items():
                if k == p.layer:
                    continue
                g = unary_union(gs)
                if g.buffer(1e-9).contains(pt):
                    # width of that metal across the line, capped (a plane looks infinite beyond ~30 h)
                    t = np.array([-p.d[1], p.d[0]])
                    seg = LineString([np.array(pt.coords[0]) - t * 0.05, np.array(pt.coords[0]) + t * 0.05]).intersection(g)
                    dz = abs(self.stack.z[p.layer] - self.stack.z[k])
                    W = min(getattr(seg, 'length', 0.0), max(16 * dz + 2 * p.w, 20 * p.w, 2e-3))
                    if W > 2 * p.w:
                        planes[k] = W
            cop = self._coplanar_at(p.layer, pt, p.d, p.w)
            key = (p.layer, round(p.w * 1e7), tuple(sorted((k, round(W * 1e5)) for k, W in planes.items())),
                   tuple((round(a * 1e6), round(b * 1e6)) for a, b in cop))
            if key not in cache:
                g, z = line_char_xs(self.stack, p.layer, p.w, fs, planes=planes, coplanar=cop, dmax=dmax)
                cache[key] = z.real
                gaps = [min(abs(a), abs(b)) - p.w / 2 for a, b in cop]
                log(f"  {p.name}: line on M{p.layer}, w={p.w * 1e3:.3f} mm, reference metal "
                    + (", ".join([f"M{k} ({W * 1e3:.1f} mm)" for k, W in planes.items()]
                                 + [f"coplanar GND, gap {gp * 1e3:.3f} mm" for gp in gaps]) or "ground only")
                    + f": Zc = " + ", ".join(f"{v:.2f}" for v in z.real) + " ohm")
            zz = cache[key]
            p.zc_fn = (lambda f, fs=fs, zz=zz: np.interp(f, fs, zz))
            p.zc = float(zz[len(zz) // 2])

    def simulation(self, dmax=0.3e-3, step=None, fmin=1e9, fmax=10e9, characterize=True, wave_down_to=None, **kw):
        """The Simulation of this model. wave_down_to (Hz): 'auto' ports become wave ports only where the feed
        is long enough for the wave fit down to that frequency (else lumped); default: wherever the geometry
        allows (the fit degrades below ~0.4 rad of sample span, with a warning)."""
        from .sim import Simulation
        ports = self.ports(step=step, dmax=dmax, fmin=wave_down_to)
        if characterize:
            self.characterize(ports, fmin, fmax)
        kw.setdefault('lumped', list(self.lumped))
        return Simulation(self.stack, self.shapes, ports, vias=self.vias, dmax=dmax, **kw)
