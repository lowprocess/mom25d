"""
mom25d.model - a planar model ready to solve: the stack, metal per interface, vias, lumped elements and port
definitions. A subclass fills in (board.BoardModel: from a kicad2ems board)

    stack       multilayer.LayerStack (ground at z = 0, interfaces 1..N)
    iface       {copper layer name: interface index}
    shapes      {interface: [shapely geometry, metres, y up]}
    vias        [sim.Via];  lumped: [sim.SeriesLumped]
    port_specs  [dict(name, c (m), d (unit vector into the circuit), k (interface), w (m), z0, kind 'auto' |
                 'wave' | 'via' | 'gap', kref (reference interface of a lumped port), radius (m))] or, for a
                 differential pair, dict(name, legs={'+': spec, '-': spec}); port_z0 {name: ohms}
    ground, cover, notes

and the methods here turn the port definitions into solver ports (wave ports where a straight feed
allows, lumped via ports otherwise), characterise the wave ports' line impedance and build the
Simulation.
"""
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from .base import C0

MM = 1e-3


KNOWN_SPAN = 0.1          # rad: the least span of a wave port's cuts at the lowest frequency, gamma known


def _cross(g, p, d, half):
    """The pieces of copper g on the line through p across direction d, `half` each side of p."""
    t = np.array([-d[1], d[0]])
    seg = LineString([p - t * half, p + t * half]).intersection(g)
    return [seg] if seg.geom_type == 'LineString' else list(getattr(seg, 'geoms', []))


def _smooth(fs, v):
    """f -> v, the polynomial through the characterised values (held beyond the ends): smooth, where linear
    interpolation's kinks cost the adaptive sweep solves."""
    fs, v = np.asarray(fs, float), np.asarray(v)
    c = np.polyfit(fs / fs[-1], v, min(len(fs) - 1, 3))
    return lambda f: np.polyval(c, np.clip(f, fs[0], fs[-1]) / fs[-1])


ZONE_DIST = 0.5e-3      # zone='auto': the mesh detail within this distance of the signal copper,
ZONE_H = 1.5            # or this many times the thickest dielectric (a GCPW's fence vias on 0.8 mm: 0.82 mm out),
ZONE_VIA = 1.0e-3       # and within this distance of a signal via (its return vias)

class PlanarModel:
    lumped = ()

    def describe(self):
        L = ["stack (from ground up):"]
        cu0 = [n for n, k in self.iface.items() if k == 0]
        if cu0:
            L.append(f"  metal {cu0[0]} on interface 0 (the bottom)")
        for j, l in enumerate(self.stack.layers):
            cu = [n for n, k in self.iface.items() if k == j + 1]
            L.append(f"  D{j + 1} {l.name:<14s} h={l.h * 1e3:.3f} mm er={l.er:.2f} tand={l.tand:.4f}"
                     + (f"   -> metal {cu[0]} on interface {j + 1}" if cu else ''))
        L.append((f"  ground: {self.ground} (z=0)" if self.ground else "  no ground: the whole stack, air below")
                 + (f", PEC cover: {self.cover}" if self.cover else ''))
        L.append("metal interfaces: " + ", ".join(f"{n}->M{k}" for n, k in self.iface.items() if k in self.shapes))
        L.append(f"vias: {len(self.vias)}   ports: " + ", ".join(
            sp['name'] + (' (differential)' if 'legs' in sp else '') for sp in self.port_specs))
        return "\n".join(L + self.notes)

    # ---- ports ----------------------------------------------------------------------------------------
    def _feed(self, g, c, d, step):
        """Width of the copper strip through c across direction d, and straight length along d."""
        def width(s):
            p = c + d * s
            return next((q.length for q in _cross(g, p, d, 5e-3) if q.distance(Point(*p)) < 1e-9), 0.0)
        w0, s = width(step / 2), step / 2
        if w0 == 0:
            return 0.0, 0.0
        while abs(width(s + step) - w0) <= 0.02 * w0 and s <= 0.2:
            s += step
        return w0, s

    def _port_geom(self, sp):
        """(centre, direction, interface, copper there) of a port definition, checked to be on that copper."""
        c, k = np.asarray(sp['c'], float), sp['k']
        g = unary_union(self.shapes[k]) if k in self.shapes else None
        if g is None or not g.buffer(1e-6).contains(Point(*c)):
            raise ValueError(f"{sp['name']} at ({c[0] / MM:.3f}, {-c[1] / MM:.3f}) mm (KiCad) is not on the copper of "
                             f"interface M{k}")
        return c, np.asarray(sp['d'], float), k, g

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
        step, out = step or dmax, []
        # a wave port of width w on the feed from c along d, in the window (gap offset, first / last cut, spacing)
        wave = lambda k, c, d, w, win, **kw: LinePort(k, c + d * win[0], d, w, -win[0], s0=win[1] - win[0],
                                                     s1=win[2] - win[0], step=win[3], **kw)
        for sp in self.port_specs:
            name = sp['name']
            if 'legs' in sp:                                              # differential pair
                legs = [(c, d, k, *self._feed(g, c, d, step)) for c, d, k, g in
                        (self._port_geom(sp['legs'][sgn]) for sgn in ('+', '-'))]
                (cp, dp_, kp, wp, Lp), (cn, dn, kn, wn, Ln) = legs
                if kp != kn or not np.allclose(dp_, dn, atol=1e-6):
                    raise ValueError(f"{name}: + and - legs must be on the same layer with the same direction")
                win, need = self._wave_window(max(wp, wn), min(Lp, Ln), step)
                if win is None:
                    raise ValueError(f"{name}: straight coupled feed too short for a differential wave port "
                                     f"({min(Lp, Ln) * 1e3:.2f} mm, need {need * 1e3:.2f} mm)")
                out.append(DiffPort(wave(kp, cp, dp_, wp, win), wave(kp, cn, dp_, wn, win), 100.0, 25.0, name=name))
                continue
            c, d, k, g = self._port_geom(sp)
            kind, z0 = sp.get('kind', 'auto'), sp['z0']
            if kind in ('auto', 'wave'):
                w, L = self._feed(g, c, d, step)
                win, need = self._wave_window(w, L, step)
                if win is not None and kind == 'auto' and fmin:
                    # with gamma from the line characterisation (characterize: gamma_fn) the two wave amplitudes
                    # need the cuts to span KNOWN_SPAN at the lowest frequency
                    er = self.stack.layers[k - 1].er
                    beta = 2 * np.pi * fmin * np.sqrt(0.5 * (er + 1)) / C0      # a lower bound
                    if (win[2] - win[1]) * beta < KNOWN_SPAN:
                        need = max(need, L - (win[2] - win[1]) + KNOWN_SPAN / beta)
                        win = None
                if win is not None:
                    out.append(wave(k, c, d, w, win, zc=z0, name=name))
                    continue
                if kind == 'wave':
                    raise ValueError(f"{name}: straight feed too short for a wave port ({L * 1e3:.2f} mm, need {need * 1e3:.2f} mm)")
                self.notes.append(f"{name}: no straight feed of {need * 1e3:.1f} mm for a wave port"
                                  + (f" down to {fmin / 1e9:g} GHz" if fmin else '') + f" (found {L * 1e3:.2f} mm) - lumped port")
                kind = 'via'
            if kind == 'via':
                kr = sp.get('kref', 0)                                     # its reference: below, or above it
                v = Via(c[0], c[1], min(kr, k), max(kr, k), radius=sp.get('radius', 0.1 * MM), name=f"{name}_via")
                out.append(ViaPort(v, z0=z0, name=name, sign=1 if kr < k else -1))
            elif kind == 'gap':
                out.append(GapPort(k, c, d, sp.get('width', 0.3 * MM), z0=z0, name=name))
            else:
                raise ValueError(f"{name}: unknown port kind {kind}")
        return out

    def _planes_at(self, layer, pt, d, w):
        """Metal on other layers covering point pt (reference planes), with its width across d (capped)."""
        planes = {}
        for k, gs in self.shapes.items():
            if k == layer or not (g := unary_union(gs)).buffer(1e-9).contains(pt):
                continue
            dz = abs(self.stack.z[layer] - self.stack.z[k])
            W = min(sum(q.length for q in _cross(g, np.array(pt.coords[0]), d, 0.05)), max(16 * dz + 2 * w, 20 * w, 2e-3))
            if W > 2 * w:
                planes[k] = W
        return planes

    def _coplanar_at(self, layer, pt, d, w, max_gap=None):
        """Same-layer metal beside the trace at pt (grounded-CPW side grounds): [(y0, y1)] offsets
        across the line (trace centred at 0), widths capped; gaps wider than max_gap are ignored."""
        h = self.stack.z[layer] if layer < len(self.stack.z) else self.stack.z[-1]
        max_gap = max_gap or max(5 * h, 3 * w)
        t, p0 = np.array([-d[1], d[0]]), np.array(pt.coords[0])
        iv = sorted(tuple(sorted(((np.array(c) - p0) @ t) for c in (q.coords[0], q.coords[-1])))
                    for q in _cross(unary_union(self.shapes.get(layer, [])), p0, d, 0.05) if q.length > 0)
        i = next((i for i, (a, b) in enumerate(iv) if a <= 1e-9 <= b or a <= -1e-9 <= b), None)   # the trace
        if i is None:
            return []
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
        from .sim import line_char_xs, DiffPort, line_char_pair
        fs = np.linspace(max(fmin, fmax / 3), fmax, n)    # the fit is well conditioned here; Zc ~ flat
        # the centre-fed line must keep the source's near field (a few widths and heights) out of the fit window:
        # 16 mm for a 3 mm line on 1.6 mm FR4 gave Zc 1.6 / 24 / 33 ohm and beta below k0, 48 mm 48-51 ohm;
        # a 0.38 mm line on 0.2 mm was right at 16 mm
        from .base import C0
        H = float(self.stack.z[-1])
        length_for = lambda w: max(16e-3, 10 * (w + H))
        # the reference planes no wider than the line and 3 H each side: the field is gone by then (Zc and gamma
        # within 0.01 % of a 13 mm plane's on sma_launch), and a wide plane had been most of the unknowns
        narrow = lambda planes, w: {k: min(W, w + 6 * H) for k, W in planes.items()}
        cache = {}
        for p in ports:
            if isinstance(p, DiffPort):
                a, b = p.pos, p.neg
                pt = Point(*(0.5 * (a.c + b.c) + a.d * 0.5 * (a.s0 + a.s1)))
                gap = abs((a.c - b.c) @ a.t) - 0.5 * (a.w + b.w)
                planes = narrow(self._planes_at(a.layer, pt, a.d, a.w), a.w + b.w + gap)
                ge, ze, go, zo = line_char_pair(self.stack, a.layer, 0.5 * (a.w + b.w), gap, fs, planes=planes, dmax=dmax,
                                                length=length_for(a.w + b.w + gap))
                zd, zcm = 2 * zo.real, ze.real / 2
                p.zd_fn, p.zc_fn = _smooth(fs, zd), _smooth(fs, zcm)
                p.zdiff, p.zcomm = float(zd[len(zd) // 2]), float(zcm[len(zcm) // 2])
                log(f"  {p.name}: coupled pair on M{a.layer}, w={a.w * 1e3:.3f} mm, gap={gap * 1e3:.3f} mm: "
                    f"Zdiff = " + ", ".join(f"{v:.1f}" for v in zd) + ", Zcomm = " + ", ".join(f"{v:.1f}" for v in zcm) + " ohm")
                continue
            if not isinstance(p, LinePort):
                continue
            pt = Point(*(p.c + p.d * 0.5 * (p.s0 + p.s1)))
            planes = narrow(self._planes_at(p.layer, pt, p.d, p.w), p.w)
            cop = self._coplanar_at(p.layer, pt, p.d, p.w)
            key = (p.layer, round(p.w * 1e7), tuple(sorted((k, round(W * 1e5)) for k, W in planes.items())),
                   tuple((round(a * 1e6), round(b * 1e6)) for a, b in cop))
            if key not in cache:
                g, z = line_char_xs(self.stack, p.layer, p.w, fs, planes=planes, coplanar=cop, dmax=dmax,
                                    length=length_for(p.w))
                cache[key] = (z.real, g)
                if np.any(np.abs(g.imag) < 0.95 * 2 * np.pi * fs / C0):       # slower than light: a failed fit
                    self.notes.append(f"WARNING {p.name}: the line characterisation fit failed (beta below k0): "
                                      f"Zc {', '.join(f'{v:.1f}' for v in z.real)} ohm is not reliable - lumped ports")
                gaps = [min(abs(a), abs(b)) - p.w / 2 for a, b in cop]
                log(f"  {p.name}: line on M{p.layer}, w={p.w * 1e3:.3f} mm, reference metal "
                    + (", ".join([f"M{k} ({W * 1e3:.1f} mm)" for k, W in planes.items()]
                                 + [f"coplanar GND, gap {gp * 1e3:.3f} mm" for gp in gaps]) or "ground only")
                    + ": Zc = " + ", ".join(f"{v:.2f}" for v in z.real) + " ohm")
            zz, gg = cache[key]
            p.zc_fn, p.zc = _smooth(fs, zz), float(zz[len(zz) // 2])
            # gamma / f is nearly flat; the wave fit takes it at every frequency (sim.sparams)
            p.gamma_fn = (lambda f, gf=_smooth(fs, gg / fs): f * gf(f))

    def simulation(self, dmax=0.3e-3, step=None, fmin=1e9, fmax=10e9, characterize=True, wave_down_to=None, **kw):
        """The Simulation of this model. wave_down_to (Hz): 'auto' ports become wave ports only where the feed
        is long enough for the wave fit down to that frequency (else lumped); default: wherever the geometry
        allows (the fit degrades below KNOWN_SPAN of sample span, with a warning). metal= (sim.Metal): conductor
        loss (default lossless)."""
        from .sim import Simulation
        ports = self.ports(step=step, dmax=dmax, fmin=wave_down_to)
        if characterize:
            self.characterize(ports, fmin, fmax, dmax=dmax)        # the cells the feeds get: the same gamma
        kw.setdefault('lumped', list(self.lumped))
        if getattr(self, 'signal', None) is not None:
            kw.setdefault('signal', self.signal)
        zone = kw.pop('zone', None)
        if zone is not None and zone != 'off':
            # (experimental) mesh detail only near the signal copper: zone = the distance (m) around it
            sig, kw['zone'] = getattr(self, 'signal', None), None
            if sig is not None:
                d = max(ZONE_DIST, ZONE_H * getattr(self, 'h_max', 0.0)) if zone == 'auto' else float(zone)
                kw['zone'] = unary_union([sig.buffer(d)] + [Point(x, y).buffer(ZONE_VIA)
                                                            for x, y in getattr(self, 'signal_vias', [])])
        return Simulation(self.stack, self.shapes, ports, vias=self.vias, dmax=dmax, **kw)
