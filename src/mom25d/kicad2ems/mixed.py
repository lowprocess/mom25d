"""
mixed: mom25d for the board, its long straight runs of trace as transmission lines (segmentation).

A long uniform line costs mom25d unknowns all along it, and the line model already knows it: per unit length
its gamma and Zc. Here the middle of each long, lone, straight run of signal trace is cut out of the board;
mom25d solves what is left with a de-embedded wave port on each face of each cut (characterised on the line's own
cross-section: the planes over it, coplanar ground beside it), and each cut is put back as a transmission line
of the cut's length with that gamma and Zc:

    S = S_xx + S_xc L (I - S_cc L)^-1 S_cx       x: the board's ports, c: the cut ports, L: the lines

What it leaves out is the coupling of a cut stretch to anything but its own line: a cut is made only where no
other signal copper (any layer) comes within `keep_out` of it, and not through vias, parts or ports. Reported as
its own solver, 'mixed' (mom25d + lines), next to mom25d's.
"""
import time

import numpy as np

from kicad2ems.api.solver import Result, Option
from .mom import MoM25D, _metal
from ..base import C0
from kicad2ems.api.solver import stop

G = 'mixed'                                     # its section in kicad2ems's GUI
CUT_MIN = 3.0           # mm: the shortest stretch worth cutting out


def _feed_areas(ports):
    """The copper each wave port's fit samples (gap to last cut, plus its width either side), so a cut keeps out."""
    from shapely.geometry import LineString
    from mom25d.geo import LinePort
    out = []
    for p in ports:
        for q in (p.pos, p.neg) if hasattr(p, 'pos') else (p,):
            if isinstance(q, LinePort):
                far = q.s1 if q.s1 is not None else q.ref
                a, b = q.c + q.d * min(0.0, q.ref), q.c + q.d * (max(far, q.ref) + 2 * q.w)
                out.append(LineString([a, b]).buffer(2 * q.w))
    return out


def plan_cuts(model, data, dmax, fmin, cut_min=CUT_MIN * 1e-3, keep_out=None, log=print, feeds=()):
    """The stretches of trace to cut out: [(Run of the cut, the run it is in)]. model: mom25d BoardModel; data: its
    board dict (nets per polygon); dmax (m); fmin (Hz): the wave ports on the cut faces need their feed."""
    from shapely.geometry import Point, LineString, shape
    from shapely.ops import unary_union
    from mom25d.line import find_runs, Run
    from mom25d.model import KNOWN_SPAN
    MM = 1e-3
    if model.signal is None:
        log("mixed: the board names no ground net - no signal trace to cut")
        return []
    gnets = set(data.get('ground_nets') or [])
    sig_k = {}
    for lay, ps in data.get('copper', {}).items():
        k = model.iface.get(lay)
        if k is not None:
            g = [shape(p['polygon']) for p in ps if p.get('net') not in gnets]
            if g:
                from shapely import affinity
                sig_k[k] = affinity.scale(unary_union(g), MM, MM, origin=(0, 0)).buffer(0)
    geoms = {k: unary_union(gs) for k, gs in model.shapes.items()}
    runs = find_runs(geoms, {k: dmax for k in geoms}, signal=model.signal, w_max=3e-3, min_len=1.0, margin=1.0)
    vias = [Point(v.x, v.y) for v in model.vias]
    ports = [Point(*sp['c']) for sp in model.port_specs if 'c' in sp] + \
            [Point(*leg['c']) for sp in model.port_specs if 'legs' in sp for leg in sp['legs'].values()]
    parts = [el.lp.line(0.0) for el in model.lumped] if model.lumped else []
    cuts = []
    for r in runs:
        w = 2 * r.a
        win, need0 = model._wave_window(w, 1.0, dmax)
        er = model.stack.layers[r.k - 1].er
        beta = 2 * np.pi * fmin * np.sqrt(0.5 * (er + 1)) / C0
        need = max(need0, 1.0 - (win[2] - win[1]) + KNOWN_SPAN / beta)        # the feed each cut face needs
        need += 1e-6                                                          # (rounding: the port still fits)
        xc0, xc1 = r.x0 + need, r.x1 - need
        if xc1 - xc0 < cut_min:
            continue
        cut = Run(r.k, r.c0, r.t, r.a, xc0, xc1)
        rect = cut.polygon()
        h = model.stack.layers[r.k - 1].h                    # the dielectric under the trace
        d = keep_out or max(3 * w, 4 * h, 0.5e-3)            # no other signal copper this close
        near = rect.buffer(d)
        if any(rect.buffer(max(2 * w, 3 * h)).contains(p) for p in vias) or any(near.contains(p) for p in ports):
            continue
        if any(near.intersects(LineString(l)) for l in parts) or any(rect.intersects(a) for a in feeds):
            continue
        own = Run(r.k, r.c0, r.t, r.a * (1 + 1e-6), xc0 - 2 * d, xc1 + 2 * d).polygon(pad=1e-7)
        crowded = False
        for k, g in sig_k.items():
            other = g.difference(own) if k == r.k else g
            if other.intersects(near):
                crowded = True
                break
        if crowded:
            continue
        cuts.append((cut, r))
    return cuts


def connect(S, cut_idx, lines):
    """The board's ports' S from the network S (n x n) whose ports cut_idx = [(a, b), ...] are joined in pairs by
    two-ports lines = [S_line (2 x 2), ...] (same reference impedance on each side of each joint)."""
    n = S.shape[0]
    c = [i for ab in cut_idx for i in ab]
    x = [i for i in range(n) if i not in c]
    L = np.zeros((len(c), len(c)), complex)
    for j, Sl in enumerate(lines):
        L[2 * j:2 * j + 2, 2 * j:2 * j + 2] = Sl
    Sxx, Sxc, Scx, Scc = S[np.ix_(x, x)], S[np.ix_(x, c)], S[np.ix_(c, x)], S[np.ix_(c, c)]
    return Sxx + Sxc @ L @ np.linalg.solve(np.eye(len(c)) - Scc @ L, Scx)


def line_S(gamma, zc, length, z0):
    """S of a uniform line (gamma 1/m, Zc ohm, length m) referenced to z0 at both ends."""
    gl = gamma * length
    sh, ch = np.sinh(gl), np.cosh(gl)
    den = (zc * zc + z0 * z0) * sh + 2 * zc * z0 * ch
    s11 = (zc * zc - z0 * z0) * sh / den
    s21 = 2 * zc * z0 / den
    return np.array([[s11, s21], [s21, s11]])


def solve_mixed(m, freq, dmax=0.3, sigma=None, max_solves=60, cut_min=CUT_MIN, keep_out=None, log=print,
                progress=None, on_step=None, port='auto', hold=None, **mom_kw):
    """S over the board's ports (kicad2ems order, as mom.solve returns them), with the long straight runs cut out of
    mom25d's model and put back as lines. mom_kw: the mom25d mesh options (quads, zone, lines, signal_vias, ...)."""
    from mom25d.board import BoardModel
    from mom25d.geo import LinePort
    from mom25d.run import solve_band
    from mom25d.api import ZONE_OUT, LINE_STEP
    from kicad2ems.api.solver import mixed_mode
    t0 = time.time()
    data = m.to_dict()
    freq = np.asarray(freq, float)
    fmin, fmax = float(freq[0]), float(freq[-1])
    dm = dmax * 1e-3
    metal, t_mm = _metal(m, sigma)
    model = BoardModel(data, port='via' if port == 'lumped' else port, log=lambda *a: None)
    # the board's own ports, from the uncut copper: a cut would shorten their straight feeds (wave ports turned
    # lumped, and so different terminations); the cuts keep out of their feeds
    # (each wave port's feed capped at what its fit needs down to fmin - a port fits over its whole straight feed
    # otherwise, and leaves nothing of the line to cut)
    from mom25d.model import KNOWN_SPAN
    er_min = min(l.er for l in model.stack.layers)
    beta_lo = 2 * np.pi * fmin * np.sqrt(0.5 * (er_min + 1)) / C0
    feed = model._feed

    def capped(g, c, d, step):
        w, L = feed(g, c, d, step)
        win, need0 = model._wave_window(w, 1.0, step)
        if win is None:
            return w, L
        need = max(need0, 1.0 - (win[2] - win[1]) + 1.2 * KNOWN_SPAN / beta_lo)
        return w, min(L, need)
    model._feed = capped
    base = model.ports(step=None, dmax=dm, fmin=fmin)
    model._feed = feed
    cuts = plan_cuts(model, data, dm, fmin, cut_min=cut_min * 1e-3, keep_out=keep_out and keep_out * 1e-3, log=log,
                     feeds=_feed_areas(base))
    removed = 0.0
    cut_ports = []
    for j, (c, r) in enumerate(cuts):
        rect = c.polygon(pad=1e-7)
        model.shapes[c.k] = [model.shapes[c.k][0].difference(rect).buffer(0)]
        if model.signal is not None:
            model.signal = model.signal.difference(rect).buffer(0)
        removed += c.x1 - c.x0
        for end, (x, d) in enumerate(((c.x0, -c.t), (c.x1, c.t))):
            face = c.c0 + c.t * x
            L = (x - r.x0) if end == 0 else (r.x1 - x)
            win, _ = model._wave_window(2 * c.a, L, dm)
            g0, s0, s1, pstep = win
            cut_ports.append(LinePort(c.k, face + d * g0, d, 2 * c.a, -g0, s0=s0 - g0, s1=s1 - g0, step=pstep,
                                      zc=50.0, name=f"cut{j + 1}{'ab'[end]}"))
    if cuts:
        log(f"mixed: {len(cuts)} stretch(es) of trace as lines, {removed * 1e3:.1f} mm in all: " +
            ", ".join(f"M{c.k} w={2 * c.a * 1e3:.3f} mm {1e3 * (c.x1 - c.x0):.1f} mm" for c, _ in cuts))
    else:
        log("mixed: no long, lone straight run to cut - this is mom25d alone")
    def ports(**kw):
        return list(base) + cut_ports
    model.ports = ports
    planes = {k: 3 * dm for k, gs in model.shapes.items() if sum(g.area for g in gs) > 0.5 * model.outline_area}
    zone, lines = mom_kw.pop('zone', None), mom_kw.pop('lines', None)
    sim = model.simulation(dmax=dm, dmax_layer=planes, fmin=fmin, fmax=fmax, solver='auto', verbose=False,
                           wave_down_to=fmin, metal=metal, **mom_kw,
                           **(dict(zone=zone, dmax_out=ZONE_OUT * dm) if zone else {}),
                           **(dict(line_basis=lines, line_step=LINE_STEP) if lines and lines != 'off' else {}))
    names = [p.name for p in sim.ports]
    zref = dict(model.port_z0)
    for p in cut_ports:
        zref[p.name] = float(p.zc)
    f, Sn, info = solve_band(sim, fmin, fmax, points=len(freq), zref=zref, max_solves=max_solves, log=log,
                             progress=progress, on_step=on_step)
    pairs = [(names.index(f"cut{j + 1}a"), names.index(f"cut{j + 1}b")) for j in range(len(cuts))]
    S = []
    for i, fr in enumerate(f):
        Ls = []
        for j, (c, r) in enumerate(cuts):
            pa = cut_ports[2 * j]
            Ls.append(line_S(pa.gamma_fn(fr), pa.zc_fn(fr), c.x1 - c.x0, pa.zc))
        S.append(connect(Sn[i], pairs, Ls) if cuts else Sn[i])
    S = np.array(S)
    keep = [n for n in names if not n.startswith('cut')]
    if len(f) != len(freq) or not np.allclose(f, freq):
        S = np.stack([np.stack([np.interp(freq, f, S[:, i, j].real) + 1j * np.interp(freq, f, S[:, i, j].imag)
                                for j in range(S.shape[2])], -1) for i in range(S.shape[1])], 1)
    order = [keep.index(p.name) for p in m.ports]
    S, out_names, out_zref = mixed_mode(S[:, order][:, :, order], m)
    unknowns = int(sim.qT.shape[1]) if getattr(sim, 'qT', None) is not None else int(sim.Ne + sim.Nv)
    info = dict(seconds=time.time() - t0, solves=int(info['solves']), unknowns=unknowns, cuts=len(cuts),
                cut_mm=round(removed * 1e3, 3), notes=list(model.notes) + list(sim.log.notes),
                samples_ghz=[round(x / 1e9, 6) for x in sorted(info['solved'])])
    if hold is not None:                         # for the current dumps (Mixed.outputs)
        hold.update(sim=sim, model=model, cuts=cuts, cut_ports=cut_ports, pairs=pairs)
    return S, out_names, out_zref, info


def dump_fields(h):
    """current_dumps' currents(f) and extra(...) for a mixed solution h (solve_mixed's hold): mom25d's currents of
    each port driven alone (1 V gap source) combined into the joined network's state, as mom25d's own dumps: 1 V
    on the driven port's gap, the board's other ports' gaps shorted, the cut ports loaded by their lines. The cut
    stretches are drawn from their lines' waves: on the trace with mom25d's own profile across it (taken just outside
    the faces), and on the planes over and under it their return current, the line's image spread over its height."""
    sim, cuts, cut_ports, pairs = h['sim'], h['cuts'], h['cut_ports'], h['pairs']
    n = len(sim.ports)
    c = [i for ab in pairs for i in ab]
    x = [i for i in range(n) if i not in c]
    waves = []                                    # per call (frequency): {column: (a, b)}

    def currents(f):
        sim.sparams(f, keep=True)                                  # its currents and waves: sim.last
        I, A, B = sim.last['I'], sim.last['A'], sim.last['B']
        L = np.zeros((len(c), len(c)), complex)
        for j, (cut, _) in enumerate(cuts):
            tau = np.exp(-cut_ports[2 * j].gamma_fn(f) * (cut.x1 - cut.x0))
            L[2 * j, 2 * j + 1] = L[2 * j + 1, 2 * j] = tau          # each line matched to its ports' own Zc
        out = np.zeros((I.shape[0], n), complex)
        w = {}
        for jj in x:
            # mom25d's convention: 1 V on the driven port's gap, the other board ports' gaps shorted; the cut
            # ports' gap voltages are whatever makes their waves those of the lines: A_c V = L B_c V
            V = np.zeros(n, complex)
            V[jj] = 1.0
            if c:
                M = A[c] - L @ B[c]
                V[c] = np.linalg.solve(M[:, c], -M[:, jj])
            out[:, jj] = I @ V
            w[jj] = (A @ V, B @ V)
        waves.append((f, w))
        return out

    # the planes over and under each cut stretch (they carry its return current), and their share of it
    model = h['model']
    zk = model.stack.z
    refs = []
    for cut, _ in cuts:
        from shapely.geometry import Point
        mid = cut.c0 + cut.t * 0.5 * (cut.x0 + cut.x1)
        planes = model._planes_at(cut.k, Point(*mid), cut.t, 2 * cut.a)
        below = max([q for q in planes if q < cut.k] + [0])
        above = [q for q in planes if q > cut.k]
        near = [(below, zk[cut.k] - zk[below])] + ([(min(above), zk[min(above)] - zk[cut.k])] if above else [])
        tot = sum(1 / hh for _, hh in near)
        refs.append([(q, hh, (1 / hh) / tot) for q, hh in near if q > 0])     # the ground (q = 0) is not meshed

    def extra(k, i, j, gx, gy, Jx, Jy):
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        f, w = waves[i]
        if j not in w:
            return Jx, Jy
        a, b = w[j]
        Jx, Jy = Jx.copy(), Jy.copy()
        P = np.c_[X.ravel(), Y.ravel()]
        for m, (cut, _) in enumerate(cuts):
            on_trace, planes = cut.k == k, {q: (hh, sh) for q, hh, sh in refs[m]}
            if not on_trace and k not in planes:
                continue
            pa = pairs[m][0]
            p = cut_ports[2 * m]
            g, zc = p.gamma_fn(f), float(p.zc_fn(f))
            u, v = cut.local(P)
            line = lambda s_: (b[pa] * np.exp(-g * s_) - a[pa] * np.exp(g * s_)) / np.sqrt(zc)   # along +t, A
            if on_trace:
                # the profile across the trace: mom25d's own, from the columns just outside the two faces
                dx = abs(gx[1] - gx[0]) if len(gx) > 1 else 1e-4
                prof, wsum = np.zeros(len(u)), 0.0
                off = max(5 * dx, 4 * cut.a)                     # clear of the cut ports' gaps
                for xf in (cut.x0 - off, cut.x1 + off):
                    col = (np.abs(u - xf) < 0.75 * dx) & (np.abs(v) < cut.a)
                    Jt = (Jx.ravel()[col] * cut.t[0] + Jy.ravel()[col] * cut.t[1])
                    I_face = line(xf - cut.x0)
                    if col.any() and abs(I_face) > 0:
                        pr = np.real(Jt / I_face)
                        prof += np.interp(v, v[col][np.argsort(v[col])], pr[np.argsort(v[col])])
                        wsum += 1
                # the line's current from the profile columns inward: past the faces too, over the cut ports'
                # gap sources (their near field is an artefact of the cut, not of the board)
                on = (u >= cut.x0 - off) & (u <= cut.x1 + off) & (np.abs(v) < cut.a)
                if wsum:
                    prof = prof / wsum
                else:                                                   # the isolated strip's, if no columns
                    prof = 1 / (np.pi * np.sqrt(np.maximum(cut.a ** 2 - v ** 2, (0.05 * cut.a) ** 2)))
                J = np.zeros(len(u), complex)
                J[on] = line(u[on] - cut.x0) * prof[on]
                keep = ~on.reshape(X.shape)                              # replace, not add, on the trace
                Jx, Jy = Jx * keep, Jy * keep
            else:
                # the plane's return current under the stretch: the image of the line, spread over ~ its height
                hh, sh = planes[k]
                on = (u >= cut.x0) & (u <= cut.x1)
                J = np.zeros(len(u), complex)
                J[on] = -sh * line(u[on] - cut.x0) * (hh / np.pi) / (hh * hh + v[on] ** 2)
            Jx += (J * cut.t[0]).reshape(X.shape)
            Jy += (J * cut.t[1]).reshape(X.shape)
        return Jx, Jy
    return currents, extra


class Mixed(MoM25D):
    name, aliases, label = 'mixed', ('segmented',), 'mixed (mom25d + lines)'
    in_all = False                  # --solvers all: a variant of mom25d, named on its own
    order = 30
    capabilities = dict(MoM25D.capabilities, ports='yes: as mom25d',
                        diff_pairs='yes: as mom25d (coupled stretches are not cut)',
                        radiation='approximated: as mom25d, except from the stretches put back as lines')
    options = [
        Option('mixed_cut', CUT_MIN, "mixed: the shortest stretch of straight trace put back as a line, mm (default "
               "{default}); its mom25d mesh options are mom25d's (--mom-dmax, --mom-zone, ...)", float, name='cut',
               gui='Shortest stretch cut out (mm)', group=G, cli=dict(metavar='MM')),
    ]

    def assess(self, run):
        out = super().assess(run)
        if run.s.mom_stack == 'whole':                              # its lines are cut against the ground plane
            out.append("mom_stack = whole: mixed cuts its lines over the infinite ground (mom25d models the whole "
                       "stack)")
        return out

    def solve(self, run, prep):
        s = run.s
        self._held = {}
        S, names, zref, info = solve_mixed(
            run.m, run.freq, dmax=s.mom_dmax, sigma=s.sigma, max_solves=s.mom_solves, cut_min=s.mixed_cut, log=run.log,
            port=s.mom_port, hold=self._held,
            quads={'off': False, 'on': True}.get(s.mom_quads, s.mom_quads),
            zone='auto' if s.mom_zone == 'on' else None, lines=s.mom_lines,
            signal_vias=s.mom_signal_vias, stitching_vias=s.mom_stitching_vias,
            progress=lambda k, n, f: (run.status('mixed', dict(fraction=k / n, solves=k, f_ghz=f / 1e9)), stop.check()))
        info.update(kpi=f"{info['unknowns']:,} unknowns, {info['cuts']} line(s), {info['seconds']:.0f} s",
                    about=f"mixed: mom25d with {info['cuts']} long straight stretch(es) of trace ({info['cut_mm']:g} mm) "
                          f"cut out and put back as transmission lines (gamma and Zc from each line's cross-section): "
                          f"{info['unknowns']:,} unknowns, {info['solves']} EM solves in {info['seconds']:.0f} s. The cut "
                          f"stretches couple to nothing but their own line.")
        run.log(f"mixed: {info['unknowns']:,} unknowns, {info['cuts']} line(s), {info['solves']} EM solves, "
                f"{info['seconds']:.0f} s")
        return Result(S, names, zref, info['seconds'], info=info, comment=f"mixed: mom25d + lines ({run.name})")

    def outputs(self, run, prep, res):
        """The current dumps (--mom-dump) of the joined solution: <out>/mixed/exc_<port>/."""
        h = getattr(self, '_held', None)
        if not run.s.mom_dump or not h:
            return
        from types import SimpleNamespace
        currents, extra = dump_fields(h)
        self._dumps(run, SimpleNamespace(sim=h['sim'], model=h['model']), currents=currents, extra=extra,
                    unit='1 V gap source')

    def estimate(self, run):
        return None
