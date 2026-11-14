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

Two traces of one width that run side by side within that distance are cut together, as a coupled line
(plan_pair_cuts): each face of the cut is a mixed-mode wave port (mom25d's DiffPort: the pair's differential and
common modes, each with its own gamma and impedance from the pair's cross-section), and the stretch is put back
as two lines, one a mode, which a symmetric pair's modes are. The traces need not be a differential pair of the
board: a coupler's lines or two signals routed together are cut the same way.
"""
import time

import numpy as np

from kicad2ems.api.solver import Result, Option
from .mom import MoM25D, _metal, run_finish, _stack, apertures, aperture_text, ports_below
from ..base import C0
from kicad2ems.api.solver import stop

G = 'mixed'                                     # its section in kicad2ems's GUI
CUT_MIN = 3.0           # mm: the shortest stretch worth cutting out
# A pair's far-end coupling is the difference of its two modes' speeds (a few percent of either), so its faces'
# ports and its lines have to have both closer than a lone line's: on two 0.38 mm traces 0.3 mm apart, 40 mm,
# from 2 GHz, |S41| 0.086 for mom25d's 0.103 with a lone cut's feed and the 16 mm characterisation line, 0.099
# with these (and max |dS| 0.022 -> 0.006)
PAIR_SPAN = 0.4         # rad: what the cuts of a pair's face port span at the lowest frequency (a lone line's: 0.1)
PAIR_CHAR = 2.4         # rad: the fit window of the line a pair's modes are characterised on, at its lowest frequency
PAIR_CHAR_MAX = 64e-3   # m: that line at most


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


def _context(model, data, dmax):
    """What the cuts are planned from: (the straight runs of signal trace, the signal copper by interface, the
    vias, the ports and the parts' lines), or None where the board names no ground net."""
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    from shapely import affinity
    from mom25d.line import find_runs
    MM = 1e-3
    if model.signal is None:
        return None
    gnets = set(data.get('ground_nets') or [])
    sig_k = {}
    for lay, ps in data.get('copper', {}).items():
        k = model.iface.get(lay)
        if k is not None:
            g = [shape(p['polygon']) for p in ps if p.get('net') not in gnets]
            if g:
                sig_k[k] = affinity.scale(unary_union(g), MM, MM, origin=(0, 0)).buffer(0)
    geoms = {k: unary_union(gs) for k, gs in model.shapes.items()}
    runs = find_runs(geoms, {k: dmax for k in geoms}, signal=model.signal, w_max=3e-3, min_len=1.0, margin=1.0)
    vias = [Point(v.x, v.y) for v in model.vias]
    ports = [Point(*sp['c']) for sp in model.port_specs if 'c' in sp] + \
            [Point(*leg['c']) for sp in model.port_specs if 'legs' in sp for leg in sp['legs'].values()]
    parts = [el.lp.line(0.0) for el in model.lumped] if model.lumped else []
    return runs, sig_k, vias, ports, parts


def _need(model, k, w, dmax, fmin, span=None):
    """The straight feed a wave port on a cut face needs, for a line of width w on interface k down to fmin: its
    cuts spanning `span` rad there (default mom25d's KNOWN_SPAN)."""
    from mom25d.model import KNOWN_SPAN
    win, need0 = model._wave_window(w, 1.0, dmax)
    er = model.stack.layers[k - 1].er
    beta = 2 * np.pi * fmin * np.sqrt(0.5 * (er + 1)) / C0
    return max(need0, 1.0 - (win[2] - win[1]) + (span or KNOWN_SPAN) / beta) + 1e-6      # (rounding: the port still fits)


def _clear(rect, legs, k, w, h, keep_out, ctx, feeds):
    """Nothing a cut may not have near it is near `rect` (the cut stretch, or the box of a pair's two): no via, port
    or part, no feed of the board's own ports, and no signal copper but the cut's own traces (`legs`: their
    polygons, run on past the cut) within keep_out (default 3 widths, 4 heights, 0.5 mm)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    runs, sig_k, vias, ports, parts = ctx
    d = keep_out or max(3 * w, 4 * h, 0.5e-3)
    near = rect.buffer(d)
    if any(rect.buffer(max(2 * w, 3 * h)).contains(p) for p in vias) or any(near.contains(p) for p in ports):
        return False
    if any(near.intersects(LineString(l)) for l in parts) or any(rect.intersects(a) for a in feeds):
        return False
    own = unary_union(legs)
    return not any((g.difference(own) if kk == k else g).intersects(near) for kk, g in sig_k.items())


def plan_cuts(model, data, dmax, fmin, cut_min=CUT_MIN * 1e-3, keep_out=None, log=print, feeds=(), ctx=None,
              skip=()):
    """The stretches of trace to cut out: [(Run of the cut, the run it is in)]. model: mom25d BoardModel; data: its
    board dict (nets per polygon); dmax (m); fmin (Hz): the wave ports on the cut faces need their feed. skip: the
    runs (their index in the context's) a pair's cut has taken."""
    from mom25d.line import Run
    ctx = ctx or _context(model, data, dmax)
    if ctx is None:
        log("mixed: the board names no ground net - no signal trace to cut")
        return []
    cuts = []
    for i, r in enumerate(ctx[0]):
        if i in skip:
            continue
        w = 2 * r.a
        need = _need(model, r.k, w, dmax, fmin)                               # the feed each cut face needs
        xc0, xc1 = r.x0 + need, r.x1 - need
        if xc1 - xc0 < cut_min:
            continue
        cut = Run(r.k, r.c0, r.t, r.a, xc0, xc1)
        h = model.stack.layers[r.k - 1].h                    # the dielectric under the trace
        d = keep_out or max(3 * w, 4 * h, 0.5e-3)            # no other signal copper this close
        own = Run(r.k, r.c0, r.t, r.a * (1 + 1e-6), xc0 - 2 * d, xc1 + 2 * d).polygon(pad=1e-7)
        if _clear(cut.polygon(), [own], r.k, w, h, keep_out, ctx, feeds):
            cuts.append((cut, r))
    return cuts


def plan_pair_cuts(model, data, dmax, fmin, cut_min=CUT_MIN * 1e-3, keep_out=None, feeds=(), ctx=None):
    """The stretches where two traces run side by side, to cut out together as a coupled line:
    ([(Run of one trace's cut, Run of the other's along the first's direction, the feed left at each face)], the
    runs they take: their index in the context's). Two straight runs on one interface, parallel, of one width (within 5 %), no further
    apart than a lone cut keeps other copper away (3 widths, 4 heights): where they overlap, less the feed a
    mixed-mode wave port needs at each end, and clear as a lone cut is of everything but the two."""
    from mom25d.line import Run
    ctx = ctx or _context(model, data, dmax)
    if ctx is None:
        return [], set()
    runs, out, used = ctx[0], [], set()
    for i, ra in enumerate(runs):
        for j in range(i + 1, len(runs)):
            rb = runs[j]
            if i in used or j in used or ra.k != rb.k or abs(ra.t @ rb.t) < 0.9999:
                continue
            if abs(ra.a - rb.a) > 0.05 * max(ra.a, rb.a):
                continue                                     # (not a symmetric pair: its modes are not d and c)
            off = float((rb.c0 + rb.t * rb.x0 - ra.c0) @ ra.n)          # the second's centre line across the first's
            w, h = ra.a + rb.a, model.stack.layers[ra.k - 1].h         # (the mean width)
            gap = abs(off) - ra.a - rb.a
            if gap <= 1e-9 or gap > (keep_out or max(3 * w, 4 * h, 0.5e-3)):
                continue                                     # further apart: each on its own, where it is clear
            ends = sorted(float((rb.c0 + rb.t * x - ra.c0) @ ra.t) for x in (rb.x0, rb.x1))
            lo, hi = max(ra.x0, ends[0]), min(ra.x1, ends[1])           # where both run, along the first
            need = _need(model, ra.k, 2 * max(ra.a, rb.a), dmax, fmin, span=PAIR_SPAN)
            xc0, xc1 = lo + need, hi - need
            if xc1 - xc0 < cut_min:
                continue
            ca = Run(ra.k, ra.c0, ra.t, ra.a, xc0, xc1)
            cb = Run(ra.k, ra.c0 + ra.n * off, ra.t, rb.a, xc0, xc1)
            box = Run(ra.k, ra.c0 + ra.n * off / 2, ra.t, abs(off) / 2 + max(ra.a, rb.a), xc0, xc1).polygon()
            d = keep_out or max(3 * w, 4 * h, 0.5e-3)
            own = [Run(c.k, c.c0, c.t, c.a * (1 + 1e-6), xc0 - 2 * d, xc1 + 2 * d).polygon(pad=1e-7) for c in (ca, cb)]
            if _clear(box, own, ra.k, w, h, keep_out, ctx, feeds):
                out.append((ca, cb, need))
                used |= {i, j}
    return out, used


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


def solve_mixed(m, freq, dmax=0.3, sigma=None, rough=None, finish=None, max_solves=60, cut_min=CUT_MIN, keep_out=None, log=print,
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
    metal, t_mm = _metal(m, sigma, rough, finish)
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
    ctx, feeds = _context(model, data, dm), _feed_areas(base)
    ko = keep_out and keep_out * 1e-3
    pcuts, taken = plan_pair_cuts(model, data, dm, fmin, cut_min=cut_min * 1e-3, keep_out=ko, feeds=feeds, ctx=ctx)
    cuts = plan_cuts(model, data, dm, fmin, cut_min=cut_min * 1e-3, keep_out=ko, log=log, feeds=feeds, ctx=ctx,
                     skip=taken)
    removed = 0.0
    cut_ports = []                              # every wave port on a cut face: a LinePort, or a pair's DiffPort
    # joins: the lines the cut ports are joined by, [dict(a, b: the two ports' names, port: the one whose gamma and
    # Zc are the line's, mode, length)]; stretches: the traces cut out, [dict(run, of: [(join, share)])], each
    # one's current the joins' line currents times their shares (a pair's: I+ = Ic / 2 + Id, I- = Ic / 2 - Id)
    joins, stretches = [], []

    def take_out(c):
        rect = c.polygon(pad=1e-7)
        model.shapes[c.k] = [model.shapes[c.k][0].difference(rect).buffer(0)]
        if model.signal is not None:
            model.signal = model.signal.difference(rect).buffer(0)

    def face_port(c, x, d, win, name):
        g0, s0, s1, pstep = win
        return LinePort(c.k, c.c0 + c.t * x + d * g0, d, 2 * c.a, -g0, s0=s0 - g0, s1=s1 - g0, step=pstep, zc=50.0,
                        name=name)
    for j, (c, r) in enumerate(cuts):
        take_out(c)
        removed += c.x1 - c.x0
        for end, (x, d) in enumerate(((c.x0, -c.t), (c.x1, c.t))):
            L = (x - r.x0) if end == 0 else (r.x1 - x)
            win, _ = model._wave_window(2 * c.a, L, dm)
            cut_ports.append(face_port(c, x, d, win, f"cut{j + 1}{'ab'[end]}"))
        stretches.append(dict(run=c, of=[(len(joins), 1.0)]))
        joins.append(dict(a=f"cut{j + 1}a", b=f"cut{j + 1}b", port=cut_ports[-2], mode=None, length=c.x1 - c.x0))
    for j, (ca, cb, need) in enumerate(pcuts):
        from mom25d.sim import DiffPort
        for c in (ca, cb):
            take_out(c)
            removed += c.x1 - c.x0
        faces = []
        for end, (x, d) in enumerate(((ca.x0, -ca.t), (ca.x1, ca.t))):
            win, _ = model._wave_window(2 * max(ca.a, cb.a), need, dm)
            name = f"cutp{j + 1}{'ab'[end]}"
            faces.append(DiffPort(face_port(ca, x, d, win, name + '+'), face_port(cb, x, d, win, name + '-'),
                                  100.0, 25.0, name=name))
            er = model.stack.layers[ca.k - 1].er                  # (the line its modes are characterised on)
            beta = 2 * np.pi * max(fmin, fmax / 3) * np.sqrt(0.5 * (er + 1)) / C0
            faces[-1].char_length = min(PAIR_CHAR_MAX, 2 * (PAIR_CHAR / beta + 3e-3))
        cut_ports += faces
        nd, nc = len(joins), len(joins) + 1
        stretches += [dict(run=ca, of=[(nd, 1.0), (nc, 0.5)]), dict(run=cb, of=[(nd, -1.0), (nc, 0.5)])]
        joins += [dict(a=f"cutp{j + 1}a{md}", b=f"cutp{j + 1}b{md}", port=faces[0], mode=md, length=ca.x1 - ca.x0)
                  for md in 'dc']
    if cuts or pcuts:
        log(f"mixed: {len(cuts) + len(pcuts)} stretch(es) of trace as lines, {removed * 1e3:.1f} mm in all: " +
            ", ".join([f"M{c.k} w={2 * c.a * 1e3:.3f} mm {1e3 * (c.x1 - c.x0):.1f} mm" for c, _ in cuts] +
                      [f"M{a.k} a pair w={2 * a.a * 1e3:.3f} mm, {1e3 * (abs(float((b.c0 - a.c0) @ a.n)) - a.a - b.a):.3f} mm "
                       f"apart, {1e3 * (a.x1 - a.x0):.1f} mm" for a, b, _ in pcuts]))
    else:
        log("mixed: no long straight run, lone or one of a pair, to cut - this is mom25d alone")
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
    for p in cut_ports:                         # each cut port at its own line's impedance; a pair's: its
        zref[p.name] = float(p.zdiff) / 2 if hasattr(p, 'pos') else float(p.zc)    # differential mode's (run.out_refs)
    f, Sn, info = solve_band(sim, fmin, fmax, points=len(freq), zref=zref, max_solves=max_solves, log=log,
                             progress=progress, on_step=on_step, freq=freq)
    for jn in joins:
        q, md = jn['port'], jn['mode']
        jn['gamma'] = q.gamma_fn if md is None else q.gd_fn if md == 'd' else q.gc_fn
        jn['zc'] = q.zc_fn if md is None else q.zd_fn if md == 'd' else q.zc_fn
        jn['z0'] = float(q.zc) if md is None else float(q.zdiff) * (1.0 if md == 'd' else 0.25)   # as solved: zref
    pairs = [(names.index(jn['a']), names.index(jn['b'])) for jn in joins]
    S = []
    for i, fr in enumerate(f):
        Ls = [line_S(jn['gamma'](fr), jn['zc'](fr), jn['length'], jn['z0']) for jn in joins]
        S.append(connect(Sn[i], pairs, Ls) if joins else Sn[i])
    S = np.array(S)
    keep = [n for n in names if not n.startswith('cut')]
    if len(f) != len(freq) or not np.allclose(f, freq):
        S = np.stack([np.stack([np.interp(freq, f, S[:, i, j].real) + 1j * np.interp(freq, f, S[:, i, j].imag)
                                for j in range(S.shape[2])], -1) for i in range(S.shape[1])], 1)
    order = [keep.index(p.name) for p in m.ports]
    S, out_names, out_zref = mixed_mode(S[:, order][:, :, order], m)
    unknowns = int(sim.qT.shape[1]) if getattr(sim, 'qT', None) is not None else int(sim.Ne + sim.Nv)
    info = dict(seconds=time.time() - t0, solves=int(info['solves']), unknowns=unknowns, cuts=len(cuts) + len(pcuts),
                pairs=len(pcuts), cut_mm=round(removed * 1e3, 3), notes=list(model.notes) + list(sim.log.notes),
                samples_ghz=[round(x / 1e9, 6) for x in sorted(info['solved'])])
    if hold is not None:                         # for the current dumps (Mixed.outputs)
        hold.update(sim=sim, model=model, joins=joins, stretches=stretches, pairs=pairs)
    return S, out_names, out_zref, info


def dump_fields(h):
    """current_dumps' currents(f) and extra(...) for a mixed solution h (solve_mixed's hold): mom25d's currents of
    each port driven alone (1 V gap source) combined into the joined network's state, as mom25d's own dumps: 1 V
    on the driven port's gap, the board's other ports' gaps shorted, the cut ports loaded by their lines. The cut
    stretches are drawn from their lines' waves (a pair's two traces from its two modes'): on the trace with
    mom25d's own profile across it (taken just outside the faces), and on the planes over and under it their return
    current, the line's image spread over its height."""
    sim, joins, stretches, pairs = h['sim'], h['joins'], h['stretches'], h['pairs']
    n = len(sim.ports)
    c = [i for ab in pairs for i in ab]
    x = [i for i in range(n) if i not in c]
    waves = []                                    # per call (frequency): {column: (a, b)}

    def currents(f):
        sim.sparams(f, keep=True)                                  # its currents and waves: sim.last
        I, A, B = sim.last['I'], sim.last['A'], sim.last['B']
        L = np.zeros((len(c), len(c)), complex)
        for j, jn in enumerate(joins):
            tau = np.exp(-jn['gamma'](f) * jn['length'])
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
    for st in stretches:
        from shapely.geometry import Point
        cut = st['run']
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

        def mode_current(q, s_):                 # a join's line current along +t at s_ from the cut's start, A
            jn, pa = joins[q], pairs[q][0]
            g, zc = jn['gamma'](f), float(jn['zc'](f))
            return (b[pa] * np.exp(-g * s_) - a[pa] * np.exp(g * s_)) / np.sqrt(zc)
        for m, st in enumerate(stretches):
            cut = st['run']
            on_trace, planes = cut.k == k, {q: (hh, sh) for q, hh, sh in refs[m]}
            if not on_trace and k not in planes:
                continue
            u, v = cut.local(P)
            line = lambda s_: sum(share * mode_current(q, s_) for q, share in st['of'])
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
                        diff_pairs='yes: as mom25d (a pair\'s straight stretches are cut as a coupled line)',
                        radiation='approximated: as mom25d, except from the stretches put back as lines',
                        dispersion='no: constant er and tan d (--diel wideband is mom25d\'s alone)')
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
        elif run.s.mom_stack == 'auto' and _stack(run.s, run.m) == 'whole':
            ap = apertures(run.m, run.s)
            why = aperture_text(ap) if ap else f"port(s) {', '.join(ports_below(run.m, run.s))} below the ground"
            out.append(f"{why}: mixed cuts its lines over an infinite ground (mom25d models the board with the whole "
                       f"stack; --mom-stack ground forces mixed anyway)")
        return out

    def solve(self, run, prep):
        s = run.s
        self._held = {}
        S, names, zref, info = solve_mixed(
            run.m, run.freq, dmax=s.mom_dmax, sigma=s.sigma, rough=getattr(s, 'rough', None), finish=run_finish(run),
            max_solves=s.mom_solves,
            cut_min=s.mixed_cut, log=run.log,
            port=s.mom_port, hold=self._held,
            quads={'off': False, 'on': True}.get(s.mom_quads, s.mom_quads),
            zone='auto' if s.mom_zone == 'on' else None, lines=s.mom_lines,
            signal_vias=s.mom_signal_vias, stitching_vias=s.mom_stitching_vias,
            progress=lambda k, n, f: (run.status('mixed', dict(fraction=k / n, solves=k, f_ghz=f / 1e9)), stop.check()))
        info.update(kpi=f"{info['unknowns']:,} unknowns, {info['cuts']} line(s), {info['seconds']:.0f} s",
                    about=f"mixed: mom25d with {info['cuts']} long straight stretch(es) of trace ({info['cut_mm']:g} mm"
                          + (f", {info['pairs']} of them two traces side by side as a coupled line" if info.get('pairs')
                             else '') +
                          f") cut out and put back as transmission lines (gamma and Zc from each line's cross-section): "
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
