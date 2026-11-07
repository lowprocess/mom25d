"""
mom25d as a kicad2ems solver plugin: the board of the run (Board.to_dict(), the content of <name>_board.json)
solved by mom25d, which reads that format itself.

    S, names, zref, info = solve(board, freq, dmax=0.3)

MoM assumes laterally infinite dielectrics over an infinite ground plane: the board outline does not exist
for it, and edge walls are rows of vias.
"""
import json
import os
import numpy as np

from kicad2ems.api.solver import Option, floatlist
from kicad2ems.api.solver import Solver, Result, Deviation
from kicad2ems.api.solver import stop

G = 'mom25d'
# --solvers auto (docs/architecture.md#choosing-solvers): seconds per EM solve on 24 cores, fitted on the examples
# run alone: T0 per pair of metal interfaces (the layered-medium tables of each frequency), plus the dense fill
# (DENSE_N2 N^2, up to 3.5k unknowns) or the direct solver's LU (LU_N3 N^3), plus VIA2 per pair of vias (the
# via-via kernels, from a table since: sma_launch's 181 vias 2.5 s a solve, 26 s before); a sweep takes about
# SOLVES solves (11..22 on the examples). demo_transition: 12.1 s a solve measured, 12.4 s estimated.
T0, DENSE_N2, LU_N3, VIA2, SOLVES, REF_CORES = 1.0, 1.6e-7, 3.7e-12, 7.6e-5, 16, 24
# with a GPU (RX 7800 XT, sma_launch, 16.1k unknowns): the fill, the complex64 LU and the refinement 6.3 s a solve
# (hybrid), the tiled path 10.5 s (its fill block by block, the symmetric near field)
LU_GPU_N3, LU_TILED_N3 = 1.5e-12, 2.5e-12
ANT_FREQS = 11                  # antenna mode: frequencies of the far field over the band (+ the pattern one)


def _far_field_grid(run):
    """Antenna mode: the far field's frequencies over the band (Hz; the pattern's is added after the sweep)."""
    return np.linspace(run.fmin, run.fmax, ANT_FREQS)


def _far_field_seeds(run):
    """The frequencies the sweep solves among its own samples and keeps the currents of, so the far field needs no
    solves of its own (the pattern frequency's aside, when it is the resonance): ANT_FREQS solves fewer on an
    antenna, antenna_vivaldi's 37 -> 26."""
    return list(_far_field_grid(run)) if run.antenna else []


def mom25d_status():
    """(usable, one-line description) of mom25d for --check and the solver checks before a run: where it
    comes from, its version and whether its compiled core is built (the NumPy fallback is ~5x slower)."""
    try:
        import mom25d
        import triangle                                             # noqa: F401  (mom25d's default mesher)
    except Exception as e:                                          # a missing or broken dependency
        return False, f"{type(e).__name__}: {e}"
    try:
        from mom25d import _core                                    # noqa: F401
        core = 'compiled core'
    except ImportError:
        core = "no compiled core (NumPy, ~5x slower: python -m mom25d.core.build)"
    return True, f"{mom25d.__version__} ({os.path.dirname(mom25d.__file__)}), {core}"


def run_finish(run):
    """The run's surface finish (kicad2ems.api 2.4: resolve_finish), or None with an older kicad2ems."""
    fin = getattr(run, 'finish', None)
    if fin is not None or not getattr(run.s, 'sigma', None):
        return fin
    try:
        from kicad2ems.api.solver import resolve_finish
    except ImportError:
        return None
    return resolve_finish(run.s, run.m)


def _metal(m, sigma, rough=None, finish=None):
    """mom25d's conductor loss for --sigma: the conductivity with the copper thickness of the ports' layer
    (openEMS models the same copper as conducting sheets of that conductivity), --rough (um RMS: Hammerstad at
    each frequency) and the surface finish (kicad2ems.finish.Finish: on the outer copper, weighted by its finished
    faces' share of the strip's loss at each frequency). None without sigma."""
    if not sigma:
        return None, None
    from mom25d.sim import Metal
    lays = [p.layer for p in m.ports if p.layer in m.cu_t] or [l for l in m.cu_order if l in m.copper]
    t_mm = m.cu_t.get(lays[0], 0.035) if lays else 0.035
    fin, on = (), {}
    if finish:
        fin = [(s, t) for _, s, t in finish.layers]
        on = {lay: finish.share(lay) for lay in finish.faces if finish.share(lay)}
    return Metal(float(sigma), t_mm * 1e-3, float(rough or 0.0) * 1e-6, finish=fin, on=on), t_mm


def _via_sides(v, words):
    """--mom-signal-vias / --mom-stitching-vias: one of `words`, or a number of sides of at least 3."""
    v = str(v).strip().lower()
    if v in words or (v.isdigit() and int(v) >= 3):
        return v
    raise ValueError(f"{v!r}: {', '.join(words)} or a number of sides (3 or more)")


def factorisations(runlog):
    """(on the GPU, on the CPU): the matrix factorisations of a sweep, from mom25d's stage timings. A solve whose
    complex64 LU failed to precondition counts on both (it is factored again in double on the CPU)."""
    names = [(stage, detail) for stage, _, detail in getattr(runlog, "records", ())]
    gpu = sum(st in ('LU factor (GPU)', 'LDLT factor (GPU, tiled)') for st, _ in names)
    cpu = sum(st in ('LU factor', 'BLR assemble + factor') or (st == 'LU solve' and detail.startswith('N='))
              for st, detail in names)
    return gpu, cpu


def reused(runlog):
    """The solves of a sweep that reused a factorisation kept from another frequency instead of their own."""
    tried = sum(stage == 'refinement (GMRES, reused factorisation)' for stage, _, _ in getattr(runlog, "records", ()))
    return tried - sum('did not precondition' in n for n in getattr(runlog, "notes", ()))


def solve(m, freq, dmax=0.3, port='auto', max_solves=60, sigma=None, rough=None, finish=None, log=print, progress=None,
          hold=None, tiled='auto', on_step=None, core='auto', quads=False, zone=None,
          signal_vias='round', stitching_vias='auto', lines=None, keep_at=(), stack='ground', quad_tol=35.0,
          coarsen=None, planes='pec', stitching_pads='drop', wideband=None):
    """S-parameters of the board m on `freq` (Hz) from mom25d, as kicad2ems reports them: (S (nf, n, n) in
    m.user_ports order with differential pairs as (d, c) modes, names, reference impedances, info dict).
    dmax in mm (planes 3x, refined near the copper of other layers). sigma (S/m): conductor loss, as --sigma
    gives openEMS (default: lossless copper, like openEMS's PEC sheets). progress(solves, at most, f). hold: a
    dict that receives mom25d's result as 'res' (its model and simulation, for the current dumps). keep_at:
    frequencies (Hz) the sweep solves and keeps the currents of (sim.kept: the current dumps'). stack: 'ground'
    (an infinite ground plane) or 'whole' (every copper layer meshed, mom25d.board ground='none'). planes,
    stitching_pads, wideband: mom25d.board's (wideband: the frequency, Hz, of a Djordjevic-Sarkar stack's er and
    tand, or None)."""
    from mom25d.api import solve_board
    from kicad2ems.api.solver import mixed_mode
    pairs, kw = _board_kw(m, dmax, port, quads=quads, quad_tol=quad_tol, coarsen=coarsen, zone=zone,
                          signal_vias=signal_vias, stitching_vias=stitching_vias, lines=lines, stack=stack,
                          planes=planes, stitching_pads=stitching_pads, wideband=wideband)
    if wideband:
        from mom25d.multilayer import djordjevic_sarkar
        for l in m.layers:
            e = djordjevic_sarkar(l['er'], l['tand'], wideband, np.array([freq[0], freq[-1]]))
            log(f"  mom25d: wideband dielectric {l['name']}: er {e[0].real:.3f} -> {e[1].real:.3f}, tand "
                f"{-e[0].imag / e[0].real:.4f} -> {-e[1].imag / e[1].real:.4f} ({freq[0] / 1e9:g} -> "
                f"{freq[-1] / 1e9:g} GHz; Djordjevic-Sarkar, matched at {wideband / 1e9:g} GHz)")
    metal, t_mm = _metal(m, sigma, rough, finish)
    if metal is not None:
        log(f"  mom25d: conductor loss sigma {sigma:g} S/m, {t_mm * 1e3:g} um copper" +
            (f", {rough:g} um RMS rough (Hammerstad)" if rough else "") +
            (f", {finish.name} on {', '.join(sorted(metal.on))}" if metal.on else ""))
    kw.update(metal=metal, max_solves=max_solves, log=lambda t: log(f"  mom25d:{t}") if 'AFS' in t else None,
              progress=progress, tiled=tiled, on_step=on_step, core=core, keep_at=keep_at)
    try:
        res = solve_board(m.to_dict(), freq, pairs=pairs, **kw)
    except ValueError as e:
        # a pair whose straight coupled feed is too short for a differential wave port (a pad, a connector pin):
        # with port auto, its legs as single-ended ports (wave or lumped each), the modes taken below
        if not (pairs and port == 'auto' and 'differential wave port' in str(e)):
            raise
        log(f"  mom25d: {e} - the pairs' legs as single-ended ports, the modes from them")
        pairs = False
        res = solve_board(m.to_dict(), freq, pairs=False, **kw)
    if hold is not None:
        hold['res'] = res
    for line in res.model.describe().splitlines()[1:] + list(res.sim.log.notes):
        log(f"  mom25d: {line}")
    log(f"mom25d: {res.unknowns:,} unknowns, solver {res.solver}, ports " +
        ", ".join(f"{p['name']} ({p['kind']})" for p in res.ports))
    gpu_n, cpu_n = factorisations(res.sim.log)
    n_re = reused(res.sim.log)
    log(f"mom25d: factorisations: {gpu_n} on the GPU, {cpu_n} on the CPU"         # benchmarks: where the solves ran
        + (f"; {n_re} solves reused one" if n_re else ''))
    S = res.S
    if pairs:                                                       # mom25d's modal ports -> kicad2ems order
        _, names, zref = mixed_mode(np.zeros((1, len(m.ports), len(m.ports))), m)
        idx = [res.names.index(n) for n in names]
        S = S[:, idx][:, :, idx]
    else:
        order = [res.names.index(p.name) for p in m.ports]
        S, names, zref = mixed_mode(S[:, order][:, :, order], m)
    info = dict(seconds=res.seconds, solves=res.solves, unknowns=res.unknowns, triangles=res.triangles,
                solver=res.solver, dmax_mm=dmax, ports=res.ports, notes=res.notes,
                samples_ghz=[round(f / 1e9, 6) for f in sorted(res.solved)], afs=res.afs, afs_tol=res.afs_tol)
    log(f"  mom25d: solved at {', '.join(f'{f:g}' for f in info['samples_ghz'])} GHz")
    return S, names, zref, info


def _board_kw(m, dmax, port, quads=False, quad_tol=35.0, coarsen=None, zone=None, signal_vias='round',
              stitching_vias='auto', lines=None, stack='ground', planes='pec', stitching_pads='drop',
              wideband=None):
    """(pairs, mom25d.api's board and mesh options) of solve()'s settings: what the run meshes, and the preview."""
    kind = 'via' if port == 'lumped' else port
    pairs = any(len(legs) == 2 for _, legs in m.user_ports) and kind != 'via'
    return pairs, dict(dmax=dmax * 1e-3, port=kind, quads=quads, quad_tol=quad_tol, coarsen=coarsen, zone=zone,
                       planes=planes, stitching_pads=stitching_pads, wideband=wideband, signal_vias=signal_vias,
                       stitching_vias=stitching_vias, lines=lines,
                       **(dict(ground='none') if stack == 'whole' else {}))


def preview_mesh(m, fmin, fmax, dmax=0.3, port='auto', **mesh):
    """(BoardModel, Simulation) of the mesh solve() would solve on, without solving (mesh_only: no quad cells
    paired yet). mesh: solve()'s mesh options (quads, coarsen, zone, vias, lines, stack, planes, ...)."""
    from mom25d.api import build_board
    pairs, kw = _board_kw(m, dmax, port, **mesh)
    try:
        return build_board(m.to_dict(), fmin, fmax, pairs=pairs, log=lambda *a: None, mesh_only=True,
                           characterize=False, **kw)
    except ValueError as e:                     # as solve(): a pair too short for a differential wave port
        if not (pairs and port == 'auto' and 'differential wave port' in str(e)):
            raise
        return build_board(m.to_dict(), fmin, fmax, pairs=False, log=lambda *a: None, mesh_only=True,
                           characterize=False, **kw)


def merged_diagonals(sim):
    """The node pairs (k, 2) of the triangle edges a run merged away into quad cells (sim.quad_info['elim']),
    for the viewers to leave out; empty without quads, or where the solver ignored them (sim.qT None)."""
    qi = getattr(sim, 'quad_info', None)
    if getattr(sim, 'qT', None) is None or not qi or 'quads' not in qi:
        return np.zeros((0, 2), int)
    r, elim = sim.rwg, np.asarray(qi['elim'], int)
    A, B = r.tris[r.tp[elim]], r.tris[r.tm[elim]]
    shared = (A[:, :, None] == B[:, None, :]).any(2)                # the edge's two nodes are in both triangles
    return np.sort(A[shared].reshape(-1, 2), 1)


def save_mesh(run, sim, model, solver='mom25d'):
    """mom25d's mesh for the 3D view and the plan (meshes.py): each meshed layer's triangles at its copper's
    height, and the diagonals merged into quad cells (`hidden`, a run's). solver: the file's name, 'mom25d' (the
    preview) or 'mom25d-run' (the mesh a run solved on)."""
    from kicad2ems.api.solver import save_mesh
    lay = {k: l for l, k in model.iface.items()}
    z = np.array([run.m.cu_z.get(lay.get(int(k)), np.nan) for k in sim.tri_layer])
    tris = np.asarray(sim.rwg.tris)
    keep = np.isfinite(z)
    nz = np.zeros(len(sim.rwg.nodes))                               # each layer is meshed on its own: no shared nodes
    nz[tris[keep].ravel()] = np.repeat(z[keep], 3)
    nodes = np.column_stack([sim.rwg.nodes[:, :2] * 1e3, nz])
    hidden = merged_diagonals(sim)
    hidden = hidden[np.isfinite(nz[hidden]).all(1) & (nz[hidden[:, 0]] == nz[hidden[:, 1]])] if len(hidden) else hidden
    what = 'mom25d mesh (3D view)' if solver == 'mom25d' else 'mom25d solved mesh (3D view)'
    run.wrote(what, save_mesh(run, solver, 'tris', nodes=nodes, tris=tris[keep], hidden=hidden))


HOW = {'dense': 'dense, NumPy', 'double': 'direct LU in double on the CPU',
       'hybrid': 'complex64 LU on the GPU, double refinement on the host', 'tiled': 'tiled L L^T on the GPU',
       'gpu-lu': 'complex64 LU on the GPU, matrix-free refinement', 'cpu-mixed': 'complex64 LU on the CPU',
       'blr': 'block low-rank, no dense matrix', 'aca': 'H-matrix and GMRES, no dense matrix'}


def memory_plan_for(n, tiled='auto'):
    """mom25d.sim.memory_plan of n unknowns on this machine: the memory available now, the GPU mom25d would use."""
    from mom25d.sim import memory_plan, gpu_info
    from mom25d.fast import _core
    from mom25d.gpu import get_backend
    g = get_backend('auto') if _core is not None else None
    return memory_plan(n, _core is not None, None, gpu_info(g, n) if g is not None else None, tiled=tiled)


def _model_kw(s):
    """mom25d.board.BoardModel's options from kicad2ems's settings: --mom-planes, --mom-stitching-pads, and --diel
    (the FDTD solvers' option, which kicad2ems's examples set: wideband is the Djordjevic-Sarkar model matched at
    --diel-fref; fixed, mom25d's constant er and tand)."""
    wide = getattr(s, 'diel', 'fixed') == 'wideband'
    return dict(planes=getattr(s, 'mom_planes', 'pec'), stitching_pads=getattr(s, 'mom_stitching_pads', 'drop'),
                wideband=getattr(s, 'diel_fref', 1.0) * 1e9 if wide else None)


def _mesh_kw(s, m):
    """solve()'s mesh options from kicad2ems's settings (the run's and the preview's) for the board m: --mom-quads,
    --mom-quad-tol, --mom-coarsen, --mom-zone, the vias, --mom-lines, --mom-stack (resolved: _stack) and the
    BoardModel's."""
    return dict(quads={'off': False, 'on': True}.get(s.mom_quads, s.mom_quads), quad_tol=s.mom_quad_tol,
                coarsen=s.mom_coarsen or None, zone='auto' if s.mom_zone == 'on' else None,
                signal_vias=s.mom_signal_vias, stitching_vias=s.mom_stitching_vias, lines=s.mom_lines,
                stack=_stack(s, m), **_model_kw(s))


_held_apertures = [None, None]                  # (board, its apertures): the last board's, asked for several times


def apertures(m, s):
    """mom25d.board.ground_apertures of the ground mom25d would put its infinite plane at (the board's ground
    layer, moved up by --mom-planes pec as mom25d.board.pec_planes does): [(polygon, mm2, layer)], [] for a complete
    ground or none."""
    if _held_apertures[0] is m and _held_apertures[1][0] == _model_kw(s)['planes']:
        return _held_apertures[1][1]
    from mom25d.board import ground_apertures, pec_planes
    g, out = _solid_ground(m), []
    if g is not None:
        data = m.to_dict()
        if _model_kw(s)['planes'] == 'pec':
            g, _ = pec_planes(data, g)
        out = [(poly, a, g) for poly, a in ground_apertures(data, g)]
    _held_apertures[:] = [m, (_model_kw(s)['planes'], out)]
    return out


def ports_below(m, s):
    """The ports on copper below the ground mom25d would put its infinite plane at (which the ground stack cannot
    have: rf_msvia's P2 on B.Cu under the In2.Cu ground)."""
    g = _solid_ground(m)
    if g is None:
        return []
    if _model_kw(s)['planes'] == 'pec':
        from mom25d.board import pec_planes
        g, _ = pec_planes(m.to_dict(), g)
    return [p.name for p in m.ports if m.cu_z.get(p.layer, np.inf) < m.cu_z[g] - 1e-9]


def _stack(s, m):
    """--mom-stack for the board m: auto is whole where the ground has apertures under the signal copper (a slot,
    a defected or partial ground: apertures()), which the infinite plane would fill in, or a port is below the
    ground (ports_below), else ground."""
    if s.mom_stack != 'auto':
        return s.mom_stack
    return 'whole' if apertures(m, s) or ports_below(m, s) else 'ground'


def _apertures_said(m, s):
    """One line on the ground's apertures for the log, or None: why auto chose the whole stack, or a warning
    that --mom-stack ground fills them in."""
    ap = apertures(m, s)
    if s.mom_stack == 'auto' and not ap and ports_below(m, s):
        return (f"mom25d: stack auto: whole, every copper layer meshed: port(s) {', '.join(ports_below(m, s))} below "
                f"the ground")
    if not ap or s.mom_stack == 'whole':
        return None
    where = aperture_text(ap)
    if s.mom_stack == 'auto':
        return f"mom25d: stack auto: whole, every copper layer meshed: {where}; --mom-stack ground fills them in"
    return (f"WARNING: mom25d: {where}: --mom-stack ground puts a solid infinite plane there, so what they do is "
            f"not modelled (rf_dgs |dS| 0.74 against FDTD); --mom-stack whole or auto models them")


def aperture_text(ap):
    """apertures()' result in a few words."""
    return f"{len(ap)} aperture{'s' if len(ap) > 1 else ''} in the ground {ap[0][2]} under the signal copper " \
            f"({sum(a for _, a, _ in ap):.1f} mm2; the largest {ap[0][1]:.1f} mm2 at " \
            f"({ap[0][0].centroid.x:.1f}, {-ap[0][0].centroid.y:.1f}) mm)"


def _solid_ground(m):
    """The board's ground layer, else the lowest copper layer covering half the outline (mom25d's choice), or
    None."""
    if m.ground:
        return m.ground
    half = 0.5 * m.outline.area
    return next((l for l in m.cu_order if sum(p.area for p in m.copper.get(l, [])) >= half), None)


class MoM25D(Solver):
    api = (2, 3)                    # kicad2ems.api (2.3: the roughness row)
    name, aliases, label = 'mom25d', ('mom',), 'mom25d (MoM)'
    consumes, previewed, rank, order = 'board', False, 1, 20     # order: after openEMS in kicad2ems's lists
    capabilities = dict(ports='yes: lumped, or de-embedded wave ports on a long enough feed', gap_ports='no',
                        diff_pairs='yes', parts='yes: one series element across the gap',
                        vias='yes: over the layers with metal at the via',
                        outline='ignored: laterally infinite dielectric', planes='yes: meshed (slow on plane-heavy boards)',
                        edge_walls='approximated: via rows every 0.5 mm', thickness='ignored: sheets',
                        conductor_loss='yes', roughness='yes: Hammerstad at each frequency',
                        finish='approximated: on the outer copper, weighted by the finished faces\' share of the strip\'s loss (the port trace\'s), at each frequency',
                        dispersion='yes with --diel wideband (Djordjevic-Sarkar at each frequency), else constant er and tan d', radiation='yes', cavity='no',
                        antenna='approximated: over an infinite ground plane (patches, arrays; not an antenna that '
                                'uses the board edge or a finite ground)')
    options = [
        Option('mom_dmax', 0.3, "largest MoM cell along the copper, mm (default {default}; planes 3x, refined near "
               "traces)", float, name='dmax', gui='Largest cell along the copper (mm)', group=G, cli=dict(metavar='MM')),
        Option('mom_port', 'auto', "auto (default): de-embedded wave ports where a straight feed allows, else lumped",
               choices=('auto', 'wave', 'lumped'), name='port', gui='Ports', group=G),
        Option('mom_stack', 'auto', "the stack mom25d models: auto (default: whole where the ground layer has "
               "apertures under the signal copper - a slot, a defected or partial ground - else ground), ground (an "
               "infinite plane at the ground layer, its copper not meshed, the copper below it left out) or whole (every copper layer meshed, the "
               "planes too, air above and below: ports and copper on any layer, radiation below the board and a "
               "finite ground in antenna mode; lumped ports; more unknowns)", choices=('auto', 'ground', 'whole'),
               name='stack', gui='MoM stack', group=G),
        Option('mom_quads', 'off', "experimental: merge near-rectangular triangle pairs into quad cells, a "
               "constant-divergence subspace with fewer unknowns and a cheaper LU (docs/mom.md#quads): off (default), "
               "on (wherever they pair up, the copper's edge strip kept as triangles), centre (also only on signal "
               "copper: no pours or pads) or all (the edge strip too: fewer unknowns, S up to 0.03 off)",
               choices=('off', 'on', 'centre', 'all'), name='quads', gui='MoM quad cells (experimental)', group=G),
        Option('mom_quad_tol', 35.0, "the most a quad cell's corner may differ from 90 degrees (default {default}; "
               "60 merges nearly every triangle, with S up to 0.03 off on a resonant filter)", float,
               name='quad_tol', gui='MoM quad tolerance (degrees)', group=G, cli=dict(metavar='DEG')),
        Option('mom_coarsen', 0.0, "experimental: cells grow away from corners, vias and ports, up to this times "
               "--mom-dmax along straight runs and inside wide copper (the edge rings kept; docs/mom.md#coarsen): "
               "0 (default: off), e.g. 2", float, name='coarsen', gui='MoM coarsening (experimental)', group=G,
               cli=dict(metavar='X')),
        Option('mom_zone', 'off', "experimental: mesh detail only within 0.5 mm of the signal copper (every net but "
               "the ground nets): no edge rings, coarse cells and square stitching-via pads elsewhere; about a third "
               "of the unknowns at the same accuracy (docs/mom.md#zone): off (default), on", choices=('off', 'on'),
               name='zone', gui='MoM signal zone (experimental)', group=G),
        Option('mom_planes', 'pec', "pec (default): solid planes (one piece over the board, no holes, ground nets "
               "only) as perfect conductors, not meshed: the highest one under every other copper layer the ground, "
               "the lowest one over them a PEC cover (rf_stripline 15,563 unknowns to 8,671); or mesh: every plane "
               "meshed (docs/mom.md#planes)", choices=('pec', 'mesh'), name='planes', gui='MoM solid planes', group=G),
        Option('mom_stitching_pads', 'drop', "drop (default): a ground via's pad with nothing else on its layer (a "
               "stitching via through a signal layer; not a part's or a port's pad) left out, the via kept as its "
               "barrel (rf_stripline with the planes PEC: 8,671 unknowns to 1,337); or keep (docs/mom.md#planes)",
               choices=('drop', 'keep'), name='stitching_pads', gui='MoM lone stitching-via pads', group=G),
        Option('mom_lines', 'off', "experimental: line basis functions on straight runs of trace - per column along "
               "the run a few functions with the edge-peaked current profile across it, in place of the triangles' "
               "unknowns inside the run (docs/mom.md#lines): off (default), on (even + odd profile: coupled lines), "
               "even (isolated lines only), full (even, flat and odd)", choices=('off', 'on', 'even', 'full'),
               name='lines', gui='MoM line functions (experimental)', group=G),
        Option('mom_signal_vias', 'round', "how vias on signal nets are meshed: round (default: pads and antipads "
               "as drawn, an octagon of the via radius), square, or a number of sides N >= 3 (pads, antipads and "
               "the barrel as N-gons; fewer unknowns, docs/mom.md#vias)", str, name='signal_vias',
               conv=lambda v: _via_sides(v, ('round', 'square')),
               gui='MoM signal vias', group=G, cli=dict(metavar='round|square|N')),
        Option('mom_stitching_vias', 'auto', "how vias on ground nets (stitching, fences, edge walls) are meshed: "
               "auto (default: square outside --mom-zone, else round), round, square or N sides", str,
               conv=lambda v: _via_sides(v, ('auto', 'round', 'square')),
               name='stitching_vias', gui='MoM stitching vias', group=G, cli=dict(metavar='auto|round|square|N')),
        Option('mom_tiled', 'auto', "mixed precision as complex64 L L^T tiles on the GPU, no dense matrix anywhere "
               "(half the GPU memory of an LU, streamed from the host past it): auto (default) where the dense "
               "GPU solver does not fit (it is faster where it fits: 9.9 s against 14.3 s a solve on the SMA "
               "launch), on, off", choices=('auto', 'on', 'off'), name='tiled', gui='MoM tiled GPU solver', group=G),
        Option('mom_core', 'auto', "mom25d's kernels: auto (default: the compiled core where it is built, NumPy's "
               "dense solver for boards up to 3,500 unknowns), compiled (the core for every board: 4-8x faster on "
               "small ones, |S| within about 0.005 of the dense solver's; stops if the core is not built) or python "
               "(NumPy only: slower, for checking the core)", choices=('auto', 'compiled', 'python'), name='core',
               gui='Core (compiled / python)', group=G),
        Option('mom_solves', 60, "most EM solves of the adaptive sweep (default {default})", int, name='solves', gui='Most EM solves of the sweep',
               group=G, cli=dict(metavar='N')),
        Option('mom_dump', [], "surface-current dumps at these frequencies (GHz): |J| maps of each signal layer, and "
               "with --dump-animate one period of Re(J e^jphi), the standing waves; the ports driven: --dump-port "
               "(default: the first)", list, conv=floatlist, name='dump', gui='Current dumps (GHz, comma-separated)',
               group=G, cli=dict(metavar='GHz,...')),
        Option('mom_near', [], "with --mom-dump: also the E and H fields on planes this far (mm) above the top of the "
               "board, from the currents (e.g. 0.5,2); at least about --mom-dmax above the copper to resolve it", list,
               conv=floatlist, name='near', gui='Near-field planes (mm above the top)', group=G, cli=dict(metavar='MM,...')),
        Option('mom_near_volume', [], "with --mom-dump: also E and H in a volume above the board, from H0 to H1 mm above "
               "its top in N planes (H0,H1[,N]; N default 12): the GUI's 3D view", list, conv=floatlist,
               name='near_volume', gui='Near-field volume (mm above the top: from,to[,planes])', group=G,
               cli=dict(metavar='H0,H1[,N]')),
    ]

    def validate(self, s):
        if getattr(s, 'mom_core', 'auto') == 'compiled':
            from mom25d.fast import _built
            if _built is None:
                return ("mom_core = compiled, but mom25d's compiled core is not built (python -m mom25d.core.build, "
                        "or setup.sh): auto or python run on NumPy")
        return None

    def resources(self):
        """The GPU mom25d's LU would use (MOM25D_GPU), or why there is none."""
        from mom25d.gpu import get_backend
        b = get_backend('auto')
        if b is None:
            return dict(gpus=[], why='; '.join(get_backend.why) or 'disabled (MOM25D_GPU=off)')
        total = b.mem_total() if hasattr(b, 'mem_total') else None
        return dict(gpus=[dict(device=b.device, backend=b.name, api=b.api,
                               free=b.mem_free() / 1e9, total=total / 1e9 if total else None)], why='')

    def check(self, s=None):
        return mom25d_status()

    def assess(self, run):
        m, out = run.m, []
        gaps = [p.name for p in m.ports if p.kind == 'gap']
        if gaps:
            out.append(f"gap port(s) {', '.join(gaps)}: no planar equivalent")
        if _stack(run.s, m) != 'whole' and _solid_ground(m) is None:
            out.append("no solid ground plane (no copper layer covers half the board): set ground = <layer>, or "
                       "mom_stack = whole")
        return out

    # ---- what it changes, and what it holds -------------------------------------------------------------
    WALL_PITCH, WALL_R = 0.5, 0.05                                  # mm: edge walls as via rows (mom25d.board)

    def deviations(self, run, prep):
        """What mom25d's planar model leaves out or changes, by the rules it states (mom25d.board): the ground is
        an infinite plane, copper below it and the outline are left out, a via spans only the layers with metal
        at it, edge walls are via rows, a part is one element across its gap bridged by a strip."""
        from shapely.geometry import Point, box
        from shapely.ops import unary_union
        m, out = run.m, []
        whole = _stack(run.s, m) == 'whole'                          # every copper layer meshed: no ground plane
        g = None if whole else _solid_ground(m)
        if g is None and not whole:
            return out
        kw, cover, pads = _model_kw(run.s), None, {}
        if not whole and kw['planes'] == 'pec':                     # mom25d.board.pec_planes: the ground may move up
            from mom25d.board import pec_planes
            g, cover = pec_planes(m.to_dict(), g)
        zg = min(m.cu_z.values()) if whole else m.cu_z[g]
        zc = m.cu_z[cover] if cover else np.inf
        out.append(Deviation('outline', 'outline', 'ignored: laterally infinite dielectric'))
        if not whole:
            out.append(Deviation(f"copper:{g}", 'planes', 'the infinite ground plane (not meshed)', dropped=True))
            out += [Deviation(f"aperture:{g}:{i}", 'planes', 'an aperture in the ground under the signal copper: '
                              'filled in by the infinite plane', layer=g, area=a)
                    for i, (_, a, _) in enumerate(apertures(m, run.s))]
            out += [Deviation(f"copper:{l}", 'planes', 'below the ground: left out', dropped=True)
                    for l in m.copper if m.cu_z[l] < zg - 1e-9]
        if cover:
            out.append(Deviation(f"copper:{cover}", 'planes', 'a PEC cover (not meshed)', dropped=True))
            out += [Deviation(f"copper:{l}", 'planes', 'above the cover: left out', dropped=True)
                    for l in m.copper if m.cu_z[l] > zc + 1e-9]
        metal = {l: m.copper[l] for l in m.copper if whole or zg + 1e-9 < m.cu_z[l] < zc - 1e-9}
        if kw['stitching_pads'] == 'drop':                          # mom25d.board.lone_pads: left out, the vias kept
            from shapely.geometry import shape
            from mom25d.board import lone_pads
            data = m.to_dict()
            pads = lone_pads(data, [l for l in metal if data.get('copper', {}).get(l)])
            for l, ix in pads.items():
                out += [Deviation(f"pad:{l}:{i}", 'vias', 'a lone stitching-via pad: left out (the via kept)', layer=l,
                                  area=-shape(data['copper'][l][i]['polygon']).area) for i in ix]
                metal[l] = [p for i, p in enumerate(data['copper'][l]) if i not in ix]
                metal[l] = [shape(p['polygon']) for p in metal[l]]
        metal = {l: unary_union(ps).buffer(1e-6) for l, ps in metal.items()}

        def span(x, y, z0, z1):
            ends = [z for z in (zg, zc) if not whole and np.isfinite(z) and z0 - 1e-9 <= z <= z1 + 1e-9]
            zs = ends + [m.cu_z[l] for l, cu in metal.items()
                         if z0 - 1e-9 <= m.cu_z[l] <= z1 + 1e-9 and cu.contains(Point(x, y))]
            return (min(zs), max(zs)) if len(set(zs)) >= 2 else None
        for i, (x, y, r, z0, z1) in enumerate(m.vias):
            s = span(x, y, z0, z1)
            if s is None:
                out.append(Deviation(f"via:{i}", 'vias', 'left out: metal on fewer than two layers at it', dropped=True))
            elif abs(s[0] - z0) > 1e-9 or abs(s[1] - z1) > 1e-9:
                out.append(Deviation(f"via:{i}", 'vias', 'spans only the layers with metal at it', to=[x, y, r, *s]))
        from mom25d.board import wall_points
        pts = wall_points(m.walls, self.WALL_PITCH, [(v[0], v[1]) for v in m.vias])
        for i, (axis, c, t0, t1, z0, z1) in enumerate(m.walls):
            add = [[x, y, self.WALL_R, *s] for j, x, y in pts if j == i for s in [span(x, y, z0, z1)] if s]
            out.append(Deviation(f"wall:{i}", 'edge_walls', f"a row of {len(add)} vias every {self.WALL_PITCH:g} mm",
                                 dropped=True, added=add))
        for p in m.parts:
            if p.layer not in metal:
                out.append(Deviation(f"part:{p.ref}", 'parts', f"on {p.layer}, not above the ground: left out", dropped=True))
                continue
            before = unary_union(m.copper[p.layer])
            strip = unary_union([before, box(*p.s[:2], *p.e[:2])]).area - before.area
            short = [v for v in (p.R, p.L, p.C) if v is not None] == [0.0] and p.C is None
            out.append(Deviation(f"part:{p.ref}", 'parts', 'a strip alone (0 ohm)' if short else
                                 'one series element across the gap, bridged by a strip', to=0 if short else 1,
                                 layer=p.layer, area=strip))
        return out

    def readback(self, run, prep):
        """mom25d's planar model of the board (mom25d.board.BoardModel), in board terms."""
        from mom25d.board import BoardModel
        bm = BoardModel(run.m.to_dict(), log=lambda *a: None, **_model_kw(run.s),
                        **(dict(ground='none') if _stack(run.s, run.m) == 'whole' else {}))
        zk, lay = bm._zk, {k: l for l, k in bm.iface.items()}
        ports = [leg['name'] for sp in bm.port_specs for leg in (sp['legs'].values() if 'legs' in sp else [sp])]
        return dict(vias=[[v.x * 1e3, v.y * 1e3, v.a * 1e3, float(zk[v.k_bot]), float(zk[v.k_top])] for v in bm.vias],
                    copper={lay[k]: sum(g.area for g in gs) * 1e6 for k, gs in bm.shapes.items() if k in lay},
                    parts={e.name: 1 for e in bm.lumped}, ports=ports)

    def estimate(self, run):
        """The unknowns of mom25d's mesh of the board (meshed here, the ports not characterised) and the time of
        a sweep of up to SOLVES solves."""
        from mom25d.board import BoardModel
        s = run.s
        bm = BoardModel(run.m.to_dict(), log=lambda *a: None, **_model_kw(s),
                        **(dict(ground='none') if _stack(s, run.m) == 'whole' else {}))
        dmax = s.mom_dmax * 1e-3
        planes = {k: 3 * dmax for k, gs in bm.shapes.items() if sum(g.area for g in gs) > 0.5 * bm.outline_area}
        sim = bm.simulation(dmax=dmax, dmax_layer=planes, fmin=run.fmin, fmax=run.fmax, characterize=False,
                            verbose=False, wave_down_to=run.fmin, mesh_only=True)
        n, solves, layers = int(sim.Ne + sim.Nv), min(s.mom_solves, SOLVES), len(bm.shapes)
        plan = memory_plan_for(n, s.mom_tiled)
        lu = LU_N3 * REF_CORES / (os.cpu_count() or 1)                 # the CPU LU scales with the cores, a GPU's not
        lu = dict(hybrid=LU_GPU_N3, tiled=LU_TILED_N3).get(plan['path'], lu)
        per = (T0 * layers * (layers + 1) / 2 + VIA2 * len(bm.vias) ** 2 + (DENSE_N2 * n ** 2 if n <= 3500 else 0)) \
            * REF_CORES / (os.cpu_count() or 1) + (lu * n ** 3 if n > 3500 else 0)
        return dict(seconds=solves * per, gb=plan['host'] / 1e9,
                    gpu_gb=plan['gpu'] / 1e9 or None, how=HOW.get(plan['path'], plan['path']),
                    what=f"{n:,} unknowns, ~{solves} EM solves")

    def prepare(self, run):
        said = _apertures_said(run.m, run.s)
        if said:
            run.log(said)
        what = mom25d_status()[1]
        if 'no compiled core' in what:                              # ~5x slower: say how to build it
            run.log(f"note: mom25d {what}")
        if run.s.preview and mom25d_status()[0]:                   # the mesh alone, for the 3D view and the plan
            s = run.s                                               # (the run's mesh options: its structured runs,
            bm, sim = preview_mesh(run.m, run.fmin, run.fmax, dmax=s.mom_dmax, port=s.mom_port, **_mesh_kw(s, run.m))
            save_mesh(run, sim, bm)                                 # coarsening and vias; the quads come with a run)

    def solve(self, run, prep):
        s = run.s
        self._held = {}
        port = s.mom_port
        if run.antenna and (s.drive or s.grid or s.steer or s.taper):  # a drive that names no port: before the solve
            from kicad2ems.api.solver import settings_drive
            settings_drive(s.drive, [p.name for p in run.m.ports], s.grid, s.steer, s.taper)
        if run.antenna and port == 'auto':                          # the gap source the antenna's power is taken from
            port = 'lumped'
            run.log("mom25d: antenna mode: lumped ports (--mom-port wave for de-embedded ones)")
        S, names, zref, info = solve(run.m, run.freq, dmax=s.mom_dmax, port=port,
                                     max_solves=s.mom_solves, sigma=s.sigma, rough=getattr(s, 'rough', None), finish=run_finish(run),
                                     log=run.log, hold=self._held, tiled=s.mom_tiled,
                                     core=s.mom_core, **_mesh_kw(s, run.m),
                                     keep_at=[f * 1e9 for f in s.mom_dump or []] + _far_field_seeds(run),
                                     progress=lambda k, n, f: (run.status('mom25d', dict(fraction=k / n, solves=k,
                                                                                         f_ghz=f / 1e9)), stop.check()),
                                     on_step=lambda h: run.status('mom25d', dict(afs_solves=h['solves'],
                                                                                 change=h['change'], pred=h['pred'],
                                                                                 tol=h['tol'])))
        kinds = ", ".join(f"{p['name']} {p['kind']}" for p in info['ports'])
        info.update(kpi=f"{info['unknowns']:,} unknowns, {info['seconds']:.0f} s",
                    about=f"mom25d, a 2.5D Method-of-Moments solver: {info['unknowns']:,} unknowns ({info['triangles']:,} "
                          f"triangles, cells up to {info['dmax_mm']:g} mm), {info['solves']} EM "
                          f"solves of an adaptive sweep in {info['seconds']:.0f} s; ports: {kinds}. MoM sees laterally "
                          f"infinite dielectrics over an infinite ground plane: no board edge, no air box, no box "
                          f"resonances; radiation and surface waves are in.")
        run.log(f"mom25d: {info['solves']} EM solves, {info['seconds']:.0f} s")
        return Result(S, names, zref, info['seconds'], info=info, comment=f"mom25d 2.5D MoM ({run.name})")

    def outputs(self, run, prep, res):
        held = getattr(self, '_held', {}).get('res')
        if held is not None:                                        # beside the preview's: the mesh it solved on
            save_mesh(run, held.sim, held.model, 'mom25d-run')
        if run.antenna and held is not None:
            self._antenna(run, held, res)
        self._dumps(run, held)

    def _antenna(self, run, held, result):
        """Antenna mode: the far field of each port driven alone (mom25d.farfield; the others shorted) at
        frequencies over the band and the pattern frequency, combined for the drive (phased.combine: --drive's
        incident waves, else the one antenna port, the other ports terminated), as openEMS's antenna results
        (the same JSON, plots and report section). With several ports the per-port fields are kept
        (<name>_mom25d_array.npz) for the GUI's beam steering. Gain counts substrate loss and surface waves as
        loss; nothing radiates below the infinite ground."""
        from mom25d.farfield import far_field, radiated
        from kicad2ems.api.solver import phased
        from kicad2ems.api.solver import match_summary, describe_antenna, save_antenna, pattern_gain
        from kicad2ems.api.solver import plot_antenna, plot_pattern
        s, m, sim = run.s, run.m, held.sim
        names = [p.name for p in sim.ports]
        n = len(names)
        grid = phased.parse_grid(s.grid, names)
        pos = {p.name: (float(p.c[0]), float(p.c[1])) for p in m.ports}   # the ports' places (model mm): --aim
        if s.drive:
            a, grid = phased.settings_drive(s.drive, names, s.grid, s.steer, s.taper)
        else:
            if s.steer or s.taper or s.aim:
                run.log("note: --steer, --taper and --aim need --drive (the ports to drive): ignored")
            want = [x.strip() for x in (s.excite.split(',') if isinstance(s.excite, str) else s.excite or [])]
            a = np.eye(n, dtype=complex)[next((j for j, nm in enumerate(names) if nm in want), 0)]
        driven = [j for j in range(n) if abs(a[j]) > 0]
        k = driven[0]
        zc = np.array([p.zc for p in sim.ports], float)
        Sa = result.S @ a                                                  # (nf, n): reflected waves over the band
        s11 = Sa[:, k] / a[k]                                              # the active reflection of the first
        match = match_summary(run.freq, s11, m.ports[k].z0)
        f_pat = float(s.ff_freq[0]) * 1e9 if s.ff_freq else match['f_res']
        if s.drive and s.aim:                                              # phases for the pattern frequency
            a = phased.aim(a, names, pos, s.aim[0], s.aim[1], f_pat)
            Sa = result.S @ a
            s11 = Sa[:, k] / a[k]
            match = match_summary(run.freq, s11, m.ports[k].z0)
        ffr = np.unique(np.concatenate([_far_field_grid(run), [f_pat]]))
        th, ph = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)
        up = th <= 90 if sim.st.bottom == 'pec' else np.ones(len(th), bool)   # the whole stack radiates below too
        what = ', '.join(phased.format_drive(a, names)) if len(driven) > 1 else names[k]
        run.log(f"mom25d: antenna, {what} driven: far field of {n} port{'s' if n > 1 else ''} at {len(ffr)} "
                f"frequencies")
        Ys, Eth, Eph = [], [], []
        ip = int(np.argmin(np.abs(ffr - f_pat)))
        kept = getattr(sim, 'kept', {})
        for i, f in enumerate(ffr):
            stop.check()
            I = kept.get(float(f))                                  # solved in the sweep (_far_field_seeds)
            if I is None:
                I = sim.currents(f, sim.excitation())
            Ys.append(np.array([np.atleast_1d(sim.port_current(I, j)) for j in range(n)]))
            et, ep = far_field(sim, np.asarray(I)[:, :n], f, th[up], ph)     # every port's at once: (n, nphi, nth)
            Eth.append(et)
            Eph.append(ep)
            if i == ip:
                I_pat = I
        data = dict(theta=th[up], phi=ph, theta_sphere=th, freq=ffr, ports=names, zc=zc, Y=np.array(Ys),
                    Eth=np.array(Eth), Eph=np.array(Eph), drive=a, f_pattern=float(ffr[ip]),
                    pos=np.array([pos.get(nm, (np.nan, np.nan)) for nm in names]))
        comb = phased.combine(data, a)
        rows, sph = comb['rows'], comb['sphere']
        dmax, pacc = sph['dmax'], [r['Pacc'] for r in rows]
        V, _ = phased.waves(data['Y'][ip], zc, a)
        I = I_pat @ V                                                      # the drive's currents at the pattern f
        step = s.ff_step
        tc = np.arange(-180, 180 + step / 2, step)
        cuts = {}
        for name, phi0 in (('xz (phi = 0)', 0.0), ('yz (phi = 90)', 90.0)):
            g = np.zeros(len(tc))
            for sign, sel in ((1, tc >= 0), (-1, tc < 0)):          # negative theta: the far side, phi + 180
                ang = np.abs(tc[sel])
                ok = ang <= (90 if sim.st.bottom == 'pec' else 180)
                if ok.any():
                    Et, Ep = far_field(sim, I, ffr[ip], ang[ok], [phi0 if sign > 0 else phi0 + 180])
                    gg = np.zeros(sel.sum())
                    gg[ok] = 4 * np.pi * radiated(Et, Ep, ang[ok], [0.0])[0][0] / pacc[ip]
                    g[sel] = gg
            cuts[name] = dict(angle=list(map(float, tc)), gain_dbi=[float(v) for v in 10 * np.log10(np.maximum(g, 1e-6))])
        pa_ = np.arange(0, 360 + step / 2, step)
        Et, Ep = far_field(sim, I, ffr[ip], [90.0], pa_)
        g = 4 * np.pi * radiated(Et, Ep, [90.0], pa_)[0][:, 0] / pacc[ip]
        cuts['xy (theta = 90)'] = dict(angle=list(map(float, pa_)), gain_dbi=[float(v) for v in 10 * np.log10(np.maximum(g, 1e-6))])
        sph_file = run.path('_mom25d_sphere.npz')
        np.savez_compressed(sph_file, **sph)
        eff = rows[ip]['eff']
        e_pat = sph['e_norm'][ip]
        ant = dict(match=match, rows=rows, f_pattern=float(ffr[ip]), cuts=cuts, port=names[k], solver='mom25d',
                   sphere=dict(theta=th, phi=ph, gain=pattern_gain(e_pat, dmax[ip], eff).T),
                   sphere_file=os.path.basename(sph_file),
                   model='infinite ground plane: no radiation below it, no board edge' if sim.st.bottom == 'pec' else
                   'the whole stack: its planes meshed, radiation above and below; laterally infinite dielectric')
        if n > 1:                                                          # the per-port fields: the GUI steers
            arr_file = run.path('_mom25d_array.npz')
            phased.save(arr_file, **data)
            run.wrote('antenna, each port\'s far field, mom25d (beam steering)', arr_file)
            ant['array_file'] = os.path.basename(arr_file)
            ant['drive'] = phased.format_drive(a, names)
            if grid:
                ant['grid'] = phased.format_grid(grid, names)             # the GUI's steering starts with it
            if s.drive and (s.taper or s.steer or s.aim):                # the GUI's Reset: these settings again
                ant['drive_base'] = phased.format_drive(phased.parse_drive(s.drive, names), names)
                ant['taper'] = s.taper or 'uniform'
                ant['steer'] = [float(x) for x in (list(s.steer or []) + [0.0, 0.0])[:2]]
                if s.aim:
                    ant['aim'] = [float(x) for x in s.aim]
            ant['peak'] = comb['peak']
            ant['active_db'] = {names[j]: float(20 * np.log10(max(abs(comb['active'][ip][j]), 1e-15)))
                                for j in driven}
            if len(driven) > 1:
                ant['port'] = ', '.join(ant['drive'])
                pk = comb['peak']
                run.log(f"  mom25d phased drive {ant['port']}: beam at theta {pk['theta']:g}, phi {pk['phi']:g} deg; "
                        f"active S11 " + ', '.join(f"{p} {v:.1f} dB" for p, v in ant['active_db'].items()) +
                        f" at {ffr[ip] / 1e9:.4g} GHz")
        for line in describe_antenna(match, rows, ant['f_pattern']):
            run.log(f"  mom25d {line}")
        tag = '' if 'antenna' not in run.summary else '_mom25d'          # openEMS's antenna results come first
        run.figure(plot_antenna(run.freq, s11, m.ports[k].z0, ant, f"{run.name} - antenna (mom25d)"),
                   "_mom25d_antenna.png", f"antenna{tag}", 'antenna (mom25d): match, gain, efficiency')
        run.figure(plot_pattern(ant, f"{run.name} - gain pattern (mom25d)"), "_mom25d_pattern.png", f"pattern{tag}",
                   'antenna (mom25d): patterns')
        js = run.path('_mom25d_antenna.json')
        save_antenna(ant, js)
        run.wrote('antenna results, mom25d (JSON)', js)
        run.wrote('antenna sphere, mom25d', sph_file)
        run.summary['antenna' + tag] = {k_: v for k_, v in ant.items() if k_ != 'sphere'}

    def _dumps(self, run, held, **kw):
        """The current dumps (--mom-dump): <out>/<solver>/exc_<port>/, laid out like openEMS's frequency-domain dumps
        (the GUI's Fields tab reads both), and their maps and animations next to the other outputs. kw: current_dumps'
        currents / extra / unit (the mixed solver's)."""
        if not run.s.mom_dump or held is None:
            return
        from kicad2ems.api.solver import DumpSource, save_fd_outputs
        tag = self.name
        # dump_port may be supplied by the optional FDTD plugin. MoM alone
        # still supports its default (first-port) current/near-field dumps.
        for port, d in current_dumps(run, held, [f * 1e9 for f in run.s.mom_dump], getattr(run.s, 'dump_port', None), run.s.mom_near,
                                     tag=tag, volume=getattr(run.s, 'mom_near_volume', None), **kw):
            for name in d['names']:
                src = DumpSource(d['dir'], name, 'FD')
                ptag = port.replace('+', 'p').replace('-', 'n')
                for f, q in save_fd_outputs(src, run.path(f"_{tag}_{ptag}_{name}"), animate=run.s.dump_animate):
                    label = f"{tag} {name} ({port} driven, {f / 1e9:g} GHz)"
                    run.wrote(('current map ' if q.endswith('.png') else 'current animation ') + label, q)
                    if q.endswith('.png'):
                        run.summary['dumps'].append(dict(label=f"{tag} {name} ({port} driven)", png=run.rel(q)))
            run.log(f"  {tag} current dumps ({port} driven): {d['dir']}")


def _grid(nodes, tris, step, nmax=600):
    """Grid lines (mm) over the triangles' extent, `step` apart (at most nmax per axis)."""
    lo, hi = nodes[tris].reshape(-1, 2).min(0), nodes[tris].reshape(-1, 2).max(0)
    return [np.linspace(lo[a], hi[a], int(min(nmax, max(2, np.ceil((hi[a] - lo[a]) / step) + 1)))) for a in (0, 1)]


def current_dumps(run, res, freqs, ports=None, near=(), tag='mom25d', currents=None, extra=None,
                  unit='1 V gap source', volume=None):
    """mom25d's surface current density J (A/m) on each meshed layer at `freqs` (Hz), its RWG basis functions
    sampled onto a grid (0 off the copper), and with `near` (mm) the E field on planes that far above the top
    of the stack, with H too, and with `volume` (h0, h1[, n] mm) E and H in that box above the top (mom25d.farfield.
    near_fields), for the ports driven (names, default the first) by mom25d's 1 V
    gap sources. Writes dump_*_fd.npz and dump_info.json per port under <out>/mom25d/exc_<port>/; returns
    [(port, dict(dir, names))]. tag: the solver (the directory under <out>). currents(f) -> (N x len(sim.ports)) the
    RWG currents of each port driven (default: sim.currents of the excitations); extra(k, f_index, column, gx, gy, Jx,
    Jy) -> (Jx, Jy): layer k's grid with what the solver adds (mixed: the stretches put back as lines)."""
    from mom25d.farfield import current_on_grid, current_deposit, near_fields
    from kicad2ems.api.solver import Dump, write_info
    from kicad2ems.api.solver import write_npz
    sim, model = res.sim, res.model
    names = [p.name for p in sim.ports]
    want = [n.strip() for n in (ports.split(',') if isinstance(ports, str) else ports or [])]
    cols = [j for j, n in enumerate(names) if n in want or n.rstrip('+-') in want] if want else [0]
    if not cols:
        raise ValueError(f"--dump-port {ports}: no mom25d port of that name ({', '.join(names)})")
    lay = {k: l for l, k in model.iface.items()}
    layers = [k for k, idx in sorted(sim.tri_idx.items()) if len(idx) and k in lay]
    nodes = sim.rwg.nodes[:, :2] * 1e3
    kept = getattr(sim, 'kept', None) or {}                               # solved by the sweep at these: no solve
    currents = currents or (lambda f: kept[f] if f in kept else sim.currents(f, sim.excitation()))
    I_f = [(stop.check(), currents(f))[1] for f in freqs]
    step = max(run.s.mom_dmax / 3, 0.02)
    ztop = float(model._zk[-1])
    out = []
    for j in cols:
        d = os.path.join(run.s.out, tag, f"exc_{names[j]}")
        os.makedirs(d, exist_ok=True)
        dumps, units = [], {}
        for k in layers:
            gx, gy = _grid(nodes, sim.rwg.tris[sim.tri_idx[k]], step)
            data = np.zeros((len(freqs), len(gx), len(gy), 1, 3), complex)
            for i, I in enumerate(I_f):
                Jx, Jy, _ = current_on_grid(sim, I[:, j], gx * 1e-3, gy * 1e-3, k)
                if extra is not None:
                    Jx, Jy = extra(k, i, j, gx * 1e-3, gy * 1e-3, Jx, Jy)
                data[i, :, :, 0, 0], data[i, :, :, 0, 1] = Jx, Jy
            z = float(model._zk[k])
            dump = Dump(f"J@{lay[k]}", 'J', (gx[0], gy[0], z), (gx[-1], gy[-1], z), list(freqs), 2, lay[k])
            write_npz(os.path.join(d, dump.name + '.npz'), dict(x=gx, y=gy, z=np.array([z])), freqs, data)
            dumps.append(dump)
            units[dump.name] = f'A/m, {unit}'
        allt = np.concatenate([sim.tri_idx[k] for k in layers])

        def grid(h, most=1023):
            # one grid over all the copper, padded so the FFT's wrap-around stays far from the board
            pad = max(5.0, 3 * h)
            lo = nodes[sim.rwg.tris[allt]].reshape(-1, 2).min(0) - pad
            hi = nodes[sim.rwg.tris[allt]].reshape(-1, 2).max(0) + pad
            st_ = max(step, (hi - lo).max() / most, h / 4)               # near_fields pads to 2x for its FFT
            return np.arange(lo[0], hi[0], st_), np.arange(lo[1], hi[1], st_)

        def fields(gx, gy, hs):
            E = np.zeros((len(freqs), len(gx), len(gy), len(hs), 3), complex)
            H = np.zeros_like(E)
            for i, (f, I) in enumerate(zip(freqs, I_f)):
                stop.check()
                LJ = {k: current_deposit(sim, I[:, j], gx * 1e-3, gy * 1e-3, k) for k in layers}
                E[i], H[i] = near_fields(sim, LJ, gx * 1e-3, gy * 1e-3, f, [h * 1e-3 for h in hs])
            return E, H

        def add(field, label, gx, gy, zs, data, normal, u):
            dump = Dump(f"{field}@{label}", field, (gx[0], gy[0], zs[0]), (gx[-1], gy[-1], zs[-1]), list(freqs),
                        normal, label)
            write_npz(os.path.join(d, dump.name + '.npz'), dict(x=gx, y=gy, z=np.asarray(zs, float)), freqs, data)
            dumps.append(dump)
            units[dump.name] = f'{u}, {unit}'
        for h in near or ():
            gx, gy = grid(h)
            E, H = fields(gx, gy, [h])
            label = f"top+{h:g}mm"
            add('E', label, gx, gy, [ztop + h], E, 2, 'V/m')
            add('H', label, gx, gy, [ztop + h], H, 2, 'A/m')
        if volume:
            h0, h1 = float(volume[0]), float(volume[1])
            n = int(volume[2]) if len(volume) > 2 else 12
            hs = list(np.linspace(h0, h1, max(n, 2)))
            gx, gy = grid(h0, most=160)                                  # a volume: about 160 x 160 x n points
            E, H = fields(gx, gy, hs)
            add('E', 'volume', gx, gy, [ztop + h for h in hs], E, None, 'V/m')
            add('H', 'volume', gx, gy, [ztop + h for h in hs], H, None, 'A/m')
        write_info(d, dumps, run.m)
        info = os.path.join(d, 'dump_info.json')                            # J: surface current, A/m
        with open(info, encoding='utf-8') as fh:
            meta = json.load(fh)
        for x in meta['dumps']:
            x['unit'] = units[x['name']]
        with open(info, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
        out.append((names[j], dict(dir=d, names=[x.name for x in dumps])))
    return out
