"""
mom25d.cli - KiCad / Gerber board or AppCSXCAD / openEMS geometry (.xml) -> S-parameters

  mom25d board.kicad_pcb --pos board-top.pos --nets RF,GND --ground B.Cu \\
         --fmin 0.5 --fmax 10 --out results/
  mom25d board.kicad_pcb --config board.mom25d.toml      (same keys as the options)
  mom25d board.xml --fmin 1 --fmax 6        (a CSX file, e.g. from kicad2ems: see mom25d.csx)

A --config file holds any of the options (dashes or underscores; board too); the command line wins over
it, and its paths (board, pos, out) are relative to the file. Outputs in --out: <name>.s<N>p
(Touchstone, reference impedance per port in the header comment), <name>_dashboard.png,
<name>_summary.json (stack, ports, mesh, timings, diagnostics). Frequencies in GHz, lengths in mm (KiCad
units). By default the sweep is adaptive (rational fit, --points samples on the output grid); --linear N
solves at N equally spaced frequencies instead. MOM25D_TRACEBACK=1 shows the Python traceback of an error.
"""
import argparse
import difflib
import json
import os
import sys
import time

import numpy as np

try:
    import tomllib
except ImportError:                                   # Python < 3.11
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

PATH_KEYS = ('board', 'pos', 'out')                   # relative to the config file


def write_touchstone(path, f, S, zref, comment='', names=None):
    """Touchstone v1 when all reference impedances are equal, v2.0 (per-port [Reference]) otherwise."""
    n = S.shape[1]
    zref = np.asarray(zref, float)
    v2 = not np.allclose(zref, zref[0])
    with open(path, 'w') as fh:
        fh.write(f"! mom25d {comment}\n")
        if names:
            fh.write("! ports: " + ", ".join(f"{i + 1}={nm}" for i, nm in enumerate(names)) + "\n")
        if v2:
            fh.write("[Version] 2.0\n")
        fh.write(f"# Hz S RI R {zref[0]:.6g}\n")
        if v2:
            fh.write(f"[Number of Ports] {n}\n")
            if n == 2:
                fh.write("[Two-Port Data Order] 21_12\n")
            fh.write(f"[Number of Frequencies] {len(f)}\n")
            fh.write("[Reference] " + " ".join(f"{z:.6g}" for z in zref) + "\n[Network Data]\n")
        for fi, s in zip(f, S):
            vals = [s[i, j] for i in range(n) for j in range(n)] if n != 2 else [s[0, 0], s[1, 0], s[0, 1], s[1, 1]]
            per_line = 4 if n > 2 else len(vals)
            for k in range(0, len(vals), per_line):
                pre = f"{fi:.9e}" if k == 0 else " " * 15
                fh.write(pre + " " + " ".join(f"{v.real:.9e} {v.imag:.9e}" for v in vals[k:k + per_line]) + "\n")
        if v2:
            fh.write("[End]\n")


from .run import renorm, out_refs                     # noqa: E402  (kept importable from here)


def _config_path(value, base, key):
    """A path from the config file: relative to its folder; one that only exists relative to the working
    directory (configs written for the repository root) is used as it is, with a note."""
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return value
    here = os.path.normpath(os.path.join(base, value))
    exists = (lambda p: os.path.isdir(os.path.dirname(p) or '.')) if key == 'out' else os.path.exists
    if not exists(here) and exists(value) and os.path.normpath(value) != here:
        print(f"config: {key} = {value!r} is not next to the config; using it relative to the working directory "
              f"(write it relative to the config file)")
        return value
    return here


def config_defaults(ap, path):
    """The options of a TOML config as parser defaults (the command line still wins). Unknown keys are an
    error that names the option they are probably a typo of."""
    if tomllib is None:
        ap.error("--config needs Python 3.11+, or: python -m pip install tomli")
    try:
        with open(path, 'rb') as fh:
            cfg = tomllib.load(fh)
    except OSError as e:
        ap.error(f"--config {path}: {e.strerror}")
    except tomllib.TOMLDecodeError as e:
        ap.error(f"--config {path}: {e}")
    known = set(vars(ap.parse_args([]))) - {'config', 'gpu_info'}
    base = os.path.dirname(os.path.abspath(path))
    out, unknown = {}, []
    for k, v in cfg.items():
        k = k.replace('-', '_')
        if k not in known:
            near = difflib.get_close_matches(k, sorted(known), n=1, cutoff=0.8)
            unknown.append(k + (f" (did you mean {near[0]}?)" if near else ''))
            continue
        if isinstance(v, dict):                                     # [dmax_layer] / [wref] tables
            v = [f"{kk}={vv}" for kk, vv in v.items()]
        elif isinstance(v, list) and k in ('nets', 'region'):
            v = ",".join(map(str, v))
        if k in PATH_KEYS and isinstance(v, str):
            v = _config_path(v, base, k)
        out[k] = v
    if unknown:
        ap.error(f"--config {path}: unknown key(s) {', '.join(unknown)}")
    return out


def main(argv=None):
    try:
        return _main(argv)
    except (FileNotFoundError, IsADirectoryError, ValueError) as e:  # a file or a setting that cannot be used
        if os.environ.get('MOM25D_TRACEBACK'):
            raise
        msg = f"{e.strerror}: {e.filename}" if isinstance(e, OSError) and e.filename else str(e)
        print(f"mom25d: error: {msg}", file=sys.stderr)
        return 2


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('board', nargs='?', help='.kicad_pcb or .xml (AppCSXCAD / openEMS); Gerber: kicad2ems')
    ap.add_argument('--config', help='TOML file with any of the options below')
    ap.add_argument('--pos', help='pick-and-place file (.pos / .csv) listing the port components')
    ap.add_argument('--nets', help='comma-separated nets to include (KiCad only; default all)')
    ap.add_argument('--ground', help='copper layer used as the infinite ground (z=0); default B.Cu (.xml: its hints, '
                                     'else the lowest solid plane)')
    ap.add_argument('--port', default='auto', choices=['auto', 'wave', 'via'],
                    help='.xml ports: wave (de-embedded), via (lumped) or auto (wave where the feed allows)')
    ap.add_argument('--mesher', default='auto',
                    choices=['auto', 'polygon', 'tensor', 'gmsh', 'gmsh-rings', 'gmsh-graded', 'distmesh', 'distmesh-rings'],
                    help='surface mesher (gmsh: pip install gmsh; see the README)')
    ap.add_argument('--cover', help='copper layer used as a PEC cover (stripline-only cross sections)')
    ap.add_argument('--region', help='crop x0,y0,x1,y1 in mm (KiCad coordinates)')
    ap.add_argument('--no-mask', action='store_true', help='ignore the solder mask layer')
    ap.add_argument('--fmin', type=float, default=0.5); ap.add_argument('--fmax', type=float, default=10.0)
    ap.add_argument('--points', type=int, default=401, help='output frequency points (adaptive sweep)')
    ap.add_argument('--linear', type=int, help='solve at N linearly spaced points instead of adaptive')
    ap.add_argument('--afs-tol', type=float, default=2e-3)
    ap.add_argument('--dmax', type=float, default=0.3, help='max cell size along conductors (mm)')
    ap.add_argument('--dmax-layer', action='append', default=[], help='per layer, e.g. In1.Cu=0.8 (repeatable)')
    ap.add_argument('--wref', action='append', default=[], help='edge-grading width per layer, e.g. F.Cu=0.35')
    ap.add_argument('--ncross', type=int, default=4)
    ap.add_argument('--near-fac', type=float, default=4.0, help='near-field radius in cells (accuracy vs memory)')
    ap.add_argument('--solver', default='auto', choices=['auto', 'dense', 'direct', 'blr', 'aca'])
    ap.add_argument('--gpu', default='auto', choices=['auto', 'cupy', 'torch', 'off'],
                    help="GPU for the LU of the mixed-precision direct solver (ROCm or CUDA via CuPy / PyTorch)")
    ap.add_argument('--gpu-info', action='store_true', help="print the detected GPU and exit")
    ap.add_argument('--precision', default='auto', choices=['auto', 'double', 'mixed'],
                    help="direct solver: 'mixed' stores Z in complex64 and refines to double accuracy (auto: mixed when the core is built)")
    ap.add_argument('--z0', type=float, default=50.0, help='output reference impedance')
    ap.add_argument('--sigma', type=float, help='conductor conductivity S/m (enables metal loss)')
    ap.add_argument('--out', default='mom25d_out')
    return ap


def parse(argv=None):
    """(parser, options): the command line, over a --config file's values, over the defaults."""
    ap = parser()
    a = ap.parse_args(argv)
    if a.config:                                        # the file's values become defaults: typed options win
        ap.set_defaults(**config_defaults(ap, a.config))
        a = ap.parse_args(argv)
    return ap, a


def _main(argv=None):
    ap, a = parse(argv)

    if a.gpu_info:
        from . import gpu as mom25d_gpu
        if a.gpu != 'auto':
            os.environ.setdefault('MOM25D_GPU', a.gpu)
        print(mom25d_gpu.describe())
        return 0
    if not a.board:
        ap.error("board file required")
    from .sim import Metal
    from . import viz as viz
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.board))[0].replace('-job', '')
    region = tuple(float(v) for v in a.region.split(',')) if a.region else None
    z0 = a.z0
    if a.board.lower().endswith('.xml'):
        from .csx import CSXModel
        model = CSXModel(a.board, ground=a.ground, cover=a.cover, port=a.port, f_loss=0.5 * (a.fmin + a.fmax) * 1e9)
        z0 = model.port_z0 if a.z0 == 50.0 else a.z0                 # the ports' own impedances unless given
        brd = None
    else:
        from .kicad import KiCadBoard
        brd = KiCadBoard(a.board)
        nets = [n.strip() for n in a.nets.split(',')] if a.nets else None
        model = brd.build(nets=nets, ground=a.ground or 'B.Cu', cover=a.cover, region=region, pos_file=a.pos,
                          include_mask=not a.no_mask)
    if brd is not None:
        print(brd.describe_stackup())
    print(model.describe())
    kv = lambda lst: {model.iface[k]: float(v) * 1e-3 for k, v in (s.split('=') for s in lst)}
    sim = model.simulation(dmax=a.dmax * 1e-3, dmax_layer=kv(a.dmax_layer), w_ref=kv(a.wref) or None,
                           ncross=a.ncross, solver=a.solver, verbose=True, fmin=a.fmin * 1e9, fmax=a.fmax * 1e9,
                           near_fac=a.near_fac, precision=a.precision, gpu=a.gpu,
                           metal=Metal(a.sigma) if a.sigma else (Metal(model.sigma, model.sheet_t or 35e-6)
                                                                  if getattr(model, 'sigma', None) else None),
                           mesher=a.mesher, title=name)
    sim.layer_names = {k: n for n, k in model.iface.items()}
    zc = np.array([p.zc for p in sim.ports])
    if any(getattr(p, 'mode', None) for p in sim.ports):
        print("mixed-mode ports: " + ", ".join(p.name for p in sim.ports))
    if a.linear:
        f = np.linspace(a.fmin, a.fmax, a.linear) * 1e9
        res = sim.sweep(f, currents_at=0.5 * (f[0] + f[-1]))
        S = np.array([renorm(s, np.array([p.zc_fn(fi) if getattr(p, 'zc_fn', None) else p.zc for p in sim.ports]),
                             out_refs(sim, z0)) for fi, s in zip(f, res.S)])
        res.S = S
        nsolve = len(f)
    else:
        from .afs import adaptive_sweep
        def solve(fr):
            print(f"  solving {fr / 1e9:.4f} GHz", flush=True)
            Sr = sim.sparams(fr)
            return renorm(Sr, np.array([p.zc for p in sim.ports]), out_refs(sim, z0))
        f, S, fs, Ss, hist = adaptive_sweep(solve, a.fmin * 1e9, a.fmax * 1e9, tol=a.afs_tol,
                                            ngrid=a.points, log=print)
        nsolve = len(fs)
        # one extra solve at mid-band keeping currents, for the dashboard
        from types import SimpleNamespace
        fm = fs[len(fs) // 2]
        Sm = sim.sparams(fm, keep=True)
        res = SimpleNamespace(f=f, S=S, I=sim.last['I'], fI=fm, gamma=np.zeros((len(f), len(zc))),
                              resid=np.tile(sim.last['resid'], (len(f), 1)),
                              recip=np.abs(S - S.transpose(0, 2, 1)).max((1, 2)),
                              passiv=np.array([np.linalg.svd(s, compute_uv=False).max() for s in S]))
    ts = os.path.join(a.out, f"{name}.s{S.shape[1]}p")
    write_touchstone(ts, f, S, out_refs(sim, z0), comment=f"{name} ({nsolve} EM solves)",
                     names=[p.name for p in sim.ports])
    dash = viz.dashboard(sim, res, os.path.join(a.out, f"{name}_dashboard.png"))
    # lowest frequency where every wave port's sample span covers >= 0.4 rad (from the port Zc lines'
    # phase velocity estimate: eps_eff from the characterised gamma is not stored, use the fitted one)
    valid_from = None
    try:
        from .geo import LinePort
        spans = [abs(s[-1] - s[0]) for p, s in zip(sim.ports, sim.samples) if isinstance(p, LinePort)]
        if spans:
            fv = sim.last['f']; gv = np.nanmax(np.abs(sim.last['gamma'].imag) / fv)   # beta/f
            valid_from = float(0.4 / (gv * min(spans)))
            print(f"wave-port separation reliable above ~{valid_from / 1e9:.2f} GHz")
    except Exception:
        pass
    summary = dict(board=a.board, valid_from_GHz=None if valid_from is None else round(valid_from / 1e9, 3), ports=[getattr(p, 'name', '') for p in sim.ports],
                   port_kind=[type(p).__name__ for p in sim.ports], zc=list(map(float, zc)),
                   unknowns=int(sim.Ne + sim.Nv), solver=sim.solver, em_solves=int(nsolve),
                   fmin=a.fmin, fmax=a.fmax, wall_time_s=round(time.time() - t0, 1),
                   max_passivity=float(np.max(res.passiv)), max_reciprocity_err=float(np.max(res.recip)),
                   stack=model.describe().splitlines(), timings={k: round(v, 2) for k, v in sim.log.totals().items()},
                   notes=sim.log.notes)
    with open(os.path.join(a.out, f"{name}_summary.json"), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {ts}, {dash}  ({nsolve} EM solves, {time.time() - t0:.0f} s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
