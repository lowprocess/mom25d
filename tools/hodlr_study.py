"""The hierarchical solver (solver='hodlr', mom25d.hodlr) against the dense direct one on kicad2ems boards: time,
memory, the ranks per tree level and |dS| at one frequency.

    python tools/hodlr_study.py board.json f_GHz [tol,tol,...] [--no-dense] [--zone] [--pec] [--drop] [--whole] [--refine]
                                [--leaf N] [--sym]
"""
import json, sys, time
import numpy as np


def build(board, solver, zone=False, **kw):
    from mom25d.board import BoardModel
    from mom25d.api import ZONE_OUT
    dmax = 0.3e-3
    m = BoardModel(board, port='via', log=lambda *a: None, **kw.pop('model', {}))
    planes = {k: 3 * dmax for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
    z = dict(zone='auto', dmax_out=ZONE_OUT * dmax) if zone else {}
    return m.simulation(dmax=dmax, dmax_layer=planes, fmin=1e9, fmax=10e9, verbose=False, characterize=False,
                        solver=solver, precision='double', gpu='off', **z, **kw)


def main():
    a = sys.argv[1:]
    board, f = json.load(open(a[0])), float(a[1]) * 1e9
    tols = [float(t) for t in a[2].split(',')] if len(a) > 2 and not a[2].startswith('--') else [1e-4, 1e-6]
    leaf = int(a[a.index('--leaf') + 1]) if '--leaf' in a else 256
    zone = '--zone' in a
    # every plane and pad meshed (the measurements in ROADMAP.md item 6), unless --pec / --drop
    model = dict(planes='pec' if '--pec' in a else 'mesh', stitching_pads='drop' if '--drop' in a else 'keep')
    if '--whole' in a:                                           # every copper layer meshed (a port on B.Cu)
        model['ground'] = 'none'
    Sref = None
    if '--no-dense' not in a:
        s = build(board, 'direct', zone, model=model)
        s.sparams(f)                                             # tables and the core's set-up out of the timing
        t = time.time(); Sref = s.sparams(f); td = time.time() - t
        print(f"dense direct: N={s.Ne + s.Nv}  {td:.1f} s a solve  ({16 * (s.Ne + s.Nv) ** 2 / 1e9:.1f} GB complex128)",
              flush=True)
    for tol in tols:
        s = build(board, 'hodlr', zone, model=model)
        s.hodlr_tol, s.hodlr_leaf, s.hodlr_refine = tol, leaf, '--refine' in a
        s.hodlr_symmetric = '--sym' in a
        if '--once' not in a:                                    # a warm-up solve, out of the timing
            s.sparams(f)
        n0 = len(s.log.records)
        t = time.time(); S = s.sparams(f); th = time.time() - t
        st = s.fast.hodlr.stats()
        stages = {}
        for name, dt, _ in s.log.records[n0:]:
            stages[name] = stages.get(name, 0.0) + dt
        dS = f"{np.abs(S - Sref).max():.2e}" if Sref is not None else 'n/a'
        print(f"hodlr tol {tol:g}: N={st['n']}  {th:.1f} s a solve (build+factor {st['seconds']} s), {st['MB']} MB "
              f"(dense {st['dense_MB']} MB), entries {st['entries']}, |dS| {dS}", flush=True)
        print(f"   ranks per level (max, mean): {st['ranks']}" + (f"; GMRES iterations {s.fast.iters}" if '--refine' in a else ''),
              flush=True)
        print('   stages: ' + '; '.join(f"{k} {v:.1f} s" for k, v in stages.items()), flush=True)


if __name__ == '__main__':
    main()
