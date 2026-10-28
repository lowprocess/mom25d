"""The mesh configurations (tools/mesh_configs.py) against a fine triangle reference on kicad2ems boards: unknowns, a
solve's time (double precision, CPU) and the largest |dS| at three frequencies. For kicad2ems's docs
(docs/mom.md#coarsen).

    python tools/quad_configs_study.py out.json board.json:f1,f2,f3 [board.json:f1,f2,f3 ...]   # GHz; then the plot
    python tools/quad_configs_study.py out.json --plot-only out.png
"""
import functools, json, os, sys, time
import numpy as np

CONFIGS = [  # (label, Simulation options, augmenting-path depth: 0 = the old greedy pairing)
    ('triangles', dict(quads=False), 12),
    ('quads 25°, greedy (before)', dict(quads='all', quad_tol=25, structured=False), 0),
    ('--mom-quads on', dict(quads=True, quad_tol=35), 12),
    ('fast: 60°, --mom-coarsen 2', dict(quads=True, quad_tol=60, coarsen=2), 12),
]
REF = dict(dmax=0.2e-3)                       # the fine reference: triangles
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # categorical slots 1-4, validated (light surface)
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def build(board, dmax=0.3e-3, **kw):
    from mom25d.board import BoardModel
    m = BoardModel(board, port='via', log=lambda *a: None)
    planes = {k: 3 * dmax for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
    return m.simulation(dmax=dmax, dmax_layer=planes, fmin=1e9, fmax=6e9, verbose=False, characterize=False,
                        solver='direct', precision='double', gpu='off', **kw)


def run(board, F, **kw):
    s = build(board, **kw)
    s.sparams(F[0])                                                   # warm-up: the tables
    ts, S = [], []
    for f in F:
        t = time.time(); S.append(s.sparams(f).ravel()); ts.append(time.time() - t)
    N = s.qT.shape[1] if s.qT is not None else s.Ne + s.Nv
    return np.array(S), int(N), float(np.median(ts))


def study(out, specs):
    from mom25d import quads as Q
    res = json.load(open(out)) if os.path.exists(out) else {}
    m0 = Q._match
    for spec in specs:
        path, fs = spec.rsplit(':', 1)
        name = os.path.basename(path).replace('_board.json', '')
        board, F = json.load(open(path)), np.array([float(f) for f in fs.split(',')]) * 1e9
        Sref, _, _ = run(board, F, **REF)
        r = res.setdefault(name, {})
        try:
            for label, kw, depth in CONFIGS:
                Q._match = functools.partial(m0, depth=depth)
                S, N, t = run(board, F, **kw)
                r[label] = dict(N=N, t=t, dS=np.abs(S - Sref).max(1).tolist())
                print(f"{name:14s} {label:28s} N={N:6d} {t:5.2f} s  |dS| {np.round(r[label]['dS'], 4).tolist()}",
                      flush=True)
        finally:
            Q._match = m0
        json.dump(res, open(out, 'w'), indent=1)
    return res


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    boards = list(res)
    labels = [c[0] for c in CONFIGS]
    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    x = np.arange(len(boards))
    w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        xs = x - 0.4 + w * (i + 0.5)
        err = [max(res[b][lab]['dS']) for b in boards]
        spd = [res[b]['triangles']['t'] / res[b][lab]['t'] for b in boards]
        kw = dict(width=w, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)   # a gap between bars
        ax_e.bar(xs, err, **kw)
        ax_s.bar(xs, spd, **kw)
    ax_e.set_ylabel('largest |ΔS| from a fine reference', color=INK2)
    ax_e.set_title('Error (lower is better)', loc='left', color=INK, fontsize=11)
    ax_s.set_ylabel('solve speed-up over triangles (×)', color=INK2)
    ax_s.set_title('Speed per solve (higher is better)', loc='left', color=INK, fontsize=11)
    ax_s.axhline(1.0, color=INK2, lw=0.8, zorder=4)
    for ax in (ax_e, ax_s):
        ax.set_facecolor(SURFACE)
        ax.set_xticks(x, boards, rotation=25, ha='right', color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(bottom=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('#c9c8c3')
    h, l = ax_e.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(labels), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    print(path)


if __name__ == '__main__':
    out = sys.argv[1]
    if '--plot-only' in sys.argv:
        plot(json.load(open(out)), sys.argv[sys.argv.index('--plot-only') + 1])
    else:
        res = study(out, sys.argv[2:])
        plot(res, os.path.splitext(out)[0] + '.png')
