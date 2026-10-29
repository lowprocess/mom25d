"""The mesh configurations side by side on one board: triangles, the old quad cells (25 degrees, greedy pairing),
--mom-quads on (35 degrees, matching, structured runs) and the fast one (60 degrees, coarsen 2). Quad cells are drawn
without their diagonal. For kicad2ems's docs (docs/mom.md#coarsen).

    python tools/mesh_configs.py <name>_board.json out.png [x0,y0,x1,y1 (mm, a window)] [title] [stitching]
"""
import functools, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
from mom25d import quads as Q
from mom25d.board import BoardModel

SURFACE, INK, EDGE = '#fcfcfb', '#0b0b0b', '#52514e'
FILL = dict(quad='#d4e4f7', tri='#f8cbb6', fan='#b5e6d3')        # tints of the palette's blue, orange, aqua
CONFIGS = [  # (title, Simulation options, augmenting-path depth: 0 = the old greedy pairing)
    ('triangles (quads off)', dict(quads=False), 12),
    ('quads, 25°, greedy pairing (before)', dict(quads='all', quad_tol=25, structured=False), 0),
    ('--mom-quads on: 35°, matching, structured runs, the edge strip kept', dict(quads=True, quad_tol=35), 12),
    ('fast: 60°, structured, --mom-coarsen 2', dict(quads=True, quad_tol=60, coarsen=2), 12),
]
CONFIGS_STITCHING = [  # tools/mesh_configs.py ... stitching: round against square stitching vias
    ('fast, stitching vias round (default)', dict(quads=True, quad_tol=60, coarsen=2), 12),
    ('fast, --mom-stitching-vias square: each via snapped to a cell of the grid',
     dict(quads=True, quad_tol=60, coarsen=2, stitching_vias='square'), 12),
]


def simulation(board, dmax=0.3e-3, **kw):
    m = BoardModel(board, port='via', log=lambda *a: None)
    planes = {k: 3 * dmax for k, gs in m.shapes.items() if sum(g.area for g in gs) > 0.5 * m.outline_area}
    return m.simulation(dmax=dmax, dmax_layer=planes, fmin=1e9, fmax=6e9, verbose=False, characterize=False,
                        solver='direct', **kw)


def draw(ax, s, title, win, seen):
    r, k = s.rwg, max(s.layers)
    on = s.tri_layer == k
    P = r.nodes * 1e3
    elim = s.quad_info['elim'] if getattr(s, 'qT', None) is not None else np.zeros(0, int)
    merged = np.zeros(len(r.tris), bool)
    merged[r.tp[elim]] = merged[r.tm[elim]] = True
    fan = np.zeros(len(r.tris), bool)
    fan[s.F] = True
    col = np.where(fan, FILL['fan'], np.where(merged, FILL['quad'], FILL['tri']))
    P_c = P[r.tris[on]].mean(1)
    vis = np.ones(on.sum(), bool) if not win else \
        (P_c[:, 0] >= win[0]) & (P_c[:, 0] <= win[2]) & (P_c[:, 1] >= win[1]) & (P_c[:, 1] <= win[3])
    seen.update(col[on][vis])
    ax.add_collection(PolyCollection(P[r.tris[on]], facecolors=col[on], edgecolors='none'))
    E = np.unique(np.sort(r.tris[on][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1), axis=0)
    gone = {tuple(np.intersect1d(r.tris[r.tp[d]], r.tris[r.tm[d]])) for d in elim}   # the merged diagonals
    E = np.array([e for e in E if tuple(e) not in gone])
    ax.add_collection(LineCollection(P[E], colors=EDGE, linewidths=0.4))
    N = s.qT.shape[1] if getattr(s, 'qT', None) is not None else s.Ne + s.Nv
    ax.set_title(f"{title}\n{N:,} unknowns, {merged[on].sum() / on.sum():.0%} of the triangles in quads",
                 fontsize=10, color=INK, loc='left')
    ax.set_aspect('equal')
    if win:
        ax.set_xlim(win[0], win[2]); ax.set_ylim(win[1], win[3])
    else:
        ax.autoscale_view()
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color('#d0cfca')
    ax.tick_params(colors=EDGE, labelsize=8)


def main():
    import json
    board = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    win = tuple(map(float, sys.argv[3].split(','))) if len(sys.argv) > 3 and sys.argv[3] else None
    title = sys.argv[4] if len(sys.argv) > 4 else 'mom25d mesh configurations'
    aspect = (win[3] - win[1]) / (win[2] - win[0]) if win else 0.5
    n = len(CONFIGS if len(sys.argv) <= 5 else CONFIGS_STITCHING)
    fig, axs = plt.subplots(n, 1, figsize=(10, 1.2 + n * (10 * aspect + 0.7)), sharex=True,
                            sharey=True, facecolor=SURFACE)
    configs = CONFIGS if len(sys.argv) <= 5 else [c for c in CONFIGS_STITCHING]
    m0, seen = Q._match, set()
    try:
        for ax, (t, kw, depth) in zip(axs, configs):
            Q._match = functools.partial(m0, depth=depth)
            draw(ax, simulation(board, **kw), t, win, seen)
    finally:
        Q._match = m0
    axs[-1].set_xlabel('mm', color=EDGE, fontsize=9)
    labels = dict(quad='quad cell (no diagonal)', tri='triangle', fan='via fan (stays triangles)')
    fig.legend(handles=[Patch(color=FILL[k], label=v) for k, v in labels.items() if FILL[k] in seen],
               loc='upper right', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.97, 1.0))
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main()
