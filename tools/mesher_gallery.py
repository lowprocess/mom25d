"""
Mesher gallery: what the meshers physically produce on one board (a kicad2ems <name>_board.json), and what
that does to the solution.

    python tools/mom25d/mesher_gallery.py out/board_board.json [--f0 3.25] [--band 2.5,4.0] [--out gallery.npz]
    python tools/mom25d/mesher_gallery.py out/board_board.json --plot [--zoom x0,x1,y0,y1] [--cut x_mm] [--png docs/...]

The first form meshes the board with every mesher (dmax 0.3 mm, ncross 6, the study's default point, plus the
fine Triangle reference), solves the surface current at f0 (port 1 driven) and an adaptive S sweep over the
band, one case after the other (no overlap, so the timings are comparable), and caches everything in the npz.
The second plots from the cache: the mesh and |J| in a zoom window, |J| along a horizontal cut across the traces (the edge singularity),
triangle quality and S21 against the reference.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mom25d.board import BoardModel                                # noqa: E402

CASES = [('triangle', 0.1, 10, 'reference: Triangle 0.1 mm, ncross 10'),
         ('triangle', 0.3, 6, 'Triangle (default)'),
         ('gmsh-rings', 0.3, 6, 'gmsh, offset rings'),
         ('gmsh-graded', 0.3, 6, 'gmsh, isotropic graded'),
         ('distmesh', 0.3, 6, 'DistMesh, isotropic'),
         ('distmesh-rings', 0.3, 6, 'DistMesh in offset rings')]


def compute(board, f0, band, out):
    from mom25d.run import solve_band
    data = dict(np.load(out, allow_pickle=True)) if os.path.exists(out) else {}
    for i, (mesher, d, n, label) in enumerate(CASES):
        if f'c{i}_nodes' in data:
            continue
        print(f"{label} ...", flush=True)
        m = BoardModel(json.load(open(board)), port='via', log=lambda *a: None)
        t0 = time.time()
        sim = m.simulation(dmax=d * 1e-3, fmin=band[0], fmax=band[1], solver='direct', verbose=False,
                           mesher=mesher, ncross=n, characterize=False)
        t_mesh = time.time() - t0
        t1 = time.time()
        I = sim.currents(f0, sim.excitation())
        t_f0 = time.time() - t1
        Jx, Jy = sim.current_density(I[:, 0])
        t2 = time.time()
        f, S, info = solve_band(sim, band[0], band[1], points=301, zref=m.port_z0, log=lambda *a: None)
        r = sim.rwg
        data.update({f'c{i}_nodes': r.nodes, f'c{i}_tris': r.tris, f'c{i}_J': np.stack([Jx, Jy]),
                     f'c{i}_f': f, f'c{i}_S21': S[:, 1, 0], f'c{i}_S11': S[:, 0, 0],
                     f'c{i}_stats': np.array([sim.Ne + sim.Nv, len(r.tris), t_mesh, t_f0, time.time() - t2,
                                              info['solves']])})
        print(f"  {sim.Ne + sim.Nv} unknowns, mesh {t_mesh:.1f} s, one solve {t_f0:.1f} s, "
              f"sweep {time.time() - t2:.1f} s ({info['solves']} solves)", flush=True)
        np.savez(out, **data)
        del sim


def plot(npz, zoom, cut, png, f0, close, band):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.tri import Triangulation
    from mom25d.mesh_gmsh import quality
    D = np.load(npz)
    n = len(CASES)
    x0, x1, y0, y1 = zoom
    cx0, cx1, cy0, cy1 = close
    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(4, n, height_ratios=[1.15, 0.95, 0.55, 1.0], hspace=0.32, wspace=0.28)
    Jref = None
    xs = np.linspace(x0, x1, 3000)
    prof = []
    for i, (mesher, d, nc, label) in enumerate(CASES):
        nd, tr, J = D[f'c{i}_nodes'] * 1e3, D[f'c{i}_tris'], D[f'c{i}_J']
        st = D[f'c{i}_stats']
        P = nd[tr]
        lo, hi = P.min(1), P.max(1)
        amin, ar = quality(nd, tr)
        mag = np.hypot(np.abs(J[0]), np.abs(J[1]))
        if i == 0:
            Jref = mag.max()
        # close-up of the mesh, coloured by aspect ratio
        ax = fig.add_subplot(gs[0, i])
        sel = (hi[:, 0] > cx0) & (lo[:, 0] < cx1) & (hi[:, 1] > cy0) & (lo[:, 1] < cy1)
        pc = PolyCollection(P[sel], array=np.log10(ar[sel]), cmap='viridis', ec='k', lw=0.3)
        pc.set_clim(0, np.log10(30))
        ax.add_collection(pc)
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect('equal')
        ax.set_title(f"{label}\n{int(st[0]):,} unknowns, {int(st[1]):,} triangles", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == n - 1:
            cb = fig.colorbar(pc, ax=ax, fraction=0.05, ticks=np.log10([1, 2, 5, 10, 30]))
            cb.ax.set_yticklabels(['1', '2', '5', '10', '30'])
            cb.set_label('aspect ratio', fontsize=8)
        # |J| over the wider window
        ax = fig.add_subplot(gs[1, i])
        sel = (hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1)
        db = 20 * np.log10(mag / Jref + 1e-30)
        pc = PolyCollection(P[sel], array=np.clip(db[sel], -40, 0), cmap='inferno', ec='face', lw=0.1)
        pc.set_clim(-40, 0)
        ax.add_collection(pc)
        ax.add_patch(plt.Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec='c', lw=0.8))
        ax.axhline(cut, color='r', lw=0.8, ls='--')
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.set_facecolor('#d8d8d8')
        ax.set_title(f"|J| at {f0 / 1e9:.2f} GHz, P1 driven", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == n - 1:
            cb = fig.colorbar(pc, ax=ax, fraction=0.05)
            cb.set_label('|J| (dB rel. reference max)', fontsize=8)
        # quality
        ax = fig.add_subplot(gs[2, i])
        ax.hist(amin, bins=np.linspace(0, 60, 31), color='C0', alpha=0.8)
        ax.set_xlabel('minimum angle (deg)', fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title(f"min angle median {np.median(amin):.0f} deg, AR p95 {np.percentile(ar, 95):.1f}", fontsize=8)
        # |J| sampled along the cut (one value per triangle: the steps are the cells)
        k = Triangulation(nd[:, 0], nd[:, 1], tr).get_trifinder()(xs, np.full_like(xs, cut))
        prof.append(np.where(k >= 0, mag[np.maximum(k, 0)] / Jref, np.nan))
    gsb = gs[3, :].subgridspec(1, 3, width_ratios=[1.25, 1.25, 1.0], wspace=0.25)
    on = np.isfinite(prof[0])
    edges = xs[1:][on[1:] != on[:-1]]
    pieces = [(edges[j], edges[j + 1]) for j in range(0, len(edges) - 1, 2)]
    pieces = [pieces[0], pieces[-1]]                               # the first and the last trace in the window
    for q, (a, b) in enumerate(pieces):
        ax = fig.add_subplot(gsb[q])
        for i, (mesher, d, nc, label) in enumerate(CASES):
            ax.plot(xs, prof[i], lw=2.4 if i == 0 else 1.2, color='k' if i == 0 else f'C{i - 1}', label=label)
        pad = 0.04 * (b - a)
        ax.set_xlim(a - pad, b + pad)
        ax.set_xlabel(f'x (mm) across the trace at y = {cut} mm (red cut)', fontsize=9)
        ax.set_ylabel('|J| / reference max', fontsize=9)
        ax.set_title('surface current across a trace: each flat step is one triangle', fontsize=9)
        ax.grid(alpha=0.3)
        if q == 0:
            ax.legend(fontsize=7, frameon=False, loc='upper center')
    ax = fig.add_subplot(gsb[2])
    for i, (mesher, d, nc, label) in enumerate(CASES):
        f, s21 = D[f'c{i}_f'], D[f'c{i}_S21']
        st = D[f'c{i}_stats']
        ax.plot(f / 1e9, 20 * np.log10(np.abs(s21)), lw=2.4 if i == 0 else 1.2, color='k' if i == 0 else f'C{i - 1}',
                label=f"{'reference' if i == 0 else mesher}: {st[4]:.0f} s, {int(st[5])} solves")
    ax.set_xlim(*band); ax.set_ylim(-40, 0); ax.grid(alpha=0.3)
    ax.set_xlabel('f (GHz)', fontsize=9); ax.set_ylabel('|S21| (dB)', fontsize=9)
    ax.set_title('|S21|, lumped ports; sweep time (cases run one at a time)', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='lower center')
    fig.savefig(png, dpi=120, bbox_inches='tight')
    print('wrote', png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('board')
    ap.add_argument('--f0', type=float, default=3.25)
    ap.add_argument('--band', default='2.5,4.0')
    ap.add_argument('--out', default='mesher_gallery.npz')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--zoom', default='', help='x0,x1,y0,y1 in mm (mom25d coordinates)')
    ap.add_argument('--cut', type=float, default=None, help='y of the |J| cross-section (a horizontal cut), mm')
    ap.add_argument('--close', default='', help='x0,x1,y0,y1 in mm: the mesh close-up (default: zoom)')
    ap.add_argument('--fplot', default='', help='f0,f1 in GHz: the S21 axis (default: the band)')
    ap.add_argument('--png', default='mesher_gallery.png')
    a = ap.parse_args()
    if a.plot:
        z = [float(v) for v in a.zoom.split(',')]
        c = [float(v) for v in a.close.split(',')] if a.close else z
        fb = [float(v) for v in (a.fplot or a.band).split(',')]
        plot(a.out, z, a.cut if a.cut is not None else 0.5 * (z[2] + z[3]), a.png, a.f0 * 1e9, c, fb)
    else:
        compute(a.board, a.f0 * 1e9, [float(v) * 1e9 for v in a.band.split(',')], a.out)


if __name__ == '__main__':
    main()
