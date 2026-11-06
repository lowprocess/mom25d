"""
python -m mom25d.pdn: a power rail's DC drop and PDN impedance from a KiCad board (or a kicad2ems board.json).

    python -m mom25d.pdn board.kicad_pcb --rail +3V3 --volts 3.3 --load U21=2 --load U14=1 --out pdn_3v3
    python -m mom25d.pdn board.kicad_pcb --rail +3V3 --volts 3.3 --spread-load 5 --out pdn_3v3   # 5 A shared by pad count
    python -m mom25d.pdn --describe --json          # the options, for the GUI's PDN page

It writes <out>/pdn.json (the rail tree, each load's voltage and drop, the series parts, the hottest vias, the plane
pairs, each part's connection and mount inductance, the impedance each load sees and where it is over the target,
the notes), <out>/z.csv, and the figures <out>/dc_power_<layer>.png, dc_ground_<layer>.png (drop or rise, current
per mm, vias by current) and pdn_z.png. Its progress goes to stdout, one line per step ('pdn: ...').
"""
import argparse
import json
import os
import sys
import time

import numpy as np

DESCRIBE_FORMAT = 1
G_RAIL, G_DC, G_PDN, G_MESH = 'Rail', 'DC', 'PDN impedance', 'Mesh'
OPTIONS = [
    # key, flag, kind, default, help, group, choices
    ('rail', '--rail', 'str', '', "the rail's net (the tree grows from it through series parts)", G_RAIL, None),
    ('ground', '--ground', 'str', 'GND', "its ground net(s), comma-separated", G_RAIL, None),
    ('volts', '--volts', 'float', 3.3, "the rail's voltage at its source", G_RAIL, None),
    ('source', '--source', 'str', '', "the part it is fed from (default: the power inductor feeding the tree)", G_RAIL, None),
    ('ground_source', '--ground-source', 'str', '', "the part whose ground pads its return goes back to (default: "
     "the parts on that inductor's switch node)", G_RAIL, None),
    ('load', '--load', 'list', [], "a load: REF=AMPS (repeat)", G_RAIL, None),
    ('spread_load', '--spread-load', 'float', 0.0, "amps shared by +rail pad count over every part but R, C, L "
     "(an assumption, for a first look when the loads are not known)", G_RAIL, None),
    ('max_ohms', '--max-ohms', 'float', 1.0, "the largest resistor the tree follows (sense and 0 ohm links)", G_RAIL, None),
    ('through', '--through', 'str', '', "parts the tree follows whatever they are, comma-separated", G_RAIL, None),
    ('stop', '--stop', 'str', '', "parts the tree never follows, comma-separated", G_RAIL, None),
    ('plating', '--plating', 'float', 0.025, "via plating thickness, mm", G_DC, None),
    ('dmax', '--dmax', 'float', 1.0, "the largest mesh cell, mm (finer at the source and the loads)", G_MESH, None),
    ('vrm_r', '--vrm-r', 'float', 1e-3, "the regulator's output resistance, ohms", G_PDN, None),
    ('vrm_l', '--vrm-l', 'float', 30e-9, "the regulator's output inductance (its loop's bandwidth), H", G_PDN, None),
    ('fmin', '--fmin', 'float', 1e4, "lowest frequency, Hz", G_PDN, None),
    ('fmax', '--fmax', 'float', 2e9, "highest frequency, Hz", G_PDN, None),
    ('points', '--points', 'int', 160, "frequencies (log-spaced)", G_PDN, None),
    ('target', '--target', 'float', 0.0, "the target impedance, ohms (0: volts x ripple / step)", G_PDN, None),
    ('ripple', '--ripple', 'float', 0.05, "the allowed ripple, a fraction of the voltage", G_PDN, None),
    ('step', '--step', 'float', 0.0, "the load step, A (0: half the loads' current)", G_PDN, None),
    ('no_pdn', '--no-pdn', 'bool', False, "the DC drop only", G_PDN, None),
]


def describe():
    import mom25d
    return dict(format=DESCRIBE_FORMAT, version=getattr(mom25d, '__version__', '?'),
                options=[dict(key=k, flag=f, kind=t, default=d, help=h, group=g, choices=c)
                         for k, f, t, d, h, g, c in OPTIONS])


def parser():
    ap = argparse.ArgumentParser(prog='python -m mom25d.pdn', description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('board', nargs='?', help='.kicad_pcb, or a kicad2ems <name>_board.json')
    ap.add_argument('--out', default='pdn', help='output folder')
    ap.add_argument('--describe', action='store_true', help='print the options and exit')
    ap.add_argument('--json', action='store_true', help='with --describe: as JSON')
    for k, f, t, d, h, g, c in OPTIONS:
        if t == 'bool':
            ap.add_argument(f, dest=k, action='store_true', help=h)
        elif t == 'list':
            ap.add_argument(f, dest=k, action='append', default=[], help=h)
        else:
            ap.add_argument(f, dest=k, type={'float': float, 'int': int, 'str': str}[t], default=d, help=h)
    return ap


def say(*a):
    print('pdn:', *a, flush=True)


def load_board(path, nets):
    """board.json's dict of `nets` (a set) from a .kicad_pcb (through kicad2ems) or a board.json."""
    if path.endswith('.json'):
        return json.load(open(path))
    from kicad2ems.importers import open_board
    from kicad2ems.board import Board
    raw = open_board(path)
    b = Board(raw, nets=sorted(nets), parts='none', need_ports=False, log=lambda *a: None)
    return b.to_dict()


def spread(board, rail_net, amps):
    users = {f['ref']: sum(p.get('net') == rail_net for p in f['pads']) for f in board['prims']['footprints']
             if not f.get('dnp') and any(p.get('net') == rail_net for p in f['pads'])
             and not f['ref'].upper().startswith(('C', 'R', 'L', 'FB'))}
    tot = sum(users.values())
    return {r: amps * n / tot for r, n in users.items()} if tot else {}


def bands(f, z, target):
    over = np.abs(z) > target
    edges = np.flatnonzero(np.diff(np.r_[0, over.astype(int), 0]))
    return [[float(f[a]), float(f[min(b, len(f)) - 1])] for a, b in zip(edges[::2], edges[1::2])]


def run(a):
    from .rail import Rail
    from .dc import solve_rail
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    ground = tuple(x.strip() for x in a.ground.split(',') if x.strip())
    if not a.rail:
        raise SystemExit('pdn: --rail is required')
    say(f'reading {os.path.basename(a.board)}')
    board = load_board(a.board, {a.rail, *ground})
    loads = {}
    for s in a.load:
        ref, _, amps = s.partition('=')
        loads[ref.strip()] = float(amps)
    assumed = False
    if not loads and a.spread_load > 0:
        loads, assumed = spread(board, a.rail, a.spread_load), True
    if not loads:
        raise SystemExit('pdn: no loads: --load REF=AMPS, or --spread-load AMPS')
    kw = dict(ground=ground, loads=loads, source=a.source or None, ground_source=a.ground_source or None,
              max_ohms=a.max_ohms, through=tuple(x for x in a.through.split(',') if x),
              stop=tuple(x for x in a.stop.split(',') if x))
    rail = Rail(board, a.rail, **kw)
    if not a.board.endswith('.json') and not rail.nets <= {a.rail}:      # the tree's other nets: read with it
        say(f'rail tree: {len(rail.nets)} nets; reading their copper')
        board = load_board(a.board, set(rail.nets) | set(ground))
        rail = Rail(board, a.rail, **kw)
    for line in rail.describe():
        say(line.strip())
    say('DC: meshing and solving')
    t = time.time()
    dc = solve_rail(rail, a.volts, dmax=a.dmax, plating=a.plating, log=lambda m: say(m))
    t_dc = time.time() - t
    out = dict(format=1, rail=a.rail, nets=sorted(rail.nets), ground=list(ground), volts=a.volts, assumed_loads=assumed,
               settings={k: getattr(a, k) for k, *_ in OPTIONS},
               tree=[dict(ref=s.part.ref, kind=s.kind, R=s.R, a=s.a, b=s.b, defaulted=s.defaulted) for s in rail.series],
               source=[p.ref for p in rail.source_parts],
               loads={r: dict(I=v['I'], V=v['V'], drop=v['drop'], power_drop=v['power_drop'], ground_rise=v['ground_rise'])
                      for r, v in dc['loads'].items()},
               series={r: v for r, v in dc['series'].items()},
               loss_W=dc['power'].loss + dc['ground'].loss,
               vias=dict(power=sorted([dict(x=v['x'], y=v['y'], I=v['I']) for v in dc['power'].vias], key=lambda v: -v['I'])[:20],
                         ground=sorted([dict(x=v['x'], y=v['y'], I=v['I']) for v in dc['ground'].vias], key=lambda v: -v['I'])[:20]),
               vertices=dict(power=dc['power'].nodes, ground=dc['ground'].nodes), seconds=dict(dc=round(t_dc, 1)),
               notes=list(rail.notes) + dc['notes'])
    say(f"DC done in {t_dc:.1f} s: worst drop {max(v['drop'] for v in out['loads'].values()) * 1e3:.2f} mV")
    figs = dc_figures(board, dc, a.volts, a.out)
    out['figures'] = figs
    if not a.no_pdn:
        from .stack import RailPDN
        say('PDN: plane pairs, parts, mesh')
        t = time.time()
        pdn = RailPDN(rail, dmax=a.dmax, vrm=dict(R=a.vrm_r, L=a.vrm_l), log=lambda m: say(m))
        t_b = time.time() - t
        f = np.logspace(np.log10(a.fmin), np.log10(a.fmax), a.points)
        say(f'PDN: {len(f)} frequencies')
        t = time.time()
        obs = [r for r in loads if pdn.dof(r) is not None]
        Z = pdn.z(f, observe=obs)
        Z0 = pdn.z(f, observe=obs, bare=True)
        t_z = time.time() - t
        step = a.step or 0.5 * sum(loads.values())
        target = a.target or a.volts * a.ripple / step
        out['pdn'] = dict(f=f.tolist(), target=target,
                          z={r: dict(mag=np.abs(Z[r]).tolist(), deg=np.degrees(np.angle(Z[r])).tolist(),
                                     bare=np.abs(Z0[r]).tolist(), over=bands(f, Z[r], target)) for r in obs},
                          cavities=[dict(rail=c.rail_layer, ground=c.ground_layer, d=c.d, er=c.er, area=c.geom.area,
                                         C=8.854e-12 * c.er * c.geom.area * 1e-6 / (c.d * 1e-3)) for c in pdn.cavities],
                          parts={n: dict(role=i['role'], mount_nH=i['mount'] * 1e9, node=i['rail'].node,
                                         rail=i['rail'].why, gnd=i['gnd'].why,
                                         element={k: v for k, v in (i['el'] or {}).items() if k != 'defaulted'},
                                         defaulted=bool((i['el'] or {}).get('defaulted')))
                                 for n, i in pdn.parts.items()},
                          vertices=pdn.nv)
        out['notes'] += pdn.notes
        out['seconds'].update(pdn_build=round(t_b, 1), pdn_sweep=round(t_z, 1))
        with open(os.path.join(a.out, 'z.csv'), 'w') as fh:
            fh.write('f_Hz,' + ','.join(f'{r}_mag,{r}_deg' for r in obs) + '\n')
            for i in range(len(f)):
                fh.write(f'{f[i]:.6g},' + ','.join(f'{abs(Z[r][i]):.6g},{np.degrees(np.angle(Z[r][i])):.3f}' for r in obs) + '\n')
        figs.append(z_figure(f, Z, Z0, target, a.out))
        worst = max(obs, key=lambda r: np.abs(Z[r]).max()) if obs else None
        if worst:
            say(f'PDN done in {t_b + t_z:.1f} s; target {target * 1e3:.1f} mOhm; {worst} over it at '
                f'{len(out["pdn"]["z"][worst]["over"])} band(s)')
    out['seconds']['total'] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, 'pdn.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    say(f'wrote {os.path.join(a.out, "pdn.json")}')
    return out


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def dc_figures(board, dc, volts, out):
    """dc_power_<layer>.png and dc_ground_<layer>.png for the layers carrying most of the current: the drop (or the
    rise), the current per mm, the vias by their current."""
    from matplotlib.tri import Triangulation
    from shapely.geometry import shape
    plt = _plt()
    x0, y0, x1, y1 = shape(board['outline']).bounds
    files = []
    for side, res in (('power', dc['power']), ('ground', dc['ground'])):
        carry = sorted(res.sheets.items(), key=lambda kv: -float(np.nansum(kv[1].J * 0 + kv[1].J)))
        for lay, sh in carry[:2]:
            tri = Triangulation(sh.nodes[:, 0], sh.nodes[:, 1], sh.tris)
            fig, ax = plt.subplots(1, 2, figsize=(14, 4.6 * (y1 - y0) / max(x1 - x0, 1) + 1.2))
            v = (volts - sh.V) * 1e3 if side == 'power' else sh.V * 1e3
            m = ax[0].tripcolor(tri, np.nan_to_num(v), shading='gouraud', cmap='viridis' if side == 'power' else 'cividis')
            fig.colorbar(m, ax=ax[0], label='mV')
            ax[0].set_title(f"{lay}: {'drop from the source' if side == 'power' else 'rise above the source'}, mV", fontsize=10)
            j = sh.J
            n = ax[1].tripcolor(tri, facecolors=j, cmap='magma', vmax=float(np.percentile(j, 99.7)) if len(j) else 1)
            fig.colorbar(n, ax=ax[1], label='A/mm')
            I = np.array([q['I'] for q in res.vias])
            if len(I) and I.max() > 0:
                XY = np.array([(q['x'], q['y']) for q in res.vias])
                ax[1].scatter(XY[:, 0], XY[:, 1], s=2 + 50 * I / I.max(), facecolors='none', edgecolors='c', lw=0.5)
            ax[1].set_title(f"{lay}: current per mm; vias by current (max {I.max() if len(I) else 0:.2f} A)", fontsize=10)
            for q in ax:
                q.set_aspect('equal'); q.set_xlim(x0, x1); q.set_ylim(y0, y1); q.set_xticks([]); q.set_yticks([])
            fig.tight_layout()
            name = f"dc_{side}_{lay.replace('.', '_')}.png"
            fig.savefig(os.path.join(out, name), dpi=110)
            plt.close(fig)
            files.append(dict(file=name, side=side, layer=lay))
    return files


def z_figure(f, Z, Z0, target, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (r, z) in enumerate(Z.items()):
        ax.loglog(f, np.abs(z), c=f'C{k % 10}', lw=1.8, label=r)
        ax.loglog(f, np.abs(Z0[r]), ':', c=f'C{k % 10}', lw=0.8)
    ax.axhline(target, c='r', lw=1, label=f'target {target * 1e3:.1f} mOhm')
    ax.set_xlabel('Hz'); ax.set_ylabel('|Z| (Ohm)'); ax.grid(True, which='both', alpha=0.3)
    ax.set_title('impedance each load sees (dotted: the planes alone)', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'pdn_z.png'), dpi=110); plt.close(fig)
    return dict(file='pdn_z.png', side='pdn', layer='')


def main(argv=None):
    a = parser().parse_args(argv)
    if a.describe:
        d = describe()
        print(json.dumps(d, indent=1) if a.json else '\n'.join(f"{o['flag']:18s} {o['help']}" for o in d['options']))
        return 0
    if not a.board:
        parser().error('a board is required')
    try:
        run(a)
    except (ValueError, KeyError) as e:
        print(f'pdn: error: {e}', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
