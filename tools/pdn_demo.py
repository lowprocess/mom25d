"""
Power-integrity demo on mom25d's meshes: a 4-layer 60 x 40 mm board with a 1.0 V rail from a regulator (VRM) on the
left to two loads on the right, its DC drop (mom25d.pdn.dc) and its PDN impedance with the decoupling capacitors
(mom25d.pdn.cavity).

    python tools/pdn_demo.py out/            # board.json, dc_*.png, pdn_z.png, summary.json

The stack: F.Cu (1.6 mm) / 0.1 mm / In1.Cu GND plane / 0.1 mm core / In2.Cu +1V0 plane / 1.4 mm / B.Cu. The +1V0
plane has a slot cut into it from the bottom edge (a split, say for another rail's routing) that leaves a 13 mm
neck between the regulator's half and the loads', and a +3V3 island in a corner.
"""
import json
import os
import sys

import numpy as np
from shapely.geometry import box, Point, mapping
from shapely.ops import unary_union

from mom25d.pdn import dc, cavity

NET, GND, V = '+1V0', 'GND', 1.0
VIA_R, AP_R = 0.15, 0.35                                    # via barrel radius, antipad radius (mm)


def grid(x0, y0, nx, ny, p):
    return [(x0 + i * p, y0 + j * p) for i in range(nx) for j in range(ny)]


def demo_board():
    outline = box(0, 0, 60, 40)
    # the parts: their +1V0 and GND pads on F.Cu, and the vias under them
    vrm_p, vrm_g = box(3, 22, 13, 32), box(3, 8, 13, 18)    # the regulator's output and ground pours
    u1_p, u1_g = box(44, 24, 52, 32), box(44, 14, 52, 22)   # load U1 (6 A): its supply and ground pad fields
    u2_p, u2_g = box(30, 30, 36, 36), box(30, 22, 36, 28)   # load U2 (2 A)
    pwr_vias = grid(5, 24, 4, 4, 2.0) + grid(45.5, 25.5, 4, 4, 1.6) + grid(31.5, 31.5, 2, 2, 2.0)
    gnd_vias = grid(5, 10, 4, 4, 2.0) + grid(45.5, 15.5, 4, 4, 1.6) + grid(31.5, 23.5, 2, 2, 2.0)
    decaps = [  # name, x, y (its +1V0 via; its GND via 1 mm below), C, ESR, ESL
        ('C1', 42.0, 34.0, 1e-6, 10e-3, 0.30e-9), ('C2', 54.0, 34.0, 1e-6, 10e-3, 0.30e-9),
        ('C3', 42.0, 23.0, 1e-6, 10e-3, 0.30e-9), ('C4', 54.0, 23.0, 1e-6, 10e-3, 0.30e-9),
        ('C5', 38.0, 37.0, 10e-6, 3e-3, 0.50e-9), ('C6', 56.0, 37.0, 10e-6, 3e-3, 0.50e-9),
        ('C7', 28.0, 33.0, 1e-6, 10e-3, 0.30e-9),
        ('C8', 15.0, 30.0, 100e-6, 5e-3, 1.0e-9), ('C9', 15.0, 25.0, 100e-6, 5e-3, 1.0e-9)]
    pwr_vias += [(x, y) for _, x, y, *_ in decaps]
    gnd_vias += [(x, y - 1.0) for _, x, y, *_ in decaps]
    holes = lambda pts: unary_union([Point(x, y).buffer(AP_R, 16) for x, y in pts])
    plane_p = box(0.5, 0.5, 59.5, 39.5).difference(box(24, 0.5, 28, 26)).difference(box(46, 0.5, 59.5, 10))
    plane_p = plane_p.difference(holes(gnd_vias))
    island = box(47, 1.5, 58.5, 9)                          # +3V3
    plane_g = box(0.5, 0.5, 59.5, 39.5).difference(holes(pwr_vias))
    pads_p = [vrm_p, u1_p, u2_p] + [Point(x, y).buffer(0.5, 8) for _, x, y, *_ in decaps]
    pads_g = [vrm_g, u1_g, u2_g] + [Point(x, y - 1.0).buffer(0.5, 8) for _, x, y, *_ in decaps]
    fcu = [dict(net=NET, polygon=mapping(g)) for g in pads_p] + [dict(net=GND, polygon=mapping(g)) for g in pads_g]
    b = dict(format='kicad2ems-board', version=1, units='mm',
             stack=dict(layers=[dict(name='core 2', z0=0.0, z1=1.4, er=4.5, tand=0.02, kind='diel'),
                                dict(name='core 1', z0=1.4, z1=1.5, er=4.2, tand=0.02, kind='diel'),
                                dict(name='prepreg', z0=1.5, z1=1.6, er=4.2, tand=0.02, kind='diel')],
                        copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='In2.Cu', z=1.4, t=0.035),
                                dict(name='In1.Cu', z=1.5, t=0.035), dict(name='F.Cu', z=1.6, t=0.035)],
                        thickness=1.6),
             outline=mapping(outline),
             copper={'F.Cu': fcu, 'In1.Cu': [dict(net=GND, polygon=mapping(plane_g))],
                     'In2.Cu': [dict(net=NET, polygon=mapping(plane_p)), dict(net='+3V3', polygon=mapping(island))],
                     'B.Cu': []},
             vias=[dict(x=x, y=y, r=VIA_R, z0=1.4, z1=1.6, net=NET) for x, y in pwr_vias] +
                  [dict(x=x, y=y, r=VIA_R, z0=1.5, z1=1.6, net=GND) for x, y in gnd_vias],
             ports=[], parts=[], walls=[], notes=[])
    loads = dict(source=dict(x=8, y=27, r=2.0, layer='F.Cu', V=V), gsource=dict(x=8, y=13, r=2.0, layer='F.Cu', V=0.0),
                 loads=[dict(name='U1', x=48, y=28, r=0.6, layer='F.Cu', I=6.0),
                        dict(name='U2', x=33, y=33, r=0.6, layer='F.Cu', I=2.0)],
                 gloads=[dict(name='U1', x=48, y=18, r=0.6, layer='F.Cu', I=6.0),
                         dict(name='U2', x=33, y=25, r=0.6, layer='F.Cu', I=2.0)],
                 decaps=decaps)
    return b, loads


def plot_dc(out, res, net, scale, label, fname, vias=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    lays = [l for l in ('F.Cu', 'In1.Cu', 'In2.Cu') if l in res.sheets]
    fig, axs = plt.subplots(len(lays), 2, figsize=(11, 3.1 * len(lays)), squeeze=False)
    for row, lay in zip(axs, lays):
        sh = res.sheets[lay]
        tri = Triangulation(sh.nodes[:, 0], sh.nodes[:, 1], sh.tris)
        v = scale(sh.V)
        a = row[0].tripcolor(tri, np.nan_to_num(v), shading='gouraud', cmap='viridis')
        fig.colorbar(a, ax=row[0], label=label)
        row[0].set_title(f'{net} on {lay}: {label}')
        j = sh.J
        b = row[1].tripcolor(tri, facecolors=j, cmap='magma', vmax=np.percentile(j, 99.5))
        fig.colorbar(b, ax=row[1], label='A/mm')
        row[1].set_title(f'{net} on {lay}: current per mm of width')
        if vias:
            I = np.array([v['I'] for v in res.vias])
            xy = np.array([(v['x'], v['y']) for v in res.vias])
            if len(xy):
                row[1].scatter(xy[:, 0], xy[:, 1], s=4 + 40 * I / max(I.max(), 1e-9), facecolors='none',
                               edgecolors='c', linewidths=0.6)
        for ax in row:
            ax.set_aspect('equal'); ax.set_xlim(0, 60); ax.set_ylim(0, 40); ax.set_xlabel('mm')
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=110)
    plt.close(fig)


def main(out):
    os.makedirs(out, exist_ok=True)
    b, L = demo_board()
    json.dump(b, open(os.path.join(out, 'board.json'), 'w'))
    import time
    t = time.time()
    seen, p, g = dc.supply(b, NET, GND, L['source'], L['loads'], ground_source=L['gsource'], ground_loads=L['gloads'],
                           dmax=0.8)
    t_dc = time.time() - t
    plot_dc(out, p, NET, lambda v: (V - v) * 1e3, 'drop from 1.0 V, mV', 'dc_power.png')
    plot_dc(out, g, GND, lambda v: v * 1e3, 'rise above 0 V, mV', 'dc_ground.png')
    vi = sorted([(v['I'], v['x'], v['y']) for v in p.vias], reverse=True)

    # PDN: the +1V0 / GND plane pair (In2 / In1, 0.1 mm), the decaps, the regulator and the loads as ports
    mount = cavity.mount_inductance(0.15, 1.0, VIA_R) + 0.2e-9         # the via pair + the pad escape
    ports = [dict(name='U1', x=47.1, y=27.1, r=VIA_R), dict(name='U2', x=31.5, y=31.5, r=VIA_R),
             dict(name='VRM', x=9.0, y=28.0, r=VIA_R, R=1e-3, L=30e-9)]
    ports += [dict(name=n, x=x, y=y, r=VIA_R, C=C, ESR=esr, ESL=esl, mount=mount) for n, x, y, C, esr, esl in L['decaps']]
    t = time.time()
    pp = cavity.PlanePair(b, NET, GND, 'In2.Cu', 'In1.Cu', ports, dmax=1.0)
    f = np.logspace(4, 9.3, 213)
    Z = pp.z(f, observe=['U1', 'U2'])
    Z0 = pp.z(f, bare=True, observe=['U1', 'U2'])
    nod = [q for q in ports if 'C' not in q]
    Znd = cavity.PlanePair(b, NET, GND, 'In2.Cu', 'In1.Cu', nod, dmax=1.0, log=lambda *a: None).z(f, observe=['U1', 'U2'])
    t_pdn = time.time() - t
    ztarget = V * 0.03 / (0.5 * 6.0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for k, name in enumerate(['U1', 'U2']):
        ax[k].loglog(f, np.abs(Z0[:, k, k]), ':', c='0.5', label='planes alone')
        ax[k].loglog(f, np.abs(Znd[:, k, k]), '--', c='C1', label='planes + VRM')
        ax[k].loglog(f, np.abs(Z[:, k, k]), c='C0', lw=2, label='planes + VRM + 9 decaps')
        ax[k].axhline(ztarget, c='C3', lw=1, label=f'target {ztarget * 1e3:.0f} mOhm')
        ax[k].set_title(f'{name}: PDN impedance at its +1V0 via'); ax[k].set_xlabel('Hz'); ax[k].grid(True, which='both', alpha=0.3)
    ax[0].set_ylabel('|Z| (Ohm)'); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'pdn_z.png'), dpi=110); plt.close(fig)
    zu1 = np.abs(Z[:, 0, 0])
    bands = []                                              # the frequency bands where U1 is over its target
    for k in np.flatnonzero(np.diff(np.r_[0, (zu1 > ztarget).astype(int), 0])):
        bands.append(float(f[min(k, len(f) - 1)]))
    bands = [[round(a), round(b)] for a, b in zip(bands[::2], bands[1::2])]
    summary = dict(
        dc=dict(seconds=round(t_dc, 2), vertices=p.nodes + g.nodes,
                loads={n: dict(V=round(s['V'], 5), drop_mV=round(s['drop'] * 1e3, 2),
                               power_drop_mV=round(s['power_drop'] * 1e3, 2), ground_rise_mV=round(s['ground_rise'] * 1e3, 2))
                       for n, s in seen.items()},
                loss_mW=round((p.loss + g.loss) * 1e3, 2),
                hottest_vias=[dict(I=round(i, 3), x=x, y=y) for i, x, y in vi[:5]],
                notes=p.notes + g.notes),
        pdn=dict(seconds=round(t_pdn, 2), vertices=len(pp.nodes), plane_C_nF=round(pp.C * 1e9, 3),
                 target_mOhm=round(ztarget * 1e3, 1), mount_nH=round(mount * 1e9, 3),
                 U1_max_mOhm=round(zu1.max() * 1e3, 1), U1_max_at_Hz=float(f[np.argmax(zu1)]),
                 U1_over_target_Hz=bands))
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'pdn_demo')
