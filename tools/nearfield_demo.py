"""
A buck converter's hot loop, three ways, on mom25d.mqs: its input capacitor tight against the regulator's VIN and
PGND pins (good), 10 mm away on a trace (far), and that with a slot in the ground plane under its return (slot).
The loop's inductance, the plane's return current, and the magnetic near field 1 and 3 mm above the board at the
switching harmonics (fsw 500 kHz: 0.5-3 MHz, and the trend to 30 MHz) for a 3 A trapezoidal input current (duty
0.33, 5 ns edges); H in planes and a volume as kicad2ems field dumps (the GUI's Fields and 3D views).

    python tools/nearfield_demo.py out/          # out/<case>/..., out/nearfield_*.png, out/summary.json
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box, mapping, Point
from shapely.ops import unary_union

from mom25d.mqs import MQS

W, H = 36.0, 26.0
Z_TOP, Z_GND = 1.6, 1.4                           # F.Cu, In1.Cu (the ground plane, 0.2 mm below)
FSW, DUTY, TR, IPK = 500e3, 0.33, 5e-9, 3.0
HARM = [1, 2, 3, 4, 5, 6, 10, 20, 60]             # 0.5 .. 3 MHz, then 5, 10, 30 MHz
VIA_R = 0.15


def harmonic_amps(n):
    """The input current's harmonics (A peak): a trapezoid 0 to IPK, duty DUTY, edges TR, at n x FSW."""
    n = np.asarray(n, float)
    return 2 * IPK * DUTY * np.abs(np.sinc(n * DUTY)) * np.abs(np.sinc(n * FSW * TR))


def layout(case):
    """board.json's dict of one case: the copper of VIN and GND, the vias, the terminals."""
    vin_pad, pgnd_pad = box(15.6, 12.0, 16.4, 14.0), box(19.6, 12.0, 20.4, 14.0)
    if case == 'good':
        c_p, c_n = box(16.0, 10.0, 17.0, 11.0), box(19.0, 10.0, 20.0, 11.0)
        vin = unary_union([vin_pad, box(15.6, 10.0, 17.0, 14.0)])
        gnd_top = unary_union([pgnd_pad, box(19.0, 10.0, 20.4, 14.0)])
        gvias = [(19.4, 12.6), (20.0, 12.6), (19.4, 11.6), (20.0, 11.6)]
        fine = box(13, 7, 23, 17)
    else:
        c_p, c_n = box(15.5, 2.5, 16.5, 3.5), box(19.5, 2.5, 20.5, 3.5)
        vin = unary_union([vin_pad, box(15.75, 3.0, 16.25, 12.0), c_p])
        gnd_top = unary_union([pgnd_pad, c_n])
        gvias = [(20.0, 12.6), (20.0, 13.4), (19.8, 3.0), (20.2, 3.0)]     # the cap's two in its ground pad
        fine = unary_union([box(13, 0.6, 24, 16)] + ([box(6, 5.5, 14, 9.5)] if case == 'slot' else []))
    plane = box(0.5, 0.5, W - 0.5, H - 0.5)
    if case == 'slot':
        plane = plane.difference(box(8.0, 6.8, W - 0.5, 7.6))        # a slot under the return, open to the right
    # the vias' antipads in the plane are not cut: the ground vias join it, the VIN copper has none
    cu = {'F.Cu': [dict(net='VIN', polygon=mapping(vin)), dict(net='GND', polygon=mapping(gnd_top))],
          'In1.Cu': [dict(net='GND', polygon=mapping(plane))], 'B.Cu': []}
    stack = dict(layers=[dict(name='core', z0=0, z1=Z_GND, er=4.5, tand=0.02, kind='diel'),
                         dict(name='prepreg', z0=Z_GND, z1=Z_TOP, er=4.2, tand=0.02, kind='diel')],
                 copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='In1.Cu', z=Z_GND, t=0.035),
                         dict(name='F.Cu', z=Z_TOP, t=0.035)], thickness=Z_TOP)
    vias = [dict(x=x, y=y, r=VIA_R, z0=Z_GND, z1=Z_TOP, net='GND') for x, y in gvias]
    terms = {'U1.VIN': [('F.Cu', vin_pad)], 'U1.PGND': [('F.Cu', pgnd_pad)],
             'C1.1': [('F.Cu', c_p)], 'C1.2': [('F.Cu', c_n)]}
    b = dict(stack=stack, copper=cu, vias=vias, outline=mapping(box(0, 0, W, H)))
    return b, terms, fine


def run_case(case, out):
    b, terms, fine = layout(case)
    t = time.time()
    q = MQS(b, {'VIN', 'GND'}, terminals=terms, dmax=1.2, fine=fine, dmax_fine=0.4)
    q.source('U1', 'U1.VIN', 'U1.PGND', part_h=0.4)                 # the switches inside the regulator
    q.element('C1', 'C1.1', 'C1.2', C=10e-6, ESR=3e-3, ESL=0.4e-9, part_h=0.6)
    f = FSW * np.array(HARM, float)
    r = q.solve(f)
    t_solve = time.time() - t
    Z = r.zc['U1']                                                  # the copper loop's own (the cap's Z left out)
    mom = np.array([q.moment(r, i) for i in range(len(f))])          # A m^2 per amp of the source
    # H on planes over the board, at each harmonic, for its current
    gx, gy = np.arange(0, W + 1e-9, 0.5), np.arange(0, H + 1e-9, 0.5)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    amps = harmonic_amps(HARM)
    planes = {}
    t = time.time()
    for hgt in (1.0, 3.0, 10.0):
        pts = np.c_[X.ravel(), Y.ravel(), np.full(X.size, Z_TOP + hgt)] * 1e-3
        Hk = q.h_field(r, list(range(len(f))), pts)                 # (n, 3, nf): every harmonic in one pass
        planes[hgt] = np.stack([Hk[:, :, i].reshape(X.shape + (3,)) * amps[i] for i in range(len(f))])
    # a volume: 0.5 to 6 mm above, 12 planes, 1 mm in x, y, at the first 6 harmonics
    vx, vy, vz = np.arange(0, W + 1e-9, 1.0), np.arange(0, H + 1e-9, 1.0), np.linspace(0.5, 6.0, 12)
    VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')
    vpts = np.c_[VX.ravel(), VY.ravel(), Z_TOP + VZ.ravel()] * 1e-3
    Hv = q.h_field(r, list(range(6)), vpts)
    vol = np.stack([Hv[:, :, i].reshape(VX.shape + (3,)) * amps[i] for i in range(6)])
    t_field = time.time() - t
    write_dumps(os.path.join(out, case), b, f, gx, gy, planes, vx, vy, vz, vol)
    J = q.current_density(r, 0)
    return dict(case=case, q=q, r=r, Z=Z, mom=mom * harmonic_amps(HARM)[:, None], f=f, planes=planes, gx=gx, gy=gy, J=J,
                t_solve=t_solve, t_field=t_field,
                tris=len(q.tris), loops=q._mats['C'].shape[1])


def write_dumps(d, b, f, gx, gy, planes, vx, vy, vz, vol):
    """H@top+1mm, H@top+3mm (every harmonic) and H@volume (the first six) as kicad2ems frequency-domain dumps, for the
    GUI (<d>/mqs/exc_U1/: dump_info.json and the npz files)."""
    from kicad2ems.api.solver import Dump, write_info, write_npz
    from shapely.geometry import shape
    exc = os.path.join(d, 'mqs', 'exc_U1')
    os.makedirs(exc, exist_ok=True)
    dumps = []
    for hgt, data in planes.items():
        z = Z_TOP + hgt
        dm = Dump(f"H@top+{hgt:g}mm", 'H', (gx[0], gy[0], z), (gx[-1], gy[-1], z), list(f), 2, f"top+{hgt:g}mm")
        write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=gx, y=gy, z=np.array([z])), f, data[:, :, :, None, :])
        dumps.append(dm)
    dm = Dump("H@volume", 'H', (vx[0], vy[0], Z_TOP + vz[0]), (vx[-1], vy[-1], Z_TOP + vz[-1]), list(f[:6]), None, 'volume')
    write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=vx, y=vy, z=Z_TOP + vz), f[:6], vol)
    dumps.append(dm)
    m = SimpleNamespace(outline=shape(b['outline']), cu_z={c['name']: c['z'] for c in b['stack']['copper']},
                        copper={l: [q for c in cs for q in getattr(shape(c['polygon']), 'geoms', [shape(c['polygon'])])]
                                for l, cs in b['copper'].items() if cs},
                        ports=[], vias=[(v['x'], v['y'], v['r'], v['z0'], v['z1']) for v in b['vias']], thickness=Z_TOP)
    write_info(exc, dumps, m)
    info = json.load(open(os.path.join(exc, 'dump_info.json')))
    for x in info['dumps']:
        x['unit'] = 'A/m, the 3 A trapezoid\'s harmonic'
    json.dump(info, open(os.path.join(exc, 'dump_info.json'), 'w'))


def draw_layouts(cases, out):
    """nearfield_layouts.png: each case's top copper, its ground plane (and slot), vias, pads, parts and hot loop."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import shape
    # the layouts: the top copper, the ground plane (its slot), the vias, the parts, the hot loop's path
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 4.6))
    for a, case in zip(ax, cases):
        b, terms, fine = layout(case)
        for c in b['copper']['In1.Cu']:
            g = shape(c['polygon'])
            for p in getattr(g, 'geoms', [g]):
                a.fill(*p.exterior.xy, color='#cfd8dc', lw=0)
                for h in p.interiors:
                    a.fill(*h.xy, color='white', lw=0)
        for c in b['copper']['F.Cu']:
            g = shape(c['polygon'])
            col = '#e5a33b' if c['net'] == 'VIN' else '#5b8c5a'
            for p in getattr(g, 'geoms', [g]):
                a.fill(*p.exterior.xy, color=col, lw=0.5, ec='k')
        for v in b['vias']:
            a.add_patch(plt.Circle((v['x'], v['y']), v['r'], color='k'))
        for name, pads in terms.items():
            (_, pg), = pads
            x0, y0, x1, y1 = pg.bounds
            a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='r', lw=1.0))
        u_c = (18.0, 13.0)
        a.add_patch(Rectangle((15.4, 11.6), 5.2, 2.8, fill=False, ec='k', lw=1.2, ls='--'))
        a.text(18.0, 14.8, 'U1 (regulator)', ha='center', fontsize=7)
        cx = 18.0
        cy = 10.5 if case == 'good' else 3.0
        a.text(cx, cy - 1.4, 'C1 10 µF (input)', ha='center', fontsize=7)
        a.add_patch(Rectangle((15.9 if case == 'good' else 15.4, cy - 0.6), 4.2 if case == 'good' else 5.2, 1.2,
                              fill=False, ec='k', lw=1.2, ls='--'))
        # the hot loop: C1+ -> VIN pin -> (switches) -> PGND pin -> C1- (top or the plane) -> back through C1
        path = [(16.3, cy), (16.0, 12.8), (20.0, 12.8), (19.9 if case == 'good' else 20.0, cy)]
        for p0, p1 in zip(path, path[1:] + path[:1]):
            a.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=10, color='crimson', lw=1.2, alpha=0.8))
        if case == 'slot':
            a.text(24, 8.0, 'slot in the ground plane', fontsize=7, color='0.3')
            a.annotate('', xy=(7.6, 7.2), xytext=(19.6, 5.4), arrowprops=dict(arrowstyle='->', color='crimson', ls=':',
                       connectionstyle='arc3,rad=-0.3'))
            a.text(9.5, 4.3, 'return detours round the slot', fontsize=6.5, color='crimson')
        a.set_xlim(4, 32); a.set_ylim(0.0, 17.5); a.set_aspect('equal')
        a.set_xlabel('mm', fontsize=7); a.tick_params(labelsize=6)
        a.set_title({'good': 'good: C1 tight against VIN and PGND, solid plane',
                     'far': 'far: C1 10 mm away, VIN on a 0.5 mm trace',
                     'slot': 'slot: as far, a slot in the plane under the return'}[case], fontsize=9)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color='#e5a33b', label='VIN (F.Cu)'), Patch(color='#5b8c5a', label='GND (F.Cu)'),
                          Patch(color='#cfd8dc', label='GND plane (In1, 0.2 mm below)'),
                          Patch(fill=False, ec='r', label='pads'), Patch(color='crimson', label='hot loop')],
                 fontsize=6.5, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'nearfield_layouts.png'), dpi=120); plt.close(fig)


def figures(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from shapely.geometry import shape
    cases = [r['case'] for r in res]
    draw_layouts([r['case'] for r in res], out)
    # 1: |H| 1 mm above at the fundamental, the same scale for all
    mags = [np.linalg.norm(np.abs(r['planes'][3.0][0]), axis=-1) for r in res]
    vmax = max(m.max() for m in mags)
    fig, ax = plt.subplots(1, len(res), figsize=(5.2 * len(res), 4.4))
    for a, r, m in zip(ax, res, mags):
        im = a.imshow(m.T, origin='lower', extent=(r['gx'][0], r['gx'][-1], r['gy'][0], r['gy'][-1]),
                      norm=LogNorm(vmin=vmax / 300, vmax=vmax), cmap='inferno')
        b, _, _ = layout(r['case'])
        for l, col in (('F.Cu', 'c'), ('In1.Cu', 'w')):
            for c in b['copper'][l]:
                g = shape(c['polygon'])
                for p in getattr(g, 'geoms', [g]):
                    a.plot(*p.exterior.xy, c=col, lw=0.6, alpha=0.7)
                    for h in p.interiors:
                        a.plot(*h.xy, c=col, lw=0.6, alpha=0.7)
        a.set_title(f"{r['case']}: |H| 3 mm above, {FSW / 1e6:g} MHz\npeak {m.max():.1f} A/m; copper loop {r['Z'][0].imag / (2 * np.pi * r['f'][0]) * 1e9:.2f} nH", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, label='A/m', shrink=0.8)
    fig.savefig(os.path.join(out, 'nearfield_maps.png'), dpi=110, bbox_inches='tight'); plt.close(fig)
    # 2: the plane's return current at the fundamental
    fig, ax = plt.subplots(1, len(res), figsize=(5.2 * len(res), 4.2))
    for a, r in zip(ax, res):
        c, J = r['J']['In1.Cu']
        mag = np.linalg.norm(np.abs(J), axis=1)
        q = r['q']; idx = q.layer_tris['In1.Cu']
        from matplotlib.tri import Triangulation
        tri = Triangulation(q.nodes[:, 0] * 1e3, q.nodes[:, 1] * 1e3, q.tris[idx])
        im = a.tripcolor(tri, facecolors=mag + 1e-6, norm=LogNorm(vmin=mag.max() / 1e3, vmax=mag.max()), cmap='viridis')
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{r['case']}: ground-plane current, 1 A at {FSW / 1e6:g} MHz", fontsize=9)
        fig.colorbar(im, ax=a, label='A/m', shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'nearfield_return.png'), dpi=110); plt.close(fig)
    # 3: the spectrum of the peak |H| at 1 and 3 mm, and the loop's L
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for k, r in enumerate(res):
        for hgt, ls in ((3.0, '-'), (10.0, '--')):
            pk = [np.linalg.norm(np.abs(r['planes'][hgt][i]), axis=-1).max() for i in range(len(r['f']))]
            ax[0].loglog(r['f'], pk, ls, c=f'C{k}', marker='o', ms=3, label=f"{r['case']}, {hgt:g} mm")
        ax[1].semilogx(r['f'], r['Z'].imag / (2 * np.pi * r['f']) * 1e9, c=f'C{k}', marker='o', ms=3, label=r['case'])
        ax[2].loglog(r['f'], np.linalg.norm(np.abs(r['mom']), axis=1) * 1e6, c=f'C{k}', marker='o', ms=3, label=r['case'])
    ax[0].axvspan(0.5e6, 3e6, color='0.9', zorder=0)
    ax[0].set_xlabel('Hz'); ax[0].set_ylabel('peak |H| (A/m)'); ax[0].grid(True, which='both', alpha=0.3)
    ax[0].set_title('peak near field at each harmonic (3 A trapezoid, 500 kHz, D 0.33, 5 ns)', fontsize=9)
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel('Hz'); ax[1].set_ylabel('nH'); ax[1].grid(True, which='both', alpha=0.3)
    ax[1].set_title("hot loop's copper inductance (the cap's own ESL left out)", fontsize=9); ax[1].legend(fontsize=8)
    ax[2].axvspan(0.5e6, 3e6, color='0.9', zorder=0)
    ax[2].set_xlabel('Hz'); ax[2].set_ylabel('|m| (mA m^2)'); ax[2].grid(True, which='both', alpha=0.3)
    ax[2].set_title('magnetic dipole moment (the field far off ~ |m| / r^3)', fontsize=9); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'nearfield_spectrum.png'), dpi=110); plt.close(fig)


def main(out):
    os.makedirs(out, exist_ok=True)
    res = [run_case(c, out) for c in ('good', 'far', 'slot')]
    figures(res, out)
    summ = {}
    for r in res:
        f = r['f']
        summ[r['case']] = dict(
            triangles=r['tris'], loops=r['loops'], solve_s=round(r['t_solve'], 1), field_s=round(r['t_field'], 1),
            L_nH={f'{fi / 1e6:g} MHz': round(z.imag / (2 * np.pi * fi) * 1e9, 3) for fi, z in zip(f, r['Z'])},
            R_mOhm={f'{fi / 1e6:g} MHz': round(z.real * 1e3, 2) for fi, z in zip(f, r['Z'])},
            peak_H_1mm={f'{fi / 1e6:g} MHz': round(float(np.linalg.norm(np.abs(r['planes'][1.0][i]), axis=-1).max()), 3)
                        for i, fi in enumerate(f)},
            peak_H_3mm={f'{fi / 1e6:g} MHz': round(float(np.linalg.norm(np.abs(r['planes'][3.0][i]), axis=-1).max()), 3)
                        for i, fi in enumerate(f)},
            peak_H_10mm={f'{fi / 1e6:g} MHz': round(float(np.linalg.norm(np.abs(r['planes'][10.0][i]), axis=-1).max()), 4)
                         for i, fi in enumerate(f)},
            moment_mAm2={f'{fi / 1e6:g} MHz': round(float(np.linalg.norm(np.abs(r['mom'][i]))) * 1e6, 4)
                         for i, fi in enumerate(f)})
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    for c, s in summ.items():
        print(c, 'L nH', list(s['L_nH'].values())[:3], 'H 3mm', list(s['peak_H_3mm'].values())[:2], 'H 10mm',
              list(s['peak_H_10mm'].values())[:2], 'm', list(s['moment_mAm2'].values())[:2], 'solve', s['solve_s'], 's')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'nearfield_demo')
