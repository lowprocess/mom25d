"""Crop CM5 Minima's U702 converter and plot its copper loops and magnetic near field.

    OPENBLAS_NUM_THREADS=2 python tools/cm5_converter_fields.py CM5_MINIMA_3_board.json --out cm5-fields
    python tools/cm5_converter_fields.py --out cm5-fields --render-only

Independent 1 A RMS harmonic drives: VIN->PGND (hot-loop component), and
PGND->SW (output ripple component). They are transfer functions, not an
assumed converter operating spectrum. Inductor winding/core leakage, switch
node E and far-field radiation are not modelled. Package paths are collapsed
to the top copper plane. All six copper layers are retained.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from shapely.geometry import Point, LineString, box, shape, mapping
from shapely.ops import unary_union

from mom25d.base import MU0
from mom25d.biot import rwg_currents
from mom25d.mqs import MQS
from mom25d.pdn.rail import parts_of, part_value

REFS = ['U702', 'L701', 'C704', 'C705', 'C706', 'C708', 'C709', 'C707', 'R704', 'R705', 'R706']
VIN, GND, SW, VOUT = '+5V', 'GND', '/PCIe-M2/M2_LX', 'M2_3V3'
NETS = {VIN, GND, SW, VOUT}
CASES = ['hot', 'ripple']
CASE_TITLES = ['Input hot-loop component', 'Output inductor-ripple component']
COLORS = {VIN: '#d99516', GND: '#bfcbd3', SW: '#9857b6', VOUT: '#158f9e'}


def polygons(g):
    if g.is_empty:
        return
    if g.geom_type == 'Polygon':
        yield g
    elif hasattr(g, 'geoms'):
        for part in g.geoms:
            yield from polygons(part)


def crop_board(board, region):
    """Keep real clipped copper and local primitives, with source coordinates preserved."""
    out = copy.deepcopy(board)
    out['outline'], out['region'] = mapping(shape(board['outline']).intersection(region)), list(region.bounds)
    out['request']['region'] = list(region.bounds)
    out['nets'], out['ports'], out['parts'], out['walls'], out['port_refs'] = None, [], [], [], []
    out['copper'] = {}
    for layer, records in board['copper'].items():
        out['copper'][layer] = []
        for record in records:
            g = shape(record['polygon']).intersection(region)
            for p in polygons(g):
                if p.area:
                    out['copper'][layer].append(dict(record, polygon=mapping(p)))
    out['vias'] = [v for v in board['vias'] if region.contains(Point(v['x'], v['y']))]
    prims = out['prims']
    prims['items'] = []
    for record in board['prims']['items']:
        g = shape(record['geom']).intersection(region)
        if not g.is_empty and g.area:
            prims['items'].append(dict(record, geom=mapping(g)))
    prims['tracks'] = []
    for record in board['prims']['tracks']:
        clipped = LineString(record['pts']).intersection(region)
        for line in getattr(clipped, 'geoms', [clipped]):
            if line.geom_type == 'LineString' and line.length:
                prims['tracks'].append(dict(record, pts=list(map(list, line.coords))))
    prims['vias'] = [v for v in board['prims']['vias'] if region.contains(Point(v['x'], v['y']))]
    prims['footprints'] = []
    for fp in board['prims']['footprints']:
        pads = []
        for p in fp.get('pads', []):
            if not p.get('geom'):
                continue
            clipped = shape(p['geom']).intersection(region)
            if clipped.area:
                pads.append(dict(p, geom=mapping(clipped)))
        if pads:
            prims['footprints'].append(dict(fp, pads=pads))
    out['notes'] = ['Local U702 converter crop; source board coordinates retained.',
                    'All clipped copper retained for context; explicit MQS nets and parts are in summary.json.']
    return out


def compute(args, out):
    path = Path(args.board)
    board = json.loads(path.read_text())
    parts = {p.ref: p for p in parts_of(board)}
    extent = unary_union([pad.geom for ref in REFS for pad in parts[ref].pads if pad.layers])
    region = extent.buffer(1.).envelope
    board = crop_board(board, region)
    (out/'converter_board.json').write_text(json.dumps(board)+'\n')
    from kicad2ems.board import Board
    loaded = Board.load(out/'converter_board.json', log=lambda *_: None)
    assert loaded.outline.equals(shape(board['outline']))
    parts = {p.ref: p for p in parts_of(board)}
    terms = {}
    for ref in ['U702', 'L701', 'C704', 'C705', 'C706', 'C708', 'C709']:
        for net in NETS:
            pp = [(layer, p.geom) for p in parts[ref].on({net}) for layer in p.layers]
            if pp:
                terms[f'{ref}:{net}'] = pp
    fine = unary_union([p for pp in terms.values() for _, p in pp]).buffer(.2)
    q = MQS(board, NETS, terminals=terms, dmax=args.dmax, fine=fine, dmax_fine=args.fine,
            conductor_loss='slab', conductor_faces=1, fill_workers=2)
    estimate = (4*len(q.tris)**2+3*q.ne**2)*8
    if estimate > 6.5e9:
        raise RuntimeError(f'estimated dense working set {estimate/1e9:.1f} GB; increase --dmax/--fine')
    q.source('hot', f'U702:{VIN}', f'U702:{GND}', part_h=0)
    q.source('ripple', f'U702:{GND}', f'U702:{SW}', part_h=0)
    for ref, rail in [('C704', VIN), ('C705', VIN), ('C706', VIN), ('C708', VOUT), ('C709', VOUT)]:
        q.element(ref, f'{ref}:{rail}', f'{ref}:{GND}', C=part_value(parts[ref]), ESR=.003, ESL=.4e-9, part_h=0)
    q.element('L701', f'L701:{SW}', f'L701:{VOUT}', R=.05, L=part_value(parts['L701']), part_h=0)
    freqs = np.array([.1, 1., 10.])*1e6
    primary = 1
    xmin, ymin, xmax, ymax = region.bounds
    x, y = np.linspace(xmin, xmax, 65)*1e-3, np.linspace(ymin, ymax, 53)*1e-3
    heights = np.array([1., 3.])
    volume_heights = np.linspace(.5, 5., 19)
    ztop = q.z_of.max()
    results, J, planes, volume, convergence, direct_errors, timings, residuals = [], [], [], [], {}, {}, {}, {}
    ends = q.br+[(q.term_node[p['a']], q.term_node[p['b']]) for p in q.parts]
    from scipy.sparse import coo_matrix
    a, b = np.array(ends).T; indices = np.arange(len(ends))
    A = coo_matrix((np.r_[np.ones(len(ends)), -np.ones(len(ends))],
                    (np.r_[a, b], np.r_[indices, indices])), shape=(q.n_nodes, len(ends))).tocsr()
    for case in CASES:
        start = time.perf_counter()
        r = q.solve(freqs, amps={n: float(n == case) for n in CASES})
        timings[f'{case}_solve_s'] = time.perf_counter()-start
        residuals[case] = float(abs(A@r.i.T).max())
        assert residuals[case] < 1e-8
        results.append(r)
        currents = []
        for fi in range(len(freqs)):
            aa, bb = rwg_currents(q.rwg, r.i[fi, :q.ne]/q.rwg.l)
            currents.append(aa[:, None]*q.rwg.cent+bb)
        J.append(currents)
        start = time.perf_counter()
        H, plane_info = q.h_grid(r, list(range(len(freqs))), x, y, ztop+heights*1e-3,
                                 rtol=.02, return_info=True)
        Hv, vol_info = q.h_grid(r, primary, x, y, ztop+volume_heights*1e-3, rtol=.02, return_info=True)
        planes.append(H); volume.append(Hv)
        timings[f'{case}_field_s'] = time.perf_counter()-start
        convergence[case] = dict(planes=plane_info, volume=vol_info)
        X, Y, Z = np.meshgrid(x[::16], y[::13], ztop+heights*1e-3, indexing='ij')
        points = np.stack([X, Y, Z], -1).reshape(-1, 3)
        direct = q.h_field(r, primary, points).reshape(5, 5, 2, 3)
        direct_errors[case] = float(abs(H[::16, ::13, :, :, primary]-direct).max()/abs(direct).max())
        print(case, 'direct field error', direct_errors[case], 'KCL', residuals[case], flush=True)
    layer_names = list(q.layer_tris)
    layer_id = np.array([layer_names.index(l) for l in q.layer_of])
    part_a = np.array([q._term_centre(p['a']) for p in q.parts])
    part_b = np.array([q._term_centre(p['b']) for p in q.parts])
    data = dict(nodes=q.nodes, tris=q.tris, layer_id=layer_id, layer_names=np.array(layer_names),
                triangle_z=q.z_of, J=np.array(J), I=np.array([r.i for r in results]),
                part_a=part_a, part_b=part_b, part_index=np.arange(len(q.parts))+len(q.br),
                x=x, y=y, heights_mm=heights, volume_heights_mm=volume_heights,
                H=np.array(planes), H_volume=np.array(volume), freqs=freqs, primary=np.array(primary),
                copper_z=np.array([r.zc[c] for r, c in zip(results, CASES)]))
    np.savez_compressed(out/'currents_fields.npz', **data)
    meta = dict(converter='U702 AP3441SHE-7B', inductor='L701', source_json=path.name,
                source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), crop_model_mm=list(region.bounds),
                crop_kicad_mm=[xmin, -ymax, xmax, -ymin], nets=sorted(NETS), refs=REFS, cases=CASES,
                source_drive='Independent 1 A RMS sinusoidal phasors; not a measured operating waveform',
                frequencies_hz=freqs.tolist(), primary_hz=float(freqs[primary]), triangles=len(q.tris),
                edges=q.ne, via_segments=q.nv, layers=layer_names, dmax_mm=args.dmax, fine_mm=args.fine,
                fill=q.fill_info, convergence=convergence, direct_relative_peak_error=direct_errors,
                kcl_residual_a=residuals, timings=timings,
                parts=q.parts, assumptions=dict(capacitor_ESR_ohm=.003, capacitor_ESL_h=.4e-9,
                   inductor_DCR_ohm=.05, inductor_L_h=part_value(parts['L701']), conductor_loss='one-face slab; via DC',
                   dielectric='magneto-quasistatic, explicit copper returns', package_height_mm=0,
                   omitted='inductor winding/core leakage, switch-node E, far-field radiation, nonlinear switching',
                   accuracy='Cropped coarse-mesh illustration; absolute hardware and crop/mesh convergence unvalidated'))
    meta['peak_B_uT_per_A_rms'] = {c: (MU0*1e6*np.linalg.norm(data['H'][k], axis=3)).max(axis=(0, 1)).tolist()
                                   for k, c in enumerate(CASES)}
    (out/'summary.json').write_text(json.dumps(meta, indent=2)+'\n')
    return board, data, meta


def dumps(out, q, board, data):
    from kicad2ems.api.solver import Dump, write_info, write_npz
    from mom25d.farfield import current_deposit
    sim = SimpleNamespace(rwg=q.rwg, tri_idx=q.layer_tris)
    ztop = q.z_of.max()*1e3
    model = SimpleNamespace(outline=shape(board['outline']), cu_z={c['name']: c['z'] for c in board['stack']['copper']},
                            copper={l: [p for c in cs for p in polygons(shape(c['polygon']))]
                                    for l, cs in board['copper'].items()}, ports=[],
                            vias=[(v['x'], v['y'], v['r'], v['z0'], v['z1']) for v in board['vias']], thickness=ztop)
    x, y, f = data['x']*1e3, data['y']*1e3, data['freqs']
    for ci, case in enumerate(CASES):
        target = out/'mqs'/f'exc_{case}';target.mkdir(parents=True, exist_ok=True)
        for obsolete in ('dump_H_plane_fd.npz', 'H@volume.npz'):
            (target/obsolete).unlink(missing_ok=True)
        records = []
        for hi, h in enumerate(data['heights_mm']):
            z = ztop+h
            record = Dump(f'H@top+{h:g}mm', 'H', (x[0], y[0], z), (x[-1], y[-1], z), list(f), 2, f'top+{h:g}mm')
            fields = np.moveaxis(data['H'][ci, :, :, hi], -1, 0)[:, :, :, None, :]
            write_npz(str(target/f'{record.name}.npz'), dict(x=x, y=y, z=np.array([z])), f, fields)
            records.append(record)
        z = ztop+data['volume_heights_mm']
        record = Dump('H@volume', 'H', (x[0], y[0], z[0]), (x[-1], y[-1], z[-1]), [f[int(data['primary'])]], None, 'volume')
        write_npz(str(target/f'{record.name}.npz'), dict(x=x, y=y, z=z), [f[int(data['primary'])]], data['H_volume'][ci][None])
        records.append(record)
        for lay, idx in q.layer_tris.items():
            fields = []
            for fi in range(len(f)):
                jx, jy = current_deposit(sim, data['I'][ci, fi, :q.ne]/q.rwg.l, data['x'], data['y'], lay)
                fields.append(np.stack([jx, jy, np.zeros_like(jx)], -1)[:, :, None])
            z = q.z_of[idx[0]]*1e3
            record = Dump(f'J@{lay}', 'J', (x[0], y[0], z), (x[-1], y[-1], z), list(f), 2, lay)
            write_npz(str(target/f'{record.name}.npz'), dict(x=x, y=y, z=np.array([z])), f, np.array(fields))
            records.append(record)
        write_info(str(target), records, model)
        info_path = target/'dump_info.json';info = json.loads(info_path.read_text())
        for record in info['dumps']:
            record['unit'] = 'A/m RMS per 1 A RMS source harmonic'
        info_path.write_text(json.dumps(info, indent=2))


def render(out, board, data, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors, tri
    from matplotlib.patches import Patch, FancyArrowPatch
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    import io

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 9,
                         'savefig.dpi': 160, 'font.family': 'DejaVu Sans'})
    bounds = meta['crop_model_mm'];xmin, ymin, xmax, ymax = bounds
    parts = {p.ref: p for p in parts_of(board)}
    fidx = int(data['primary'])
    labels = list(data['layer_names'])
    points = data['nodes']*1e3
    cent = points[data['tris']].mean(1)
    J = data['J'][:, fidx]
    global_mag = np.linalg.norm(J, axis=-1)
    jmax = max(1., float(np.percentile(global_mag, 99.5)))
    jnorm = colors.LogNorm(vmin=max(.05, jmax/2000), vmax=jmax, clip=True)
    cmap = plt.get_cmap('viridis')
    pdf = PdfPages(out/'converter_report.pdf')

    def style(ax, title, dark=False):
        ax.set(xlim=(xmin, xmax), ylim=(-ymin, -ymax), aspect='equal', title=title,
               xlabel='KiCad X (mm)', ylabel='KiCad Y (mm)')
        if dark:
            ax.set_facecolor('#12222f')
        ax.tick_params(labelsize=8)

    def outline(ax, layer='F.Cu', color='#d0d6df', alpha=.5):
        for c in board['copper'].get(layer, []):
            if c['net'] not in NETS:
                continue
            for p in polygons(shape(c['polygon'])):
                for ring in [p.exterior, *p.interiors]:
                    coords = np.asarray(ring.coords)
                    ax.plot(coords[:, 0], -coords[:, 1], color=color, alpha=alpha, lw=.45)

    def part_labels(ax, refs, color='white'):
        for ref in refs:
            p = parts[ref]
            ax.text(p.raw['x'], -p.raw['y'], ref, ha='center', va='center', fontsize=7, color=color,
                    bbox=dict(boxstyle='round,pad=.16', fc='#132431' if color == 'white' else 'white', ec='none', alpha=.82))

    def save(fig, name):
        fig.savefig(out/f'{name}.png', bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 7.6))
    ax.set_facecolor('#f1f4f5')
    for c in board['copper']['F.Cu']:
        for p in polygons(shape(c['polygon'])):
            col = COLORS.get(c.get('net'), '#dfdde4')
            ax.fill(*np.c_[np.asarray(p.exterior.coords)[:, 0], -np.asarray(p.exterior.coords)[:, 1]].T,
                    fc=col, ec='white', lw=.4, alpha=.92)
            for ring in p.interiors:
                xy = np.asarray(ring.coords);ax.fill(xy[:, 0], -xy[:, 1], fc='#f1f4f5', ec='none')
    for v in board['vias']:
        if v.get('net') in NETS:
            ax.add_patch(plt.Circle((v['x'], -v['y']), v['r'], fc='none', ec='#253b4e', lw=.6))
    for ref in REFS:
        pp = unary_union([p.geom for p in parts[ref].pads if p.layers])
        x0, y0, x1, y1 = pp.bounds
        ax.plot([x0, x1, x1, x0, x0], [-y0, -y0, -y1, -y1, -y0], color='#364b5c', ls='--', lw=.7)
    part_labels(ax, REFS, color='#0e2638')
    style(ax, 'CM5 Minima • U702 converter crop\nActual top copper, vias and pad extents; six layers retained in the solve')
    ax.legend(handles=[Patch(fc=COLORS[n], label=l) for n, l in [(VIN, '+5 V input'), (SW, 'Switch node'),
                                                               (VOUT, '3.3 V output'), (GND, 'Ground')]],
              loc='lower left', fontsize=8, framealpha=.95)
    fig.text(.13, .018, 'Input: C704/C705/C706 → U702.  Output: U702 → L701 → C708/C709.\n'
             'Coordinates follow KiCad (Y increases downward). Dashed boxes are pad extents, not package body outlines.', fontsize=8)
    fig.subplots_adjust(bottom=.13)
    save(fig, 'converter_crop')

    def current_panel(ax, ci, layer, phase=0., arrows=True, contacts=False, background=True):
        idx = np.flatnonzero(data['layer_id'] == labels.index(layer))
        pc = None
        if background:
            triang = tri.Triangulation(points[:, 0], -points[:, 1], data['tris'][idx])
            pc = ax.tripcolor(triang, facecolors=np.maximum(global_mag[ci, idx], jnorm.vmin),
                              norm=jnorm, cmap=cmap, shading='flat', rasterized=True)
            outline(ax, layer, alpha=.35)
        artists = []
        if arrows:
            # One representative per 0.45 mm cell: avoid a thicket at via fans.
            order = idx[np.argsort(global_mag[ci, idx])[::-1]]
            used, chosen = set(), []
            for t in order:
                cell = tuple(np.floor((cent[t]-[xmin, ymin])/.45).astype(int))
                if cell not in used and global_mag[ci, t] > jnorm.vmin*2:
                    chosen.append(t);used.add(cell)
            chosen = np.array(chosen, int)
            instant = np.real(J[ci, chosen]*np.exp(1j*phase))
            norm = np.linalg.norm(instant, axis=1)
            length = .40*np.sqrt(np.minimum(norm/jmax, 1.5))
            unit = instant/np.maximum(norm[:, None], 1e-30)
            arrows_artist = ax.quiver(cent[chosen, 0], -cent[chosen, 1], unit[:, 0]*length, -unit[:, 1]*length,
                                     angles='xy', scale_units='xy', scale=1, color='white', width=.004,
                                     headwidth=3.5, headlength=4.2, pivot='mid')
            artists.append(arrows_artist)
        if contacts:
            for pi, part in enumerate(meta['parts']):
                if part['name'] not in CASES+['L701', 'C704', 'C705', 'C706', 'C708', 'C709']:
                    continue
                amp = float(np.real(data['I'][ci, fidx, data['part_index'][pi]]*np.exp(1j*phase)))
                if abs(amp) < .015:
                    continue
                a, b = data['part_a'][pi]*1e3, data['part_b'][pi]*1e3
                if amp < 0:
                    a, b = b, a
                arrow = FancyArrowPatch((a[0], -a[1]), (b[0], -b[1]), arrowstyle='-|>', mutation_scale=13,
                                        lw=1.7 if part['kind'] == 'I' else 1.2,
                                        color='#ff738e' if part['kind'] == 'I' else '#ffe07b', zorder=5)
                ax.add_patch(arrow);artists.append(arrow)
        if background:
            style(ax, f'{CASE_TITLES[ci]} • {layer}', dark=True)
        return pc, artists

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4), constrained_layout=True)
    for ci in range(2):
        for li, layer in enumerate(['F.Cu', 'In1.Cu']):
            pc, _ = current_panel(axes[ci, li], ci, layer, contacts=li == 0)
            part_labels(axes[ci, li], ['U702', 'L701', 'C704', 'C705', 'C706', 'C708', 'C709'])
    cb = fig.colorbar(pc, ax=axes.ravel(), shrink=.8, pad=.02)
    cb.set_label('Sheet current |J|, A/m RMS per 1 A RMS drive\nColor clipped at the 99.5th percentile')
    fig.suptitle('Computed copper current loops at 1 MHz • independent 1 A RMS drives\n'
                 'White: phase-zero directions • pink: source • gold: lumped branches', fontsize=14)
    save(fig, 'current_loops')

    field = MU0*1e6*np.linalg.norm(data['H'][..., fidx], axis=4)
    bmax = float(field.max())
    bnorm = colors.LogNorm(vmin=max(bmax/1000, 1e-4), vmax=bmax)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), constrained_layout=True)
    extent = [data['x'][0]*1e3, data['x'][-1]*1e3, -data['y'][0]*1e3, -data['y'][-1]*1e3]
    for ci in range(2):
        for hi, h in enumerate(data['heights_mm']):
            ax = axes[ci, hi]
            im = ax.imshow(field[ci, :, :, hi].T, origin='lower', extent=extent,
                           norm=bnorm, cmap='magma', interpolation='nearest')
            outline(ax, color='#b8d5e2', alpha=.4)
            part_labels(ax, ['U702', 'L701', 'C705', 'C708', 'C709'])
            style(ax, f'{CASE_TITLES[ci]} • {h:g} mm above F.Cu')
            peak = float(field[ci, :, :, hi].max())
            ax.text(.02, .97, f'Peak {peak:.2f} µT/A', transform=ax.transAxes, va='top', color='white',
                    bbox=dict(fc='#172330', alpha=.8, ec='none', pad=3))
    fig.colorbar(im, ax=axes.ravel(), shrink=.8, pad=.02, label='Magnetic flux density |B|, µT RMS per A RMS')
    fig.suptitle('Copper/interconnect magnetic near field at 1 MHz\n'
                 'Independent 1 A RMS drives • inductor coil/core leakage excluded', fontsize=14)
    save(fig, 'magnetic_fields')

    fig, axes = plt.subplots(2, len(labels), figsize=(22, 8), constrained_layout=True)
    for ci in range(2):
        for li, layer in enumerate(labels):
            pc, _ = current_panel(axes[ci, li], ci, layer, contacts=layer == 'F.Cu')
            axes[ci, li].set_title(f'{CASES[ci]} • {layer}', fontsize=10)
    fig.colorbar(pc, ax=axes.ravel(), shrink=.75, pad=.01, label='|J|, A/m RMS per A RMS (same scale on every layer)')
    fig.suptitle('Return-current distribution through all six copper layers • 1 MHz', fontsize=15)
    save(fig, 'all_layer_currents')

    volume = MU0*1e6*np.linalg.norm(data['H_volume'], axis=-1)
    cut_y = -parts['U702'].on({VIN})[0].xy[1]
    yi = int(np.argmin(abs(-data['y']*1e3-cut_y)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ci, ax in enumerate(axes):
        im = ax.pcolormesh(data['x']*1e3, data['volume_heights_mm'], volume[ci, :, yi].T,
                           shading='auto', cmap='magma', norm=colors.LogNorm(vmin=bnorm.vmin, vmax=volume.max()))
        ax.set(xlabel='KiCad X (mm)', ylabel='Height above F.Cu (mm)',
               title=f'{CASE_TITLES[ci]} • vertical slice at KiCad Y = {-data["y"][yi]*1e3:.2f} mm')
    fig.colorbar(im, ax=axes, shrink=.85, label='|B|, µT RMS per A RMS')
    fig.suptitle('Field decay away from the cropped converter • 1 MHz', fontsize=14)
    save(fig, 'field_cross_section')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ci, ax in enumerate(axes):
        for hi, h in enumerate(data['heights_mm']):
            peak = MU0*1e6*np.linalg.norm(data['H'][ci, :, :, hi], axis=2).max(axis=(0, 1))
            ax.loglog(data['freqs'], peak, 'o-', label=f'{h:g} mm above F.Cu')
        ax.set(xlabel='Frequency (Hz)', ylabel='Peak |B| in scan (µT RMS per A RMS)', title=CASE_TITLES[ci])
        ax.grid(True, which='both', alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Sampled magnetic-field transfer • three frequencies, not a switching spectrum', fontsize=13)
    save(fig, 'frequency_response')

    # Phase animation: colors remain RMS current; vectors and signed Bz move.
    for ci, case in enumerate(CASES):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
        _, artists0 = current_panel(axes[0], ci, 'F.Cu', arrows=False)
        _, artists1 = current_panel(axes[1], ci, 'In1.Cu', arrows=False)
        hz = data['H'][ci, :, :, 0, 2, fidx]*MU0*1e6*np.sqrt(2)
        lim = max(float(abs(hz).max()), 1e-10)
        im = axes[2].imshow(np.real(hz).T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-lim, vmax=lim)
        style(axes[2], 'Instantaneous Bz • 1 mm above copper')
        outline(axes[2], alpha=.4)
        fig.colorbar(im, ax=axes[2], shrink=.8, label='Bz (µT), instantaneous')
        title = fig.suptitle('')
        frames = []
        for phase in np.linspace(0, 2*np.pi, 24, endpoint=False):
            for a in artists0+artists1:
                a.remove()
            _, artists0 = current_panel(axes[0], ci, 'F.Cu', phase, contacts=True, background=False)
            _, artists1 = current_panel(axes[1], ci, 'In1.Cu', phase, background=False)
            im.set_data(np.real(hz*np.exp(1j*phase)).T)
            title.set_text(f'{CASE_TITLES[ci]} • 1 MHz • 1 A RMS • phase {np.degrees(phase):.0f}°')
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba[:, :, :3].copy()).convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
        frames[0].save(out/f'{case}_phase.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
        plt.close(fig)
    pdf.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('board', nargs='?')
    ap.add_argument('--out', default='cm5-fields')
    ap.add_argument('--render-only', action='store_true')
    ap.add_argument('--dmax', type=float, default=3.)
    ap.add_argument('--fine', type=float, default=1.2)
    args = ap.parse_args()
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        board = json.loads((out/'converter_board.json').read_text())
        data = dict(np.load(out/'currents_fields.npz'))
        meta = json.loads((out/'summary.json').read_text())
    else:
        if not args.board:
            ap.error('board JSON is required unless --render-only')
        board, data, meta = compute(args, out)
    from mom25d.base import RWG
    rwg = RWG(data['nodes'], data['tris'])
    proxy = SimpleNamespace(rwg=rwg, ne=rwg.Ne, z_of=data['triangle_z'],
                            layer_tris={str(lay): np.flatnonzero(data['layer_id'] == k)
                                        for k, lay in enumerate(data['layer_names'])})
    dumps(out, proxy, board, data)
    from kicad2ems.fields.frames import find_dumps
    found = find_dumps(str(out))
    assert len(found) == 18
    for source in found:
        _, field = source.load(0)
        assert np.isfinite(field).all() and field.shape[-1] == 3
    with (out/'branch_currents.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'frequency_Hz', 'part', 'magnitude_A_RMS', 'phase_deg', 'real_A_RMS', 'imag_A_RMS'])
        for ci, case in enumerate(CASES):
            for fi, freq in enumerate(data['freqs']):
                for pi, part in enumerate(meta['parts']):
                    v = data['I'][ci, fi, data['part_index'][pi]]
                    writer.writerow([case, freq, part['name'], abs(v), np.angle(v, deg=True), v.real, v.imag])
    render(out, board, data, meta)
    report = '''# CM5 U702 converter: current loops and magnetic near field

The saved board JSON is the actual local copper/primitive crop, with source
coordinates and all six layers retained. The solver includes +5V, GND, SW and
M2_3V3 copper; C704/C705/C706, L701 and C708/C709 are explicit series elements.

Plots use independent **1 A RMS harmonic drives**, at **1 MHz**. They are
normalized transfer functions, not an assumed switching spectrum or hardware
measurement. `hot` draws VIN to PGND; `ripple` injects PGND to SW. Phases and
amplitudes can be combined linearly for a specified operating waveform.

- converter_crop.png: actual top copper and converter pad extents.
- current_loops.png: computed top/nearest-ground currents and branch arrows.
- all_layer_currents.png: each source's distribution through all six layers.
- magnetic_fields.png: RMS magnetic field, 1 and 3 mm above top copper.
- field_cross_section.png: vertical decay through the sampled field volume.
- frequency_response.png: field transfer at 100 kHz, 1 MHz and 10 MHz; not an EMI spectrum.
- hot_phase.gif / ripple_phase.gif: instantaneous direction and signed Bz.
- currents_fields.npz: complex branch/sheet currents and vector H data in SI units.
- branch_currents.csv: capacitor/inductor/source current magnitude and phase.
- mqs/: standard kicad2ems current, field-plane and field-volume dumps.
- summary.json: excitation, mesh, convergence, direct-field checks and assumptions.

The inductor's 2.2 uH comes from the board, but its magnetic winding/core
leakage is **not** modelled: field plots show copper/interconnect currents and
straight package filaments collapsed to the pad plane. Capacitor ESR/ESL and
inductor DCR are explicit assumptions. Copper uses one-face slab loss; barrel
resistance is DC. The model omits switch-node E and far-field radiation. The
crop and coarse mesh are illustrative; absolute hardware and crop/mesh
convergence are not established. FFT convergence does not establish those.
'''
    (out/'README.md').write_text(report)
    with zipfile.ZipFile(out/'cm5_converter_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if path.is_file() and path.suffix != '.zip':
                z.write(path, path.relative_to(out))
    print('Results:', out, flush=True)


if __name__ == '__main__':
    main()
