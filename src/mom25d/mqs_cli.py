"""Converter input-loop near fields, from KiCad or a kicad2ems board JSON."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union

OPTIONS = [
    ('converter', 'str', '', 'Regulator reference; empty selects a unique discovered converter', 'Converter'),
    ('inductor', 'str', '', 'Power inductor reference override', 'Converter'),
    ('vin', 'str', '', 'Input supply net override', 'Converter'),
    ('ground', 'str', 'GND', 'Ground net', 'Converter'),
    ('capacitors', 'str', '', 'Input capacitors, comma-separated; empty discovers nearby parts', 'Converter'),
    ('cap_radius', 'float', 8., 'Maximum distance from regulator to input capacitors, mm', 'Converter'),
    ('fsw', 'float', 500e3, 'Switching frequency, Hz', 'Operating point'),
    ('vin_volts', 'float', 0., 'Input voltage; with vout sets duty (0 uses duty)', 'Operating point'),
    ('vout', 'float', 0., 'Output voltage; with vin-volts sets duty', 'Operating point'),
    ('duty', 'float', .33, 'Duty fraction for assumed rectangular input-current pulse', 'Operating point'),
    ('current', 'float', 1., 'Pulse amplitude, A; approximates buck load current', 'Operating point'),
    ('edge_time', 'float', 5e-9, 'Edge smoothing time, s', 'Operating point'),
    ('harmonics', 'str', '1,2,3,4,5,6', 'Positive harmonic numbers, comma-separated', 'Operating point'),
    ('heights', 'str', '1,3', 'Scan heights above top copper, mm; 2+ heights also give a volume', 'Scan'),
    ('scan_points', 'int', 25, 'Points along each lateral scan axis', 'Scan'),
    ('field_tol', 'float', .02, 'Successive FFT-grid peak-relative tolerance per height/frequency', 'Scan'),
    ('margin', 'float', 3., 'Copper crop margin around selected component pads, mm', 'Mesh'),
    ('dmax', 'float', 1.5, 'Nominal coarse mesh spacing, mm', 'Mesh'),
    ('fine', 'float', .6, 'Nominal spacing near selected component pads, mm', 'Mesh'),
    ('fill_backend', 'str', 'auto', 'auto, openmp, threaded, numpy or cupy', 'Mesh'),
    ('fill_workers', 'int', 0, 'CPU fill workers (0 chooses automatically)', 'Mesh'),
    ('fill_chunk', 'int', 16384, 'Maximum triangle pairs per fill tile', 'Mesh'),
    ('fill_device', 'int', 0, 'CuPy device index', 'Mesh'),
    ('esr', 'float', .003, 'Assumed ESR per input capacitor, ohms', 'Parts'),
    ('esl', 'float', .4e-9, 'Assumed ESL per input capacitor, H', 'Parts'),
    ('part_height', 'float', .5, 'Assumed part filament height above pads, mm', 'Parts'),
]


def describe():
    from . import __version__
    return dict(format=1, version=__version__, options=[dict(key=k, flag='--'+k.replace('_','-'), kind=t,
                default=d, help=h, group=g, choices=None) for k,t,d,h,g in OPTIONS])


def parser():
    ap=argparse.ArgumentParser(prog='python -m mom25d.mqs', description=__doc__)
    ap.add_argument('board', nargs='?')
    ap.add_argument('--out', default='mqs')
    ap.add_argument('--describe', action='store_true')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--list-converters', action='store_true', help='describe candidate converters without meshing')
    for k,t,d,h,g in OPTIONS:
        ap.add_argument('--'+k.replace('_','-'),type={'str':str,'int':int,'float':float}[t],default=d,help=h)
    return ap


def load_board(path):
    path=Path(path)
    if path.suffix=='.json':
        return json.loads(path.read_text())
    from kicad2ems.board import Board
    from kicad2ems.importers import open_board
    return Board(open_board(str(path)), need_ports=False, log=lambda *_:None).to_dict()


def discover(board, ground='GND'):
    """Structural candidates only: an inductor-to-IC switch net, and VIN/GND capacitors.

    Ambiguity is returned, never resolved by picking the first net or part.
    """
    from .pdn.rail import parts_of, classify
    parts=parts_of(board)
    regulators=[p for p in parts if p.ref.upper().startswith('U') and ground in p.nets()]
    caps=[p for p in parts if p.ref.upper().startswith('C') and ground in p.nets()]
    candidates=[]
    for ind in parts:
        kind=classify(ind)
        if not kind or kind[0]!='inductor':
            continue
        for reg in regulators:
            switch=(ind.nets() & reg.nets())-{ground}
            for sw in sorted(switch):
                inputs=[]
                for net in sorted(reg.nets()-{ground,sw}-ind.nets()):
                    cc=[p for p in caps if net in p.nets()]
                    if cc:
                        inputs.append(dict(vin=net,capacitors=[p.ref for p in cc]))
                if inputs:
                    candidates.append(dict(converter=reg.ref,inductor=ind.ref,switch=sw,inputs=inputs))
    return candidates


def select(board, a):
    from .pdn.rail import parts_of, part_value
    parts={p.ref:p for p in parts_of(board)}
    candidates=[c for c in discover(board,a.ground) if (not a.converter or c['converter']==a.converter)
                and (not a.inductor or c['inductor']==a.inductor)]
    if a.converter and a.vin:
        ref=a.converter  # Explicit input-loop selection also works without a recognizable switch/inductor.
    else:
        refs=sorted({c['converter'] for c in candidates})
        if len(refs)!=1:
            raise ValueError('select --converter and --vin; candidates: '+(', '.join(refs) or 'none'))
        ref=refs[0]
    if ref not in parts:
        raise ValueError(f'unknown converter {ref}')
    reg=parts[ref]
    centre=unary_union([p.geom for p in reg.pads]).centroid
    local=[p for p in parts.values() if p.ref.upper().startswith('C') and a.ground in p.nets()
           and min((pad.geom.distance(centre) for pad in p.pads),default=float('inf'))<=a.cap_radius]
    if a.vin:
        vin=a.vin
    else:
        options={i['vin'] for c in candidates for i in c['inputs']
                 if any(i['vin'] in p.nets() for p in local)}
        if len(options)!=1:
            raise ValueError('ambiguous input supply; set --vin from: '+', '.join(sorted(options)))
        vin=options.pop()
    caps=[x.strip() for x in a.capacitors.split(',') if x.strip()] if a.capacitors else sorted(p.ref for p in local if vin in p.nets())
    if not caps or len(caps)!=len(set(caps)):
        raise ValueError('select one or more unique --capacitors')
    if vin==a.ground:
        raise ValueError('input supply and ground must differ')
    terms={}
    for name in [ref,*caps]:
        if name not in parts:
            raise ValueError(f'unknown part {name}')
        for net in [vin,a.ground]:
            pads=[(layer,p.geom) for p in parts[name].on({net}) for layer in p.layers]
            if not pads:
                raise ValueError(f'{name} has no copper pads on {net}')
            terms[f'{name}:{net}']=pads
    values={c:part_value(parts[c]) for c in caps}
    if any(v is None or v<=0 for v in values.values()):
        raise ValueError('each selected capacitor needs a positive capacitance value')
    return dict(converter=ref,vin=vin,ground=a.ground,capacitors=caps,values=values),terms


def numbers(text, name):
    try:
        x=np.array([float(t) for t in text.split(',')])
    except ValueError as exc:
        raise ValueError(f'{name} must be comma-separated numbers') from exc
    if not len(x) or not np.isfinite(x).all() or (x<=0).any() or len(np.unique(x))!=len(x):
        raise ValueError(f'{name} must contain unique positive finite values')
    return np.sort(x)


def validate(a):
    for k in ['fsw','cap_radius','dmax','fine','margin','field_tol']:
        if not np.isfinite(getattr(a,k)) or getattr(a,k)<=0:
            raise ValueError(f'{k} must be positive and finite')
    for k in ['current','edge_time','esr','esl','part_height','vin_volts','vout']:
        if not np.isfinite(getattr(a,k)) or getattr(a,k)<0:
            raise ValueError(f'{k} must be nonnegative and finite')
    if bool(a.vin_volts)!=bool(a.vout):
        raise ValueError('set both --vin-volts and --vout')
    duty=a.vout/a.vin_volts if a.vin_volts else a.duty
    if not np.isfinite(duty) or not 0<duty<1:
        raise ValueError('duty must be between 0 and 1')
    if a.scan_points<2 or a.fill_workers<0 or a.fill_chunk<1 or a.fill_device<0:
        raise ValueError('invalid scan points or fill controls')
    n=numbers(a.harmonics,'harmonics')
    if not (n==np.floor(n)).all():
        raise ValueError('harmonics must be integers')
    return n,numbers(a.heights,'heights'),duty


def write_outputs(out, board, q, r, H, x, y, heights, summary):
    from kicad2ems.api.solver import Dump, write_info, write_npz
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exc=out/'mqs'/'exc_converter'
    exc.mkdir(parents=True,exist_ok=True)
    top=max(c['z'] for c in board['stack']['copper'])
    z=top+heights
    dumps=[]
    for k,h in enumerate(heights):
        label=f'top+{h:g}mm'
        dm=Dump('H@'+label,'H',(x[0],y[0],z[k]),(x[-1],y[-1],z[k]),r.f.tolist(),2,label)
        write_npz(exc/(dm.name+'.npz'),dict(x=x,y=y,z=z[k:k+1]),r.f,np.moveaxis(H[:,:,k:k+1],-1,0))
        dumps.append(dm)
    if len(z)>1:
        dm=Dump('H@volume','H',(x[0],y[0],z[0]),(x[-1],y[-1],z[-1]),r.f.tolist(),None,'volume')
        write_npz(exc/(dm.name+'.npz'),dict(x=x,y=y,z=z),r.f,np.moveaxis(H,-1,0))
        dumps.append(dm)
    copper={}
    for lay,records in board['copper'].items():
        polys=[]
        for record in records:
            if record['net'] not in q.nets:continue
            g=shape(record['polygon']).intersection(shape(summary['crop']))
            polys.extend(p for p in getattr(g,'geoms',[g]) if p.geom_type=='Polygon' and not p.is_empty)
        copper[lay]=polys
    m=SimpleNamespace(outline=shape(summary['crop']),cu_z={c['name']:c['z'] for c in board['stack']['copper']},
        copper=copper,ports=[],vias=[(v['x'],v['y'],v['r'],v['z0'],v['z1']) for v in board['vias']
                                    if v['net'] in q.nets and shape(summary['crop']).covers(shape({'type':'Point','coordinates':[v['x'],v['y']]}))],
        thickness=board['stack'].get('thickness',top))
    # The frontend also discovers files absent from the metadata. Remove only
    # obsolete generated H dumps in this command's reserved excitation folder.
    generated={d.name+'.npz' for d in dumps}
    for stale in exc.glob('dump_H_*_fd.npz'):
        if stale.name not in generated:
            stale.unlink()
    write_info(str(exc),dumps,m)
    mag=np.linalg.norm(H,axis=3)
    figures=[]
    for k,h in enumerate(heights):
        fig,ax=plt.subplots()
        im=ax.pcolormesh(x,y,mag[:,:,k,0].T,shading='auto');fig.colorbar(im,ax=ax,label='|H|, A/m peak')
        ax.set(xlabel='mm',ylabel='mm',aspect='equal',title=f'{r.f[0]:g} Hz, {h:g} mm above top copper')
        name=f'H_top_{h:g}mm.png';fig.tight_layout();fig.savefig(out/name,dpi=120);plt.close(fig)
        figures.append(name)
    fig,ax=plt.subplots();ax.loglog(r.f,np.max(mag,axis=(0,1,2)),marker='o')
    ax.set(xlabel='Hz',ylabel='Peak |H| across scan, A/m');ax.grid(True)
    fig.tight_layout();fig.savefig(out/'spectrum.png',dpi=120);plt.close(fig)
    summary['figures']=figures+['spectrum.png']
    np.savez_compressed(out/'mesh.npz',nodes=q.nodes,tris=q.tris,layer_of=q.layer_of)


def run(a):
    from .mqs import MQS
    from shapely.geometry import mapping
    start=time.perf_counter()
    n,heights,duty=validate(a)
    print('mqs: reading board',flush=True)
    board=load_board(a.board)
    selected,terms=select(board,a)
    extent=unary_union([p for pp in terms.values() for _,p in pp])
    crop=extent.buffer(a.margin).envelope
    print(f"mqs: {selected['converter']}, {selected['vin']}, capacitors {', '.join(selected['capacitors'])}",flush=True)
    q=MQS(board,{selected['vin'],a.ground},terminals=terms,region=crop,dmax=a.dmax,
          fine=extent.buffer(1).envelope,dmax_fine=a.fine,fill_backend=a.fill_backend,
          fill_workers=a.fill_workers or None,fill_chunk=a.fill_chunk,fill_device=a.fill_device if a.fill_backend=='cupy' else None)
    ref=selected['converter'];vin=selected['vin']
    q.source(ref,f'{ref}:{vin}',f'{ref}:{a.ground}',part_h=a.part_height)
    for cap,value in selected['values'].items():
        q.element(cap,f'{cap}:{vin}',f'{cap}:{a.ground}',C=value,ESR=a.esr,ESL=a.esl,part_h=a.part_height)
    r=q.solve(n*a.fsw)
    impedance=r.zc[ref].copy()
    amps=2*a.current*duty*np.abs(np.sinc(n*duty))*np.abs(np.sinc(n*a.fsw*a.edge_time))
    r.i*=amps[:,None]
    moments=np.array([q.moment(r,i) for i in range(len(n))])
    solve_s=time.perf_counter()-start
    xmin,ymin,xmax,ymax=crop.bounds
    x=np.linspace(xmin,xmax,a.scan_points);y=np.linspace(ymin,ymax,a.scan_points)
    top=max(c['z'] for c in board['stack']['copper'])
    H,info=q.h_grid(r,list(range(len(n))),x*.001,y*.001,(top+heights)*.001,rtol=a.field_tol,return_info=True)
    summary=dict(format=1,selection=selected,settings={k:getattr(a,k) for k,*_ in OPTIONS},crop=mapping(crop),
        frequencies_Hz=r.f.tolist(),harmonic_peak_A=amps.tolist(),copper_R_ohm=impedance.real.tolist(),
        copper_L_H=(impedance.imag/(2*np.pi*r.f)).tolist(),dipole_real_Am2=moments.real.tolist(),dipole_imag_Am2=moments.imag.tolist(),
        heights_above_top_mm=heights.tolist(),peak_H_Am=np.linalg.norm(H,axis=3).max(axis=(0,1)).T.tolist(),
        triangles=len(q.tris),fill=q.fill_info,field_convergence=info,
        notes=['Input-loop magnetic model; switch-node E and output-inductor leakage are excluded.',
               'Pulse spectrum, ESR, ESL and part filament heights are explicit assumptions.',
               'FFT convergence does not establish copper mesh or crop convergence.'],
        seconds=dict(solve=solve_s,total=time.perf_counter()-start))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_outputs(out,board,q,r,H,x,y,heights,summary)
    summary['seconds']['total']=time.perf_counter()-start
    (out/'mqs.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'mqs: wrote {out / "mqs.json"}',flush=True)
    return summary


def main(argv=None):
    ap=parser();a=ap.parse_args(argv)
    if a.describe:
        d=describe();print(json.dumps(d) if a.json else '\n'.join(f"{o['flag']}: {o['help']}" for o in d['options']));return 0
    if not a.board:ap.error('a board is required')
    try:
        if a.list_converters:
            print(json.dumps(discover(load_board(a.board),a.ground),indent=2))
        else:run(a)
    except (ValueError,KeyError,OSError,RuntimeError) as exc:
        print(f'mqs: error: {exc}',file=sys.stderr);return 2
    return 0
