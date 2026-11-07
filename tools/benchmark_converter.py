"""Benchmark a real converter from a kicad2ems board JSON or KiCad file.

CM5 Minima (CERN-OHL-S, KiCad demos), download separately:
https://gitlab.com/kicad/code/kicad/-/raw/master/demos/cm5_minima/CM5_MINIMA_3.kicad_pcb

    python tools/benchmark_converter.py CM5_MINIMA_3.kicad_pcb --out cm5.json

Defaults select U702's +5V/GND input loop and C704/C705/C706. ESR/ESL are
explicit benchmark assumptions, not extracted component measurements.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from shapely.ops import unary_union

from mom25d.mqs import MQS
from mom25d.pdn.rail import parts_of, part_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('board')
    ap.add_argument('--converter', default='U702')
    ap.add_argument('--capacitors', default='C704,C705,C706')
    ap.add_argument('--vin', default='+5V')
    ap.add_argument('--ground', default='GND')
    ap.add_argument('--margin', type=float, default=3, help='crop margin (mm)')
    ap.add_argument('--dmax', type=float, default=1.5)
    ap.add_argument('--fine', type=float, default=.6)
    ap.add_argument('--out', default='converter-benchmark.json')
    ap.add_argument('--fill-backend', default='auto')
    ap.add_argument('--fill-workers', type=int)
    ap.add_argument('--fill-chunk', type=int, default=16384)
    ap.add_argument('--fill-device', type=int)
    args = ap.parse_args()
    path = Path(args.board)
    start = time.perf_counter()
    if path.suffix == '.kicad_pcb':
        from kicad2ems.board import Board
        from kicad2ems.importers.kicad import KiCadBoard
        board = Board(KiCadBoard(str(path)), need_ports=False, log=lambda *_: None).to_dict()
    else:
        board = json.loads(path.read_text())
    parts = {p.ref: p for p in parts_of(board)}
    names = [args.converter] + args.capacitors.split(',')
    terminals = {}
    for ref in names:
        for net, suffix in ((args.vin, 'vin'), (args.ground, 'gnd')):
            pads = [(layer, pad.geom) for pad in parts[ref].on({net}) for layer in pad.layers]
            if not pads:
                raise ValueError(f'{ref}: no pads on {net}')
            terminals[f'{ref}.{suffix}'] = pads
    extent = unary_union([poly for pads in terminals.values() for _, poly in pads])
    region = extent.buffer(args.margin).envelope
    fine = extent.buffer(1.0).envelope
    q = MQS(board, {args.vin, args.ground}, terminals=terminals, region=region,
            dmax=args.dmax, fine=fine, dmax_fine=args.fine, fill_backend=args.fill_backend,
            fill_workers=args.fill_workers, fill_chunk=args.fill_chunk, fill_device=args.fill_device)
    q.source(args.converter, f'{args.converter}.vin', f'{args.converter}.gnd', part_h=.4)
    for ref in names[1:]:
        value = part_value(parts[ref])
        if value is None or value <= 0:
            raise ValueError(f'{ref}: invalid capacitance')
        q.element(ref, f'{ref}.vin', f'{ref}.gnd', C=value, ESR=.003, ESL=.4e-9, part_h=.5)
    setup = time.perf_counter() - start
    start = time.perf_counter()
    r = q.solve([1e3, 1e6, 1e7])
    solve_time = time.perf_counter() - start
    xmin, ymin, xmax, ymax = extent.bounds
    x, y = np.linspace(xmin-1, xmax+1, 17)*1e-3, np.linspace(ymin-1, ymax+1, 17)*1e-3
    heights = q.z_of.max() + np.array([1, 3])*1e-3
    start = time.perf_counter()
    H, info = q.h_grid(r, [0, 1, 2], x, y, heights, rtol=.02, return_info=True)
    fft_time = time.perf_counter() - start
    X, Y, Z = np.meshgrid(x[::4], y[::4], heights, indexing='ij')
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    direct = q.h_field(r, [0, 1, 2], pts).reshape(5, 5, 2, 3, 3)
    report = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  converter=args.converter, capacitors=names[1:], region_mm=region.bounds,
                  dmax_mm=args.dmax, fine_mm=args.fine, layers=list(q.layer_tris), triangles=len(q.tris),
                  edges=q.ne, vias=q.nv, loops=q._mats['C'].shape[1],
                  setup_s=setup, solve_s=solve_time, fft_field_s=fft_time, fill=q.fill_info,
                  field_convergence=info,
                  direct_relative_max_error=float(abs(H[::4, ::4]-direct).max()/abs(direct).max()),
                  copper_matrix_bytes=sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
                                          if hasattr(a, 'indptr') else a.nbytes for a in q._copper),
                  frequencies_hz=r.f.tolist(),
                  loop_R_ohm=r.zc[args.converter].real.tolist(),
                  loop_L_h=(r.zc[args.converter].imag/(2*np.pi*r.f)).tolist(),
                  assumptions=dict(capacitor_ESR_ohm=.003, capacitor_ESL_h=.4e-9,
                                   conductor='DC sheets', cropped_return=True))
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
