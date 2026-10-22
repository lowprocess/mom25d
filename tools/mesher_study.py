"""
Mesher study: accuracy per unknown of mom25d's polygon mesh (Triangle, offset rings at the copper edges) on
a board (a kicad2ems <name>_board.json), against a fine reference mesh.

    python tools/mom25d/mesher_study.py out/board_board.json [--freqs 2.9,3.1,...] [--out study.json]

For every (mesher, dmax, ncross) it reports unknowns, mesh / fill+solve time and the error of S against
the reference: max |dS| (complex) over the frequencies and the error of |S21| in dB where the reference's
|S21| is above -30 dB.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mom25d.board import BoardModel                                # noqa: E402
from mom25d.run import renorm                                      # noqa: E402


def run(board, mesher, dmax, ncross, freqs):
    m = BoardModel(json.load(open(board)), port='via', log=lambda *a: None)
    t0 = time.time()
    sim = m.simulation(dmax=dmax, fmin=freqs[0], fmax=freqs[-1], solver='direct', verbose=False, mesher=mesher,
                       ncross=ncross, characterize=False)
    tm = time.time() - t0
    zr = np.array([m.port_z0[p.name] for p in sim.ports])
    t1 = time.time()
    S = np.array([renorm(sim.sparams(f), np.array([p.zc for p in sim.ports]), zr) for f in freqs])
    return dict(mesher=mesher, dmax_mm=dmax * 1e3, ncross=ncross, unknowns=int(sim.Ne + sim.Nv),
                triangles=int(len(sim.rwg.tris)), t_setup=tm, t_solve=(time.time() - t1) / len(freqs),
                S=[[[complex(v).real, complex(v).imag] for v in row.ravel()] for row in S], shape=list(S.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('board')
    ap.add_argument('--freqs', default='2.0,2.9,3.1,3.2,3.25,3.3,3.4,3.6,5.0')
    ap.add_argument('--ref', default='triangle:0.08:10', help='reference mesher:dmax_mm:ncross')
    ap.add_argument('--cases', default='triangle;0.4,0.3,0.2,0.15;4,6,8')
    ap.add_argument('--out', default='mesher_study.json')
    a = ap.parse_args()
    freqs = np.array([float(v) for v in a.freqs.split(',')]) * 1e9
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r['mesher'], round(r['dmax_mm'], 4), r['ncross']) for r in results}
    rm, rd, rn = a.ref.split(':')
    todo = [(rm, float(rd), int(rn))]
    for grp in a.cases.split('|'):
        ms, ds, ns = grp.split(';')
        todo += [(m_, float(d), int(n)) for m_ in ms.split(',') for d in ds.split(',') for n in ns.split(',')]
    for mesher, d, n in todo:
        if (mesher, round(d, 4), n) in done:
            continue
        print(f"{mesher:12s} dmax {d:.3f} mm ncross {n} ...", flush=True)
        results.append(run(a.board, mesher, d * 1e-3, n, freqs))
        json.dump(results, open(a.out, 'w'), indent=0)
    ref = next(r for r in results if (r['mesher'], round(r['dmax_mm'], 4), r['ncross']) == (rm, round(float(rd), 4), int(rn)))
    S0 = np.array([[complex(*v) for v in row] for row in ref['S']])
    print(f"\nreference {rm} dmax {rd} mm ncross {rn}: {ref['unknowns']} unknowns, {ref['t_solve']:.1f} s per frequency")
    print(f"{'mesher':12s} {'dmax':>5s} {'nx':>3s} {'unknowns':>8s} {'setup s':>7s} {'solve s':>7s} {'max|dS|':>8s} {'dB S21 >-30':>12s}")
    for r in sorted(results, key=lambda r: (r['mesher'], -r['dmax_mm'], r['ncross'])):
        if r is ref:
            continue
        S = np.array([[complex(*v) for v in row] for row in r['S']])
        dS = np.abs(S - S0).max()
        P = int(np.sqrt(S.shape[1]))
        s21, s21r = S[:, P], S0[:, P]                                # (2,1) in row-major ravel
        pb = 20 * np.log10(np.abs(s21r)) > -30                      # where the transmission is meaningful
        ddb = np.abs(20 * np.log10(np.abs(s21[pb]) / np.abs(s21r[pb]))).max() if pb.any() else np.nan
        r['err'] = dict(max_dS=float(dS), max_ddb_s21=float(ddb))
        print(f"{r['mesher']:12s} {r['dmax_mm']:5.2f} {r['ncross']:3d} {r['unknowns']:8d} {r['t_setup']:7.2f} "
              f"{r['t_solve']:7.2f} {dS:8.4f} {ddb:12.3f}")
    json.dump(results, open(a.out, 'w'), indent=0)


if __name__ == '__main__':
    main()
