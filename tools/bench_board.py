"""Per-stage time and peak memory of mom25d on the demo board (cropped), for optimisation work.
   python tools/mom25d/bench_board.py [solver] [region x0,y0,x1,y1] [precision]"""
import os, sys, time, subprocess
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)
import numpy as np
from kicad2ems.board import Board
from kicad2ems.importers.kicad import KiCadBoard
from mom25d.board import BoardModel


def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM'):
            return int(l.split()[1]) / 2**20


solver = sys.argv[1] if len(sys.argv) > 1 else 'direct'
precision = sys.argv[3] if len(sys.argv) > 3 else 'double'
region = tuple(map(float, sys.argv[2].split(','))) if len(sys.argv) > 2 else (0, 3, 24, 7)
base = os.path.join(ROOT, 'examples', 'demo_transition')
if not os.path.exists(base + '.kicad_pcb'):
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'examples', 'make_demo_board.py'), base])
b = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], ground='B.Cu', pos_file=base + '-top.pos', region=region,
          log=lambda *a: None)
m = BoardModel(b.to_dict(), log=lambda *a: None)
t = time.time()
sim = m.simulation(dmax=0.3e-3, dmax_layer={2: 0.8e-3}, w_ref={1: 0.15e-3, 2: 1e-3, 3: 0.35e-3}, ncross=4,
                   verbose=False, solver=solver, near_fac=3.0, characterize=False, precision=precision)
t_init = time.time() - t
t = time.time(); S = sim.sparams(4e9); t_f = time.time() - t
print(f"solver={solver}/{precision} core={'yes' if getattr(sim, 'fast', None) and sim.fast.core else 'no'} "
      f"N={sim.Ne}+{sim.Nv} init {t_init:.1f}s  per-frequency {t_f:.1f}s  peak RSS {rss():.2f} GB")
for k, v in sim.log.totals().items():
    print(f"   {k:28s} {v:7.1f} s")
np.save(os.path.join(os.path.dirname(__file__), f'bench_S_{solver}_{precision}.npy'), S)
print("S =", np.round(S, 4).tolist())
