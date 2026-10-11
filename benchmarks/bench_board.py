"""Per-stage time and peak memory on the demo KiCad board (cropped), for optimisation work.
   python benchmarks/bench_board.py [solver] [region]"""
import os, sys, time, resource, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from mom25d.kicad import KiCadBoard

def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM'):
            return int(l.split()[1]) / 2**20
solver = sys.argv[1] if len(sys.argv) > 1 else 'direct'
precision = sys.argv[3] if len(sys.argv) > 3 else 'double'
region = tuple(map(float, sys.argv[2].split(','))) if len(sys.argv) > 2 else (0, 3, 24, 7)
ex = os.path.join(os.path.dirname(__file__), '..', 'examples')
if not os.path.exists(os.path.join(ex, 'demo_transition.kicad_pcb')):
    subprocess.check_call([sys.executable, os.path.join(ex, 'make_demo_board.py'), os.path.join(ex, 'demo_transition')])
m = KiCadBoard(os.path.join(ex, 'demo_transition.kicad_pcb')).build(
    nets=['RF', 'GND'], ground='B.Cu', pos_file=os.path.join(ex, 'demo_transition-top.pos'),
    include_mask=False, region=region)
t = time.time()
sim = m.simulation(dmax=0.3e-3, dmax_layer={2: 0.8e-3}, w_ref={1: 0.15e-3, 2: 1e-3, 3: 0.35e-3}, ncross=4,
                   verbose=False, solver=solver, near_fac=3.0, characterize=False, precision=precision)
t_init = time.time() - t
t = time.time(); S = sim.sparams(4e9); t_f = time.time() - t
print(f"solver={solver}/{precision} core={'yes' if getattr(sim, 'fast', None) and sim.fast.core else 'no'} N={sim.Ne}+{sim.Nv} init {t_init:.1f}s  per-frequency {t_f:.1f}s  peak RSS {rss():.2f} GB")
for k, v in sim.log.totals().items():
    print(f"   {k:28s} {v:7.1f} s")
np.save(os.path.join(os.path.dirname(__file__), f'bench_S_{solver}_{precision}.npy'), S)
print("S =", np.round(S, 4).tolist())
