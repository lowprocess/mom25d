import os, sys, json, time, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, ROOT)
from kicad2ems.board import Board
from kicad2ems.importers.kicad import KiCadBoard
from mom25d.board import BoardModel
base = os.path.join(ROOT, 'examples', 'demo_transition')
D = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], ground='B.Cu', pos_file=base + '-top.pos',
          region=(0, 3, 24, 7), log=lambda *a: None).to_dict()
BASE = dict(dmax=0.3e-3, dmax_layer={2: 0.8e-3}, w_ref={1: 0.15e-3, 2: 1e-3, 3: 0.35e-3}, ncross=4, near_fac=3.0)
def make(**kw):
    a = dict(BASE); a.update(kw)
    m = BoardModel(D, log=lambda *a: None)
    return m.simulation(verbose=False, solver='direct', characterize=False, **a)
VARIANTS = {
  'default (triangle, ncross 4)': {},
  'gmsh (layers)': dict(mesher='gmsh'),
  'gmsh-rings': dict(mesher='gmsh-rings'),
  'gmsh-graded': dict(mesher='gmsh-graded'),
  'grade 1.5': dict(grade=1.5),
  'grade 2': dict(grade=2.0),
  'ncross 3': dict(ncross=3),
  'ncross 2': dict(ncross=2),
  'dmax 0.4': dict(dmax=0.4e-3),
  'dmax 0.5': dict(dmax=0.5e-3),
  'ground layer 1.2 mm': dict(dmax_layer={2: 1.2e-3}),
  'REF dmax 0.2 ncross 6': dict(dmax=0.2e-3, dmax_layer={2: 0.5e-3}, ncross=6),
}
if __name__ == '__main__':
    for name, kw in VARIANTS.items():
        try:
            s = make(**kw)
            per = {int(k): int((s.edge_layer == k).sum()) for k in s.layers}
            print(f"{name:28s} N={s.Ne + s.Nv:6d}  per layer {per}  {s.mesher_used}", flush=True)
        except Exception as e:
            print(f"{name:28s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
