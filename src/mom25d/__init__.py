"""
mom25d: a 2.5D (planar) Method-of-Moments solver for PCB structures.

    from mom25d.multilayer import Layer, LayerStack     # dielectric stack above a ground plane
    from mom25d.sim import Simulation, Via, ViaPort     # polygons, vias, ports -> S-parameters
    from mom25d.board import BoardModel                 # a kicad2ems board (<name>_board.json) as a model
    from mom25d.api import solve_board                  # a board -> S-parameters, one call

Boards reach it through kicad2ems (`kicad2ems board.kicad_pcb --solvers mom`), which reads KiCad and
Gerber files and writes the board every solver gets; mom25d reads that file without kicad2ems.
"""
__version__ = '261028-r1'

import os as _os

# the compiled core's OpenMP threads sleep between its parallel regions instead of spinning: spinning, they hold the
# cores the LU's OpenBLAS threads need (rf_stubnotch, 1.5k unknowns: the LU 0.22 s a solve against 0.026 s alone,
# 7 s with OMP_WAIT_POLICY=ACTIVE). Read when libgomp loads (the core's first import); a value set already wins
_os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
