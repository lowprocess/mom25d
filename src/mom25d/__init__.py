"""
mom25d: a 2.5D (planar) Method-of-Moments solver for PCB structures.

    from mom25d.multilayer import Layer, LayerStack     # dielectric stack above a ground plane
    from mom25d.sim import Simulation, Via, ViaPort     # polygons, vias, ports -> S-parameters
    from mom25d.board import BoardModel                 # a kicad2ems board (<name>_board.json) as a model
    from mom25d.api import solve_board                  # a board -> S-parameters, one call

Boards reach it through kicad2ems (`kicad2ems board.kicad_pcb --solvers mom`), which reads KiCad and
Gerber files and writes the board every solver gets; mom25d reads that file without kicad2ems.
"""
__version__ = '261019-r2'
