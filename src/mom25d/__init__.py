"""
mom25d: a 2.5D (planar) Method-of-Moments solver for PCB structures.

    from mom25d.multilayer import Layer, LayerStack     # dielectric stack above a ground plane
    from mom25d.sim import Simulation, Via, ViaPort     # polygons, vias, ports -> S-parameters
    from mom25d.csx import CSXModel                     # an AppCSXCAD / openEMS geometry (.xml) as a model
    from mom25d.kicad import KiCadBoard                 # a KiCad board as a model

Command line: `mom25d board.kicad_pcb ...` or `mom25d model.xml ...` (see mom25d.cli).
"""
__version__ = '0.5.0'
