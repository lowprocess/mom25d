"""
mom25d as a kicad2ems solver plugin, found through the 'kicad2ems.solvers' entry points of mom25d's pyproject:

    mom          MoM25D, the solver 'mom25d' (alias mom): the board of the run solved by mom25d
    mixed        Mixed, the solver 'mixed': mom25d for the board, its long straight runs as transmission lines

Only kicad2ems imports these (it must be installed); mom25d itself never does.
"""
