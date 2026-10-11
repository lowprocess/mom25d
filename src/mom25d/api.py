"""
mom25d.api - one call for a program that hands mom25d a board and wants S-parameters back, without reaching
into the Simulation's internals (kicad2ems's MoM solver is the first user of this seam).

    from mom25d.api import solve_board
    res = solve_board('out/filter_board.json', np.linspace(1e9, 6e9, 401), dmax=0.3e-3)
    res.S, res.names, res.zref        # (nf, n, n) at the ports' reference impedances, port names, ohms
    res.ports, res.unknowns, res.solves, res.seconds, res.notes

The board is kicad2ems's file format (see board.BoardModel): the path of a <name>_board.json, or its content.
Fields of Result are the stable part; Result.model and Result.sim give the objects behind it for anything
else (and change with mom25d).
"""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    f: np.ndarray                   # Hz: the requested frequencies
    S: np.ndarray                   # (nf, n, n) at zref
    names: list                     # port names (a differential pair with pairs=True: P1d / P1c modes)
    zref: list                      # output reference impedance per port, ohms
    ports: list                     # [dict(name, kind='wave' | 'lumped', zc=line impedance of a wave port)]
    unknowns: int                   # RWG edges + via chains
    triangles: int
    solver: str                     # 'dense', 'direct', 'blr', 'aca'
    precision: str                  # 'double' or 'mixed'
    mesher: str                     # what was asked for ('auto': Triangle)
    solves: int                     # EM solves of the sweep
    solved: list                    # the frequencies solved (Hz)
    seconds: float                  # model, mesh, port characterisation and sweep
    core: bool                      # the compiled core was used
    notes: list = field(default_factory=list)
    model: object = None            # the BoardModel
    sim: object = None              # the Simulation


def solve_board(board, freq, dmax=0.3e-3, plane_factor=3.0, port='auto', pairs=False, mesher='auto', metal=None,
                ground=None, max_solves=60, afs_tol=5e-3, log=print, progress=None):
    """S-parameters of a board (a <name>_board.json path, or its content) on `freq` (Hz).

    dmax (m): largest cell along the copper; layers whose metal covers more than half the board (planes) get
    plane_factor x dmax (refined near the copper of other layers). port: 'auto' (de-embedded wave ports where
    the straight feed spans enough for the fit down to the lowest frequency, else lumped), 'wave' or 'via'
    (lumped). pairs: P<n>+ / P<n>- ports as one mixed-mode port. metal: a sim.Metal (conductor loss), or None
    (lossless). log: lines of progress (the adaptive sweep's); progress(solves, max_solves, f): after each
    EM solve."""
    from .board import BoardModel
    from .fast import _core
    from .geo import LinePort
    from .run import solve_band
    name = os.path.basename(board) if isinstance(board, str) else 'board'
    if isinstance(board, str):
        with open(board) as fh:
            board = json.load(fh)
    freq = np.asarray(freq, float)
    fmin, fmax = float(freq[0]), float(freq[-1])
    t0 = time.time()
    model = BoardModel(board, ground=ground, port=port, pairs=pairs, log=log)
    planes = {k: plane_factor * dmax for k, gs in model.shapes.items()
              if sum(g.area for g in gs) > 0.5 * model.outline_area}
    sim = model.simulation(dmax=dmax, dmax_layer=planes, fmin=fmin, fmax=fmax, solver='auto', verbose=False,
                           wave_down_to=fmin, mesher=mesher, title=name, metal=metal)
    f, S, info = solve_band(sim, fmin, fmax, points=len(freq), zref=model.port_z0, max_solves=max_solves,
                            afs_tol=afs_tol, log=log, progress=progress)
    if len(f) != len(freq) or not np.allclose(f, freq):              # the sweep's grid onto the requested one
        S = np.stack([np.stack([np.interp(freq, f, S[:, i, j].real) + 1j * np.interp(freq, f, S[:, i, j].imag)
                                for j in range(S.shape[2])], -1) for i in range(S.shape[1])], 1)
    ports = [dict(name=p.name, kind='wave' if isinstance(p, LinePort) else 'lumped',
                  zc=float(getattr(p, 'zc', 0) or 0)) for p in sim.ports]
    return Result(f=freq, S=S, names=list(info['names']), zref=list(info['zref']), ports=ports,
                  unknowns=int(sim.Ne + sim.Nv), triangles=int(len(sim.rwg.tris)), solver=sim.solver,
                  precision=getattr(sim, 'precision', 'double'), mesher=mesher, solves=int(info['solves']),
                  solved=list(info['solved']), seconds=time.time() - t0, core=_core is not None,
                  notes=list(model.notes) + list(sim.log.notes), model=model, sim=sim)
