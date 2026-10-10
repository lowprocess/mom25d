"""
mom25d.api - one call for a program that hands mom25d a geometry file and wants S-parameters back, without
reaching into the Simulation's internals (kicad2openems's MoM solver is the first user of this seam).

    from mom25d.api import solve_csx
    res = solve_csx('board.xml', np.linspace(1e9, 6e9, 401), dmax=0.3e-3)
    res.S, res.names, res.zref        # (nf, n, n) at the ports' reference impedances, port names, ohms
    res.ports, res.unknowns, res.solves, res.seconds, res.notes

The file is an AppCSXCAD / openEMS geometry (.xml), with the optional <mom25d> hints (see mom25d.csx).
Fields of Result are the stable part; Result.model and Result.sim give the objects behind it for anything
else (and change with mom25d).
"""
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
    model: object = None            # the CSXModel
    sim: object = None              # the Simulation


def solve_csx(path, freq, dmax=0.3e-3, plane_factor=3.0, port='auto', pairs=False, mesher='auto', metal='model',
              ground=None, f_loss=None, max_solves=60, afs_tol=5e-3, log=print):
    """S-parameters of the geometry file `path` on `freq` (Hz).

    dmax (m): largest cell along the copper; layers whose metal covers more than half the board (planes) get
    plane_factor x dmax (refined near the copper of other layers). port: 'auto' (de-embedded wave ports where
    the straight feed spans enough for the fit down to the lowest frequency, else lumped), 'wave' or 'via'
    (lumped). pairs: P<n>+ / P<n>- hint ports as one mixed-mode port. metal: 'model' (the file's conducting
    sheets; lossless without), None (lossless) or a sim.Metal. log: progress lines (the adaptive sweep's)."""
    from .csx import CSXModel
    from .fast import _core
    from .geo import LinePort
    from .run import solve_band
    freq = np.asarray(freq, float)
    fmin, fmax = float(freq[0]), float(freq[-1])
    t0 = time.time()
    model = CSXModel(path, ground=ground, port=port, pairs=pairs, f_loss=f_loss or 0.5 * (fmin + fmax), log=log)
    planes = {k: plane_factor * dmax for k, gs in model.shapes.items()
              if sum(g.area for g in gs) > 0.5 * model.outline_area}
    kw = {} if isinstance(metal, str) and metal == 'model' else dict(metal=metal)
    sim = model.simulation(dmax=dmax, dmax_layer=planes, fmin=fmin, fmax=fmax, solver='auto', verbose=False,
                           wave_down_to=fmin, mesher=mesher, title=os.path.basename(path), **kw)
    f, S, info = solve_band(sim, fmin, fmax, points=len(freq), zref=model.port_z0, max_solves=max_solves,
                            afs_tol=afs_tol, log=log)
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
