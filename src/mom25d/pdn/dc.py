"""
DC voltage drop (IR drop) of a power net and its return: the steady conduction in the copper, each layer's copper
a resistive sheet (sigma x thickness, linear triangles on mom25d's mesh), the vias plated barrels between the
layers they join, the source's pad held at its voltage and each load drawing its current from its pad.

    r = solve(board, '+1V0', source=dict(x=5, y=-20, r=1.5, layer='F.Cu', V=1.0),
              loads=[dict(name='U1', x=40, y=-20, r=1.0, layer='F.Cu', I=4.0)], dmax=1.0)
    r.loads['U1']['V']           # the load's pad voltage (mean over its pad)
    r.sheets['In2.Cu'].V         # node voltages; .J the current per unit width of each triangle, A/mm
    r.vias                       # [dict(x, y, I, layers)] the current each via carries (its largest segment)

The return (ground) is solved alike with the source at 0 V and the loads putting their current back
(solve(..., ground=True)); the voltage a load sees is its power pad's minus its ground pad's
(supply(board, net, ground_net, ...)).

A pad is a disc of radius r around its point: the source holds every mesh vertex of it at V; a load draws
an equal share of its current from each. A via joins the copper of its net on every layer between its ends
that it lies on: from the ring of its radius on one layer to the next, through a barrel of plating thickness
`plating` (mm), its resistance rho dz / area.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from .mesh import net_sheets, stiffness, gradient

SIGMA_CU = 5.8e7                # S/m
PAD_G = 1e3                     # S: a ring vertex to its via's junction (a pad ring of good copper, equipotential)


@dataclass
class DCResult:
    net: str
    sheets: dict                    # {layer: Sheet with .V (node volts), .J (A/mm per triangle), .off (dof offset)}
    vias: list                      # [dict(x, y, r, I, segments=[(layer_a, layer_b, I)])]
    loads: dict                     # {name: dict(V, I, layer)}
    source: dict
    loss: float                     # W in the copper
    nodes: int
    notes: list = field(default_factory=list)


def _barrel_g(r_mm, dz_mm, plating_mm, sigma):
    ri = max(r_mm - plating_mm, 0.0)
    area = np.pi * (r_mm ** 2 - ri ** 2) * 1e-6               # m2
    return sigma * area / (dz_mm * 1e-3)


def solve(board, net, source, loads, dmax=1.0, sigma=SIGMA_CU, plating=0.025, ground=False, log=print):
    """The DC solution of `net` (see the module docstring). source: dict(x, y, r, layer, V); loads: [dict(name, x, y,
    r, layer, I)], amps drawn (ground=True: returned, the source at 0 V)."""
    vias = [v for v in board['vias'] if v.get('net') == net]
    pts = [(source['x'], source['y'], source['r'])] + [(l['x'], l['y'], l['r']) for l in loads] + \
          [(v['x'], v['y'], v['r']) for v in vias]
    nl = 1 + len(loads)
    sheets = net_sheets(board, net, pts, dmax=dmax)
    if not sheets:
        raise ValueError(f"no copper of net {net!r}")
    off = 0
    for sh in sheets.values():
        sh.off = off
        off += len(sh.nodes)
    n_mesh = off
    rows, cols, vals = [], [], []

    def link(a, b, g):
        rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([g, g, -g, -g])

    blocks = []
    for sh in sheets.values():
        K = stiffness(sh.nodes, sh.tris, sigma * sh.t * 1e-3).tocoo()
        blocks.append((K.row + sh.off, K.col + sh.off, K.data))
    # vias: a junction per layer it joins, tied to the ring there, barrels between consecutive junctions
    via_out, junction = [], off
    for k, v in enumerate(vias):
        i = nl + k
        on = sorted([(sh.z, lay) for lay, sh in sheets.items() if sh.at(i) and v['z0'] - 1e-6 <= sh.z <= v['z1'] + 1e-6])
        if len(on) < 2:
            via_out.append(dict(x=v['x'], y=v['y'], r=v['r'], segments=[], I=0.0, junctions=[]))
            continue
        js = []
        for z, lay in on:
            sh = sheets[lay]
            ring = sh.at(i)[1:] or sh.at(i)
            for q in ring:
                link(junction, sh.off + q, PAD_G / len(ring))
            js.append((junction, lay, z))
            junction += 1
        segs = []
        for (ja, la, za), (jb, lb, zb) in zip(js, js[1:]):
            g = _barrel_g(v['r'], zb - za, plating, sigma)
            link(ja, jb, g)
            segs.append((la, lb, ja, jb, g))
        via_out.append(dict(x=v['x'], y=v['y'], r=v['r'], segments=segs, I=0.0, junctions=js))
    n = junction
    r_ = np.concatenate([b[0] for b in blocks] + [np.asarray(rows, int)])
    c_ = np.concatenate([b[1] for b in blocks] + [np.asarray(cols, int)])
    d_ = np.concatenate([b[2] for b in blocks] + [np.asarray(vals, float)])
    G = coo_matrix((d_, (r_, c_)), shape=(n, n)).tocsr()

    src = sheets.get(source['layer'])
    if src is None or not src.at(0):
        raise ValueError(f"source pad at ({source['x']}, {source['y']}) is not on {net} copper on {source['layer']}")
    fixed = np.array([src.off + q for q in src.at(0)])
    V0 = 0.0 if ground else float(source['V'])
    inj = np.zeros(n)
    for j, l in enumerate(loads):
        sh = sheets.get(l['layer'])
        if sh is None or not sh.at(1 + j):
            raise ValueError(f"load {l.get('name', j)} at ({l['x']}, {l['y']}) is not on {net} copper on {l['layer']}")
        ids = np.array([sh.off + q for q in sh.at(1 + j)])
        inj[ids] += (1 if ground else -1) * l['I'] / len(ids)
    # copper not connected to the source: left out (NaN), said
    _, lab = connected_components(G, directed=False)
    live = lab == lab[fixed[0]]
    notes = []
    if not live.all():
        lost = (~live[:n_mesh]).sum()
        notes.append(f"{net}: {lost} mesh vertices on copper not connected to the source (left out)")
        if np.abs(inj[~live]).sum() > 0:
            raise ValueError(f"{net}: a load is on copper not connected to the source")
    free = np.flatnonzero(live & ~np.isin(np.arange(n), fixed))
    V = np.full(n, np.nan)
    V[fixed] = V0
    Gff = G[free][:, free]
    V[free] = spsolve(Gff.tocsc(), inj[free] - G[free][:, fixed] @ V[fixed])
    Vz = np.nan_to_num(V)
    loss = float(Vz @ (G @ Vz))

    for lay, sh in sheets.items():
        sh.V = V[sh.off:sh.off + len(sh.nodes)]
        gv = gradient(sh.nodes, sh.tris, np.nan_to_num(sh.V))      # V/mm
        sh.J = sigma * sh.t * 1e-3 * np.hypot(gv[:, 0], gv[:, 1])   # A per mm of width
    for vo in via_out:
        segs = []
        for la, lb, ja, jb, g in vo.pop('segments'):
            segs.append((la, lb, float(g * (Vz[ja] - Vz[jb]))))
        vo['segments'] = segs
        vo['I'] = max([abs(s[2]) for s in segs], default=0.0)
        vo.pop('junctions')
    out_loads = {}
    for j, l in enumerate(loads):
        sh = sheets[l['layer']]
        out_loads[l.get('name', f'L{j + 1}')] = dict(V=float(np.mean(sh.V[sh.at(1 + j)])), I=l['I'], layer=l['layer'])
    log(f"dc {net}: {n_mesh:,} vertices on {len(sheets)} layer(s), {sum(1 for v in via_out if v['segments'])} vias, "
        f"{loss * 1e3:.2f} mW in the copper")
    return DCResult(net, sheets, via_out, out_loads, dict(source), loss, n_mesh, notes)


def supply(board, net, ground_net, source, loads, ground_source=None, ground_loads=None, **kw):
    """The power net and its return solved, and what each load sees: {name: dict(V, drop, power_drop,
    ground_rise)}, plus the two DCResults. ground_source / ground_loads: the ground pads (default: the power pads'
    positions on the ground net, which a pad usually is not: give them)."""
    p = solve(board, net, source, loads, **kw)
    g = solve(board, ground_net, ground_source or source, ground_loads or loads, ground=True, **kw)
    seen = {}
    for name, lp in p.loads.items():
        lg = g.loads[name]
        seen[name] = dict(V=lp['V'] - lg['V'], drop=source['V'] - (lp['V'] - lg['V']),
                          power_drop=source['V'] - lp['V'], ground_rise=lg['V'], I=lp['I'])
    return seen, p, g
