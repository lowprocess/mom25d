"""
PDN impedance from a plane pair: the power plane over its ground plane as a 2D cavity (the voltage between
the planes, V(x, y), on linear triangles over the area where both are copper), with the decoupling capacitors, the
regulator and the load attached at their vias as ports. DC to the first cavity modes and beyond, as long
as the planes are close compared with the wavelength (d << lambda): the field between them is then uniform
across d.

    pdn = PlanePair(board, power='+1V0', ground='GND', power_layer='In2.Cu', ground_layer='In1.Cu',
                    ports=[dict(name='U1', x=40, y=-20, r=0.15),
                           dict(name='C1', x=36, y=-16, r=0.15, C=10e-6, ESR=3e-3, ESL=0.4e-9),
                           dict(name='VRM', x=5, y=-20, r=0.15, R=1e-3, L=20e-9)], dmax=1.0)
    f = np.logspace(3, 9, 241)
    Z = pdn.z(f)                      # (nf, nports, nports): the port impedance matrix with every element attached
    Z0 = pdn.z(f, bare=True)          # the planes alone

The planes, per unit area and per square:
  shunt     Y = j w C' (1 - j tand),  C' = eps0 / sum(d_i / er_i)       (the dielectric between them)
  series    Z = 2 Rs(f) + j w mu0 d                                    (both planes' surface resistance, the
                                                                         inductance of the space between)
  -div(grad V / Z) + Y V = J (the current into the cavity at the ports); the edges are open (no current out).
Rs(f) = 1 / (sigma delta (1 - exp(-t / delta))): 1 / (sigma t) at DC, the skin resistance once delta << t.

A port is the vertices of a disc of radius r around its point, held at one voltage (a via through the cavity).
Its element (a decap: C, ESR, ESL, mount; a regulator: R, L; a load: none) is a series impedance between the
planes there. `mount` (H) is the inductance of the vias from the part's pads down to the cavity, added to its
ESL: mount_inductance(h, pitch, r) estimates it for a via pair.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from ..base import MU0, EPS0
from .mesh import copper_of, mesh_sheet, stiffness, mass, stack_z

SIGMA_CU = 5.8e7


def fill_holes(geom, area, cover=None):
    """geom with its holes smaller than `area` (mm2) filled, and those `cover` (shapely: the antipads of the vias
    through it, a field of them merged into one hole) covers but for 5% of their area."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [geom] if geom.geom_type == 'Polygon' else [g for g in geom.geoms if g.geom_type == 'Polygon']

    def keep(h):
        hp = Polygon(h)
        if hp.area < area:
            return False
        return not (cover is not None and hp.difference(cover).area < 0.05 * hp.area)
    return unary_union([Polygon(p.exterior, [h for h in p.interiors if keep(h)]) for p in polys])


def mount_inductance(h, pitch, r):
    """The loop inductance (H) of a power-ground via pair of height h, centres `pitch` apart, radius r (any one unit):
    mu0 h / pi * ln(pitch / r), two parallel wires shorted at one end."""
    return MU0 * (h * 1e-3) / np.pi * np.log(pitch / r)


def surface_resistance(f, t_m, sigma=SIGMA_CU):
    f = np.maximum(np.asarray(f, float), 1e-3)
    delta = np.sqrt(1.0 / (np.pi * f * MU0 * sigma))
    return 1.0 / (sigma * delta * (1 - np.exp(-t_m / delta)))


def element_z(p, w):
    """The series impedance of a port's element at angular frequency w (np.inf: none)."""
    if 'C' in p:
        return p.get('ESR', 0.0) + 1j * w * (p.get('ESL', 0.0) + p.get('mount', 0.0)) + 1.0 / (1j * w * p['C'])
    if 'R' in p or 'L' in p:
        return p.get('R', 0.0) + 1j * w * (p.get('L', 0.0) + p.get('mount', 0.0))
    return np.inf


class PlanePair:
    def __init__(self, board=None, power=None, ground=None, power_layer=None, ground_layer=None, ports=(), dmax=1.0,
                 geom=None, d=None, er=None, tand=0.0, t=0.035, sigma=SIGMA_CU, hole_fill=2.0, log=print):
        """From a board (board.json's dict): the overlap of `power` on power_layer and `ground` on ground_layer
        (holes in either smaller than hole_fill mm2 filled: the antipads), the dielectric between from the
        stack. Or a geometry of its own: geom (shapely, mm), d (mm), er, tand, t."""
        if geom is None:
            zt = stack_z(board)
            gp, gg = copper_of(board, power, power_layer), copper_of(board, ground, ground_layer)
            if gp is None or gg is None:
                raise ValueError(f"no {power} on {power_layer} or no {ground} on {ground_layer}")
            # the antipads (the holes vias pass through) filled: a via through the cavity is a port at its point,
            # and a small hole changes the plane's inductance little
            geom = fill_holes(gp, hole_fill).intersection(fill_holes(gg, hole_fill)).buffer(0)
            za, zb = sorted([zt[power_layer][0], zt[ground_layer][0]])
            slabs = [(max(l['z0'], za), min(l['z1'], zb), l['er'], l.get('tand', 0.0))
                     for l in board['stack']['layers'] if min(l['z1'], zb) > max(l['z0'], za)]
            d = sum(b - a for a, b, _, _ in slabs)
            er = d / sum((b - a) / e for a, b, e, _ in slabs)                  # the series (effective) permittivity
            tand = sum((b - a) / e * tn for a, b, e, tn in slabs) / sum((b - a) / e for a, b, e, _ in slabs)
            t = min(zt[power_layer][1], zt[ground_layer][1])
        self.geom, self.d, self.er, self.tand, self.t, self.sigma = geom, d, er, tand, t, sigma
        self.ports = [dict(p) for p in ports]
        pts = [(p['x'], p['y'], p['r']) for p in self.ports]
        nodes, tris, rings, _ = mesh_sheet(geom, pts, dmax)
        missing = [p['name'] for p, r in zip(self.ports, rings) if r is None]
        if missing:
            raise ValueError(f"port(s) {', '.join(missing)} not on the plane pair")
        self.nodes, self.tris = nodes, tris
        # one unknown per port (its disc held at one voltage), one per other vertex
        dof = -np.ones(len(nodes), int)
        for k, r in enumerate(rings):
            dof[r] = k
        rest = np.flatnonzero(dof < 0)
        dof[rest] = len(self.ports) + np.arange(len(rest))
        self.dof, self.n = dof, len(self.ports) + len(rest)
        P = coo_matrix((np.ones(len(nodes)), (np.arange(len(nodes)), dof)), shape=(len(nodes), self.n)).tocsr()
        self.K = (P.T @ stiffness(nodes, tris) @ P).tocsc()                    # dimensionless (2D)
        self.M = (P.T @ mass(nodes, tris) @ P).tocsc() * 1e-6                  # m2
        self.area = float(self.M.sum())
        self.C = EPS0 * er * self.area / (d * 1e-3)
        log(f"plane pair: {geom.area:.0f} mm2, d {d:.3g} mm, er {er:.3g}, C {self.C * 1e9:.3g} nF; "
            f"{len(nodes):,} vertices, {len(self.ports)} ports")

    def y_planes(self, f):
        w = 2 * np.pi * f
        zs = 2 * surface_resistance(f, self.t * 1e-3, self.sigma) + 1j * w * MU0 * self.d * 1e-3
        ysh = 1j * w * EPS0 * self.er * (1 - 1j * self.tand) / (self.d * 1e-3)
        return self.K / zs + self.M * ysh

    def z(self, f, bare=False, observe=None):
        """(nf, k, k) impedance matrix between the planes at the ports (observe: their names, default every port),
        with each port's element attached (bare: the planes alone)."""
        f = np.atleast_1d(np.asarray(f, float))
        names = [p['name'] for p in self.ports]
        obs = [names.index(o) for o in (observe or names)]
        out = np.zeros((len(f), len(obs), len(obs)), complex)
        E = np.zeros((self.n, len(obs)))
        E[obs, np.arange(len(obs))] = 1.0
        for i, fi in enumerate(f):
            Y = self.y_planes(fi)
            if not bare:
                y = np.zeros(self.n, complex)
                for k, p in enumerate(self.ports):
                    ze = element_z(p, 2 * np.pi * fi)
                    if np.isfinite(ze):
                        y[k] = 1.0 / ze
                Y = Y + diags(y)
            X = splu(Y.tocsc()).solve(E.astype(complex))
            out[i] = X[obs]
        return out


def rect_cavity_z(a, b, d, er, tand, ports, f, modes=80, sigma=SIGMA_CU, t=0.035):
    """The analytic impedance matrix (nf, k, k) of a rectangular plane pair a x b (mm, corner at the origin),
    spacing d (mm), between square ports (x, y, w) (mm): the cavity's double sum over its modes (m, n),
        Z_ij = j w mu0 d / (a b) sum_mn chi_m^2 chi_n^2 / (k_mn^2 - k^2) f_mn(i) f_mn(j),
        f_mn(p) = cos(m pi x / a) cos(n pi y / b) sinc(m pi w / 2a) sinc(n pi w / 2b),
    chi_0 = 1, chi_m = sqrt(2); k^2 = w^2 mu eps (1 - j tand) (1 - j 2 Rs / (w mu0 d)) (the planes' loss)."""
    f = np.atleast_1d(np.asarray(f, float))
    a, b, d = a * 1e-3, b * 1e-3, d * 1e-3
    m = np.arange(modes)
    chi = np.where(m == 0, 1.0, np.sqrt(2.0))
    kx, ky = m * np.pi / a, m * np.pi / b
    kmn2 = kx[:, None] ** 2 + ky[None, :] ** 2
    F = []
    for x, y, w in ports:
        x, y, w = x * 1e-3, y * 1e-3, w * 1e-3
        fx = chi * np.cos(kx * x) * np.sinc(kx * w / 2 / np.pi)
        fy = chi * np.cos(ky * y) * np.sinc(ky * w / 2 / np.pi)
        F.append(fx[:, None] * fy[None, :])
    F = np.array(F)                                                      # (k, m, n)
    out = np.zeros((len(f), len(ports), len(ports)), complex)
    for i, fi in enumerate(f):
        w = 2 * np.pi * fi
        rs = surface_resistance(fi, t * 1e-3, sigma)
        k2 = w ** 2 * MU0 * EPS0 * er * (1 - 1j * tand) * (1 - 1j * 2 * rs / (w * MU0 * d))
        g = 1.0 / (kmn2 - k2)
        out[i] = 1j * w * MU0 * d / (a * b) * np.einsum('imn,mn,jmn->ij', F, g, F)
    return out
