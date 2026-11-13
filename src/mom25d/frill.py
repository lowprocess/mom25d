"""
mom25d.frill - a via through a hole (antipad) in a PEC plane inside the stack

The plane (LayerStack.pec) is not meshed: it shorts the stack's network, and what is above it does not see what is
below. A signal via through one of its holes joins the two: its current runs on through the hole (one chain, as
through any interface it does not touch), and the hole carries one more unknown, the voltage V across it.

Model
-----
The field in the hole is the coaxial one, E_rho = V / (rho ln(b / a)) for a < rho < b (a: the via's radius, b: the
hole's), V the via's potential over the plane's. Closing the hole with PEC and putting that field on both faces
of the plane (a magnetic frill each side) leaves two problems, each with the plane as its ground or cover, and
the field of the hole in each is TM and axisymmetric. In the spectral transmission-line network (multilayer.py)
it is the network driven at the plane by the voltage

    e(kr) V,    e(kr) = (J0(kr a) - J0(kr b)) / (kr ln(b / a))          (the order-1 transform of E_rho)

in place of its short (the field in a hole of a thin plane is not quite the coaxial one: `profile` has the shapes
used, a first one like it and others that redistribute the field across the hole, each with its own unknown;
what follows holds for each): T_k(kr) = LayerStack.transfer is the voltage it leaves at interface k. With the
exp(+jwt), J0 conventions of via.py:

  on interface k           E_t = -grad_t Psi_k,    Psi_k(rho) = int e T_k J0(kr rho) dkr
                           (the potential of the hole on that interface: V at the via, 0 far away; exact, not
                           only statically, the field being TM)
  along a via in layer j   int E_z dz = int kr (T_j - T_{j-1}) e / kz_j^2  J0(kr rho) kr dkr
                           = -(Psi_j - Psi_{j-1}) - int (T_j - T_{j-1}) e k_j^2 / (kr^2 - k_j^2) J0(kr rho) dkr
  at the hole              H_phi(+) - H_phi(-) = int (Yu + Yd) e V J1(kr rho) kr dkr     (Yu, Yd: the network's
                           admittances up and down from the plane)

The unknown V takes one equation, H_phi continuous across the hole, tested with the hole's own profile. By
reciprocity the row is the transpose of the column the hole's field makes in the other equations, so the system
stays symmetric:

    [ Z    c ] [I]   [E]        c_m = -<f_m, E of the hole at 1 V>  = -int div f_m Psi   (an edge; a chain's
    [ c^T -Y ] [V] = [0]              ends on pads the same, by their fans' charge), plus the chains' remainder
                                Y_fg = 2 pi int (Yu + Yd) e_f e_g J0(kr d_fg) kr dkr     (holes of one plane)

A chain's potential terms at the plane itself cancel between its two sides (a via through the hole), or are zero
(a via onto the plane, outside every hole), so only its ends on pads and the remainder count. Statically this is
the hole's capacitance into both sides, charged by what the pads' charges induce where the hole is: the current
through the hole alone makes no voltage across it.

The first azimuthal order (Frill.dipoles)
-----------------------------------------
A trace that crosses the hole's rim is not axisymmetric: its return current on the plane has to go round the
hole, and its charge finds no plane under it there. The field this puts in the hole is, to first order, one that
goes as cos(phi) and sin(phi) about the via:

    E = rho^ R(rho) cos(phi) + phi^ F(rho) sin(phi)        ("c"; "s" is the same turned a quarter turn)

with F = 0 at the via and at the rim (the field along a conductor's edge), R free there (`dipole_shapes`: the
shapes of the hole of a thin plane with no via, Bouwkamp's, and the via's own effect on them). Such a field is not
a gradient: its spectrum has a part along k (TM, the network's 'e' mode, as the axisymmetric field's) and a part
across it (TE, the 'h' mode):

    E~(k) = 2 pi [ e1(kr) cos(al) k^  -  h1(kr) sin(al) (z x k^) ]                    (al: the direction of k)
    e1 = int [R J1'(kr rho) - F J1(kr rho) / (kr rho)] rho drho,   h1 = int [R J1 / (kr rho) - F J1'] rho drho

  on interface k   the TM part is a potential again, Psi_k = cos(phi) psi_k(rho), psi_k = -int T_k e1 J1(kr rho) dkr
                   (tested with the edges' divergence, the chains' ends and remainder as above, J0 -> J1);
                   the TE part is a field with no potential, tested on the edges themselves:
                   E = (U0 + U2 cos 2phi, U2 sin 2phi),  U_n = 1/2 int Th_k h1 J_n(kr rho) kr dkr
                   (Th: the transfer of the 'h' network); a vertical via does not see it
  at the hole      Y_cc = pi int [(Yu + Yd)_e e1 e1' + (Yu + Yd)_h h1 h1'] kr dkr for two shapes of one hole (c and
                   s do not couple there, nor either with the axisymmetric shapes); between holes, `Y`.
"""
import os

import numpy as np
from scipy.special import jv, j0, j1

from .via import sommerfeld_contour, hankel, Table1D, _layers


MODES = 2               # axisymmetric field shapes a hole carries (Frill.modes): each one more unknown
DIPOLES = int(os.environ.get('MOM25D_HOLE_DIPOLES', 2))     # shapes of the first azimuthal order (Frill.dipoles):
                        # each two more unknowns (its cos and its sin). Two: a trace over a 0.45 mm hole 0.011 in
                        # |S| from the meshed plane at 10 GHz (0.065 with none; three and four change nothing)


class Frill:
    """A hole of radius b in the PEC plane on interface k, around the via `via` at its axis. a: the inner radius
    of the gap (default the via's; its pad's where the via has one in the hole). modes: how many axisymmetric
    field shapes the hole carries (profile's m = 0..modes - 1), each with its own unknown. dipoles: how many
    shapes of the first azimuthal order (dipole_shapes' j = 0..dipoles - 1), each two unknowns (cos, sin): what
    a trace crossing the rim adds."""
    def __init__(self, via, k, b, modes=None, a=None, dipoles=None):
        self.via, self.k, self.b, self.modes = via, int(k), float(b), int(modes or MODES)
        self.a = float(a or via.a)
        self.dipoles = DIPOLES if dipoles is None else int(dipoles)
        assert self.b > self.a, "a hole is wider than its via"
        assert 0 <= self.dipoles <= len(DIPOLE_SHAPES)

    @property
    def xy(self):
        return np.array([self.via.x, self.via.y])

    @property
    def kinds(self):
        """The hole's unknowns: m (an axisymmetric shape), then ('c', j) and ('s', j) of each dipole shape."""
        return list(range(self.modes)) + [(o, j) for j in range(self.dipoles) for o in 'cs']

    @property
    def count(self):
        return self.modes + 2 * self.dipoles


NQ = 96                 # Gauss-Legendre points across a hole (rho = b sin t)


def _nodes(a, b):
    """(t0, t, w): the quadrature in t across the hole, t0 = asin(a / b) at the via to pi / 2 at the rim."""
    t0 = np.arcsin(a / b)
    x, wq = np.polynomial.legendre.leggauss(NQ)
    return t0, 0.5 * (np.pi / 2 - t0) * x + 0.5 * (np.pi / 2 + t0), 0.5 * (np.pi / 2 - t0) * wq


def bessel_matrices(rho, kr, orders=(0, 1, 2)):
    """{n: J_n(rho x kr)}, (len(rho), len(kr)): the complex Bessel functions only on the lifted part of the
    contour, the fast real J0 and J1 (and J2 from them) on the rest."""
    rho = np.atleast_1d(np.asarray(rho, float))
    real = np.abs(kr.imag) < 1e-300
    out = {n: np.empty((len(rho), len(kr)), complex) for n in orders}
    z = rho[:, None] * kr.real[None, real]
    J0, J1 = j0(z), j1(z)
    for n in orders:
        if n == 2:                                          # (2 J1 / z - J0 loses its digits at small z)
            zs = np.where(z > 0.05, z, 1.0)
            out[n][:, real] = np.where(z > 0.05, 2 * J1 / zs - J0, z * z / 8 * (1 - z * z / 12 * (1 - z * z / 32)))
        else:
            out[n][:, real] = J0 if n == 0 else J1
        if (~real).any():
            out[n][:, ~real] = jv(n, rho[:, None] * kr[None, ~real])
    return out


def profile(kr, a, b, m=0, J=None):
    """e_m(kr) of a hole: the order-1 Hankel transform of its m-th field shape g_m(rho), a < rho < b:

        g_m = cos(m pi u) / (rho sqrt(1 - rho^2 / b^2)) / arccosh(b / a),    rho = b sin t,  u = (t - t0) / (pi/2 - t0)

    (t0 = asin(a / b)). m = 0 is the coaxial field at the via, 1 / rho, with the 1 / sqrt(distance) of a thin
    sheet's edge at the hole's rim, and 1 V across the hole; the others redistribute the field across it.
    e_m = b / arccosh(b / a) int_t0^{pi/2} cos(m pi u) J1(kr b sin t) dt (Gauss-Legendre). J: bessel_matrices of
    the hole's nodes (b sin t) and kr, where a caller has them."""
    t0, t, wq = _nodes(a, b)
    wq = wq * np.cos(m * np.pi * (t - t0) / (np.pi / 2 - t0))
    J = J or bessel_matrices(b * np.sin(t), kr, (1,))
    return b / np.arccosh(b / a) * (wq @ J[1])


# R(rho), F(rho) of the first order's shapes, as functions of (cos t, a / rho): rho = b sin t. With no via the
# hole of a thin plane under a uniform tangential H has R = 1 / cos t + cos t, F = -2 cos t (Bouwkamp); a thin
# conductor on the axis of a transverse field makes them (1 + a^2 / rho^2) and (1 - a^2 / rho^2) of that.
DIPOLE_SHAPES = (
    (lambda c, q: (1 + q * q) / c, None),                   # the rim's singular field, across the rim
    (None, lambda c, q: (1 - q * q) * c),                   # the field along the hole's circles
    (lambda c, q: (1 + q * q) * c, None),                   # across, without the singularity
    (None, lambda c, q: (1 - q * q) * c ** 3),
)


def dipole_shapes(kr, a, b, j, J=None):
    """(e1, h1)(kr) of the hole's j-th shape of the first azimuthal order, E = (rho^ R cos(phi) + phi^ F sin(phi)) / b
    for a < rho < b (DIPOLE_SHAPES; 1 / b: an amplitude in volts):

        e1 = int [R J1'(kr rho) - F J1(kr rho) / (kr rho)] rho drho      (its part along k: TM)
        h1 = int [R J1(kr rho) / (kr rho) - F J1'(kr rho)] rho drho      (its part across k: TE)

    (Gauss-Legendre in t, rho = b sin t: the rim's 1 / sqrt is the Jacobian's.) J: as profile's."""
    R, F = DIPOLE_SHAPES[j]
    t0, t, wq = _nodes(a, b)
    rho, c = b * np.sin(t), np.cos(t)
    wq = wq * rho * c                                       # rho drho = rho b cos t dt, and the shape's 1 / b
    J = J or bessel_matrices(rho, kr, (0, 1))
    j1z = J[1] / np.multiply.outer(rho, kr)
    j1p = J[0] - j1z
    r = R(c, a / rho) * wq if R else 0 * wq
    f = F(c, a / rho) * wq if F else 0 * wq
    return r @ j1p - f @ j1z, r @ j1z - f @ j1p


def bessel_int(n, spec, kr, dk, rho, J=None):
    """int spec(kr) J_n(kr rho) dkr on the contour, for an array of rho: (len(rho),), or (nspec, len(rho)).
    J: bessel_matrices of rho and kr, where a caller has them."""
    spec = np.asarray(spec)
    J = (J or bessel_matrices(rho, kr, (n,)))[n]
    out = (np.atleast_2d(spec) * dk[None, :]) @ J.T
    return out[0] if spec.ndim == 1 else out


class FrillKernels:
    """
    Per-frequency kernels of the holes `frills` (Frill) of a stack's internal PEC planes. A hole's unknowns are
    its field shapes' amplitudes; `unknowns` lists them, (frill, m), in the order of the system's columns.
      psi[(shape, k)]        : Table1D of Psi_k(rho), the shape's potential on the metal interface k
      rem(shape, span, rho)  : the remainder of a chain through the layers of `span` (k_bot, k_top) at rho
      Y(u, v)                : the admittance between two unknowns (of one hole, or of two holes of one plane)
    shape = (plane, a, b, m): holes alike share their kernels.
    """
    def __init__(self, f, stack, frills, interfaces, rho_max, nrho=200):
        self.w = w = 2 * np.pi * f
        self.st = stack = stack.at(f)
        self.kr, self.dk, _ = kr, dk, _ = sommerfeld_contour(f, stack, rho_max)
        self.rho_max, self.frills = rho_max, list(frills)
        self.unknowns = [(fr, m) for fr in self.frills for m in fr.kinds]
        every = {self.shape(*u) for u in self.unknowns}
        self.shapes = sorted(s for s in every if not isinstance(s[3], tuple))       # the axisymmetric ones
        self.dipoles = sorted(s for s in every if isinstance(s[3], tuple))          # (plane, a, b, ('d', j))
        Jq = {ab: bessel_matrices(ab[1] * np.sin(_nodes(*ab)[1]), kr, (0, 1)) for ab in {s[1:3] for s in every}}
        self.e = {s: profile(kr, s[1], s[2], s[3], Jq[s[1:3]]) for s in self.shapes}
        self.e1, self.h1 = {}, {}
        for s in self.dipoles:
            self.e1[s], self.h1[s] = dipole_shapes(kr, s[1], s[2], s[3][1], Jq[s[1:3]])
        self._T = {}
        rho = rho_max * np.linspace(0, 1, nrho) ** 2
        keys = [(s, k) for s in self.shapes for k in interfaces if stack.reaches(s[0], k)]
        self.psi = {}
        if keys:
            specs = np.array([2 * np.pi * self.e[s] * self.T(s[0], k) / kr for s, k in keys])
            for key, v in zip(keys, hankel(specs, kr, dk, rho)):
                self.psi[key] = Table1D(rho, v, rho_max)
        # the first order: psi (its TM part's potential, to go with cos or sin of the angle) and U0, U2 (its TE
        # part's field), radial tables on each interface
        keys = [(s, k) for s in self.dipoles for k in interfaces if stack.reaches(s[0], k)]
        self.U = {}
        if keys:
            tm = np.array([-self.e1[s] * self.T(s[0], k) for s, k in keys])
            te = np.array([0.5 * self.h1[s] * self.T(s[0], k, 'h') * kr for s, k in keys])
            J = bessel_matrices(rho, kr)
            for key, p1, u0, u2 in zip(keys, bessel_int(1, tm, kr, dk, rho, J), bessel_int(0, te, kr, dk, rho, J),
                                       bessel_int(2, te, kr, dk, rho, J)):
                self.psi[key] = Table1D(rho, p1, rho_max)
                self.U[key] = Table1D(rho, np.stack([u0, u2], 1), rho_max)
        planes = {s[0] for s in every}
        Y0, tn, sc, Yd, Yu = stack._network(kr, w, 'e')
        self._Y = {p: Yu[p] + Yd[p] for p in planes}
        if self.dipoles:
            Y0, tn, sc, Yd, Yu = stack._network(kr, w, 'h')
            self._Yh = {p: Yu[p] + Yd[p] for p in planes}

    @staticmethod
    def shape(fr, m=0):
        """What a hole's unknown shares with the like unknowns of holes alike: (plane, a, b, m) of an axisymmetric
        shape, (plane, a, b, ('d', j)) of a dipole shape (its cos and its sin: one set of kernels)."""
        return fr.k, round(fr.a, 9), round(fr.b, 9), (('d', m[1]) if isinstance(m, tuple) else m)

    def T(self, p, k, mode='e'):
        if (p, k, mode) not in self._T:
            self._T[(p, k, mode)] = self.st.transfer(self.kr, self.w, p, k, mode)
        return self._T[(p, k, mode)]

    def rem(self, shape, span, rho):
        """-int_chain E_z dz less its potential part, for a chain through `span`'s layers at distance(s) rho from
        the hole's axis: sum over its layers of int (T_j - T_{j-1}) e k_j^2 / (kr^2 - k_j^2) J0(kr rho) dkr."""
        p, kr = shape[0], self.kr
        k0 = self.w / 299792458.0
        spec = 0.0
        for j in _layers(span):
            dT = self.T(p, j) - self.T(p, j - 1)
            if np.any(dT):
                k2 = k0 ** 2 * self.st.eps(j)
                spec = spec + dT * k2 / (kr ** 2 - k2)
        if np.isscalar(spec):
            return np.zeros(np.size(rho), complex)
        if isinstance(shape[3], tuple):                     # the first order: with cos or sin of the chain's angle
            return bessel_int(1, -self.e1[shape] * spec, kr, self.dk, np.atleast_1d(rho))
        return hankel(2 * np.pi * self.e[shape] * spec / kr, kr, self.dk, np.atleast_1d(rho))

    def Y(self, u, v):
        """The admittance between two unknowns (frill, m): 0 for holes of different planes. With i = u, j = v,
        d and be the distance and direction from v's hole to u's, and the angular integrals of exp(j kr d cos(al - be)):

            axisymmetric, both          2 pi int Ye e e' J0 kr dkr
            axisymmetric i, c (s) j     -2 pi cos(be) (sin) int Ye e e1' J1 kr dkr          (the other way: +)
            c, c (s, s)                 pi int [Ye e1 e1' (J0 -+ J2 cos 2be) + Yh h1 h1' (J0 +- J2 cos 2be)] kr dkr
            c, s                        pi sin 2be int [-Ye e1 e1' + Yh h1 h1'] J2 kr dkr
        """
        su, sv = self.shape(*u), self.shape(*v)
        if su[0] != sv[0]:
            return 0.0
        du, dv = isinstance(u[1], tuple), isinstance(v[1], tuple)
        same = u[0] is v[0]
        if same and (du != dv or (du and u[1][0] != v[1][0])):
            return 0.0                                      # one hole: orders and orientations apart
        kr, dk, Ye = self.kr, self.dk, self._Y[su[0]]
        sep = u[0].xy - v[0].xy
        d, be = (0.0, 0.0) if same else (float(np.hypot(*sep)), float(np.arctan2(sep[1], sep[0])))
        J = lambda n: jv(n, kr * d) if d else (np.ones_like(kr) if n == 0 else np.zeros_like(kr))
        tot = lambda spec: complex(np.sum(spec * kr * dk))
        if not du and not dv:
            return 2 * np.pi * tot(Ye * self.e[su] * self.e[sv] * J(0))
        if du != dv:
            ang = (np.cos(be) if (v[1] if dv else u[1])[0] == 'c' else np.sin(be)) * (-1 if dv else 1)
            e0, e1 = (self.e[su], self.e1[sv]) if dv else (self.e[sv], self.e1[su])
            return 2 * np.pi * ang * tot(Ye * e0 * e1 * J(1))
        ee, hh = Ye * self.e1[su] * self.e1[sv], self._Yh[su[0]] * self.h1[su] * self.h1[sv]
        if u[1][0] == v[1][0]:
            sg = np.cos(2 * be) * (1 if u[1][0] == 'c' else -1)
            return np.pi * tot(ee * (J(0) - sg * J(2)) + hh * (J(0) + sg * J(2)))
        return np.pi * np.sin(2 * be) * tot((hh - ee) * J(2))
