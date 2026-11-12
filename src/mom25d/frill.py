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
"""
import numpy as np
from scipy.special import jv

from .via import sommerfeld_contour, hankel, Table1D, _layers


MODES = 2               # field shapes a hole carries (Frill.modes): each one more unknown


class Frill:
    """A hole of radius b in the PEC plane on interface k, around the via `via` at its axis. a: the inner radius
    of the gap (default the via's; its pad's where the via has one in the hole). modes: how many field shapes
    the hole carries (profile's m = 0..modes - 1), each with its own unknown."""
    def __init__(self, via, k, b, modes=None, a=None):
        self.via, self.k, self.b, self.modes = via, int(k), float(b), int(modes or MODES)
        self.a = float(a or via.a)
        assert self.b > self.a, "a hole is wider than its via"

    @property
    def xy(self):
        return np.array([self.via.x, self.via.y])


def profile(kr, a, b, m=0):
    """e_m(kr) of a hole: the order-1 Hankel transform of its m-th field shape g_m(rho), a < rho < b:

        g_m = cos(m pi u) / (rho sqrt(1 - rho^2 / b^2)) / arccosh(b / a),    rho = b sin t,  u = (t - t0) / (pi/2 - t0)

    (t0 = asin(a / b)). m = 0 is the coaxial field at the via, 1 / rho, with the 1 / sqrt(distance) of a thin
    sheet's edge at the hole's rim, and 1 V across the hole; the others redistribute the field across it.
    e_m = b / arccosh(b / a) int_t0^{pi/2} cos(m pi u) J1(kr b sin t) dt (Gauss-Legendre)."""
    t0 = np.arcsin(a / b)
    x, wq = np.polynomial.legendre.leggauss(96)
    t = 0.5 * (np.pi / 2 - t0) * x + 0.5 * (np.pi / 2 + t0)
    wq = 0.5 * (np.pi / 2 - t0) * wq * np.cos(m * np.pi * (t - t0) / (np.pi / 2 - t0))
    return b / np.arccosh(b / a) * (jv(1, np.multiply.outer(kr, b * np.sin(t))) @ wq)


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
        self.unknowns = [(fr, m) for fr in self.frills for m in range(fr.modes)]
        self.shapes = sorted({self.shape(*u) for u in self.unknowns})
        self.e = {s: profile(kr, s[1], s[2], s[3]) for s in self.shapes}
        self._T = {}
        rho = rho_max * np.linspace(0, 1, nrho) ** 2
        keys = [(s, k) for s in self.shapes for k in interfaces if stack.reaches(s[0], k)]
        self.psi = {}
        if keys:
            specs = np.array([2 * np.pi * self.e[s] * self.T(s[0], k) / kr for s, k in keys])
            for key, v in zip(keys, hankel(specs, kr, dk, rho)):
                self.psi[key] = Table1D(rho, v, rho_max)
        Y0, tn, sc, Yd, Yu = stack._network(kr, w, 'e')
        self._Y = {p: Yu[p] + Yd[p] for p in {s[0] for s in self.shapes}}

    @staticmethod
    def shape(fr, m=0):
        return fr.k, round(fr.a, 9), round(fr.b, 9), m

    def T(self, p, k):
        if (p, k) not in self._T:
            self._T[(p, k)] = self.st.transfer(self.kr, self.w, p, k)
        return self._T[(p, k)]

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
        return hankel(2 * np.pi * self.e[shape] * spec / kr, kr, self.dk, np.atleast_1d(rho))

    def Y(self, u, v):
        """The admittance between two unknowns (frill, m): 0 for holes of different planes."""
        su, sv = self.shape(*u), self.shape(*v)
        if su[0] != sv[0]:
            return 0.0
        d = 0.0 if u[0] is v[0] else float(np.hypot(*(u[0].xy - v[0].xy)))
        spec = (2 * np.pi) ** 2 * self._Y[su[0]] * self.e[su] * self.e[sv]
        return complex(hankel(spec, self.kr, self.dk, np.array([d]))[0])
