"""
mom25d.slots - openings in a PEC plane of the stack (slot planes)

A plane of the stack (LayerStack.pec) stays a perfect conductor, not meshed, and each of its openings - a slot, a
defected-ground cut, the aperture of an aperture-coupled patch - is meshed on its own: its unknowns are the
tangential electric field in the opening, so the plane costs about its openings' area instead of its own. The
frill holes (frill.py) are the case of a round hole round a via, with a few fixed field shapes; here the field is
free over the opening, in RWG functions.

Model
-----
The field in an opening is E = z x M on its mesh, M a sum of RWG functions (their normal component is zero at the
opening's rim, so E along the rim is: the PEC's). Closing the opening and putting E on both faces of the plane
(a magnetic current each side) leaves two problems, each with the plane as its ground or cover, as for a frill
(frill.py, with the same conventions: exp(+jwt), the spectral network of multilayer.py, T_k = LayerStack.transfer
the voltage at interface k of the network driven by a unit voltage at the plane). A field element E dA at the
origin of the opening gives, on a metal interface k on either side:

    its TM part, a potential   Psi_k = (E . rho^) P_k(rho),     P_k = -(1/2 pi) int T_k^e J1(kr rho) dkr
    its TE part, a field       E_k = (U0_k I + U2_k R2(phi)) E,  U_n = (1/4 pi) int T_k^h J_n(kr rho) kr dkr
                               R2 = [[cos 2 phi, sin 2 phi], [sin 2 phi, -cos 2 phi]]
    along a via in layer span  -int E_z dz less its potential part: (E . rho^) Rem(rho),
                               Rem = -(1/2 pi) int sum_j (T_j - T_{j-1}) k_j^2 / (kr^2 - k_j^2) J1(kr rho) dkr

(a frill's shapes are these, superposed over its hole: tests/test_slots.py checks it). The unknowns v_n (the
amplitudes of E_n = z x f_n) take one equation each, H continuous across the opening, tested with f_n; by
reciprocity the system stays symmetric, as frill.py's:

    [ Z    c ] [I]   [E]      c_mn = -<f_m, E_k of E_n> = -int div f_m Psi_k(E_n) - int f_m . E_k^TE(E_n) (edges),
    [ c^T -Y ] [v] = [0]             the chains' ends on pads by their fans' charge and the remainder (frill.py)
                              Y_mn = int int f_m . f_n G_e + int int div f_m div f_n G_d      (one plane's openings)
                              G_e = (1/2 pi) int Ye J0 kr dkr,  G_d = (1/2 pi) int (Yh - Ye) / kr^2 J0 kr dkr

Ye, Yh: the TM and TE admittances into both sides of the plane (Yu + Yd). Y is the dual of the strips' Z
(Vh I + (Ve - Vh) k^k^ there, Ye I + (Yh - Ye) k^k^ here), and is filled the same way: G_e / (jw) as its "A" and
jw G_d as its "q", their 1/rho parts eps0 (eps_below + eps_above) / (2 pi) and 1 / (pi mu0) taken out and
integrated analytically (base.singular_moments).

Between an opening's triangle and a strip's, the kernels are smooth (the plane is no metal interface): one point
each where they are far apart, quadrature where they are near (the opening's triangles subdivided, the field
peaking over the plane's distance to the strip, each pair subdivided to the scale of its own distance). An
opening's field radiates (farfield.py).

A via ending on the plane outside the openings is not at zero potential there: only a round hole's axisymmetric
field leaves the plane at zero, while another field's TM part has a potential on the plane, P_p = -1 / (2 pi rho)
(T = 1), whose gradient cancels its TE part's field on the metal. The via is charged it at that end, as its fans
are at their ends (+ at its top, - at its bottom). Without it a port via's potential at its pad stood alone, and a
slot's effect on the line came out about doubled (S21 -12 dB at 8 GHz where the ground meshed gives -5.1 dB; 2026-
11-15). frill.py's first-order shapes leave such a potential too, and its chains do not see it: added there, it
moved demo_transition from 0.003 to 0.011 at 8 GHz against the finely meshed plane, so it is left out of frill
until that is understood (kicad2ems ROADMAP).
"""
import numpy as np

from .base import MU0, EPS0, RWG, qform, combine, singular_moments, quad_pts, moments7, quad_moments, \
    _BAR3, _W3, _BAR6, _W6
from .frill import bessel_int
from .via import sommerfeld_contour, hankel, Table1D, _layers

NEAR = 4.0              # an opening's triangle and a strip's are near within NEAR x (the larger's size + the height)
SUB_MAX = 3             # the opening's triangles subdivided at most 4^SUB_MAX times for a near strip


class Slot:
    """An opening in the PEC plane on interface k: poly, a shapely Polygon in metres (holes allowed: an island
    inside the opening is plane too, not floating metal: say so with a hole only where it is joined to the plane,
    else it is left out). name: for the log."""
    def __init__(self, k, poly, name=''):
        self.k, self.poly, self.name = int(k), poly, name
        self.rwg = None                                     # mesh(): the opening's RWG functions

    @property
    def count(self):
        return 0 if self.rwg is None else self.rwg.Ne

    def mesh(self, dmax, ncross=6, grade=1.0, fine=None, dmax_fine=None, w_max=None):
        """The opening's triangles: edge rings at its rim (where its field peaks), cells up to dmax, and up to
        dmax_fine inside `fine` (where copper on the other layers is near: as a meshed plane is refined).
        w_max: the edge rings' reference width at most (default 4 dmax)."""
        from .geo import mesh_polygons
        w = 2 * self.poly.area / max(self.poly.length, 1e-12)      # its width, about: (twice area / perimeter)
        nodes, tris, _ = mesh_polygons(self.poly, dmax, min(w, w_max or 4 * dmax), ncross, grade,
                                       fine=fine, dmax_fine=dmax_fine)
        self.rwg = r = RWG(nodes, tris)
        self.P3, self.W3 = quad_pts(r, _BAR3), r.area[:, None] * _W3[None, :]
        self.P6, self.W6 = quad_pts(r, _BAR6), r.area[:, None] * _W6[None, :]
        V = r.nodes[r.tris]
        self.size = np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2).max(1)
        self._static = None
        return r

    def field_integrals(self):
        """(Ex, Ey) (ntri x Ne, sparse): the integral of E_n = z x f_n over each of the opening's triangles."""
        r = self.rwg
        cx, cy, A = r.cent[:, 0], r.cent[:, 1], r.area
        from scipy import sparse
        D = lambda v: sparse.diags(v)
        fx = D(cx) @ r.Aal - r.Bx                            # f = Aal r - (Bx, By) on a triangle, its integral
        fy = D(cy) @ r.Aal - r.By                            # A (Aal c - B)
        return (D(A) @ (-fy)).tocsr(), (D(A) @ fx).tocsr()   # z x f = (-f_y, f_x)

    def statics(self):
        """The opening's 1/R moment integrals (frequency-independent): combine's (M0, Mr, Mrr), once."""
        if self._static is None:
            self._static = singular_moments(self.rwg)
        return self._static


def _sub_points(level):
    """Points and weights (area fractions) of a triangle's 6-point rule on its 4^level subtriangles, (n, 3) bary."""
    tri = [np.eye(3)]
    for _ in range(level):
        nxt = []
        for t in tri:
            m01, m12, m20 = (t[0] + t[1]) / 2, (t[1] + t[2]) / 2, (t[2] + t[0]) / 2
            nxt += [np.array([t[0], m01, m20]), np.array([m01, t[1], m12]), np.array([m20, m12, t[2]]),
                    np.array([m01, m12, m20])]
        tri = nxt
    bar = np.vstack([_BAR6 @ t for t in tri])
    w = np.concatenate([_W6 / len(tri)] * len(tri))
    return bar, w


class SlotKernels:
    """
    Per-frequency kernels of the openings `slots` (Slot) of a stack's PEC planes:
      table[p]        : SlotTable of plane p (its openings' own Y)
      P[(p, k)]       : Table1D of P_k(rho), the TM potential's kernel on metal interface k
      U[(p, k)]       : Table1D of (U0, U2), the TE field's
      rem(p, span)    : Table1D of Rem(rho) for a chain through `span` (k_bot, k_top)
    """
    def __init__(self, f, stack, slots, interfaces, rho_max, nrho=260):
        self.w = w = 2 * np.pi * f
        self.st = st = stack.at(f)
        self.kr, self.dk, _ = kr, dk, _ = sommerfeld_contour(f, st, rho_max)
        self.rho_max = rho_max
        self.rho = rho = rho_max * np.linspace(0, 1, nrho) ** 2
        planes = sorted({s.k for s in slots})
        self.table = {p: SlotTable(f, st, p, rho_max) for p in planes}
        self.P, self.U, self._rem = {}, {}, {}
        keys = [(p, k) for p in planes for k in interfaces if st.reaches(p, k) and k != p]
        if keys:
            Te = np.array([st.transfer(kr, w, p, k, 'e') for p, k in keys])
            Th = np.array([st.transfer(kr, w, p, k, 'h') for p, k in keys])
            P = bessel_int(1, -Te / (2 * np.pi), kr, dk, rho)
            U0 = bessel_int(0, Th * kr / (4 * np.pi), kr, dk, rho)
            U2 = bessel_int(2, Th * kr / (4 * np.pi), kr, dk, rho)
            for i, key in enumerate(keys):
                self.P[key] = Table1D(rho, P[i], rho_max)
                self.U[key] = Table1D(rho, np.stack([U0[i], U2[i]], 1), rho_max)

    def rem(self, p, span):
        """Rem(rho) of a chain through `span`'s layers (on one side of plane p)."""
        if (p, span) not in self._rem:
            kr, k0 = self.kr, self.w / 299792458.0
            spec = 0.0
            for j in _layers(span):
                dT = self.st.transfer(kr, self.w, p, j, 'e') - self.st.transfer(kr, self.w, p, j - 1, 'e')
                if np.any(dT):
                    k2 = k0 ** 2 * self.st.eps(j)
                    spec = spec + dT * k2 / (kr ** 2 - k2)
            vals = np.zeros_like(self.rho, dtype=complex) if np.isscalar(spec) else \
                bessel_int(1, -spec / (2 * np.pi), kr, self.dk, self.rho)
            self._rem[(p, span)] = Table1D(self.rho, vals, self.rho_max)
        return self._rem[(p, span)]


class SlotTable:
    """The kernels of an opening's own equation on plane p, as GreenTableML's (its 'A' and 'q', what base.combine
    and the singular moments take): A = G_e / (jw), q = jw G_d (module doc), their 1/rho parts cA / rho, cq / rho
    extracted, and the nearest image (a dielectric step or metal close to the plane) where there is one within
    IMAGE_REACH."""
    IMAGE_REACH = 0.6e-3

    def __init__(self, f, stack, p, rho_max, nrho=260, nfine=8192):
        w = 2 * np.pi * f
        self.p, self.f, self.d = p, f, 0.0
        self.cA = EPS0 * stack.eps_sum(p) / (2 * np.pi)
        self.cq = 1.0 / (np.pi * MU0)
        kr, dk, K = sommerfeld_contour(f, stack, rho_max)
        GA, Gq = self.spectral(stack, kr, w, p)
        self.dimg, self.cAi, self.cqi = 0.0, 0.0, 0.0
        near = stack.nearest_discontinuity(p) if hasattr(stack, 'nearest_discontinuity') else np.inf
        if near < self.IMAGE_REACH:
            self.dimg = 2 * near
            K1 = 18.0 / self.dimg
            ga, gq = self.spectral(stack, np.array([K1 + 0j]), 2 * np.pi, p)
            fac = K1 * np.exp(K1 * self.dimg) / (2 * np.pi)
            self.cAi = complex((ga[0] - 2 * np.pi * self.cA / K1) * fac)
            self.cqi = complex((gq[0] - 2 * np.pi * self.cq / K1) * fac)
        for d, cA, cq in self.extracted():
            e = 2 * np.pi * np.exp(-kr * d) / kr
            GA, Gq = GA - cA * e, Gq - cq * e
        rho = rho_max * np.linspace(0, 1, nrho) ** 2
        self._tab = Table1D(rho, hankel(np.stack([GA, Gq]), kr, dk, rho).T, rho_max, nfine)

    @staticmethod
    def spectral(stack, kr, w, p):
        _, _, _, Yd, Yu = stack._network(kr, w, 'e')
        Ye = Yu[p] + Yd[p]
        _, _, _, Yd, Yu = stack._network(kr, w, 'h')
        Yh = Yu[p] + Yd[p]
        return Ye / (1j * w), 1j * w * (Yh - Ye) / kr ** 2

    def extracted(self):
        return [(0.0, self.cA, self.cq)] + ([(self.dimg, self.cAi, self.cqi)] if self.dimg else [])

    def AQ(self, r):
        v = self._tab(r)
        return v[..., 0], v[..., 1]


def self_block(slots, table, w, near_fac=5.0, cache=None):
    """Y (n x n) of the openings `slots` of one plane, with `table` (SlotTable), as Simulation.impedance fills Z:
    jw (static A + A) + (static q + q) / (jw), the triangle pairs near each other by quadrature."""
    from types import SimpleNamespace
    from scipy import sparse
    from .sing import offplane_pairs, offplane_moments
    # the openings of one plane as one mesh (their triangles and functions in turn)
    nodes, tris, off = [], [], 0
    for s in slots:
        nodes.append(s.rwg.nodes); tris.append(s.rwg.tris + off); off += len(s.rwg.nodes)
    key = tuple(id(s) for s in slots)
    cache = {} if cache is None else cache
    if key not in cache:
        r = RWG(np.vstack(nodes), np.vstack(tris))
        M0, Mr, Mrr = singular_moments(r)
        V = r.nodes[r.tris]
        size = np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2).max(1)
        cache[key] = dict(r=r, M0=M0, Mr=Mr, Mrr=Mrr, size=size, P3=quad_pts(r, _BAR3),
                          W3=r.area[:, None] * _W3[None, :], img={})
        # the order of the functions: the one mesh's edges, mapped back to each opening's own (Slot.rwg)
        cache[key]['order'] = _edge_order(r, slots)
    c = cache[key]
    r = c['r']
    sym = lambda M: 0.5 * (M + M.T)
    rho = np.sqrt(((r.cent[:, None, :] - r.cent[None, :, :]) ** 2).sum(-1))
    GA, GQ = table.AQ(rho)
    wgt = r.area[:, None] * r.area[None, :]
    out = moments7(GA * wgt, GQ * wgt, r.cent[:, None], r.cent[None])
    size = np.maximum(c['size'][:, None], c['size'][None, :])
    p, q = np.nonzero(rho < near_fac * size)
    for M, v in zip(out, quad_moments(c['P3'], c['W3'], p, q, table)):
        M[p, q] = v
    if table.dimg:                                           # the nearest image, analytically for every pair
        if table.dimg not in c['img']:
            pp, qq = np.repeat(np.arange(len(r.tris)), len(r.tris)), np.tile(np.arange(len(r.tris)), len(r.tris))
            c['img'][table.dimg] = (pp, qq, offplane_pairs(r.nodes, r.tris, pp, qq, table.dimg))
        pp, qq, m = c['img'][table.dimg]
        for M, v in zip(out, offplane_moments(table.cAi, table.cqi, m)):
            M[pp, qq] += v
    A0, Axo, Ayo, _, _, Arr, Q0 = out
    SA = table.cA * combine(r, c['Mrr'], c['Mr'][..., 0], c['Mr'][..., 1], c['M0'])
    Y = 1j * w * (SA + combine(r, sym(Arr), Axo, Ayo, sym(A0))) + \
        (table.cq * qform(r.D, c['M0'], r.D) + qform(r.D, sym(Q0), r.D)) / (1j * w)
    Y = 0.5 * (Y + Y.T)
    o = c['order']
    return Y[np.ix_(o, o)]


def _edge_order(r, slots):
    """For the edges of the slots' own meshes in turn, their index among the joint mesh's edges r (matched by their
    end points; the joint mesh has each opening's nodes and triangles, in order). The sign of a function (its T+
    and T-) is the same in both: it follows the triangles' order, which the joint mesh keeps."""
    look = {tuple(np.round(r.ends[e].ravel(), 12)): e for e in range(r.Ne)}
    out = [look[tuple(np.round(s.rwg.ends[e].ravel(), 12))] for s in slots for e in range(s.rwg.Ne)]
    return np.array(out, int)


def _levels(size, dz):
    """How often a triangle of `size` is subdivided for a kernel peaking over the height dz: to about dz / 2."""
    return int(np.clip(np.ceil(np.log2(max(size, 1e-12) / max(0.5 * dz, 1e-12))), 0, SUB_MAX))


def _tm_te_moments(dx, dy, wgt, ru, P, U):
    """The far / near pair moments at observation - source offsets (dx, dy) (any shape, summed over the last axes by
    the caller), weights wgt, the source points ru (x, y) (same shape) and the observation points ro: the TM
    potential's (M1, M2, M3: int g_x, int g_y, int g_y x' - g_x y', g = P rho^) and, with U, the TE tensor's pieces
    (T_xx, T_xy, T_yy) for _te_moments."""
    rho = np.hypot(dx, dy)
    inv = np.divide(1.0, rho, out=np.zeros_like(rho), where=rho > 0)
    cs, sn = dx * inv, dy * inv
    p = P(rho) * wgt
    gx, gy = p * cs, p * sn
    tm = (gx, gy, gy * ru[0] - gx * ru[1])
    if U is None:
        return tm, None
    u = U(rho)
    c2, s2 = cs * cs - sn * sn, 2 * sn * cs
    u0, u2 = u[..., 0] * wgt, u[..., 1] * wgt
    return tm, (u0 + u2 * c2, u2 * s2, u0 - u2 * c2)


def _te_moments(T, ro, ru):
    """From the tensor pieces T = (Txx, Txy, Tyy) at pairs of points (ro observation, ru source): T0 (xx, xy, yy),
    Ro = r . T (x, y), Ws = T . w' (w' = z x r' = (-y', x')) and RW = r . T . w' (module coupling's moments)."""
    txx, txy, tyy = T
    wx, wy = -ru[1], ru[0]
    rox, roy = ro[0] * txx + ro[1] * txy, ro[0] * txy + ro[1] * tyy
    wsx, wsy = txx * wx + txy * wy, txy * wx + tyy * wy
    return (txx, txy, tyy), (rox, roy), (wsx, wsy), ro[0] * wsx + ro[1] * wsy


def _plane_potential(rho):
    """P on the plane itself: -(1/2 pi) int J1(kr rho) dkr = -1 / (2 pi rho)."""
    return -1.0 / (2 * np.pi * np.maximum(rho, 1e-12))


def coupling(sim, slots, kern, B, C, col0):
    """The rows and columns of the openings' unknowns: c into B (the strips' edges) and into C's chain rows (their
    fans and remainder), from column col0 on, in the order of `slots` (each Slot.count). module doc."""
    r, st, Nc = sim.rwg, kern.st, len(sim.chains)
    F = np.asarray(sim.F, int)
    pos_F = {int(t): i for i, t in enumerate(F)}
    col = col0
    for s in slots:
        rs, n = s.rwg, s.count
        cols = slice(col, col + n)
        Aal_s, Bx_s, By_s = rs.Aal, rs.Bx, rs.By
        vs = lambda M1, M2, M3: np.asarray(M3 @ Aal_s + M1 @ By_s - M2 @ Bx_s)
        Vs = s.rwg.nodes[s.rwg.tris]
        for k in sim.layers:
            if (s.k, k) not in kern.P:
                continue
            idx = sim.tri_idx[k]
            if not len(idx):
                continue
            dz = abs(st.z[k] - st.z[s.k])
            P, U = kern.P[(s.k, k)], kern.U[(s.k, k)]
            ct, cu = r.cent[idx], rs.cent
            dx, dy = ct[:, 0, None] - cu[None, :, 0], ct[:, 1, None] - cu[None, :, 1]
            wgt = r.area[idx][:, None] * rs.area[None, :]
            ru = (cu[None, :, 0] + 0 * dx, cu[None, :, 1] + 0 * dy)
            ro = (ct[:, 0, None] + 0 * dx, ct[:, 1, None] + 0 * dy)
            (M1, M2, M3), T = _tm_te_moments(dx, dy, wgt, ru, P, U)
            T0, Ro, Ws, RW = _te_moments(T, ro, ru)
            # near pairs: both triangles subdivided to the height's scale
            size = np.maximum(sim.tri_size[idx][:, None], s.size[None, :])
            pi_, qi_ = np.nonzero(dx ** 2 + dy ** 2 + dz ** 2 < (NEAR * (size + 0 * dx)) ** 2)
            if len(pi_):
                # each near pair's rule from its own distance: subdivided to the scale of the closer of its distance
                # and the height (the kernel's), grouped by (strip level, opening level)
                dd = np.sqrt(dx[pi_, qi_] ** 2 + dy[pi_, qi_] ** 2)
                st_, su_ = sim.tri_size[idx][pi_], s.size[qi_]
                deff = np.maximum(dd - 0.5 * (st_ + su_), dz)
                lev = lambda sz: np.clip(np.ceil(np.log2(np.maximum(sz, 1e-12) / np.maximum(0.5 * deff, 1e-12))),
                                         0, SUB_MAX).astype(int)
                Lt_, Lu_ = lev(st_), lev(su_)
                for Lt, Lu in sorted(set(zip(Lt_.tolist(), Lu_.tolist()))):
                    g = np.flatnonzero((Lt_ == Lt) & (Lu_ == Lu))
                    gp, gq = pi_[g], qi_[g]
                    bt, wt = _sub_points(Lt)
                    bu, wu = _sub_points(Lu)
                    Pt = np.einsum('qk,pkc->pqc', bt, r.nodes[r.tris[idx[gp]]])       # (np, nt, 2)
                    Pu = np.einsum('qk,pkc->pqc', bu, Vs[gq])                         # (np, nu, 2)
                    Wt = r.area[idx[gp]][:, None] * wt[None, :]
                    Wu = s.rwg.area[gq][:, None] * wu[None, :]
                    chunk = max(1, 2_000_000 // (Pt.shape[1] * Pu.shape[1]))
                    for c0 in range(0, len(gp), chunk):
                        sl = slice(c0, c0 + chunk)
                        a, b = Pt[sl][:, :, None, :], Pu[sl][:, None, :, :]
                        ddx, ddy = a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]
                        ww = Wt[sl][:, :, None] * Wu[sl][:, None, :]
                        rub = (b[..., 0] + 0 * ddx, b[..., 1] + 0 * ddy)
                        roa = (a[..., 0] + 0 * ddx, a[..., 1] + 0 * ddy)
                        (m1, m2, m3), t = _tm_te_moments(ddx, ddy, ww, rub, P, U)
                        t0, ro_, ws_, rw_ = _te_moments(t, roa, rub)
                        ii, jj = gp[sl], gq[sl]
                        for M, v in ((M1, m1), (M2, m2), (M3, m3), (RW, rw_)):
                            M[ii, jj] = v.sum((1, 2))
                        for Ms, vs_ in ((T0, t0), (Ro, ro_), (Ws, ws_)):
                            for M, v in zip(Ms, vs_):
                                M[ii, jj] = v.sum((1, 2))
            # TM: the potential over each strip triangle, tested with the divergence (and the fans' charge)
            psi = vs(M1, M2, M3)                                           # (len(idx), n)
            Dk = r.D[idx]
            B[:, cols] -= np.asarray(Dk.T @ psi)
            if len(F) and Nc:
                here = [(pos_F[int(t)], j) for j, t in enumerate(idx) if int(t) in pos_F]
                if here:
                    fi, li = np.array(here).T
                    C[:Nc, cols] -= np.asarray(sim.dF[fi, :Nc].T @ psi[li])
            # TE: the field on the edges themselves: -<f, T E>, f = Aal r - (Bx, By), E = Aal' w' + (By', -Bx')
            (txx, txy, tyy), (rox, roy), (wsx, wsy) = T0, Ro, Ws
            Aal_k, Bx_k, By_k = r.Aal[idx], r.Bx[idx], r.By[idx]
            te = Aal_k.T @ (RW @ Aal_s + rox @ By_s - roy @ Bx_s)
            te = te - Bx_k.T @ (wsx @ Aal_s) - By_k.T @ (wsy @ Aal_s)
            te = te - Bx_k.T @ (txx @ By_s - txy @ Bx_s) - By_k.T @ (txy @ By_s - tyy @ Bx_s)
            B[:, cols] -= np.asarray(te)
        # the chains' remainder (their part of int E_z dz that is not the potential's), by span
        if Nc:
            spans = {}
            for c, (v, kb, kt) in enumerate(sim.chains):
                spans.setdefault((kb, kt), []).append(c)
            for span, cs in spans.items():
                # (a chain does not run through the plane: a via through it is split there, its parts ending on
                # the plane; Simulation refuses one through an opening)
                if not any(st.reaches(s.k, j) or j == s.k for j in span):
                    continue
                xy = np.array([(sim.chains[c][0].x, sim.chains[c][0].y) for c in cs])
                dx = xy[:, 0, None, None] - s.P6[None, :, :, 0]
                dy = xy[:, 1, None, None] - s.P6[None, :, :, 1]
                ru = (s.P6[None, :, :, 0] + 0 * dx, s.P6[None, :, :, 1] + 0 * dy)
                R = kern.rem(s.k, span)
                if np.any(R.v):
                    (n1, n2, n3), _ = _tm_te_moments(dx, dy, s.W6[None] + 0 * dx, ru, R, None)
                    C[np.array(cs), cols] += vs(n1.sum(-1), n2.sum(-1), n3.sum(-1))
                if s.k in span:
                    # its end on the plane: the opening's potential there, P_p = -1 / (2 pi rho) (T = 1 on the
                    # plane). Not zero outside an opening: only a round hole's axisymmetric field leaves the plane
                    # at zero; another's TM part has a potential there whose gradient its TE part cancels. With
                    # the fans' sign (dF: +1 at a chain's bottom, -1 at its top), the plane at its bottom: -Psi_p
                    (n1, n2, n3), _ = _tm_te_moments(dx, dy, s.W6[None] + 0 * dx, ru, _plane_potential, None)
                    sign = -1.0 if span[0] == s.k else 1.0
                    C[np.array(cs), cols] += sign * vs(n1.sum(-1), n2.sum(-1), n3.sum(-1))
        col += n
    return col


def spectrum(slots, v, kx, ky):
    """(Ex, Ey) (nk[, nexc]): the openings' field transformed at the plane-wave directions (kx, ky), sum of
    int E e^{j k.rho} (far field, farfield.py), with v their unknowns in the order of `slots`."""
    Ex = Ey = 0
    col = 0
    for s in slots:
        n = s.count
        ix, iy = s.field_integrals()
        vv = v[col:col + n]
        ph = np.exp(1j * (np.outer(kx, s.rwg.cent[:, 0]) + np.outer(ky, s.rwg.cent[:, 1])))
        Ex, Ey = Ex + ph @ (ix @ vv), Ey + ph @ (iy @ vv)
        col += n
    return Ex, Ey
