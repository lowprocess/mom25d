"""
mom25d.biot - the magnetic field of surface currents on triangles and of straight filaments, by Biot-Savart:
H(r) = 1/(4 pi) int J(r') x (r - r') / |r - r'|^3 dS'. Exact for the quasi-static near field (the board much
smaller than a wavelength, or the point much nearer than one).

    J = rwg_currents(rwg, I)                         # (a, b) per triangle: J(r) = a r + b on it, A/m
    H = h_sheets([(rwg_nodes, tris, a, b, z), ...], pts)   # (n, 3) complex, A/m
    H += h_segments(seg_a, seg_b, I_seg, pts)        # filaments (vias): end points (m, 3), currents

The surface integral is a 6-point rule per triangle, subdivided where a point is near the triangle (closer than
`near` x its size): each triangle split into 4, twice at most; accurate to about 1% for points at a tenth of a
triangle's size and more above the sheet.
"""
import numpy as np

from .base import _BAR6, _W6, tri_area2


def rwg_currents(rwg, I):
    """(a, b): J = a r + b on each triangle (a (Nt,), b (Nt, 2)) from RWG coefficients I (Ne,); I (Ne, k): a (Nt, k),
    b (Nt, 2, k)."""
    I = np.asarray(I)
    multi = I.ndim > 1
    I2 = I if multi else I[:, None]
    k = I2.shape[1]
    a = np.zeros((len(rwg.tris), k), complex)
    b = np.zeros((len(rwg.tris), 2, k), complex)
    for t, alpha, v in ((rwg.tp, rwg.l / (2 * rwg.area[rwg.tp]), rwg.vp), (rwg.tm, -rwg.l / (2 * rwg.area[rwg.tm]), rwg.vm)):
        ia = I2 * alpha[:, None]
        np.add.at(a, t, ia)
        np.add.at(b, t, -ia[:, None, :] * v[:, :, None])
    return (a, b) if multi else (a[:, 0], b[:, :, 0])


def _split4(V):
    """Each triangle (M, 3, 2) into its 4 midpoint triangles (4M, 3, 2)."""
    m01, m12, m20 = (V[:, 0] + V[:, 1]) / 2, (V[:, 1] + V[:, 2]) / 2, (V[:, 2] + V[:, 0]) / 2
    return np.concatenate([np.stack(t, 1) for t in ((V[:, 0], m01, m20), (m01, V[:, 1], m12),
                                                      (m20, m12, V[:, 2]), (m01, m12, m20))])


def _h_tris(V, a, b, z, pts, chunk=4_000_000):
    """H at pts (n, 3) of triangles V (M, 3, 2) at height z (M,), current J = a r + b, by the 6-point rule. a, b may
    carry a last axis of excitations (frequencies): a (M, k), b (M, 2, k), giving H (n, 3, k). The geometry is
    made once per block of points and the currents of every excitation applied by one matrix product each:
    J has no z part, so H_x = sum J_y R_z / r^3, H_y = -sum J_x R_z / r^3, H_z = sum (J_x R_y - J_y R_x) / r^3."""
    multi = np.ndim(a) > 1
    a2 = a if multi else a[:, None]
    b2 = b if multi else b[:, :, None]
    e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    Q = np.einsum('qk,tkc->tqc', _BAR6, V)                          # (M, 6, 2)
    w = (area[:, None] * _W6[None]).reshape(-1)                     # (M6,)
    Jx = (a2[:, None, :] * Q[..., 0:1] + b2[:, None, 0, :]).reshape(-1, a2.shape[1]) * w[:, None]
    Jy = (a2[:, None, :] * Q[..., 1:2] + b2[:, None, 1, :]).reshape(-1, a2.shape[1]) * w[:, None]
    sx, sy, sz = Q[..., 0].reshape(-1), Q[..., 1].reshape(-1), np.repeat(z, 6)
    H = np.zeros((len(pts), 3, a2.shape[1]), complex)
    step = max(1, chunk // max(len(sx), 1))
    for p0 in range(0, len(pts), step):
        P = pts[p0:p0 + step]
        Rx, Ry, Rz = P[:, 0:1] - sx[None], P[:, 1:2] - sy[None], P[:, 2:3] - sz[None]
        r3 = (Rx * Rx + Ry * Ry + Rz * Rz) ** 1.5
        inv = np.where(r3 > 1e-30, 1.0 / np.where(r3 > 1e-30, r3, 1.0), 0.0)
        Az, Ay, Ax = Rz * inv, Ry * inv, Rx * inv
        H[p0:p0 + step, 0] = Az @ Jy
        H[p0:p0 + step, 1] = -(Az @ Jx)
        H[p0:p0 + step, 2] = Ay @ Jx - Ax @ Jy
    H /= 4 * np.pi
    return H if multi else H[:, :, 0]


def h_sheets(sheets, pts, near=1.5, levels=2):
    """H (n, 3) complex at points pts (n, 3), m, of the sheets [(nodes (N, 2) m, tris, a, b, z)]: each a
    triangle mesh at height z carrying J = a r + b on each triangle (A/m). With a (M, k) and b (M, 2, k) for k
    excitations (frequencies): H (n, 3, k), the geometry made once."""
    pts = np.asarray(pts, float)
    H = None
    for nodes, tris, a, b, z in sheets:
        V = nodes[tris]
        size = np.sqrt(np.abs(tri_area2(V)) / 2)
        cen = V.mean(1)
        # the triangles near any point: refined
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[pts[:, :2], (pts[:, 2] - z) ])
        dist, _ = tree.query(np.c_[cen, np.zeros(len(cen))])
        far = dist > near * 2 * size
        h = _h_tris(V[far], a[far], b[far], np.full(far.sum(), z), pts)
        Vn, an, bn = V[~far], a[~far], b[~far]
        for _ in range(levels):
            if not len(Vn):
                break
            Vn = _split4(Vn)
            an, bn = np.concatenate([an] * 4), np.concatenate([bn] * 4)
        if len(Vn):
            h = h + _h_tris(Vn, an, bn, np.full(len(Vn), z), pts)
        H = h if H is None else H + h
    return H


def h_segments(p0, p1, I, pts):
    """H (n, 3) at pts of straight filaments from p0 to p1 ((m, 3) m) carrying I (m,) amps (or (m, k): H (n, 3, k)):
    the finite-segment Biot-Savart, H = I / (4 pi d) (cos t1 - cos t2) around each."""
    p0, p1, pts = np.asarray(p0, float), np.asarray(p1, float), np.asarray(pts, float)
    I = np.asarray(I)
    multi = I.ndim > 1
    I2 = I if multi else I[:, None]
    H = np.zeros((len(pts), 3, I2.shape[1] if len(p0) else 1), complex)
    if len(p0):
        L = p1 - p0
        ln = np.linalg.norm(L, axis=1)
        u = L / ln[:, None]
        for k in range(len(p0)):
            r0 = pts - p0[k]
            s_ = r0 @ u[k]                                          # along the filament
            perp = r0 - s_[:, None] * u[k]
            d = np.linalg.norm(perp, axis=1)
            ok = d > 1e-9
            c1 = s_ / np.sqrt(s_ ** 2 + d ** 2)
            c2 = (s_ - ln[k]) / np.sqrt((s_ - ln[k]) ** 2 + d ** 2)
            g = np.where(ok, (c1 - c2) / (4 * np.pi * np.where(ok, d, 1)), 0)
            dirn = np.cross(u[k], np.where(ok[:, None], perp / np.where(ok, d, 1)[:, None], 0))
            H += (dirn * g[:, None])[:, :, None] * I2[k][None, None, :]
    return H if multi else H[:, :, 0]
