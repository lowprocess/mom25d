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

from .base import _BAR6, _W6


def rwg_currents(rwg, I):
    """(a, b): J = a r + b on each triangle (a (Nt,), b (Nt, 2)) from RWG coefficients I (Ne,)."""
    a = np.zeros(len(rwg.tris), complex)
    b = np.zeros((len(rwg.tris), 2), complex)
    I = np.asarray(I)
    for t, alpha, v in ((rwg.tp, rwg.l / (2 * rwg.area[rwg.tp]), rwg.vp), (rwg.tm, -rwg.l / (2 * rwg.area[rwg.tm]), rwg.vm)):
        np.add.at(a, t, I * alpha)
        np.add.at(b, t, -(I * alpha)[:, None] * v)
    return a, b


def _split4(V):
    """Each triangle (M, 3, 2) into its 4 midpoint triangles (4M, 3, 2)."""
    m01, m12, m20 = (V[:, 0] + V[:, 1]) / 2, (V[:, 1] + V[:, 2]) / 2, (V[:, 2] + V[:, 0]) / 2
    return np.concatenate([np.stack(t, 1) for t in ((V[:, 0], m01, m20), (m01, V[:, 1], m12),
                                                      (m20, m12, V[:, 2]), (m01, m12, m20))])


def _h_tris(V, a, b, z, pts, chunk=2_000_000):
    """H at pts (n, 3) of triangles V (M, 3, 2) at height z (M,), current J = a r + b, by the 6-point rule."""
    e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    Q = np.einsum('qk,tkc->tqc', _BAR6, V)                          # (M, 6, 2)
    Jq = a[:, None, None] * Q + b[:, None, :]                       # (M, 6, 2)
    w = area[:, None] * _W6[None]                                   # (M, 6)
    src = np.concatenate([Q.reshape(-1, 2), np.repeat(z, 6)[:, None]], 1)    # (M6, 3)
    J3 = np.concatenate([Jq.reshape(-1, 2), np.zeros((len(src), 1))], 1) * w.reshape(-1)[:, None]
    H = np.zeros((len(pts), 3), complex)
    step = max(1, chunk // max(len(src), 1))
    for p0 in range(0, len(pts), step):
        P = pts[p0:p0 + step]
        R = P[:, None, :] - src[None]                               # (np, M6, 3)
        r3 = np.linalg.norm(R, axis=-1) ** 3
        r3[r3 < 1e-30] = np.inf
        H[p0:p0 + step] = np.cross(J3[None], R).__truediv__(r3[..., None]).sum(1)
    return H / (4 * np.pi)


def h_sheets(sheets, pts, near=1.5, levels=2):
    """H (n, 3) complex at points pts (n, 3), m, of the sheets [(nodes (N, 2) m, tris, a, b, z)]: each a
    triangle mesh at height z carrying J = a r + b on each triangle (A/m)."""
    pts = np.asarray(pts, float)
    H = np.zeros((len(pts), 3), complex)
    for nodes, tris, a, b, z in sheets:
        V = nodes[tris]
        size = np.sqrt(np.abs(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])) / 2)
        cen = V.mean(1)
        # the triangles near any point: refined
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[pts[:, :2], (pts[:, 2] - z) ])
        dist, _ = tree.query(np.c_[cen, np.zeros(len(cen))])
        far = dist > near * 2 * size
        H += _h_tris(V[far], a[far], b[far], np.full(far.sum(), z), pts)
        Vn, an, bn = V[~far], a[~far], b[~far]
        for _ in range(levels):
            if not len(Vn):
                break
            Vn = _split4(Vn)
            an, bn = np.tile(an, 4), np.tile(bn, (4, 1))
        if len(Vn):
            H += _h_tris(Vn, an, bn, np.full(len(Vn), z), pts)
    return H


def h_segments(p0, p1, I, pts):
    """H (n, 3) at pts of straight filaments from p0 to p1 ((m, 3) m) carrying I (m,) amps: the finite-segment
    Biot-Savart, H = I / (4 pi d) (cos t1 - cos t2) around each."""
    p0, p1, pts = np.asarray(p0, float), np.asarray(p1, float), np.asarray(pts, float)
    H = np.zeros((len(pts), 3), complex)
    if not len(p0):
        return H
    L = p1 - p0
    ln = np.linalg.norm(L, axis=1)
    u = L / ln[:, None]
    for k in range(len(p0)):
        r0 = pts - p0[k]
        s = r0 @ u[k]                                               # along the filament
        perp = r0 - s[:, None] * u[k]
        d = np.linalg.norm(perp, axis=1)
        ok = d > 1e-9
        c1 = s / np.sqrt(s ** 2 + d ** 2)
        c2 = (s - ln[k]) / np.sqrt((s - ln[k]) ** 2 + d ** 2)
        mag = np.where(ok, I[k] / (4 * np.pi * np.where(ok, d, 1)) * (c1 - c2), 0)
        dirn = np.cross(u[k], np.where(ok[:, None], perp / np.where(ok, d, 1)[:, None], 0))
        H += mag[:, None] * dirn
    return H
