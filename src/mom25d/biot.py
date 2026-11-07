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


def h_grid(model, res, fi, x, y, heights, refine=1):
    """Open-boundary H on uniform x/y grids (m), at absolute z `heights` (m).

    Sheet currents are conservatively deposited then convolved with the
    magnetostatic Biot-Savart kernel by FFT. Linear convolution avoids periodic
    images; the explicitly meshed copper supplies every return path. The
    source grid expands to contain all copper, even for a small observation
    window. Filaments are evaluated analytically. For reliable sheet fields
    use grid spacing well below distance to copper (and check convergence).
    Shape (nx, ny, nh, 3), with an extra last axis for a list of fi.
    """
    from types import SimpleNamespace
    from scipy.fft import fft2, ifft2, next_fast_len
    from .farfield import current_deposit

    x, y, heights = [np.asarray(v, float) for v in (x, y, np.atleast_1d(heights))]
    for axis in (x, y):
        if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or
                np.diff(axis).min() <= 0 or not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-8, atol=1e-15)):
            raise ValueError('x and y must be increasing uniform grids (m)')
    if heights.ndim != 1 or not np.isfinite(heights).all():
        raise ValueError('heights must be finite absolute z coordinates (m)')
    if np.any(np.isclose(heights[:, None], np.unique(model.z_of)[None], rtol=0, atol=1e-15)):
        raise ValueError('FFT H must be evaluated away from copper planes')
    if not isinstance(refine, (int, np.integer)) or refine < 1:
        raise ValueError('refine must be a positive integer')
    dx, dy = (x[1] - x[0]) / refine, (y[1] - y[0]) / refine
    grids, offsets = [], []
    for dim, axis, step in ((0, x, dx), (1, y, dy)):
        lo = min(0, int(np.floor((model.nodes[:, dim].min() - axis[0]) / step)) - 2)
        hi = max(refine * (len(axis) - 1), int(np.ceil((model.nodes[:, dim].max() - axis[0]) / step)) + 2)
        grids.append(axis[0] + np.arange(lo, hi + 1) * step)
        offsets.append(-lo)
    gx, gy = grids
    nx, ny = len(gx), len(gy)
    shape = (next_fast_len(3*nx - 2), next_fast_len(3*ny - 2))
    rx = np.arange(1 - nx, nx)[:, None] * dx
    ry = np.arange(1 - ny, ny)[None, :] * dy
    multi = np.ndim(fi) > 0
    fis = list(np.atleast_1d(fi))
    out = np.zeros((len(x), len(y), len(heights), 3, len(fis)), complex)
    crop = (slice(nx - 1 + offsets[0], nx - 1 + offsets[0] + refine * len(x), refine),
            slice(ny - 1 + offsets[1], ny - 1 + offsets[1] + refine * len(y), refine))
    sim = SimpleNamespace(rwg=model.rwg, tri_idx=model.layer_tris)
    for lay, idx in model.layer_tris.items():
        # Bound spectral working memory when refinement or harmonic count grows.
        block_size = max(1, (64 * 1024**2) // (np.prod(shape) * 16 * 6))
        for first in range(0, len(fis), block_size):
            spectra = [[], []]
            for k in fis[first:first + block_size]:
                jx, jy = current_deposit(sim, res.i[k, :model.ne] / model.rwg.l, gx, gy, lay)
                spectra[0].append(fft2(jx, s=shape))
                spectra[1].append(fft2(jy, s=shape))
            jx, jy = [np.stack(s, axis=-1) for s in spectra]
            for h, height in enumerate(heights):
                dz = height - model.z_of[idx[0]]
                kernel = dx*dy / (4*np.pi * (rx*rx + ry*ry + dz*dz)**1.5)
                kx, ky, kz = [fft2(v * kernel, s=shape)[..., None] for v in (rx, ry, dz)]
                for c, spectrum in enumerate((jy*kz, -jx*kz, jx*ky - jy*kx)):
                    out[:, :, h, c, first:first + block_size] += ifft2(spectrum, axes=(0, 1))[crop]
    X, Y, Z = np.meshgrid(x, y, heights, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    p0, p1, currents = [], [], []
    for k, (vx, vy, rr, za, zb) in enumerate(model.vseg):
        p0.append([vx, vy, za]); p1.append([vx, vy, zb])
        currents.append(res.i[fis, model.ne + k])
    for k, part in enumerate(model.parts):
        offset = np.array([0, 0, part['part_h'] * 1e-3])
        p0.append(model._term_centre(part['a']) + offset)
        p1.append(model._term_centre(part['b']) + offset)
        currents.append(res.i[fis, model.ne + model.nv + k])
    if p0:
        out += h_segments(p0, p1, np.asarray(currents), pts).reshape(out.shape)
    return out if multi else out[..., 0]
