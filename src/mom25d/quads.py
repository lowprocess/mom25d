"""
mom25d.quads - quadrilateral cells on straight runs, as a constant-divergence subspace of the RWG basis.

Two triangles that share an edge and together make a near-rectangle (the strips between the edge rings
along a straight trace, the cells of the tensor mesher) are merged into one quad cell: the current is
constrained to have the same divergence (charge density) on both, which fixes the current across their
shared diagonal from the currents through the quad's four sides. The diagonal stops being an unknown.

What is left on a quad has the degrees of freedom of a rooftop (one per side, the charge uniform over the
cell) and is still a conforming RWG current, so nothing downstream changes: the matrix is filled on the
triangles as before and projected, Z_q = T^T Z T, E_q = T^T E, I = T I_q (Galerkin with the merged
functions as both bases and tests). An n x m strip of rectangles split into triangles has ~3nm edge
unknowns; as quads ~2nm: a third fewer, and the LU (N^3) about 3.4x cheaper on those cells.

    T, info = quad_subspace(rwg, n_total, exclude_tris=..., exclude_edges=..., tol_deg=25)
"""
import numpy as np
from scipy import sparse


def _quad_quality(nodes, a, u, b, v):
    """Largest deviation (degrees) of the interior angles of the quad a-u-b-v from 90; inf if not convex."""
    P = nodes[[a, u, b, v]]
    worst, sgn = 0.0, 0.0
    for i in range(4):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % 4]
        e0, e1 = p0 - p1, p2 - p1
        cr = e1[0] * e0[1] - e1[1] * e0[0]
        if cr == 0 or (sgn and np.sign(cr) != sgn):
            return np.inf
        sgn = np.sign(cr)
        c = e0 @ e1 / (np.linalg.norm(e0) * np.linalg.norm(e1))
        worst = max(worst, abs(np.degrees(np.arccos(np.clip(c, -1, 1))) - 90.0))
    return worst


def quad_subspace(rwg, n_total, exclude_tris=(), exclude_edges=(), tol_deg=25.0):
    """T (n_total x n_reduced, CSC) mapping the merged unknowns to the RWG (+ via) unknowns, and a dict of
    counts. Unknowns at index >= rwg.Ne (vias) pass through. exclude_tris: triangles that must stay
    triangles (via fans: the via's charge is spread over them); exclude_edges: edges that must stay unknowns
    (port gaps, sample cuts, lumped elements)."""
    Ne, tris, nodes = rwg.Ne, rwg.tris, rwg.nodes
    bad_t = np.zeros(len(tris), bool)
    bad_t[np.asarray(list(exclude_tris), int)] = True
    bad_e = np.zeros(Ne, bool)
    bad_e[np.asarray(list(exclude_edges), int)] = True
    # the edges of each triangle (RWG index or -1 on the boundary)
    tri_edges = -np.ones((len(tris), 3), int)
    fill = np.zeros(len(tris), int)
    for e in range(Ne):
        for t in (rwg.tp[e], rwg.tm[e]):
            tri_edges[t, fill[t]] = e
            fill[t] += 1
    cand = []
    for d in range(Ne):
        t1, t2 = rwg.tp[d], rwg.tm[d]
        if bad_e[d] or bad_t[t1] or bad_t[t2]:
            continue
        a = _opposite(tris[t1], rwg.ends[d], nodes)
        b = _opposite(tris[t2], rwg.ends[d], nodes)
        uu, vv = _edge_nodes(tris[t1], a)             # t1 = (a, uu, vv) counter-clockwise: quad a-uu-b-vv
        q = _quad_quality(nodes, a, uu, b, vv)
        if q <= tol_deg:
            cand.append((q, d))
    cand.sort()
    used = np.zeros(len(tris), bool)
    elim = []
    for q, d in cand:
        t1, t2 = rwg.tp[d], rwg.tm[d]
        if used[t1] or used[t2]:
            continue
        used[t1] = used[t2] = True
        elim.append(d)
    elim = np.array(sorted(elim), int)
    keep = np.setdiff1d(np.arange(n_total), elim)
    col = -np.ones(n_total, int)
    col[keep] = np.arange(len(keep))
    rows, cols, vals = list(keep), list(col[keep]), [1.0] * len(keep)
    D = rwg.D.tocsc()                                  # Nt x Ne: +l/A on T+, -l/A on T-
    for d in elim:
        t1, t2 = rwg.tp[d], rwg.tm[d]
        # rho(t1) = rho(t2):  sum_k (D[t1,k] - D[t2,k]) I_k = 0, solved for I_d
        dd = D[t1, d] - D[t2, d]
        for k in set(tri_edges[t1]) | set(tri_edges[t2]):
            if k < 0 or k == d:
                continue
            c = -(D[t1, k] - D[t2, k]) / dd
            if c != 0.0:
                assert col[k] >= 0, "an eliminated diagonal depends on another one"
                rows.append(d); cols.append(col[k]); vals.append(c)
    T = sparse.csc_matrix((vals, (rows, cols)), shape=(n_total, len(keep)))
    info = dict(quads=len(elim), tris_merged=2 * len(elim), tris=len(tris), edges=Ne, reduced=len(keep),
                full=n_total, keep=keep, elim=elim)
    return T, info


def _opposite(tri, ends, nodes):
    """The vertex of triangle `tri` not on the edge with end points `ends`."""
    for n in tri:
        if min(np.abs(nodes[n] - ends[0]).sum(), np.abs(nodes[n] - ends[1]).sum()) > 0:
            return n
    raise ValueError("degenerate triangle")


def _edge_nodes(tri, a):
    """The two vertices of `tri` other than a, in the triangle's (counter-clockwise) order after a."""
    i = list(tri).index(a)
    return tri[(i + 1) % 3], tri[(i + 2) % 3]


def project(Z, keep, elim, C):
    """T^T Z T for T = (identity on `keep`, rows `elim` = C), without forming T: Z (N x N) dense, C (ne x Nr)
    sparse. Stays in Z's dtype; returns the Nr x Nr matrix (C order)."""
    rdt = np.float32 if Z.dtype == np.complex64 else np.float64
    C = sparse.csr_matrix(C, dtype=rdt)
    Ct = C.T.tocsr()
    M = np.take(Z, keep, axis=1)                       # Z T = Z[:, keep] + Z[:, elim] C
    M += np.asarray(Ct @ np.take(Z, elim, axis=1).T).T
    Zq = np.take(M, keep, axis=0)                      # T^T (Z T) = M[keep] + C^T M[elim]
    Zq += np.asarray(Ct @ np.take(M, elim, axis=0))
    return Zq

