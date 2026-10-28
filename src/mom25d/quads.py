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


def excluded(rwg, exclude_tris=(), exclude_edges=()):
    """Boolean masks (triangles, edges) of the triangles and edges that must stay as they are."""
    bad_t, bad_e = np.zeros(len(rwg.tris), bool), np.zeros(rwg.Ne, bool)
    bad_t[np.asarray(list(exclude_tris), int)] = True
    bad_e[np.asarray(list(exclude_edges), int)] = True
    return bad_t, bad_e


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
    bad_t, bad_e = excluded(rwg, exclude_tris, exclude_edges)
    # the edges of each triangle (RWG index or -1 on the boundary)
    tri_edges, fill = -np.ones((len(tris), 3), int), np.zeros(len(tris), int)
    for e in range(Ne):
        for t in (rwg.tp[e], rwg.tm[e]):
            tri_edges[t, fill[t]] = e
            fill[t] += 1
    cand = []
    for d in range(Ne):
        t1, t2 = rwg.tp[d], rwg.tm[d]
        if bad_e[d] or bad_t[t1] or bad_t[t2]:
            continue
        a, b = _opposite(tris[t1], rwg.ends[d], nodes), _opposite(tris[t2], rwg.ends[d], nodes)
        i = list(tris[t1]).index(a)                   # t1 = (a, uu, vv) counter-clockwise: quad a-uu-b-vv
        q = _quad_quality(nodes, a, tris[t1][(i + 1) % 3], b, tris[t1][(i + 2) % 3])
        if q <= tol_deg:
            cand.append((q, d))
    elim = np.array(sorted(_match(cand, rwg.tp, rwg.tm, len(tris))), int)
    keep, col = np.setdiff1d(np.arange(n_total), elim), -np.ones(n_total, int)
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
    return T, dict(quads=len(elim), tris_merged=2 * len(elim), tris=len(tris), edges=Ne, reduced=len(keep),
                   full=n_total, keep=keep, elim=elim)


def _match(cand, tp, tm, nt, depth=12):
    """The diagonals to eliminate: a matching of the triangles over the candidate diagonals cand [(quality, edge)].
    Greedy, best quality first, then grown along augmenting paths (alternating free / matched diagonals from an
    unpaired triangle to another, up to `depth` diagonals long): each one pairs two more triangles. The search is
    a tree without blossoms, so it can miss a path, never take a wrong one. Greedy alone leaves about a tenth of
    the triangles unpaired that a maximum matching would pair (looser tolerances, Delaunay interiors)."""
    mate = -np.ones(nt, int)                          # the diagonal each triangle is paired over
    adj = [[] for _ in range(nt)]
    for q, d in sorted(cand):
        t1, t2 = tp[d], tm[d]
        adj[t1].append((t2, d)); adj[t2].append((t1, d))
        if mate[t1] < 0 and mate[t2] < 0:
            mate[t1] = mate[t2] = d
    other = lambda t, d: tm[d] if tp[d] == t else tp[d]
    for root in range(nt):
        if mate[root] >= 0 or not adj[root]:
            continue
        # BFS over even triangles (reached by a matched diagonal; the root is even): parent[t] = (odd tri, edge)
        parent, front, end = {root: None}, [root], None
        for _ in range(depth // 2):
            nxt = []
            for u in front:
                for v, d in adj[u]:
                    if v in parent or v == root:
                        continue
                    if mate[v] < 0:                    # a free triangle: the path ends here
                        parent[v], end = (u, d), v
                        break
                    w = other(v, mate[v])
                    if w in parent:
                        continue
                    parent[v], parent[w] = (u, d), (v, mate[v])
                    nxt.append(w)
                if end is not None:
                    break
            if end is not None or not nxt:
                break
            front = nxt
        if end is None:
            continue
        v = end                                        # flip the path: its free diagonals become matched
        while v != root:
            u, d = parent[v]
            mate[v] = mate[u] = d
            v = parent[u][0] if u != root else root
    return {int(d) for d in mate if d >= 0}


def _opposite(tri, ends, nodes):
    """The vertex of triangle `tri` not on the edge with end points `ends`."""
    for n in tri:
        if min(np.abs(nodes[n] - ends[0]).sum(), np.abs(nodes[n] - ends[1]).sum()) > 0:
            return n
    raise ValueError("degenerate triangle")


def project(Z, keep, elim, C, core=None):
    """T^T Z T for T = (identity on `keep` in its first len(keep) columns, rows `elim` = C), without forming T:
    Z (N x N) dense, C (ne x Nr) sparse, Nr >= len(keep) (the columns past len(keep) are combined functions on the
    eliminated unknowns, line.py). Stays in Z's dtype; returns the Nr x Nr matrix (C order). core: the compiled
    core (fast._core), which reduces Z's rows in place (Z is overwritten) on every thread: the demo transition's
    8.3k unknowns in 0.2 s against 1.4 s here."""
    if core is not None and hasattr(core, 'project') and Z.flags.c_contiguous and Z.dtype in (np.complex64, np.complex128):
        C = sparse.csr_matrix(C, dtype=np.float64)
        C.sort_indices()
        Ct = C.T.tocsr()
        i32 = lambda a: np.ascontiguousarray(a, np.int32)
        return (core.project_c64 if Z.dtype == np.complex64 else core.project)(
            Z, i32(keep), i32(elim), i32(C.indptr), i32(C.indices), C.data, i32(Ct.indptr), i32(Ct.indices), Ct.data,
            int(C.shape[1]))
    rdt = np.float32 if Z.dtype == np.complex64 else np.float64
    C = sparse.csr_matrix(C, dtype=rdt)
    Ct = C.T.tocsr()
    nk = len(keep)
    ZT = np.asarray(Ct @ np.take(Z, elim, axis=1).T).T            # Z T = [Z[:, keep], 0] + Z[:, elim] C
    ZT[:, :nk] += np.take(Z, keep, axis=1)
    Zq = np.asarray(Ct @ np.take(ZT, elim, axis=0))               # T^T (Z T) = [ZT[keep]; 0] + C^T ZT[elim]
    Zq[:nk] += np.take(ZT, keep, axis=0)
    return Zq
