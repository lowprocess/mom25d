"""
mom25d.looptree - the loop/tree basis: the MPIE right down to low frequency (pcbqs's basis, ported)

The MPIE's matrix has two parts that drift apart as the frequency falls:

    Z = jw Z_A + Z_Q / (jw),        Z_Q = Q^T Gc Q

Q the charge each unknown puts on the triangles (an RWG edge +-l on its two triangles, a via chain its end charges
spread over the fans of triangles around it), Gc the potential between unit charges on two triangles. On a
divergence-free current Z_Q is zero, but in the RWG basis that zero is a cancellation: once jw Z_A is below the
precision's rounding of Z_Q / (jw), about (k h)^2 of it (h a cell's size), the currents of the loops are lost. On
the via transition that is below ~100 MHz for a complex64 LU (mixed precision fell back to double there) and
~5 kHz for double.

The loop/tree basis keeps the two apart, as pcbqs's PEEC does (pcbqs.mqs._loops, pcbqs.peec):

  graph    a node per triangle and one for the ground (and the PEC cover: zero potential, where a via ends with
           no fan); a branch per RWG edge (T+ to T-) and per via chain (its bottom end to its top, each end on one
           triangle of its fan, or the ground). A via's charge is spread over its fan: its branch carries, besides
           its own current, small currents across the fan that move the charge to that one triangle (the fan
           paths), so every branch puts exactly +1 and -1 on two nodes
  tree     a spanning forest, breadth first from the ground (short paths); its branches carry the charge
  loops    each other branch with the tree's path back: no charge at all, so Z_Q is not in their rows or columns

The unknowns are those branches' currents (Mmap: an RWG coefficient is a current per length, and a via's fan
paths), then loop and tree amplitudes (K: a loop is its branch plus the tree's path, a tree branch itself). With
the loops scaled by s = 1/sqrt(jw) and the tree by sqrt(jw), K_s = Mmap K diag(s):

    M = K_s^T Z_A K_s + [[0, 0], [0, A_T^T Gc A_T]]
      = [[Z_A,LL + S_LL/(jw),   jw Z_A,LT + S_LT],
         [jw Z_A,TL + S_TL,     (jw)^2 Z_A,TT + jw S_TT + A_T^T Gc A_T]]

(Z_A here the current's part with the lumped elements and the metal loss S; A_T the tree's incidence, +-1). As
w -> 0 it tends to [[Z_A,LL, 0], [0, A_T^T Gc A_T]]: inductance and elastance, each well conditioned, and nothing
cancels. Z_A is filled with the tables' charge side zeroed (the compiled core's tables), Gc from fan_q over every
triangle and the near pairs' corrections (near_q); neither has the other's part, so M is right to rounding at
any frequency (fast.FastOperator._solve_loop_tree).

    lt = LoopTree(sim)                      # the graph, once a mesh
    lt.to_ideal(Z); lt.cols(Z, core); lt.rows(Z, core)          # Z <- K^T Mmap^T Z Mmap K, in place
    lt.charge_into(Z, Mq, area, core)       # + A_T^T Gc A_T on the tree (after the scaling)
    u = lt.from_ideal(lt.from_tree(x))      # basis currents from loop and tree amplitudes
"""
import numpy as np
from scipy import sparse


class LoopTree:
    """The loop/tree basis of a Simulation's unknowns (its Ne RWG edges, then its via chains; no PEC-plane holes).
    Branch j is unknown j; after the transform unknown j is branch j's loop (cotree) or itself (tree)."""

    def __init__(self, sim):
        if sim.frills:
            raise ValueError("the loop/tree basis does not take the holes of PEC planes (frills)")
        r = sim.rwg
        Ne, Nv, Nt = sim.Ne, sim.Nv, len(r.tris)
        self.Ne, self.Nv, self.N, self.Nt, self.G = Ne, Nv, Ne + Nv, Nt, Nt         # G: the ground's node
        self.dl = np.r_[1.0 / r.l, np.ones(Nv)]                    # branch current -> basis coefficient
        bu, bv = np.empty(self.N, int), np.empty(self.N, int)
        bu[:Ne], bv[:Ne] = r.tp, r.tm
        # the via chains: each end on its fan's first triangle (the fan's charge moved there by the fan paths), or
        # the ground. The paths' branch currents per unit via current: F (Ne x Nv)
        rows, cols, vals = [], [], []
        for n, (v, kb, kt) in enumerate(sim.chains):
            i = sim.vias.index(v)
            for k, sign, ends in ((kb, +1.0, bu), (kt, -1.0, bv)):
                ts = sim.fan.get((i, k))
                if ts is None or not len(ts):
                    ends[Ne + n] = self.G
                    continue
                t0 = int(ts[0])
                ends[Ne + n] = t0
                w = r.area[ts] / r.area[ts].sum()
                within = set(int(x) for x in ts)
                for t, wt in zip(ts[1:], w[1:]):
                    for e, s in _path(t0, int(t), within, r):
                        rows.append(e); cols.append(n); vals.append(sign * wt * s)
        F = sparse.csc_matrix((vals, (rows, cols)), shape=(Ne, Nv))
        self.fan_edges = np.unique(np.asarray(rows, int))
        self.F = F[self.fan_edges].toarray() if len(self.fan_edges) else np.zeros((0, Nv))
        self.bu, self.bv = bu, bv
        self._tree()

    def _tree(self):
        """The spanning forest, breadth first from the ground, then from about the middle of each component the
        ground does not reach; the tree's nodes in that order (node, parent, branch up, its sign: +1 where the
        branch runs from the node to its parent), the loops' branches and ends."""
        n_nodes = self.Nt + 1
        bu, bv = self.bu, self.bv
        # adjacency (CSR by hand: several branches between two nodes stay apart)
        a = np.r_[bu, bv]
        order = np.argsort(a, kind='stable')
        ix = np.r_[bv, bu][order]
        br = np.r_[np.arange(self.N), np.arange(self.N)][order]
        ip = np.searchsorted(a[order], np.arange(n_nodes + 1))
        depth = np.full(n_nodes, -1)
        parent, pbranch = np.full(n_nodes, -1), np.full(n_nodes, -1)
        levels = {}

        def grow(root):
            depth[root], front, d = 0, np.array([root]), 0
            while len(front):
                d += 1
                cnt = ip[front + 1] - ip[front]
                if not cnt.sum():
                    break
                src = np.repeat(front, cnt)
                pos = np.concatenate([np.arange(ip[x], ip[x + 1]) for x in front])
                far, b = ix[pos], br[pos]
                new = depth[far] < 0
                far, src, b = far[new], src[new], b[new]
                far, first = np.unique(far, return_index=True)
                src, b = src[first], b[first]
                depth[far], parent[far], pbranch[far] = d, src, b
                levels.setdefault(d, []).append((far, src, b, np.where(bu[b] == far, 1.0, -1.0)))
                front = far
        roots = []
        if (a == self.G).any():
            roots.append(self.G)
            grow(self.G)
        while (depth[:self.Nt] < 0).any():
            seed = int(np.flatnonzero(depth[:self.Nt] < 0)[0])
            root = _middle(ip, ix, seed)
            roots.append(root)
            grow(root)
        self.roots = np.array(roots, np.int32)
        self.levels = [tuple(np.concatenate(p) for p in zip(*levels[d])) for d in sorted(levels)]
        cat = lambda k: np.concatenate([lv[k] for lv in self.levels]) if self.levels else np.zeros(0)
        self.t_node, self.t_parent, self.t_branch = (cat(k).astype(np.int32) for k in range(3))
        self.t_sig = cat(3).astype(float)
        is_tree = np.zeros(self.N, bool)
        is_tree[self.t_branch] = True
        self.is_tree = is_tree
        self.tree, self.cot = np.flatnonzero(is_tree), np.flatnonzero(~is_tree)
        self.cu, self.cv = bu[self.cot], bv[self.cot]
        self.depth = int(depth.max())
        self._args = (self.t_node, self.t_parent, self.t_branch, self.t_sig, self.roots,
                      self.cot.astype(np.int32), self.cu.astype(np.int32), self.cv.astype(np.int32), n_nodes)

    # ---- the transforms ---------------------------------------------------------------------------
    def to_ideal(self, Z, scale_rows=None):
        """Z <- Mmap^T Z Mmap in place (N x N): unit branch currents. Mmap = diag(dl) (I + F): the coefficients'
        scale, then the fan paths (only the via chains' rows and columns). scale_rows(fn): _row_blocks, to scale
        on several threads."""
        dl = self.dl

        def scale(a, b):
            Z[a:b] *= dl[a:b, None]
            Z[a:b] *= dl[None, :]
        (scale_rows or (lambda fn: fn(0, len(Z))))(scale)
        fe, Ne = self.fan_edges, self.Ne
        if len(fe):
            Z[:, Ne:] += Z[:, fe] @ self.F
            Z[Ne:, :] += self.F.T @ Z[fe, :]
        return Z

    def rhs(self, E):
        """Mmap^T E (N x m)."""
        out = np.array(E, complex) * (self.dl[:, None] if np.ndim(E) == 2 else self.dl)
        if len(self.fan_edges):
            out[self.Ne:] += self.F.T @ out[self.fan_edges]
        return out

    def from_ideal(self, x):
        """Mmap x: basis currents (RWG coefficients, via currents) of branch currents."""
        out = np.array(x, complex)
        if len(self.fan_edges):
            out[self.fan_edges] += self.F @ out[self.Ne:]
        return out * (self.dl[:, None] if out.ndim == 2 else self.dl)

    def rows(self, X, core=None, block=2048):
        """X <- K^T X in place (N x m, C order): a loop's row is its branch's plus the tree path's, from the
        prefix P[node] = P[parent] + sign X[branch up] (the sum over the node's path to its root)."""
        if core is not None and hasattr(core, 'tree_rows'):
            core.tree_rows(X, *self._args)
            return X
        m = X.shape[1]
        for c0 in range(0, m, block):
            c1 = min(m, c0 + block)
            P = np.zeros((self.Nt + 1, c1 - c0), X.dtype)
            for nodes, parents, branches, sig in self.levels:
                P[nodes] = P[parents] + sig[:, None] * X[branches, c0:c1]
            X[self.cot, c0:c1] += P[self.cv] - P[self.cu]
        return X

    def cols(self, X, core=None):
        """X <- X K in place (square N x N without the core)."""
        if core is not None and hasattr(core, 'tree_cols'):
            core.tree_cols(X, *self._args)
            return X
        transpose(X)
        self.rows(X)
        return transpose(X)

    def from_tree(self, x):
        """K x (N or N x m): the branch currents of loop and tree amplitudes."""
        x = np.asarray(x)
        out = np.array(x, complex)
        inj = np.zeros((self.Nt + 1,) + x.shape[1:], complex)
        np.add.at(inj, self.cv, x[self.cot])
        np.add.at(inj, self.cu, -x[self.cot])
        for nodes, parents, branches, sig in reversed(self.levels):
            out[branches] += (sig[:, None] if x.ndim == 2 else sig) * inj[nodes]
            np.add.at(inj, parents, inj[nodes])
        return out

    def scales(self, w):
        """s: 1/sqrt(jw) on the loops, sqrt(jw) on the tree."""
        r = np.sqrt(1j * w)
        return np.where(self.is_tree, r, 1.0 / r)

    def charge_into(self, Z, Mq, area, core=None, block=512):
        """Z[tree, tree] += A_T^T Gc A_T: the potentials of the tree branches' charges (+1 on bu, -1 on bv, none
        on the ground), Gc = Mq / (area area) (Mq: the G_q moments between the triangles, Nt x Nt)."""
        t, u, v = self.tree, self.bu[self.tree], self.bv[self.tree]
        if core is not None and hasattr(core, 'charge_into'):
            core.charge_into(Z, Mq, t.astype(np.int32), u.astype(np.int32), v.astype(np.int32),
                             np.ascontiguousarray(area, float), int(self.G))
            return Z
        ia = np.r_[1.0 / area, 0.0]                         # (the ground's row: none)
        Mp = np.zeros((self.Nt + 1, self.Nt + 1), Mq.dtype)
        Mp[:-1, :-1] = Mq
        Mp *= ia[:, None]
        Mp *= ia[None, :]
        for a in range(0, len(t), block):
            b = min(len(t), a + block)
            R = Mp[u[a:b]] - Mp[v[a:b]]
            Z[np.ix_(t[a:b], t)] += R[:, u] - R[:, v]
        return Z

    def charge_operator(self, sim):
        """Q (Nt x N, sparse): the charge each basis unknown puts on the triangles (an RWG edge +-l, a chain its
        fans' shares) - for checks: Q Mmap K is zero on the loops."""
        r = sim.rwg
        e = np.arange(self.Ne)
        Q = sparse.csr_matrix((np.r_[r.l, -r.l], (np.r_[r.tp, r.tm], np.r_[e, e])), shape=(self.Nt, self.N))
        if self.Nv:
            dq = sparse.coo_matrix(sim.dF * r.area[sim.F][:, None])
            Q = Q + sparse.csr_matrix((dq.data, (sim.F[dq.row], self.Ne + dq.col)), shape=(self.Nt, self.N))
        return Q.tocsr()


def transpose(A, block=1024):
    """A <- A^T in place (square), by blocks."""
    n = A.shape[0]
    for i in range(0, n, block):
        i1 = min(n, i + block)
        A[i:i1, i:i1] = A[i:i1, i:i1].T.copy()
        for j in range(i1, n, block):
            j1 = min(n, j + block)
            t = A[i:i1, j:j1].copy()
            A[i:i1, j:j1] = A[j:j1, i:i1].T
            A[j:j1, i:i1] = t.T
    return A


def _edges_of(r, a):
    """The RWG edges of triangle a (the lists cached on r)."""
    cache = getattr(r, '_tri_edges', None)
    if cache is None:
        cache = [[] for _ in range(len(r.tris))]
        for e, (p, m) in enumerate(zip(r.tp, r.tm)):
            cache[p].append(e)
            cache[m].append(e)
        r._tri_edges = cache
    return cache[a]


def _path(t0, t, within, r):
    """[(edge, +1 / -1)]: a path of RWG edges from triangle t0 to t (+1 where it runs T+ to T-), through the
    triangles `within` (a fan) when they join them, else through any."""
    for allowed in (within, None):
        prev, front = {t0: None}, [t0]
        while front and t not in prev:
            nxt = []
            for a in front:
                for e in _edges_of(r, a):
                    b = int(r.tm[e] if r.tp[e] == a else r.tp[e])
                    if b not in prev and (allowed is None or b in allowed):
                        prev[b] = (a, int(e))
                        nxt.append(b)
            front = nxt
        if t in prev:
            out = []
            while prev[t] is not None:
                a, e = prev[t]
                out.append((e, 1.0 if r.tp[e] == a else -1.0))
                t = a
            return out[::-1]
    raise ValueError(f"triangles {t0} and {t} are not connected")


def _sweep(ip, ix, start):
    """Breadth-first depths from start (-1: not reached)."""
    depth = np.full(len(ip) - 1, -1)
    depth[start], front, d = 0, np.array([start]), 0
    while len(front):
        d += 1
        nb = np.unique(np.concatenate([ix[ip[x]:ip[x + 1]] for x in front]))
        nb = nb[depth[nb] < 0]
        depth[nb] = d
        front = nb
    return depth


def _middle(ip, ix, seed):
    """A node about the middle of seed's component: half way between the two ends of a double sweep."""
    d = _sweep(ip, ix, seed)
    comp = np.flatnonzero(d >= 0)
    end = int(comp[np.argmax(d[comp])])
    d2 = _sweep(ip, ix, end)
    far = int(comp[np.argmax(d2[comp])])
    d3 = _sweep(ip, ix, far)
    half = d2[far] // 2
    mid = comp[(d2[comp] == half) & (d3[comp] == d2[far] - half)]
    return int(mid[0]) if len(mid) else seed
