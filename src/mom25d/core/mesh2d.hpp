// mesh2d.hpp - a small 2D constrained Delaunay triangulator (the meshers' own: no library). The same file in mom25d's
// and pcbqs's cores; cdt_mesh.hpp wraps it for Python (cdt_triangulate), cdtmesh.py places the points.
//
//   mesh2d::CDT cdt(xy, n, min_dist);          // the points (their indices kept), inserted in Hilbert order
//   cdt.insert_segment(a, b) ...               // the constraints (split where they cross, or where a point is on one)
//   cdt.mark_outer();                          // the region reached from the bounding triangle without crossing one
//   cdt.refine(min_angle_rad, budget, min_edge); // optional: Ruppert's refinement of the rest
//
// The pieces, each the textbook one:
//   predicates  orient2d and incircle with a floating-point filter and an exact fallback (Shewchuk's expansion
//               arithmetic, products exact through fma): every decision on the topology is exact; only the points
//               it adds (where segments cross, the refinement's) are rounded, and a crossing too fine for doubles to
//               place between its neighbours is an error, not a broken mesh
//   insertion   a bounding triangle, points in Hilbert order located by a visibility walk from the last one,
//               Lawson's flips (a point on an edge splits it: a constraint's halves stay constraints)
//   segments    Sloan's flips: the edges crossing a segment flipped away, then the new edges made Delaunay again; a
//               point on (or within min_dist of) a segment splits it there; crossing segments are split where they
//               cross (a new point)
//   outer       the triangles reached from the bounding triangle without crossing a segment (holes stay: geo drops
//               them by their centroids)
//   refine      Ruppert / Shewchuk: encroached segments split first (at their middles, or on concentric shells about an
//               input point, so that segments meeting at a sharp angle do not split each other without end), then
//               each triangle with an angle under the bound: its circumcentre, unless that encroaches a segment
//               (the segment is split instead) or is behind one (that one is split); an angle between two segments is
//               left as it is; at most `budget` points added
//
// Triangles counter-clockwise; neighbour k is across the edge opposite vertex k. Internally the bounding triangle's
// three points come first (0..2); the caller's points are 3..n+2 and the added ones after them.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mesh2d {

// ---- predicates ------------------------------------------------------------------------------------------------
namespace pred {
using Exp = std::vector<double>;                  // a nonoverlapping expansion, increasing magnitude, no zeros

inline void two_sum(double a, double b, double &x, double &y) {
    x = a + b;
    double bv = x - a, av = x - bv;
    y = (a - av) + (b - bv);
}
inline void two_diff(double a, double b, double &x, double &y) {
    x = a - b;
    double bv = a - x, av = x + bv;
    y = (a - av) + (bv - b);
}
inline Exp diff(double a, double b) {
    double x, y;
    two_diff(a, b, x, y);
    Exp e;
    if (y != 0) e.push_back(y);
    if (x != 0) e.push_back(x);
    return e;
}
inline Exp grow(const Exp &e, double b) {          // e + b
    Exp h;
    h.reserve(e.size() + 1);
    double q = b, s, t;
    for (double ei : e) {
        two_sum(q, ei, s, t);
        if (t != 0) h.push_back(t);
        q = s;
    }
    if (q != 0) h.push_back(q);
    return h;
}
inline Exp sum(Exp e, const Exp &f) {
    for (double fi : f) e = grow(e, fi);
    return e;
}
inline Exp neg(Exp e) {
    for (double &x : e) x = -x;
    return e;
}
inline Exp scale(const Exp &e, double b) {         // e * b (products exact through fma)
    Exp h;
    if (e.empty() || b == 0) return h;
    h.reserve(2 * e.size());
    double q = e[0] * b, t = std::fma(e[0], b, -q);
    if (t != 0) h.push_back(t);
    for (size_t i = 1; i < e.size(); ++i) {
        double p1 = e[i] * b, p0 = std::fma(e[i], b, -p1), s, u;
        two_sum(q, p0, s, u);
        if (u != 0) h.push_back(u);
        double x = p1 + s;                         // fast two-sum (|p1| >= |s|)
        double w = s - (x - p1);
        if (w != 0) h.push_back(w);
        q = x;
    }
    if (q != 0) h.push_back(q);
    return h;
}
inline Exp mul(const Exp &e, const Exp &f) {
    Exp r;
    for (double fi : f) r = sum(r, scale(e, fi));
    return r;
}
inline int sign(const Exp &e) { return e.empty() ? 0 : (e.back() > 0 ? 1 : -1); }

constexpr double EPS = 1.1102230246251565e-16;    // 2^-53
constexpr double CCW_BOUND = (3.0 + 16.0 * EPS) * EPS;
constexpr double ICC_BOUND = (10.0 + 96.0 * EPS) * EPS;

// > 0: a, b, c counter-clockwise; < 0 clockwise; 0 collinear (exactly)
inline double orient(double ax, double ay, double bx, double by, double cx, double cy) {
    double l = (ax - cx) * (by - cy), r = (ay - cy) * (bx - cx), det = l - r;
    double bound = CCW_BOUND * (std::fabs(l) + std::fabs(r));
    if (det > bound || -det > bound) return det;
    Exp d = sum(mul(diff(ax, cx), diff(by, cy)), neg(mul(diff(ay, cy), diff(bx, cx))));
    return (double)sign(d);
}

// > 0: d inside the circle through a, b, c (counter-clockwise); 0 on it (exactly)
inline double incircle(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy) {
    double adx = ax - dx, ady = ay - dy, bdx = bx - dx, bdy = by - dy, cdx = cx - dx, cdy = cy - dy;
    double bc1 = bdx * cdy, bc2 = cdx * bdy, ca1 = cdx * ady, ca2 = adx * cdy, ab1 = adx * bdy, ab2 = bdx * ady;
    double al = adx * adx + ady * ady, bl = bdx * bdx + bdy * bdy, cl = cdx * cdx + cdy * cdy;
    double det = al * (bc1 - bc2) + bl * (ca1 - ca2) + cl * (ab1 - ab2);
    double perm = (std::fabs(bc1) + std::fabs(bc2)) * al + (std::fabs(ca1) + std::fabs(ca2)) * bl +
                  (std::fabs(ab1) + std::fabs(ab2)) * cl;
    double bound = ICC_BOUND * perm;
    if (det > bound || -det > bound) return det;
    Exp ex = diff(ax, dx), ey = diff(ay, dy), fx = diff(bx, dx), fy = diff(by, dy), gx = diff(cx, dx),
        gy = diff(cy, dy);
    Exp A = sum(mul(ex, ex), mul(ey, ey)), B = sum(mul(fx, fx), mul(fy, fy)), C = sum(mul(gx, gx), mul(gy, gy));
    Exp bc = sum(mul(fx, gy), neg(mul(gx, fy))), ca = sum(mul(gx, ey), neg(mul(ex, gy))),
        ab = sum(mul(ex, fy), neg(mul(fx, ey)));
    return (double)sign(sum(sum(mul(A, bc), mul(B, ca)), mul(C, ab)));
}
}  // namespace pred

struct Tri {
    int v[3];          // counter-clockwise
    int n[3];          // neighbour across the edge opposite v[k] (-1: none)
    uint8_t c;         // bit k: that edge is a segment (a constraint)
    uint8_t outer;     // outside the outline (reached from the bounding triangle without crossing a segment)
};

class CDT {
public:
    std::vector<double> X, Y;
    std::vector<Tri> T;
    std::vector<int> vtri;      // a triangle at each point
    size_t n_input = 0;         // the caller's points: internal indices 3 .. n_input + 2
    double min_dist = 0.0;

    CDT(const double *xy, size_t n, double min_dist_ = 0.0) : n_input(n), min_dist(min_dist_) {
        if (!n) return;
        double x0 = xy[0], x1 = xy[0], y0 = xy[1], y1 = xy[1];
        for (size_t i = 0; i < n; ++i) {
            double x = xy[2 * i], y = xy[2 * i + 1];
            if (!std::isfinite(x) || !std::isfinite(y)) throw std::invalid_argument("mesh2d: a point is not finite");
            x0 = std::min(x0, x), x1 = std::max(x1, x), y0 = std::min(y0, y), y1 = std::max(y1, y);
        }
        const double cx = 0.5 * (x0 + x1), cy = 0.5 * (y0 + y1);
        const double R = std::max({x1 - x0, y1 - y0, 1e-300}) * 64.0;
        X = {cx - 2 * R, cx + 2 * R, cx};
        Y = {cy - R, cy - R, cy + 2 * R};
        X.reserve(n + 3), Y.reserve(n + 3);
        for (size_t i = 0; i < n; ++i) X.push_back(xy[2 * i]), Y.push_back(xy[2 * i + 1]);
        vtri.assign(n + 3, -1);
        T.reserve(2 * n + 8);
        T.push_back(Tri{{0, 1, 2}, {-1, -1, -1}, 0, 1});
        vtri[0] = vtri[1] = vtri[2] = 0;
        // Hilbert order: each point's walk starts next to it
        std::vector<std::pair<uint64_t, int>> key(n);
        const double sx = (x1 > x0) ? 65535.0 / (x1 - x0) : 0, sy = (y1 > y0) ? 65535.0 / (y1 - y0) : 0;
        for (size_t i = 0; i < n; ++i)
            key[i] = {hilbert((uint32_t)((xy[2 * i] - x0) * sx), (uint32_t)((xy[2 * i + 1] - y0) * sy)), (int)i};
        std::sort(key.begin(), key.end());
        int hint = 0;
        for (auto &k : key) hint = insert_point(k.second + 3, hint, true);
    }

    size_t size() const { return X.size(); }

    // ---- geometry on point indices
    double orient(int a, int b, int c) const { return pred::orient(X[a], Y[a], X[b], Y[b], X[c], Y[c]); }
    double orient_p(int a, int b, double px, double py) const { return pred::orient(X[a], Y[a], X[b], Y[b], px, py); }
    bool in_circle(int t, int d) const {
        const Tri &q = T[t];
        return pred::incircle(X[q.v[0]], Y[q.v[0]], X[q.v[1]], Y[q.v[1]], X[q.v[2]], Y[q.v[2]], X[d], Y[d]) > 0;
    }
    bool in_circle_p(int t, double px, double py) const {
        const Tri &q = T[t];
        return pred::incircle(X[q.v[0]], Y[q.v[0]], X[q.v[1]], Y[q.v[1]], X[q.v[2]], Y[q.v[2]], px, py) > 0;
    }
    static int idx(const Tri &t, int v) { return t.v[0] == v ? 0 : t.v[1] == v ? 1 : t.v[2] == v ? 2 : -1; }
    static int third(const Tri &t, int a, int b) {
        for (int k = 0; k < 3; ++k)
            if (t.v[k] != a && t.v[k] != b) return t.v[k];
        return -1;
    }
    bool is_super(int v) const { return v < 3; }

    // ---- topology
    void set_nb(int t, int old, int nw) {
        if (t < 0) return;
        Tri &q = T[t];
        for (int k = 0; k < 3; ++k)
            if (q.n[k] == old) { q.n[k] = nw; return; }
    }
    static bool cbit(const Tri &t, int k) { return (t.c >> k) & 1; }
    void set_c(int t, int k, bool on) {
        if (on) T[t].c |= (uint8_t)(1 << k);
        else T[t].c &= (uint8_t)~(1 << k);
    }

    // flip the edge opposite T[t].v[i]; afterwards T[t] = (a, b, d), T[u] = (a, d, c), a = the old T[t].v[i] at 0
    void flip(int t, int i) {
        const int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
        Tri tt = T[t];
        const int a = tt.v[i], b = tt.v[i1], c = tt.v[i2];
        const int u = tt.n[i];
        Tri uu = T[u];
        const int j = idx(uu, b) == -1 ? -1 : (idx(uu, b) + 1) % 3;    // u = (d, c, b): d after b
        const int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        const int d = uu.v[j];
        const int tA = tt.n[i1], tB = tt.n[i2], uB = uu.n[j1], uC = uu.n[j2];
        const bool fA = cbit(tt, i1), fB = cbit(tt, i2), fuB = cbit(uu, j1), fuC = cbit(uu, j2);
        T[t] = Tri{{a, b, d}, {uB, u, tB}, (uint8_t)((fuB ? 1 : 0) | (fB ? 4 : 0)), tt.outer};
        T[u] = Tri{{a, d, c}, {uC, tA, t}, (uint8_t)((fuC ? 1 : 0) | (fA ? 2 : 0)), uu.outer};
        set_nb(uB, u, t);
        set_nb(tA, t, u);
        vtri[a] = vtri[b] = vtri[d] = t;
        vtri[c] = u;
    }

    // Lawson: the edges on the stack, each opposite T[t].v[k], flipped until Delaunay (segments stay)
    void legalize(std::vector<std::pair<int, int>> &st) {
        while (!st.empty()) {
            auto [t, k] = st.back();
            st.pop_back();
            if (cbit(T[t], k)) continue;
            const int u = T[t].n[k];
            if (u < 0) continue;
            const Tri &uu = T[u];
            const int b = T[t].v[(k + 1) % 3];
            const int d = uu.v[(idx(uu, b) + 1) % 3];
            if (!in_circle(t, d)) continue;
            flip(t, k);
            st.push_back({t, 0});
            st.push_back({u, 0});
        }
    }

    int new_tri(const Tri &q) {
        T.push_back(q);
        return (int)T.size() - 1;
    }

    // p inside triangle t
    void split_tri(int t, int p) {
        Tri q = T[t];
        const int a = q.v[0], b = q.v[1], c = q.v[2];
        const int t1 = (int)T.size(), t2 = t1 + 1;
        T[t] = Tri{{p, b, c}, {q.n[0], t1, t2}, (uint8_t)(q.c & 1), q.outer};
        new_tri(Tri{{p, c, a}, {q.n[1], t2, t}, (uint8_t)((q.c >> 1) & 1), q.outer});
        new_tri(Tri{{p, a, b}, {q.n[2], t, t1}, (uint8_t)((q.c >> 2) & 1), q.outer});
        set_nb(q.n[1], t, t1);
        set_nb(q.n[2], t, t2);
        vtri[p] = vtri[b] = vtri[c] = t;
        vtri[a] = t1;
        std::vector<std::pair<int, int>> st{{t, 0}, {t1, 0}, {t2, 0}};
        legalize(st);
    }

    // p on the edge opposite T[t].v[i]: both triangles split in two (a segment's halves stay segments)
    void split_edge(int t, int i, int p) {
        const int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
        Tri tt = T[t];
        const int a = tt.v[i], b = tt.v[i1], c = tt.v[i2], u = tt.n[i];
        const bool ce = cbit(tt, i), fA = cbit(tt, i1), fB = cbit(tt, i2);
        const int tA = tt.n[i1], tB = tt.n[i2];
        if (u < 0) throw std::runtime_error("mesh2d: an edge without a neighbour");
        Tri uu = T[u];
        const int j = (idx(uu, b) + 1) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        const int d = uu.v[j], uB = uu.n[j1], uC = uu.n[j2];
        const bool fuB = cbit(uu, j1), fuC = cbit(uu, j2);
        const int tn = (int)T.size(), un = tn + 1;
        const uint8_t e = ce ? 1 : 0;
        T[t] = Tri{{a, b, p}, {un, tn, tB}, (uint8_t)(e | (fB ? 4 : 0)), tt.outer};
        new_tri(Tri{{a, p, c}, {u, tA, t}, (uint8_t)(e | (fA ? 2 : 0)), tt.outer});
        T[u] = Tri{{d, c, p}, {tn, un, uC}, (uint8_t)(e | (fuC ? 4 : 0)), uu.outer};
        new_tri(Tri{{d, p, b}, {t, uB, u}, (uint8_t)(e | (fuB ? 2 : 0)), uu.outer});
        set_nb(tA, t, tn);
        set_nb(uB, u, un);
        vtri[a] = vtri[b] = vtri[p] = t;
        vtri[c] = tn;
        vtri[d] = u;
        std::vector<std::pair<int, int>> st{{t, 2}, {tn, 1}, {u, 2}, {un, 1}};
        legalize(st);
    }

    // the triangle at (px, py) by a visibility walk from t; loc 0 inside, 1 on the edge opposite v[k], 2 on v[k]
    struct Loc { int t, loc, k; };
    Loc locate(double px, double py, int t) {
        uint32_t r = 0x9e3779b9u ^ (uint32_t)t;
        const size_t limit = 4 * T.size() + 64;
        for (size_t step = 0;; ++step) {
            if (step > limit) return locate_scan(px, py);
            const Tri &q = T[t];
            r = r * 1664525u + 1013904223u;
            const int s = (int)((r >> 16) % 3);
            int next = -1;
            for (int m = 0; m < 3; ++m) {
                const int k = (s + m) % 3;
                if (orient_p(q.v[(k + 1) % 3], q.v[(k + 2) % 3], px, py) < 0) { next = q.n[k]; break; }
            }
            if (next < 0) break;
            t = next;
        }
        return classify(t, px, py);
    }
    Loc locate_scan(double px, double py) {
        for (size_t t = 0; t < T.size(); ++t) {
            const Tri &q = T[t];
            bool in = true;
            for (int k = 0; k < 3 && in; ++k) in = orient_p(q.v[(k + 1) % 3], q.v[(k + 2) % 3], px, py) >= 0;
            if (in) return classify((int)t, px, py);
        }
        throw std::runtime_error("mesh2d: a point outside the bounding triangle");
    }
    Loc classify(int t, double px, double py) const {
        const Tri &q = T[t];
        for (int k = 0; k < 3; ++k)
            if (X[q.v[k]] == px && Y[q.v[k]] == py) return {t, 2, k};
        for (int k = 0; k < 3; ++k)
            if (orient_p(q.v[(k + 1) % 3], q.v[(k + 2) % 3], px, py) == 0) return {t, 1, k};
        return {t, 0, 0};
    }

    // insert point p (its coordinates already in X, Y) from the walk's start hint; returns a triangle at it
    int insert_point(int p, int hint, bool input = false) {
        Loc L = locate(X[p], Y[p], hint);
        if (L.loc == 2) {
            if (input) throw std::invalid_argument("mesh2d: duplicate points (merge them first)");
            return L.t;
        }
        if (L.loc == 1) split_edge(L.t, L.k, p);
        else split_tri(L.t, p);
        return vtri[p];
    }
    int add_point(double x, double y) {
        X.push_back(x), Y.push_back(y), vtri.push_back(-1);
        return (int)X.size() - 1;
    }

    // ---- edges
    // a triangle with the edge a-b, and the index of its third vertex (t = -1: no such edge)
    std::pair<int, int> find_edge(int a, int b) const {
        int t = vtri[a];
        const int t0 = t;
        do {
            const Tri &q = T[t];
            const int k = idx(q, a);
            if (q.v[(k + 1) % 3] == b) return {t, (k + 2) % 3};
            if (q.v[(k + 2) % 3] == b) return {t, (k + 1) % 3};
            t = q.n[(k + 1) % 3];                  // counter-clockwise about a
        } while (t >= 0 && t != t0);
        if (t >= 0) return {-1, -1};
        for (t = T[t0].n[(idx(T[t0], a) + 2) % 3]; t >= 0;) {   // a on the hull (the bounding triangle's): clockwise
            const Tri &q = T[t];
            const int k = idx(q, a);
            if (q.v[(k + 1) % 3] == b) return {t, (k + 2) % 3};
            if (q.v[(k + 2) % 3] == b) return {t, (k + 1) % 3};
            t = q.n[(k + 2) % 3];
        }
        return {-1, -1};
    }
    void constrain(int a, int b) {
        auto [t, k] = find_edge(a, b);
        set_c(t, k, true);
        const int u = T[t].n[k];
        if (u >= 0) set_c(u, (idx(T[u], T[t].v[(k + 1) % 3]) + 1) % 3, true);
    }
    // point c on segment a-b: exactly, or within min_dist of it (between its ends)
    bool on_segment(int a, int b, int c) const {
        const double o = orient(a, b, c);
        const double dx = X[b] - X[a], dy = Y[b] - Y[a], L2 = dx * dx + dy * dy;
        const double s = ((X[c] - X[a]) * dx + (Y[c] - Y[a]) * dy) / L2;
        if (s <= 0 || s >= 1) return false;
        if (o == 0) return true;
        const double area2 = dx * (Y[c] - Y[a]) - dy * (X[c] - X[a]);
        return min_dist > 0 && std::fabs(area2) < min_dist * std::sqrt(L2);
    }
    static bool crosses(double o1, double o2) { return (o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0); }

    void insert_segment(int a, int b) {
        std::vector<std::pair<int, int>> todo{{a, b}};
        while (!todo.empty()) {
            auto [s, e] = todo.back();
            todo.pop_back();
            if (s == e) continue;
            if (find_edge(s, e).first >= 0) { constrain(s, e); continue; }
            int stop = -1, hit_t = -1, hit_k = -1;
            std::vector<std::pair<int, int>> cross;     // the edges crossed, in order (as point pairs)
            // the triangle about s that the segment leaves through
            int t = vtri[s];
            const int t0 = t;
            int first = -1, p = -1, q = -1;
            do {
                const Tri &tr = T[t];
                const int k = idx(tr, s);
                const int v1 = tr.v[(k + 1) % 3], v2 = tr.v[(k + 2) % 3];
                if (!is_super(v1) && on_segment(s, e, v1)) { stop = v1; break; }
                if (!is_super(v2) && on_segment(s, e, v2)) { stop = v2; break; }
                // v2 left of s->e (p), v1 right (q)
                if (orient(s, v1, e) > 0 && orient(s, v2, e) < 0) { first = t, p = v2, q = v1; break; }
                t = tr.n[(k + 1) % 3];
            } while (t != t0);
            if (stop < 0 && first < 0) throw std::runtime_error("mesh2d: a segment leaves no triangle");
            if (stop < 0) {
                t = first;
                while (true) {
                    const Tri &tr = T[t];
                    const int kp = idx(tr, p);
                    // the edge p-q (opposite the third vertex); p is left of s->e, q right
                    const int kk = 3 - kp - idx(tr, q);
                    if (cbit(tr, kk)) { hit_t = t, hit_k = kk; break; }
                    cross.push_back({p, q});
                    const int u = tr.n[kk];
                    const Tri &uu = T[u];
                    const int w = third(uu, p, q);             // the point past p-q
                    if (w == e) break;
                    if (!is_super(w) && on_segment(s, e, w)) { stop = w; break; }
                    const double o = orient(s, e, w);
                    if (o > 0) p = w;
                    else q = w;
                    t = u;
                }
            }
            if (hit_t >= 0) {
                // a segment in the way: split both where they cross
                const Tri &tr = T[hit_t];
                const int c1 = tr.v[(hit_k + 1) % 3], c2 = tr.v[(hit_k + 2) % 3];
                const double x1 = X[s], y1 = Y[s], x2 = X[e], y2 = Y[e], x3 = X[c1], y3 = Y[c1], x4 = X[c2], y4 = Y[c2];
                const double den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3);
                double w = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den;   // along c1-c2
                w = std::min(1.0 - 1e-12, std::max(1e-12, w));
                const int m = add_point(x3 + w * (x4 - x3), y3 + w * (y4 - y3));
                // the four triangles about it counter-clockwise, or the crossing is too fine for doubles here
                const int o1 = tr.v[hit_k], u = tr.n[hit_k], o2 = third(T[u], c1, c2);
                if (!(orient(o1, c1, m) > 0 && orient(o1, m, c2) > 0 && orient(o2, c2, m) > 0 && orient(o2, m, c1) > 0))
                    throw std::runtime_error("mesh2d: segments cross where the points are too close to resolve "
                                             "(merge them first)");
                split_edge(hit_t, hit_k, m);
                todo.push_back({m, e});
                todo.push_back({s, m});
                continue;
            }
            const int end = stop >= 0 ? stop : e;
            if (stop >= 0 && cross.empty()) {           // s's neighbour on the segment
                constrain(s, stop);
                todo.push_back({stop, e});
                continue;
            }
            flip_out(s, end, cross);
            if (stop >= 0) todo.push_back({stop, e});
        }
    }

    // Sloan: flip the edges crossing s-e away (cross: they, in order), constrain s-e, restore Delaunay
    void flip_out(int s, int e, std::vector<std::pair<int, int>> &cross) {
        std::deque<std::pair<int, int>> Q;
        for (auto &[p, q] : cross)                  // (near a point taken as on it: only those that cross s-e)
            if (p != s && q != s && p != e && q != e && crosses(orient(s, e, p), orient(s, e, q)) &&
                crosses(orient(p, q, s), orient(p, q, e)))
                Q.push_back({p, q});
        std::vector<std::pair<int, int>> made;
        size_t guard = 0, limit = 64 * (Q.size() + 4) * (Q.size() + 4);
        while (!Q.empty()) {
            if (++guard > limit) throw std::runtime_error("mesh2d: a segment could not be inserted");
            auto [p, q] = Q.front();
            Q.pop_front();
            auto [t, k] = find_edge(p, q);
            if (t < 0) continue;
            const Tri &tr = T[t];
            const int o1 = tr.v[k], u = tr.n[k];
            const Tri &uu = T[u];
            const int o2 = uu.v[(idx(uu, tr.v[(k + 1) % 3]) + 1) % 3];
            // convex: p and q on either side of o1-o2
            if (!crosses(orient(o1, o2, p), orient(o1, o2, q))) { Q.push_back({p, q}); continue; }
            flip(t, k);
            if (o1 != s && o1 != e && o2 != s && o2 != e &&
                crosses(orient(s, e, o1), orient(s, e, o2)) && crosses(orient(o1, o2, s), orient(o1, o2, e)))
                Q.push_back({o1, o2});
            else
                made.push_back({o1, o2});
        }
        if (find_edge(s, e).first < 0)
            throw std::runtime_error("mesh2d: a segment could not be inserted (" + std::to_string(s - 3) + ", " +
                                     std::to_string(e - 3) + ")");
        constrain(s, e);
        bool again = true;
        for (int pass = 0; again && pass < 1000; ++pass) {
            again = false;
            for (auto &[p, q] : made) {
                if ((p == s && q == e) || (p == e && q == s)) continue;
                auto [t, k] = find_edge(p, q);
                if (t < 0 || cbit(T[t], k)) continue;
                const int u = T[t].n[k];
                const int d = T[u].v[(idx(T[u], T[t].v[(k + 1) % 3]) + 1) % 3];
                if (!in_circle(t, d)) continue;
                const int o1 = T[t].v[k];
                flip(t, k);
                p = o1, q = d;
                again = true;
            }
        }
    }

    // ---- regions
    void mark_outer() {
        for (auto &q : T) q.outer = 0;
        std::vector<int> st;
        for (size_t t = 0; t < T.size(); ++t)
            if (is_super(T[t].v[0]) || is_super(T[t].v[1]) || is_super(T[t].v[2])) T[t].outer = 1, st.push_back((int)t);
        while (!st.empty()) {
            const int t = st.back();
            st.pop_back();
            for (int k = 0; k < 3; ++k) {
                const int u = T[t].n[k];
                if (u >= 0 && !cbit(T[t], k) && !T[u].outer) T[u].outer = 1, st.push_back(u);
            }
        }
    }

    // ---- refinement
    double min_angle_of(int t, int *at = nullptr) const {
        const Tri &q = T[t];
        double best = 10;
        for (int k = 0; k < 3; ++k) {
            const int a = q.v[k], b = q.v[(k + 1) % 3], c = q.v[(k + 2) % 3];
            const double ux = X[b] - X[a], uy = Y[b] - Y[a], vx = X[c] - X[a], vy = Y[c] - Y[a];
            const double ang = std::atan2(std::fabs(ux * vy - uy * vx), ux * vx + uy * vy);
            if (ang < best) {
                best = ang;
                if (at) *at = k;
            }
        }
        return best;
    }
    double shortest(int t) const {
        const Tri &q = T[t];
        double s = INFINITY;
        for (int k = 0; k < 3; ++k) {
            const int a = q.v[k], b = q.v[(k + 1) % 3];
            s = std::min(s, std::hypot(X[b] - X[a], Y[b] - Y[a]));
        }
        return s;
    }
    // segment a-b encroached: a point of an inner triangle beside it inside its diametral circle
    bool encroached(int a, int b) const {
        auto [t, k] = find_edge(a, b);
        if (t < 0 || !cbit(T[t], k)) return false;
        auto apex_in = [&](int tt, int kk) {
            if (T[tt].outer) return false;
            const int c = T[tt].v[kk];
            return (X[a] - X[c]) * (X[b] - X[c]) + (Y[a] - Y[c]) * (Y[b] - Y[c]) < 0;
        };
        if (apex_in(t, k)) return true;
        const int u = T[t].n[k];
        return u >= 0 && apex_in(u, (idx(T[u], T[t].v[(k + 1) % 3]) + 1) % 3);
    }
    // where to split segment a-b: its middle, or (one end a caller's point) a power of two from that end, so that
    // segments meeting there at a sharp angle are split on the same circles
    std::pair<double, double> split_point(int a, int b) const {
        const double L = std::hypot(X[b] - X[a], Y[b] - Y[a]);
        const bool ia = a >= 3 && (size_t)a < n_input + 3, ib = b >= 3 && (size_t)b < n_input + 3;
        double f = 0.5;
        if (ia != ib && L > 0) {
            const double d = std::exp2(std::round(std::log2(0.5 * L)));
            const double g = d / L;
            if (g > 0.3 && g < 0.7) f = ia ? g : 1 - g;
        }
        return {X[a] + f * (X[b] - X[a]), Y[a] + f * (Y[b] - Y[a])};
    }

    // the refinement; returns the points added
    long long refine(double min_angle, long long budget, double min_edge) {
        long long added = 0;
        std::deque<std::pair<int, int>> segq;
        std::deque<int> triq;
        auto bad = [&](int t) {
            const Tri &q = T[t];
            if (q.outer) return false;
            int at = 0;
            if (min_angle_of(t, &at) >= min_angle) return false;
            if (cbit(q, (at + 1) % 3) && cbit(q, (at + 2) % 3)) return false;   // between two segments: as it is
            return shortest(t) > min_edge;
        };
        auto around = [&](int p) {                 // the triangles at p into the queues
            int t = vtri[p];
            const int t0 = t;
            do {
                const Tri &q = T[t];
                const int k = idx(q, p);
                if (bad(t)) triq.push_back(t);
                for (int m = 0; m < 3; ++m)
                    if (cbit(q, m) && !q.outer) {
                        const int a = q.v[(m + 1) % 3], b = q.v[(m + 2) % 3];
                        const int c = q.v[m];
                        if ((X[a] - X[c]) * (X[b] - X[c]) + (Y[a] - Y[c]) * (Y[b] - Y[c]) < 0) segq.push_back({a, b});
                    }
                t = q.n[(k + 1) % 3];
            } while (t >= 0 && t != t0);
        };
        auto split_seg = [&](int a, int b) {
            auto [t, k] = find_edge(a, b);
            if (t < 0 || !cbit(T[t], k)) return false;
            if (std::hypot(X[b] - X[a], Y[b] - Y[a]) <= 2 * min_edge) return false;
            auto [x, y] = split_point(a, b);
            const int m = add_point(x, y);
            split_edge(t, k, m);
            ++added;
            around(m);
            return true;
        };
        for (size_t t = 0; t < T.size(); ++t) {
            const Tri &q = T[t];
            if (q.outer) continue;
            for (int m = 0; m < 3; ++m)
                if (cbit(q, m)) {
                    const int a = q.v[(m + 1) % 3], b = q.v[(m + 2) % 3], c = q.v[m];
                    if ((X[a] - X[c]) * (X[b] - X[c]) + (Y[a] - Y[c]) * (Y[b] - Y[c]) < 0) segq.push_back({a, b});
                }
            if (bad((int)t)) triq.push_back((int)t);
        }
        std::vector<int> stamp(T.size(), -1);
        int pass = 0;
        while (added < budget) {
            if (!segq.empty()) {
                auto [a, b] = segq.front();
                segq.pop_front();
                if (encroached(a, b)) split_seg(a, b);
                continue;
            }
            if (triq.empty()) break;
            const int t = triq.front();
            triq.pop_front();
            if (!bad(t)) continue;
            const Tri q = T[t];
            const int a = q.v[0], b = q.v[1], c = q.v[2];
            const double bx = X[b] - X[a], by = Y[b] - Y[a], cx = X[c] - X[a], cy = Y[c] - Y[a];
            const double d = 2 * (bx * cy - by * cx);
            if (d == 0) continue;
            const double b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
            const double px = X[a] + (cy * b2 - by * c2) / d, py = Y[a] + (bx * c2 - cx * b2) / d;
            // the straight walk from the triangle's centroid to its circumcentre: a segment in the way is split
            const double sx = (X[a] + X[b] + X[c]) / 3, sy = (Y[a] + Y[b] + Y[c]) / 3;
            int cur = t, blocked_k = -1;
            bool found = false;
            for (size_t step = 0; step < T.size() + 8; ++step) {
                const Tri &r = T[cur];
                int exit = -1, any = -1;
                for (int k = 0; k < 3; ++k) {
                    const int u = r.v[(k + 1) % 3], v = r.v[(k + 2) % 3];
                    if (orient_p(u, v, px, py) >= 0) continue;
                    any = k;
                    const double ou = pred::orient(sx, sy, px, py, X[u], Y[u]), ov = pred::orient(sx, sy, px, py, X[v], Y[v]);
                    if (ou * ov <= 0) { exit = k; break; }
                }
                if (any < 0) { found = true; break; }
                if (exit < 0) exit = any;
                if (cbit(r, exit)) { blocked_k = exit; break; }
                cur = r.n[exit];
            }
            if (blocked_k >= 0) {
                const Tri &r = T[cur];
                if (split_seg(r.v[(blocked_k + 1) % 3], r.v[(blocked_k + 2) % 3])) triq.push_back(t);
                continue;
            }
            if (!found || T[cur].outer) continue;
            // the cavity the circumcentre would open: a segment on its rim that the point encroaches is split instead
            ++pass;
            if (stamp.size() < T.size()) stamp.resize(T.size(), -1);
            std::vector<int> st{cur}, rim;
            stamp[cur] = pass;
            std::vector<std::pair<int, int>> enc;
            while (!st.empty()) {
                const int x = st.back();
                st.pop_back();
                for (int k = 0; k < 3; ++k) {
                    const int u = T[x].v[(k + 1) % 3], v = T[x].v[(k + 2) % 3];
                    if (cbit(T[x], k)) {
                        if ((X[u] - px) * (X[v] - px) + (Y[u] - py) * (Y[v] - py) < 0) enc.push_back({u, v});
                        continue;
                    }
                    const int nb = T[x].n[k];
                    if (nb < 0 || stamp[nb] == pass) continue;
                    if (in_circle_p(nb, px, py)) stamp[nb] = pass, st.push_back(nb);
                }
            }
            if (!enc.empty()) {
                bool any = false;
                for (auto &[u, v] : enc) any |= split_seg(u, v);
                if (any) triq.push_back(t);
                continue;
            }
            Loc L = classify(cur, px, py);
            if (L.loc == 2) continue;
            const int m = add_point(px, py);
            if (L.loc == 1) split_edge(L.t, L.k, m);
            else split_tri(L.t, m);
            ++added;
            around(m);
        }
        return added;
    }

    static uint64_t hilbert(uint32_t x, uint32_t y) {
        uint64_t d = 0;
        for (uint32_t s = 1u << 15; s > 0; s >>= 1) {
            const uint32_t rx = (x & s) > 0, ry = (y & s) > 0;
            d += (uint64_t)s * s * ((3 * rx) ^ ry);
            if (ry == 0) {
                if (rx == 1) x = 65535u - x, y = 65535u - y;
                std::swap(x, y);
            }
        }
        return d;
    }
};

}  // namespace mesh2d
