// mom25d_core.cpp - compiled kernels for mom25d (pybind11 + OpenMP)
//
//   fill_far      : centroid-rule (far-field) entries for all edge pairs, written straight into a
//                   dense Z (complex128 or complex64), symmetric, row-parallel, no temporaries
//   far_block     : the same for arbitrary row / column edge lists (ACA callbacks)
//   matvec_far    : y = Z_far x WITHOUT storing Z (triangle-aggregated, O(Nt^2) work, O(N) memory)
//   near_correct  : exact-minus-centroid correction for near triangle pairs (analytic 1/R,
//                   off-plane and image parts + quadrature of the tabulated rest), scattered into a
//                   dense Z (atomic adds), into a y = dZ x product, or returned as COO triplets
//   offplane_pairs: analytic 1/sqrt(rho^2+d^2) moments for pair lists (validation / reuse)
//   near_pairs    : the near triangle pairs (a grid search; the KD-tree + np.unique of fast.py, same order)
//   hankel_real   : sum_k J0(rho_i kr_k) W[s, k] over the real part of a Sommerfeld contour, without the
//                   (n_rho x n_k) Bessel matrix (the Green tables and the via kernels)
//   scale_c64     : Z <- D Z D (equilibration) and its complex64 copy, one pass (the GPU LU's input)
//   fan_q         : G_q moments between every triangle and the via fan triangles (centroid rule)
//   near_q        : the near pairs' G_q moment corrections alone (the loop/tree basis's charge matrix)
//   tree_cols/rows: X K and K^T X for the loop/tree basis; charge_into: its charge block into Z
//   BLRFactor     : the block low-rank LU (solve_blr's preconditioner), SciPy's BLAS and LAPACK
//   project       : T^T Z T for the quad cells / line functions (quads.project), Z's rows reduced in place
//
// Everything mirrors the NumPy implementation in mom25d_fast.py (same tables and conventions),
// which stays the fallback when this module is not built.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <math.h>
#include <cstdint>
#include <chrono>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using cd = std::complex<double>;
using cf = std::complex<float>;
using i64 = std::int64_t;           // not long: 32 bits on Windows (LLP64), 64 on Linux

// J0: POSIX's j0, which Windows' C runtime (MSVC, MinGW-w64 under -std=c++17) has only as _j0
static inline double bessel_j0(double x)
{
#ifdef _WIN32
    return _j0(x);
#else
    return ::j0(x);
#endif
}
template <class T> using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------------------------
// tabulated layered-media kernels
// ---------------------------------------------------------------------------------------------
struct Tables {
    int ntab = 0, nfine = 0, L = 0;
    const cd *A = nullptr, *Q = nullptr, *cA = nullptr, *cq = nullptr, *cAi = nullptr, *cqi = nullptr;
    const double *dr = nullptr, *d = nullptr, *dimg = nullptr;
    const int *tab = nullptr, *kind = nullptr;   // kind: 0 same interface, 1 cross (extracted), 2 cross (plain)

    inline void rest(int t, double r, cd &ga, cd &gq) const {
        double u = r / dr[t];
        const double umax = nfine - 1.000001;
        if (u > umax) u = umax;
        int i = (int)u; double f = u - i;
        const cd *a = A + (size_t)t * nfine, *q = Q + (size_t)t * nfine;
        ga = a[i] * (1 - f) + a[i + 1] * f;
        gq = q[i] * (1 - f) + q[i + 1] * f;
    }
    // full kernel at a centroid distance (singular parts added; r = 0 on the same interface -> 0)
    inline void kern(int la, int lb, double r, cd &ga, cd &gq) const {
        int t = tab[la * L + lb];
        rest(t, r, ga, gq);
        if (kind[t] == 0) {
            if (r > 0) { ga += cA[t] / r; gq += cq[t] / r; }
            if (dimg[t] > 0) { double ri = 1.0 / std::sqrt(r * r + dimg[t] * dimg[t]); ga += cAi[t] * ri; gq += cqi[t] * ri; }
        } else if (kind[t] == 1) {
            double ri = 1.0 / std::sqrt(r * r + d[t] * d[t]); ga += cA[t] * ri; gq += cq[t] * ri;
        }
    }
};

struct TabHold {   // keeps the numpy buffers alive while a Tables view is used
    arr<cd> A, Q, cA, cq, cAi, cqi; arr<double> dr, d, dimg; arr<int> tab, kind;
    Tables t;
    TabHold(py::dict tb) :
        A(tb["A"]), Q(tb["Q"]), cA(tb["cA"]), cq(tb["cq"]), cAi(tb["cAi"]), cqi(tb["cqi"]),
        dr(tb["dr"]), d(tb["d"]), dimg(tb["dimg"]), tab(tb["tab"]), kind(tb["kind"]) {
        t.ntab = (int)A.shape(0); t.nfine = (int)A.shape(1); t.L = (int)tab.shape(0);
        t.A = A.data(); t.Q = Q.data(); t.cA = cA.data(); t.cq = cq.data(); t.cAi = cAi.data();
        t.cqi = cqi.data(); t.dr = dr.data(); t.d = d.data(); t.dimg = dimg.data(); t.tab = tab.data();
        t.kind = kind.data();
    }
};

// mesh / basis description shared by the routines
struct Mesh {
    int Ne = 0, Nt = 0;
    const int *ep, *em, *lt, *te;                      // edge -> T+, T-; triangle layer; triangle -> 3 edge slots
    const double *ap, *am, *vp, *vm, *cent, *area;     // alphas (signed), free vertices, centroids, areas
    const double *ta, *tv;                             // per triangle slot: alpha, free vertex
};
struct MeshHold {
    arr<int> ep, em, lt, te; arr<double> ap, am, vp, vm, cent, area, ta, tv; Mesh m;
    MeshHold(py::dict g) :
        ep(g["ep"]), em(g["em"]), lt(g["lt"]), te(g["te"]), ap(g["ap"]), am(g["am"]), vp(g["vp"]),
        vm(g["vm"]), cent(g["cent"]), area(g["area"]), ta(g["ta"]), tv(g["tv"]) {
        m.Ne = (int)ep.shape(0); m.Nt = (int)lt.shape(0);
        m.ep = ep.data(); m.em = em.data(); m.lt = lt.data(); m.te = te.data(); m.ap = ap.data();
        m.am = am.data(); m.vp = vp.data(); m.vm = vm.data(); m.cent = cent.data(); m.area = area.data();
        m.ta = ta.data(); m.tv = tv.data();
    }
};

static inline double dist(const double *c, int p, int q) {
    double dx = c[2 * p] - c[2 * q], dy = c[2 * p + 1] - c[2 * q + 1];
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------------------------
// far field: dense fill (symmetric) - row-parallel, kernel rows for the two triangles of edge m
// ---------------------------------------------------------------------------------------------
template <class T>
void fill_far_t(py::array_t<T> Z, py::dict geom, py::dict tabs, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    const Mesh &g = mh.m; const Tables &tb = th.t;
    auto buf = Z.request();
    T *z = (T *)buf.ptr;
    const size_t ld = buf.shape[1];
    const cd jw(0, w), inv_jw = 1.0 / cd(0, w);
    const int Ne = g.Ne, Nt = g.Nt;
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        std::vector<cd> GA(2 * (size_t)Nt), GQ(2 * (size_t)Nt);
#pragma omp for schedule(dynamic, 8)
        for (int m = 0; m < Ne; ++m) {
            int pm[2] = {g.ep[m], g.em[m]};
            for (int s = 0; s < 2; ++s)
                for (int q = 0; q < Nt; ++q)
                    tb.kern(g.lt[pm[s]], g.lt[q], dist(g.cent, pm[s], q), GA[s * Nt + q], GQ[s * Nt + q]);
            double am_[2] = {g.ap[m] * g.area[pm[0]], g.am[m] * g.area[pm[1]]};
            double dx[2], dy[2];
            dx[0] = g.cent[2 * pm[0]] - g.vp[2 * m]; dy[0] = g.cent[2 * pm[0] + 1] - g.vp[2 * m + 1];
            dx[1] = g.cent[2 * pm[1]] - g.vm[2 * m]; dy[1] = g.cent[2 * pm[1] + 1] - g.vm[2 * m + 1];
            for (int n = m; n < Ne; ++n) {
                int pn[2] = {g.ep[n], g.em[n]};
                double an[2] = {g.ap[n] * g.area[pn[0]], g.am[n] * g.area[pn[1]]};
                double ex[2], ey[2];
                ex[0] = g.cent[2 * pn[0]] - g.vp[2 * n]; ey[0] = g.cent[2 * pn[0] + 1] - g.vp[2 * n + 1];
                ex[1] = g.cent[2 * pn[1]] - g.vm[2 * n]; ey[1] = g.cent[2 * pn[1] + 1] - g.vm[2 * n + 1];
                cd v = 0;
                for (int s = 0; s < 2; ++s)
                    for (int t = 0; t < 2; ++t) {
                        size_t k = s * (size_t)Nt + pn[t];
                        double dot = dx[s] * ex[t] + dy[s] * ey[t];
                        v += am_[s] * an[t] * (jw * dot * GA[k] + 4.0 * GQ[k] * inv_jw);
                    }
                z[(size_t)m * ld + n] = T(v);
                z[(size_t)n * ld + m] = T(v);
            }
        }
    }
}

// far field for arbitrary edge lists (ACA row / column callbacks)
py::array_t<cd> far_block(arr<int> rows, arr<int> cols, py::dict geom, py::dict tabs, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    const Mesh &g = mh.m; const Tables &tb = th.t;
    const int nr = (int)rows.shape(0), nc = (int)cols.shape(0);
    py::array_t<cd> out({nr, nc});
    cd *o = out.mutable_data();
    const int *R = rows.data(), *C = cols.data();
    const cd jw(0, w), inv_jw = 1.0 / cd(0, w);
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(dynamic, 4) if (nr * (i64)nc > 4096)
        for (int i = 0; i < nr; ++i) {
            int m = R[i];
            int pm[2] = {g.ep[m], g.em[m]};
            double am_[2] = {g.ap[m] * g.area[pm[0]], g.am[m] * g.area[pm[1]]};
            double dx[2] = {g.cent[2 * pm[0]] - g.vp[2 * m], g.cent[2 * pm[1]] - g.vm[2 * m]};
            double dy[2] = {g.cent[2 * pm[0] + 1] - g.vp[2 * m + 1], g.cent[2 * pm[1] + 1] - g.vm[2 * m + 1]};
            for (int j = 0; j < nc; ++j) {
                int n = C[j];
                int pn[2] = {g.ep[n], g.em[n]};
                double an[2] = {g.ap[n] * g.area[pn[0]], g.am[n] * g.area[pn[1]]};
                double ex[2] = {g.cent[2 * pn[0]] - g.vp[2 * n], g.cent[2 * pn[1]] - g.vm[2 * n]};
                double ey[2] = {g.cent[2 * pn[0] + 1] - g.vp[2 * n + 1], g.cent[2 * pn[1] + 1] - g.vm[2 * n + 1]};
                cd v = 0;
                for (int s = 0; s < 2; ++s)
                    for (int t = 0; t < 2; ++t) {
                        cd ga, gq;
                        tb.kern(g.lt[pm[s]], g.lt[pn[t]], dist(g.cent, pm[s], pn[t]), ga, gq);
                        v += am_[s] * an[t] * (jw * (dx[s] * ex[t] + dy[s] * ey[t]) * ga + 4.0 * gq * inv_jw);
                    }
                o[(size_t)i * nc + j] = v;
            }
        }
    }
    return out;
}

// y = Z_far x  without storing Z: aggregate sources per triangle, one kernel row per triangle
py::array_t<cd> matvec_far(arr<cd> x, py::dict geom, py::dict tabs, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    const Mesh &g = mh.m; const Tables &tb = th.t;
    const int Ne = g.Ne, Nt = g.Nt;
    const cd *X = x.data();
    const cd jw(0, w), inv_jw = 1.0 / cd(0, w);
    // per triangle: s0 = sum a A x,  s1 = sum a A x (c - v)
    std::vector<cd> s0(Nt, 0.0), s1x(Nt, 0.0), s1y(Nt, 0.0);
    for (int e = 0; e < Ne; ++e) {
        int p = g.ep[e], q = g.em[e];
        cd a = g.ap[e] * g.area[p] * X[e], b = g.am[e] * g.area[q] * X[e];
        s0[p] += a; s1x[p] += a * (g.cent[2 * p] - g.vp[2 * e]); s1y[p] += a * (g.cent[2 * p + 1] - g.vp[2 * e + 1]);
        s0[q] += b; s1x[q] += b * (g.cent[2 * q] - g.vm[2 * e]); s1y[q] += b * (g.cent[2 * q + 1] - g.vm[2 * e + 1]);
    }
    std::vector<cd> vAx(Nt), vAy(Nt), wQ(Nt);
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(dynamic, 16)
        for (int p = 0; p < Nt; ++p) {
            cd ax = 0, ay = 0, aq = 0;
            for (int q = 0; q < Nt; ++q) {
                if (s0[q] == 0.0 && s1x[q] == 0.0 && s1y[q] == 0.0) continue;
                cd ga, gq;
                tb.kern(g.lt[p], g.lt[q], dist(g.cent, p, q), ga, gq);
                ax += ga * s1x[q]; ay += ga * s1y[q]; aq += gq * s0[q];
            }
            vAx[p] = ax; vAy[p] = ay; wQ[p] = aq;
        }
    }
    py::array_t<cd> out(Ne);
    cd *Y = out.mutable_data();
    for (int e = 0; e < Ne; ++e) {
        int p = g.ep[e], q = g.em[e];
        cd yp = g.ap[e] * g.area[p] * (jw * ((g.cent[2 * p] - g.vp[2 * e]) * vAx[p] + (g.cent[2 * p + 1] - g.vp[2 * e + 1]) * vAy[p]) + 4.0 * wQ[p] * inv_jw);
        cd ym = g.am[e] * g.area[q] * (jw * ((g.cent[2 * q] - g.vm[2 * e]) * vAx[q] + (g.cent[2 * q + 1] - g.vm[2 * e + 1]) * vAy[q]) + 4.0 * wQ[q] * inv_jw);
        Y[e] = yp + ym;
    }
    return out;
}


// Y = Z_far X for k right-hand sides (X: Ne x k, row-major): one kernel evaluation per triangle
// pair serves all columns, so k ports cost about the same as one
py::array_t<cd> matvec_far_multi(arr<cd> x, py::dict geom, py::dict tabs, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    const Mesh &g = mh.m; const Tables &tb = th.t;
    const int Ne = g.Ne, Nt = g.Nt;
    if (x.ndim() != 2 || x.shape(0) != Ne) throw std::runtime_error("matvec_far_multi: X must be (Ne, k)");
    const int K = (int)x.shape(1);
    const cd *X = x.data();
    const cd jw(0, w), inv_jw = 1.0 / cd(0, w);
    std::vector<cd> s0((size_t)Nt * K, 0.0), s1x((size_t)Nt * K, 0.0), s1y((size_t)Nt * K, 0.0);
    std::vector<char> act(Nt, 0);
    for (int e = 0; e < Ne; ++e) {
        int p = g.ep[e], q = g.em[e];
        for (int c = 0; c < K; ++c) {
            cd xe = X[(size_t)e * K + c];
            cd a = g.ap[e] * g.area[p] * xe, b = g.am[e] * g.area[q] * xe;
            size_t ip = (size_t)p * K + c, iq = (size_t)q * K + c;
            s0[ip] += a; s1x[ip] += a * (g.cent[2 * p] - g.vp[2 * e]); s1y[ip] += a * (g.cent[2 * p + 1] - g.vp[2 * e + 1]);
            s0[iq] += b; s1x[iq] += b * (g.cent[2 * q] - g.vm[2 * e]); s1y[iq] += b * (g.cent[2 * q + 1] - g.vm[2 * e + 1]);
        }
        act[p] = act[q] = 1;
    }
    std::vector<cd> vAx((size_t)Nt * K), vAy((size_t)Nt * K), wQ((size_t)Nt * K);
    {
        py::gil_scoped_release nogil;
#pragma omp parallel
        {
            std::vector<cd> ax(K), ay(K), aq(K);
#pragma omp for schedule(dynamic, 16)
            for (int p = 0; p < Nt; ++p) {
                std::fill(ax.begin(), ax.end(), cd(0)); std::fill(ay.begin(), ay.end(), cd(0)); std::fill(aq.begin(), aq.end(), cd(0));
                for (int q = 0; q < Nt; ++q) {
                    if (!act[q]) continue;
                    cd ga, gq;
                    tb.kern(g.lt[p], g.lt[q], dist(g.cent, p, q), ga, gq);
                    const cd *S0 = &s0[(size_t)q * K], *SX = &s1x[(size_t)q * K], *SY = &s1y[(size_t)q * K];
                    for (int c = 0; c < K; ++c) { ax[c] += ga * SX[c]; ay[c] += ga * SY[c]; aq[c] += gq * S0[c]; }
                }
                for (int c = 0; c < K; ++c) {
                    vAx[(size_t)p * K + c] = ax[c]; vAy[(size_t)p * K + c] = ay[c]; wQ[(size_t)p * K + c] = aq[c];
                }
            }
        }
    }
    py::array_t<cd> out({(py::ssize_t)Ne, (py::ssize_t)K});
    cd *Y = out.mutable_data();
    for (int e = 0; e < Ne; ++e) {
        int p = g.ep[e], q = g.em[e];
        double dpx = g.cent[2 * p] - g.vp[2 * e], dpy = g.cent[2 * p + 1] - g.vp[2 * e + 1];
        double dqx = g.cent[2 * q] - g.vm[2 * e], dqy = g.cent[2 * q + 1] - g.vm[2 * e + 1];
        for (int c = 0; c < K; ++c) {
            size_t ip = (size_t)p * K + c, iq = (size_t)q * K + c;
            cd yp = g.ap[e] * g.area[p] * (jw * (dpx * vAx[ip] + dpy * vAy[ip]) + 4.0 * wQ[ip] * inv_jw);
            cd ym = g.am[e] * g.area[q] * (jw * (dqx * vAx[iq] + dqy * vAy[iq]) + 4.0 * wQ[iq] * inv_jw);
            Y[(size_t)e * K + c] = yp + ym;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// analytic 1/sqrt(rho^2 + d^2) moments: one integral analytic over a triangle (V), the other a 6-point rule over
// the other (P6, W6). The rule runs over the smaller triangle of a pair (rule_over_p, as sing.rule_over): the pair's
// moments are one value whichever of the two is the observation (Z symmetric), and the accurate ones
// ---------------------------------------------------------------------------------------------
static inline bool rule_over_p(double ap, double aq, int p, int q) {
    const double tie = 1e-9 * std::max(ap, aq);              // sing.RULE_TIE
    return std::fabs(ap - aq) <= tie ? p <= q : ap < aq;
}

static inline double tri_area(const double *nd, const int *tr, int t) {
    const double *a = nd + 2 * tr[3 * t], *b = nd + 2 * tr[3 * t + 1], *c = nd + 2 * tr[3 * t + 2];
    return 0.5 * std::fabs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
}

static inline void offplane_one(const double *V /*3x2 of q*/, const double *P6, const double *W6, double d,
                                double &m0, double mro[2], double mrs[2], double &mrr) {
    double ad = std::fabs(d);
    double e[3][2], L[3], tau[3][2], un[3][2];
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        e[i][0] = V[2 * j] - V[2 * i]; e[i][1] = V[2 * j + 1] - V[2 * i + 1];
        L[i] = std::sqrt(e[i][0] * e[i][0] + e[i][1] * e[i][1]);
        tau[i][0] = e[i][0] / L[i]; tau[i][1] = e[i][1] / L[i];
        un[i][0] = tau[i][1]; un[i][1] = -tau[i][0];
    }
    m0 = 0; mro[0] = mro[1] = 0; mrs[0] = mrs[1] = 0; mrr = 0;
    for (int k = 0; k < 6; ++k) {
        double rx = P6[2 * k], ry = P6[2 * k + 1], wq = W6[k];
        double I1 = 0, IRx = 0, IRy = 0;
        for (int i = 0; i < 3; ++i) {
            double dvx = V[2 * i] - rx, dvy = V[2 * i + 1] - ry;
            double sm = dvx * tau[i][0] + dvy * tau[i][1];
            double sp = sm + L[i];
            double t0 = dvx * un[i][0] + dvy * un[i][1];
            double R02 = t0 * t0 + d * d, R0 = std::sqrt(R02);
            double Rm = std::sqrt(R02 + sm * sm), Rp = std::sqrt(R02 + sp * sp);
            double lg = (R0 > 1e-14 * L[i]) ? std::asinh(sp / R0) - std::asinh(sm / R0) : 0.0;
            double beta = std::atan2(t0 * sp, R02 + ad * Rp) - std::atan2(t0 * sm, R02 + ad * Rm);
            I1 += t0 * lg - ad * beta;
            double eR = 0.5 * (R02 * lg + sp * Rp - sm * Rm);
            IRx += eR * un[i][0]; IRy += eR * un[i][1];
        }
        double Wx = IRx + rx * I1, Wy = IRy + ry * I1;
        m0 += wq * I1;
        mro[0] += wq * rx * I1; mro[1] += wq * ry * I1;
        mrs[0] += wq * Wx; mrs[1] += wq * Wy;
        mrr += wq * (rx * Wx + ry * Wy);
    }
}

py::tuple offplane_pairs(arr<double> nodes, arr<int> tris, arr<double> P6, arr<double> W6, arr<int> P,
                         arr<int> Q, double d) {
    const int n = (int)P.shape(0);
    py::array_t<double> m0(n), mro({n, 2}), mrs({n, 2}), mrr(n);
    double *a = m0.mutable_data(), *b = mro.mutable_data(), *c = mrs.mutable_data(), *e = mrr.mutable_data();
    const double *nd = nodes.data(), *p6 = P6.data(), *w6 = W6.data();
    const int *tr = tris.data(), *pp = P.data(), *qq = Q.data();
    {
        py::gil_scoped_release nogil;                        // (held again for the tuple: it was not, and the call failed)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const bool keep = rule_over_p(tri_area(nd, tr, pp[i]), tri_area(nd, tr, qq[i]), pp[i], qq[i]);
            const int po = keep ? pp[i] : qq[i], qi = keep ? qq[i] : pp[i];  // the rule over po, analytic over qi
            double V[6];
            for (int k = 0; k < 3; ++k) { V[2 * k] = nd[2 * tr[3 * qi + k]]; V[2 * k + 1] = nd[2 * tr[3 * qi + k] + 1]; }
            offplane_one(V, p6 + 12 * (size_t)po, w6 + 6 * (size_t)po, d, a[i], keep ? b + 2 * i : c + 2 * i,
                         keep ? c + 2 * i : b + 2 * i, e[i]);
        }
    }
    return py::make_tuple(m0, mro, mrs, mrr);
}

// ---------------------------------------------------------------------------------------------
// near-field correction: accurate - centroid, per near triangle pair
//   mode 0: atomically add into dense Z (any T);  mode 1: y += dZ x  (x given);  mode 2: COO output
// ---------------------------------------------------------------------------------------------
struct NearOut { int mode; void *Z; size_t ld; bool single; const cd *x; cd *y;
                 int *ri, *ci; cd *vv; cd *qcorr;
                 double *stat; bool stat_ready; };   // stat: the analytic parts' geometry, 2 x 6 a pair (NEAR_STAT)

// the analytic (extracted) parts of a pair's moments are geometry only - m0, mro, mrs, mrr of each extracted
// distance (tb.d, tb.dimg: the stack's) - and cost most of a pair (offplane_one: logs and arctangents at 6 points x
// 3 edges); kept across frequencies when the caller gives stat (n x NEAR_STAT doubles)
static const int NEAR_STAT = 12;

static inline void atomic_add(cd *z, cd v) {
    double *p = reinterpret_cast<double *>(z);
#pragma omp atomic
    p[0] += v.real();
#pragma omp atomic
    p[1] += v.imag();
}
static inline void atomic_add(cf *z, cd v) {
    float *p = reinterpret_cast<float *>(z);
#pragma omp atomic
    p[0] += (float)v.real();
#pragma omp atomic
    p[1] += (float)v.imag();
}

void near_core(const Mesh &g, const Tables &tb, const double *nodes, const int *tris,
               const double *P3, const double *W3, const double *P6, const double *W6,
               const int *P, const int *Q, int n, double w, NearOut &o) {
    const cd jw(0, w), inv_jw = 1.0 / cd(0, w);
#pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; ++i) {
        const int p = P[i], q = Q[i];
        const int la = g.lt[p], lb = g.lt[q];
        const int t = tb.tab[la * tb.L + lb];
        const bool same = (la == lb);
        // smooth rest by quadrature: 3-point (same interface) or 6-point (cross)
        const int np = same ? 3 : 6;
        const double *Pa = same ? P3 + 6 * (size_t)p : P6 + 12 * (size_t)p;
        const double *Pb = same ? P3 + 6 * (size_t)q : P6 + 12 * (size_t)q;
        const double *Wa = same ? W3 + 3 * (size_t)p : W6 + 6 * (size_t)p;
        const double *Wb = same ? W3 + 3 * (size_t)q : W6 + 6 * (size_t)q;
        cd M0 = 0, Mox = 0, Moy = 0, Msx = 0, Msy = 0, Mrr = 0, Mq = 0;
        for (int a = 0; a < np; ++a)
            for (int b = 0; b < np; ++b) {
                double dx = Pa[2 * a] - Pb[2 * b], dy = Pa[2 * a + 1] - Pb[2 * b + 1];
                cd ga, gq;
                tb.rest(t, std::sqrt(dx * dx + dy * dy), ga, gq);
                double wt = Wa[a] * Wb[b];
                ga *= wt; gq *= wt;
                M0 += ga; Mox += ga * Pa[2 * a]; Moy += ga * Pa[2 * a + 1];
                Msx += ga * Pb[2 * b]; Msy += ga * Pb[2 * b + 1];
                Mrr += ga * (Pa[2 * a] * Pb[2 * b] + Pa[2 * a + 1] * Pb[2 * b + 1]);
                Mq += gq;
            }
        // analytic parts: the rule over the smaller triangle of the pair, the analytic integral over the other
        const bool keep = rule_over_p(g.area[p], g.area[q], p, q);
        const int po = keep ? p : q, qi = keep ? q : p;
        double V[6];
        for (int k = 0; k < 3; ++k) { V[2 * k] = nodes[2 * tris[3 * qi + k]]; V[2 * k + 1] = nodes[2 * tris[3 * qi + k] + 1]; }
        auto add = [&](int k, double dd, cd cA, cd cq) {
            double st[6], *m = o.stat ? o.stat + NEAR_STAT * (size_t)i + 6 * k : st;
            if (!(o.stat && o.stat_ready)) {
                double mro[2], mrs[2];
                offplane_one(V, P6 + 12 * (size_t)po, W6 + 6 * (size_t)po, dd, m[0], keep ? mro : mrs,
                             keep ? mrs : mro, m[5]);
                m[1] = mro[0]; m[2] = mro[1]; m[3] = mrs[0]; m[4] = mrs[1];
            }
            M0 += cA * m[0]; Mox += cA * m[1]; Moy += cA * m[2]; Msx += cA * m[3]; Msy += cA * m[4];
            Mrr += cA * m[5]; Mq += cq * m[0];
        };
        if (tb.kind[t] == 0) {
            add(0, 0.0, tb.cA[t], tb.cq[t]);
            if (tb.dimg[t] > 0) add(1, tb.dimg[t], tb.cAi[t], tb.cqi[t]);
        } else if (tb.kind[t] == 1) add(0, tb.d[t], tb.cA[t], tb.cq[t]);
        // minus the centroid value used by the far operator
        {
            double r = dist(g.cent, p, q);
            cd ga, gq;
            tb.kern(la, lb, r, ga, gq);
            double wgt = g.area[p] * g.area[q];
            double cpx = g.cent[2 * p], cpy = g.cent[2 * p + 1], cqx = g.cent[2 * q], cqy = g.cent[2 * q + 1];
            ga *= wgt; gq *= wgt;
            M0 -= ga; Mox -= ga * cpx; Moy -= ga * cpy; Msx -= ga * cqx; Msy -= ga * cqy;
            Mrr -= ga * (cpx * cqx + cpy * cqy); Mq -= gq;
        }
        if (o.qcorr) o.qcorr[i] = Mq;
        if (o.mode == 3) continue;                  // the charge moments alone (near_q)
        // scatter to the edge pairs (halves on p) x (halves on q)
        for (int s = 0; s < 3; ++s) {
            int em_ = g.te[3 * p + s];
            if (em_ < 0) continue;
            double am_ = g.ta[3 * p + s], vx = g.tv[6 * p + 2 * s], vy = g.tv[6 * p + 2 * s + 1];
            for (int u = 0; u < 3; ++u) {
                int en = g.te[3 * q + u];
                int slot = 9 * i + 3 * s + u;
                if (en < 0) { if (o.mode == 2) o.ri[slot] = -1; continue; }
                double an = g.ta[3 * q + u], wx = g.tv[6 * q + 2 * u], wy = g.tv[6 * q + 2 * u + 1];
                cd Acomb = Mrr - (wx * Mox + wy * Moy) - (vx * Msx + vy * Msy) + (vx * wx + vy * wy) * M0;
                cd val = am_ * an * (jw * Acomb + 4.0 * Mq * inv_jw);
                if (o.mode == 0) {
                    if (o.single) atomic_add((cf *)o.Z + (size_t)em_ * o.ld + en, val);
                    else atomic_add((cd *)o.Z + (size_t)em_ * o.ld + en, val);
                } else if (o.mode == 1) {
                    atomic_add(o.y + em_, val * o.x[en]);
                } else {
                    o.ri[slot] = em_; o.ci[slot] = en; o.vv[slot] = val;
                }
            }
        }
        if (o.mode == 2) {
            for (int s = 0; s < 3; ++s)
                if (g.te[3 * p + s] < 0)
                    for (int u = 0; u < 3; ++u) o.ri[9 * i + 3 * s + u] = -1;
        }
    }
}

// quad["stat"] (n x NEAR_STAT float64, C order: written in place) and quad["stat_ready"]: near_core's kept analytic parts
static void near_stat(py::dict quad, int n, NearOut &o) {
    if (!quad.contains("stat")) return;
    py::array st = quad["stat"];
    if (st.dtype().kind() != 'f' || st.itemsize() != 8 || !(st.flags() & py::array::c_style) || st.size() != (py::ssize_t)n * NEAR_STAT)
        throw std::invalid_argument("quad['stat']: a C-ordered float64 array of n x 12");
    o.stat = static_cast<double *>(st.mutable_data());
    o.stat_ready = quad["stat_ready"].cast<bool>();
}

// mode 0: Z (dense, complex128 or complex64) += correction ; returns qcorr (per pair Q0 correction)
py::array_t<cd> near_into_dense(py::array Z, py::dict geom, py::dict tabs, py::dict quad, arr<int> P,
                                arr<int> Q, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    arr<double> nodes(quad["nodes"]), P3(quad["P3"]), W3(quad["W3"]), P6(quad["P6"]), W6(quad["W6"]);
    arr<int> tris(quad["tris"]);
    const int n = (int)P.shape(0);
    py::array_t<cd> qc(n);
    NearOut o{};
    auto buf = Z.request(true);
    o.mode = 0; o.Z = buf.ptr; o.ld = buf.shape[1]; o.single = (buf.itemsize == 8); o.qcorr = qc.mutable_data();
    near_stat(quad, (int)P.shape(0), o);
    {
        py::gil_scoped_release nogil;
        near_core(mh.m, th.t, nodes.data(), tris.data(), P3.data(), W3.data(), P6.data(), W6.data(),
                  P.data(), Q.data(), n, w, o);
    }
    return qc;
}

// mode 1: y = dZ x, nothing stored
py::array_t<cd> near_matvec(arr<cd> x, py::dict geom, py::dict tabs, py::dict quad, arr<int> P, arr<int> Q,
                            double w) {
    MeshHold mh(geom); TabHold th(tabs);
    arr<double> nodes(quad["nodes"]), P3(quad["P3"]), W3(quad["W3"]), P6(quad["P6"]), W6(quad["W6"]);
    arr<int> tris(quad["tris"]);
    py::array_t<cd> y(mh.m.Ne);
    std::fill(y.mutable_data(), y.mutable_data() + mh.m.Ne, cd(0));
    NearOut o{};
    o.mode = 1; o.x = x.data(); o.y = y.mutable_data();
    near_stat(quad, (int)P.shape(0), o);
    {
        py::gil_scoped_release nogil;
        near_core(mh.m, th.t, nodes.data(), tris.data(), P3.data(), W3.data(), P6.data(), W6.data(),
                  P.data(), Q.data(), (int)P.shape(0), w, o);
    }
    return y;
}

// mode 2: COO triplets (rows = -1 for unused slots) + per-pair Q correction
py::tuple near_coo(py::dict geom, py::dict tabs, py::dict quad, arr<int> P, arr<int> Q, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    arr<double> nodes(quad["nodes"]), P3(quad["P3"]), W3(quad["W3"]), P6(quad["P6"]), W6(quad["W6"]);
    arr<int> tris(quad["tris"]);
    const int n = (int)P.shape(0);
    py::array_t<int> ri((size_t)9 * n), ci((size_t)9 * n);
    py::array_t<cd> vv((size_t)9 * n), qc(n);
    NearOut o{};
    o.mode = 2; o.ri = ri.mutable_data(); o.ci = ci.mutable_data(); o.vv = vv.mutable_data(); o.qcorr = qc.mutable_data();
    near_stat(quad, (int)P.shape(0), o);
    {
        py::gil_scoped_release nogil;
        near_core(mh.m, th.t, nodes.data(), tris.data(), P3.data(), W3.data(), P6.data(), W6.data(),
                  P.data(), Q.data(), n, w, o);
    }
    return py::make_tuple(ri, ci, vv, qc);
}

// mode 3: the per-pair Q correction alone, nothing scattered: the loop/tree basis's charge matrix between
// triangles (looptree.py) has the accurate moments of the near pairs as fan_q's centroid values plus these
py::array_t<cd> near_q(py::dict geom, py::dict tabs, py::dict quad, arr<int> P, arr<int> Q, double w) {
    MeshHold mh(geom); TabHold th(tabs);
    arr<double> nodes(quad["nodes"]), P3(quad["P3"]), W3(quad["W3"]), P6(quad["P6"]), W6(quad["W6"]);
    arr<int> tris(quad["tris"]);
    const int n = (int)P.shape(0);
    py::array_t<cd> qc(n);
    NearOut o{};
    o.mode = 3; o.qcorr = qc.mutable_data();
    near_stat(quad, n, o);
    {
        py::gil_scoped_release nogil;
        near_core(mh.m, th.t, nodes.data(), tris.data(), P3.data(), W3.data(), P6.data(), W6.data(),
                  P.data(), Q.data(), n, w, o);
    }
    return qc;
}

// ---------------------------------------------------------------------------------------------
// the loop/tree basis (looptree.py). The tree: its non-root nodes breadth first, each with its parent, the branch up
// and that branch's sign (P[node] = P[parent] + sign X[branch]: the sum over the node's path to its root); the
// roots (P = 0). A loop: its branch and that branch's two ends (its row or column plus P[cv] - P[cu]).
// ---------------------------------------------------------------------------------------------
struct TreeView {
    const int *node, *parent, *branch, *roots, *cot, *cu, *cv; const double *sig;
    int nn, nr, nc, n_nodes;
};
static TreeView tree_view(arr<int> &node, arr<int> &parent, arr<int> &branch, arr<double> &sig, arr<int> &roots,
                          arr<int> &cot, arr<int> &cu, arr<int> &cv, int n_nodes) {
    return TreeView{node.data(), parent.data(), branch.data(), roots.data(), cot.data(), cu.data(), cv.data(),
                    sig.data(), (int)node.shape(0), (int)roots.shape(0), (int)cot.shape(0), n_nodes};
}

// X <- X K in place (X: n x m complex128, C order): a row at a time, its P over the nodes in cache
void tree_cols(py::array_t<cd> X, arr<int> node, arr<int> parent, arr<int> branch, arr<double> sig, arr<int> roots,
               arr<int> cot, arr<int> cu, arr<int> cv, int n_nodes) {
    auto buf = X.request(true);
    if (buf.ndim != 2) throw std::invalid_argument("tree_cols: X must be 2-D");
    const i64 n = buf.shape[0], m = buf.shape[1];
    cd *x0 = static_cast<cd *>(buf.ptr);
    TreeView t = tree_view(node, parent, branch, sig, roots, cot, cu, cv, n_nodes);
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        std::vector<cd> P(t.n_nodes);
#pragma omp for schedule(static)
        for (i64 r = 0; r < n; ++r) {
            cd *x = x0 + r * m;
            for (int k = 0; k < t.nr; ++k) P[t.roots[k]] = 0;
            for (int k = 0; k < t.nn; ++k) P[t.node[k]] = P[t.parent[k]] + t.sig[k] * x[t.branch[k]];
            for (int j = 0; j < t.nc; ++j) x[t.cot[j]] += P[t.cv[j]] - P[t.cu[j]];
        }
    }
}

// X <- K^T X in place: blocks of columns, a block's P (nodes x block) a thread's
void tree_rows(py::array_t<cd> X, arr<int> node, arr<int> parent, arr<int> branch, arr<double> sig, arr<int> roots,
               arr<int> cot, arr<int> cu, arr<int> cv, int n_nodes) {
    auto buf = X.request(true);
    if (buf.ndim != 2) throw std::invalid_argument("tree_rows: X must be 2-D");
    const i64 m = buf.shape[1];
    cd *x0 = static_cast<cd *>(buf.ptr);
    TreeView t = tree_view(node, parent, branch, sig, roots, cot, cu, cv, n_nodes);
    const i64 B = 32, nb = (m + B - 1) / B;
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        std::vector<cd> P((size_t)t.n_nodes * B);
#pragma omp for schedule(dynamic, 1)
        for (i64 b = 0; b < nb; ++b) {
            const i64 c0 = b * B, w = std::min(B, m - c0);
            for (int k = 0; k < t.nr; ++k) std::fill_n(&P[(size_t)t.roots[k] * B], w, cd(0));
            for (int k = 0; k < t.nn; ++k) {
                cd *p = &P[(size_t)t.node[k] * B];
                const cd *q = &P[(size_t)t.parent[k] * B], *x = x0 + (i64)t.branch[k] * m + c0;
                const double s = t.sig[k];
                for (i64 c = 0; c < w; ++c) p[c] = q[c] + s * x[c];
            }
            for (int j = 0; j < t.nc; ++j) {
                cd *x = x0 + (i64)t.cot[j] * m + c0;
                const cd *pv = &P[(size_t)t.cv[j] * B], *pu = &P[(size_t)t.cu[j] * B];
                for (i64 c = 0; c < w; ++c) x[c] += pv[c] - pu[c];
            }
        }
    }
}

// Z[tree_i, tree_j] += sum over the ends a of branch i, b of branch j of s_a s_b Mq[a, b] / (area_a area_b): the
// tree's charge block (A_T^T Gc A_T, Gc the potential between unit charges on two triangles) straight into Z. A
// branch's ends: u (+1) and v (-1), the ground (g) carrying none. Mq: the G_q moments between triangles (Nt x Nt).
void charge_into(py::array_t<cd> Z, arr<cd> Mq, arr<int> tree, arr<int> u, arr<int> v, arr<double> area, int g) {
    auto buf = Z.request(true);
    const i64 ld = buf.shape[1], Nt = Mq.shape(1);
    cd *z = static_cast<cd *>(buf.ptr);
    const cd *M = Mq.data();
    const int nt = (int)tree.shape(0);
    const int *tr = tree.data(), *pu = u.data(), *pv = v.data();
    const double *ar = area.data();
    py::gil_scoped_release nogil;
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < nt; ++i) {
        cd *row = z + (i64)tr[i] * ld;
        const int ends[2] = {pu[i], pv[i]};
        const double sg[2] = {1.0, -1.0};
        for (int ea = 0; ea < 2; ++ea) {
            const int a = ends[ea];
            if (a == g) continue;
            const cd *Ma = M + (i64)a * Nt;
            const double sa = sg[ea] / ar[a];
            for (int j = 0; j < nt; ++j) {
                cd acc = 0;
                if (pu[j] != g) acc += Ma[pu[j]] / ar[pu[j]];
                if (pv[j] != g) acc -= Ma[pv[j]] / ar[pv[j]];
                row[tr[j]] += sa * acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// BLRFactor: a block low-rank LU (blr.py's, compiled; fast.FastOperator.solve_blr). The equilibrated matrix cut
// into the tiles of a geometric cluster tree, in elimination order; tiles between well-separated clusters (`far`)
// assembled, compressed by adaptive cross approximation with full pivoting and recompressed (QR, SVD), the others
// dense; a right-looking tile LU, pivoting inside the diagonal tiles, the low-rank updates recompressed; complex64.
// solve() applies the approximate inverse (GMRES's preconditioner). BLAS and LAPACK are SciPy's
// (scipy.linalg.cython_blas / cython_lapack capsules: nothing to link); the caller keeps their own threads at one
// (threadpoolctl), the tiles being OpenMP's.
// ---------------------------------------------------------------------------------------------
namespace blrk {
static double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class T> struct Lapack {         // SciPy's BLAS and LAPACK of T (complex64: c..., complex128: z...)
    typedef typename T::value_type R;
    typedef void (*gemm_t)(char *, char *, int *, int *, int *, T *, T *, int *, T *, int *, T *, T *, int *);
    typedef void (*trsm_t)(char *, char *, char *, char *, int *, int *, T *, T *, int *, T *, int *);
    typedef void (*getrf_t)(int *, int *, T *, int *, int *, int *);
    typedef void (*geqrf_t)(int *, int *, T *, int *, T *, T *, int *, int *);
    typedef void (*ungqr_t)(int *, int *, int *, T *, int *, T *, T *, int *, int *);
    typedef void (*gesvd_t)(char *, char *, int *, int *, T *, int *, R *, T *, int *, T *, int *, T *, int *, R *,
                            int *);
    gemm_t gemm; trsm_t trsm; getrf_t getrf; geqrf_t geqrf; ungqr_t ungqr; gesvd_t gesvd;
};

static void *capsule(const char *mod, const std::string &name) {
    py::dict capi = py::module_::import(mod).attr("__pyx_capi__");
    py::object c = capi[name.c_str()];
    void *p = PyCapsule_GetPointer(c.ptr(), PyCapsule_GetName(c.ptr()));
    if (!p) throw std::runtime_error(std::string("BLRFactor: no ") + name + " in " + mod);
    return p;
}
template <class T> static const Lapack<T> &lapack() {      // first called under the GIL (the constructor)
    static const std::string k = sizeof(T) == sizeof(cf) ? "c" : "z";
    static const Lapack<T> L{(typename Lapack<T>::gemm_t)capsule("scipy.linalg.cython_blas", k + "gemm"),
                             (typename Lapack<T>::trsm_t)capsule("scipy.linalg.cython_blas", k + "trsm"),
                             (typename Lapack<T>::getrf_t)capsule("scipy.linalg.cython_lapack", k + "getrf"),
                             (typename Lapack<T>::geqrf_t)capsule("scipy.linalg.cython_lapack", k + "geqrf"),
                             (typename Lapack<T>::ungqr_t)capsule("scipy.linalg.cython_lapack", k + "ungqr"),
                             (typename Lapack<T>::gesvd_t)capsule("scipy.linalg.cython_lapack", k + "gesvd")};
    return L;
}
// C (m x n) = alpha op(A) op(B) + beta C, column major
template <class Sc> static void gemm(char ta, char tb, int m, int n, int k, Sc alpha, const Sc *A, int lda, const Sc *B, int ldb,
                 Sc beta, Sc *C, int ldc) {
    if (m <= 0 || n <= 0) return;
    lda = std::max(lda, 1); ldb = std::max(ldb, 1); ldc = std::max(ldc, 1);
    lapack<Sc>().gemm(&ta, &tb, &m, &n, &k, &alpha, const_cast<Sc *>(A), &lda, const_cast<Sc *>(B), &ldb, &beta, C, &ldc);
}
template <class Sc> static void trsm(char side, char uplo, char trans, char diag, int m, int n, const Sc *A, int lda, Sc *B, int ldb) {
    if (m <= 0 || n <= 0) return;
    Sc one(1);
    lda = std::max(lda, 1); ldb = std::max(ldb, 1);
    lapack<Sc>().trsm(&side, &uplo, &trans, &diag, &m, &n, &one, const_cast<Sc *>(A), &lda, B, &ldb);
}
// LAPACK's row interchanges (ipiv, 1-based, in order) on the m rows of A (ld x ncols)
template <class Sc> static void swaps(Sc *A, int ld, int ncols, const int *ipiv, int m) {
    for (int c = 0; c < ncols; ++c) {
        Sc *a = A + (size_t)c * ld;
        for (int i = 0; i < m; ++i) {
            int p = ipiv[i] - 1;
            if (p != i) std::swap(a[i], a[p]);
        }
    }
}

template <class Sc> struct Tile {
    char kind = 'N';                        // 'D' dense (m x n), 'L' low rank U (m x r) V (r x n), 'N' none
    int m = 0, n = 0, r = 0;
    std::vector<Sc> D, U, V;                // column major
    size_t bytes() const { return (D.size() + U.size() + V.size()) * sizeof(Sc); }
    void dense() {                          // a low-rank tile as dense
        if (kind == 'D') return;
        D.assign((size_t)m * n, Sc(0));
        if (kind == 'L' && r) gemm('N', 'N', m, n, r, Sc(1), U.data(), m, V.data(), r, Sc(0), D.data(), m);
        U.clear(); U.shrink_to_fit(); V.clear(); V.shrink_to_fit(); r = 0; kind = 'D';
    }
};

// t <- U V (m x r, r x n) recompressed: QR of U and of V^T, SVD of the r x r core, singular values above atol;
// dense where the rank does not pay (r (m + n) >= 0.6 m n) or passes rmax
template <class Sc> static void recompress(Tile<Sc> &t, int m, int n, std::vector<Sc> &U, std::vector<Sc> &V, int r, double atol, int rmax) {
    const Lapack<Sc> &L = lapack<Sc>();
    t.m = m; t.n = n; t.D.clear();
    if (r == 0) { t.kind = 'L'; t.r = 0; t.U.clear(); t.V.clear(); return; }
    if (r >= std::min(m, n)) {
        t.kind = 'D'; t.D.assign((size_t)m * n, Sc(0));
        gemm('N', 'N', m, n, r, Sc(1), U.data(), m, V.data(), r, Sc(0), t.D.data(), m);
        t.U.clear(); t.V.clear(); t.r = 0;
        return;
    }
    int info = 0, lw = -1;
    std::vector<Sc> tu(r), tv(r);
    Sc q;
    L.geqrf(&m, &r, U.data(), &m, tu.data(), &q, &lw, &info);
    int lwork = std::max(1, (int)q.real());
    std::vector<Sc> Vt((size_t)n * r);             // V^T, n x r
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < r; ++i) Vt[j + (size_t)i * n] = V[i + (size_t)j * r];
    lw = -1; L.geqrf(&n, &r, Vt.data(), &n, tv.data(), &q, &lw, &info);
    lwork = std::max(lwork, (int)q.real());
    std::vector<Sc> work(std::max(lwork, 1));
    L.geqrf(&m, &r, U.data(), &m, tu.data(), work.data(), &lwork, &info);
    L.geqrf(&n, &r, Vt.data(), &n, tv.data(), work.data(), &lwork, &info);
    std::vector<Sc> Ru((size_t)r * r, Sc(0)), Rv((size_t)r * r, Sc(0));
    for (int j = 0; j < r; ++j)
        for (int i = 0; i <= j; ++i) { Ru[i + (size_t)j * r] = U[i + (size_t)j * m]; Rv[i + (size_t)j * r] = Vt[i + (size_t)j * n]; }
    lw = -1; L.ungqr(&m, &r, &r, U.data(), &m, tu.data(), &q, &lw, &info);
    int lw2 = std::max(1, (int)q.real());
    lw = -1; L.ungqr(&n, &r, &r, Vt.data(), &n, tv.data(), &q, &lw, &info);
    lw2 = std::max(lw2, (int)q.real());
    if ((int)work.size() < lw2) work.resize(lw2);
    L.ungqr(&m, &r, &r, U.data(), &m, tu.data(), work.data(), &lw2, &info);       // U <- Q_u (m x r)
    L.ungqr(&n, &r, &r, Vt.data(), &n, tv.data(), work.data(), &lw2, &info);      // Vt <- Q_v (n x r)
    std::vector<Sc> S((size_t)r * r);
    gemm('N', 'T', r, r, r, Sc(1), Ru.data(), r, Rv.data(), r, Sc(0), S.data(), r);   // R_u R_v^T
    std::vector<Sc> S0(S), W((size_t)r * r), Zh((size_t)r * r);
    std::vector<typename Sc::value_type> sig(r), rw(5 * (size_t)r + 1);
    char js = 'S';
    lw = -1; L.gesvd(&js, &js, &r, &r, S.data(), &r, sig.data(), W.data(), &r, Zh.data(), &r, &q, &lw, rw.data(), &info);
    int lw3 = std::max(1, (int)q.real());
    std::vector<Sc> w3(lw3);
    L.gesvd(&js, &js, &r, &r, S.data(), &r, sig.data(), W.data(), &r, Zh.data(), &r, w3.data(), &lw3, rw.data(), &info);
    int k = 0;
    while (k < r && sig[k] > atol) ++k;
    if (info != 0) k = r;                           // (no SVD: keep them all)
    if (k > rmax || (double)k * (m + n) >= 0.6 * (double)m * n) {
        std::vector<Sc> T1((size_t)m * r);         // Q_u S0 Q_v^T
        gemm('N', 'N', m, r, r, Sc(1), U.data(), m, S0.data(), r, Sc(0), T1.data(), m);
        t.kind = 'D'; t.r = 0; t.U.clear(); t.V.clear();
        t.D.assign((size_t)m * n, Sc(0));
        gemm('N', 'T', m, n, r, Sc(1), T1.data(), m, Vt.data(), n, Sc(0), t.D.data(), m);
        return;
    }
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < r; ++i) W[i + (size_t)j * r] *= sig[j];
    t.kind = 'L'; t.r = k;
    t.U.assign((size_t)m * k, Sc(0)); t.V.assign((size_t)k * n, Sc(0));
    if (k) {
        gemm('N', 'N', m, k, r, Sc(1), U.data(), m, W.data(), r, Sc(0), t.U.data(), m);
        gemm('N', 'T', k, n, r, Sc(1), Zh.data(), r, Vt.data(), n, Sc(0), t.V.data(), k);
    }
}

// adaptive cross approximation with full pivoting of the dense residual R (m x n, column major, overwritten):
// U (m x k), V (k x n) until ||R||_F <= atol; false where it needs more than rmax
template <class Sc> static bool aca_full(std::vector<cd> &R, int m, int n, double atol, int rmax, std::vector<Sc> &U, std::vector<Sc> &V,
                     int &k) {
    double fro2 = 0; size_t imax = 0; double amax = -1;
    for (size_t p = 0; p < R.size(); ++p) { double a = std::norm(R[p]); fro2 += a; if (a > amax) { amax = a; imax = p; } }
    std::vector<cd> us, vs;                         // k columns of m, k rows of n
    k = 0;
    while (std::sqrt(fro2) > atol) {
        if (k >= rmax || amax <= 0) return false;
        const int is = (int)(imax % m), js = (int)(imax / m);
        const cd piv = R[imax];
        us.resize((size_t)(k + 1) * m); vs.resize((size_t)(k + 1) * n);
        cd *u = &us[(size_t)k * m], *v = &vs[(size_t)k * n];
        for (int i = 0; i < m; ++i) u[i] = R[i + (size_t)js * m];
        for (int j = 0; j < n; ++j) v[j] = R[is + (size_t)j * m] / piv;
        fro2 = 0; amax = -1;
        for (int j = 0; j < n; ++j) {
            cd *col = &R[(size_t)j * m];
            const cd vj = v[j];
            for (int i = 0; i < m; ++i) {
                col[i] -= u[i] * vj;
                double a = std::norm(col[i]);
                fro2 += a;
                if (a > amax) { amax = a; imax = i + (size_t)j * m; }
            }
        }
        ++k;
    }
    U.resize((size_t)m * k); V.resize((size_t)k * n);
    for (int c = 0; c < k; ++c)
        for (int i = 0; i < m; ++i) U[i + (size_t)c * m] = Sc(us[(size_t)c * m + i]);
    for (int j = 0; j < n; ++j)
        for (int c = 0; c < k; ++c) V[c + (size_t)j * k] = Sc(vs[(size_t)c * n + j]);
    return true;
}
}  // namespace blrk

template <class Sc> class BLRFactorT {
public:
    // order: the unknowns in elimination order, start: each tile's first (nb + 1); far: nb x nb, 1 where the tile
    // pair is compressed; geom, tabs, w: the far kernel; nptr / nidx / nval: the near correction and the extra
    // blocks (CSR, Ne x Ne); B (Ne x Nv), C (Nv x Nv): the via rows and columns; dsc: the equilibration; atol:
    // the compression's absolute tolerance (on the equilibrated matrix); max_rank: 0 for half a tile
    BLRFactorT(arr<int> order, arr<int> start, arr<uint8_t> far, py::dict geom, py::dict tabs, double w, arr<i64> nptr,
              arr<int> nidx, arr<cd> nval, arr<cd> B, arr<cd> C, arr<double> dsc, double atol, int max_rank)
        : atol_(atol) {
        using namespace blrk;
        lapack<Sc>();
        MeshHold mh(geom); TabHold th(tabs);
        const Mesh &g = mh.m; const Tables &tb = th.t;
        N_ = (int)order.shape(0); nb_ = (int)start.shape(0) - 1;
        order_.assign(order.data(), order.data() + N_);
        start_.assign(start.data(), start.data() + nb_ + 1);
        const int Ne = g.Ne, Nv = N_ - Ne, Nt = g.Nt, nb = nb_;
        if ((i64)far.size() != (i64)nb * nb) throw std::invalid_argument("BLRFactor: far must be nb x nb");
        std::vector<int> tile_of(N_), local_of(N_);
        for (int k = 0; k < nb; ++k)
            for (int p = start_[k]; p < start_[k + 1]; ++p) { tile_of[order_[p]] = k; local_of[order_[p]] = p - start_[k]; }
        T_.resize((size_t)nb * nb);
        const uint8_t *fa = far.data();
        const i64 *np_ = nptr.data(); const int *ni = nidx.data(); const cd *nv = nval.data();
        const cd *Bp = B.data(), *Cp = C.data(); const double *ds = dsc.data();
        const cd jw(0, w), inv_jw = 1.0 / cd(0, w);
        const int *ord = order_.data(), *st = start_.data(), *tof = tile_of.data(), *lof = local_of.data();
        double t0 = blrk::wall();
        {
            py::gil_scoped_release nogil;
#pragma omp parallel
            {
                std::vector<int> mark_r(Nt, -1), mark_c(Nt, -1), tr, tc;
                std::vector<cd> M, Mc, GA, GQ;
                std::vector<Sc> U, V;
#pragma omp for schedule(dynamic, 1)
                for (i64 p = 0; p < (i64)nb * nb; ++p) {
                    const int I = (int)(p / nb), J = (int)(p % nb);
                    const int m = st[I + 1] - st[I], n = st[J + 1] - st[J];
                    const int *rows = ord + st[I], *cols = ord + st[J];
                    // the triangles of the tile's edges, and the kernel between them (each triangle-pair once)
                    tr.clear(); tc.clear();
                    for (int i = 0; i < m; ++i) if (rows[i] < Ne) for (int t : {g.ep[rows[i]], g.em[rows[i]]})
                        if (mark_r[t] < 0) { mark_r[t] = (int)tr.size(); tr.push_back(t); }
                    for (int j = 0; j < n; ++j) if (cols[j] < Ne) for (int t : {g.ep[cols[j]], g.em[cols[j]]})
                        if (mark_c[t] < 0) { mark_c[t] = (int)tc.size(); tc.push_back(t); }
                    const size_t nr = tr.size(), nc = tc.size();
                    GA.resize(nr * nc); GQ.resize(nr * nc);
                    for (size_t a = 0; a < nr; ++a)
                        for (size_t b = 0; b < nc; ++b)
                            tb.kern(g.lt[tr[a]], g.lt[tc[b]], dist(g.cent, tr[a], tc[b]), GA[a * nc + b], GQ[a * nc + b]);
                    M.assign((size_t)m * n, cd(0));
                    for (int j = 0; j < n; ++j) {
                        const int c = cols[j];
                        cd *col = &M[(size_t)j * m];
                        if (c < Ne) {
                            const int pn[2] = {g.ep[c], g.em[c]};
                            const int qn[2] = {mark_c[pn[0]], mark_c[pn[1]]};
                            const double an[2] = {g.ap[c] * g.area[pn[0]], g.am[c] * g.area[pn[1]]};
                            const double ex[2] = {g.cent[2 * pn[0]] - g.vp[2 * c], g.cent[2 * pn[1]] - g.vm[2 * c]};
                            const double ey[2] = {g.cent[2 * pn[0] + 1] - g.vp[2 * c + 1], g.cent[2 * pn[1] + 1] - g.vm[2 * c + 1]};
                            for (int i = 0; i < m; ++i) {
                                const int u = rows[i];
                                if (u < Ne) {
                                    const int pm[2] = {g.ep[u], g.em[u]};
                                    const int qm[2] = {mark_r[pm[0]], mark_r[pm[1]]};
                                    const double am_[2] = {g.ap[u] * g.area[pm[0]], g.am[u] * g.area[pm[1]]};
                                    const double dx[2] = {g.cent[2 * pm[0]] - g.vp[2 * u], g.cent[2 * pm[1]] - g.vm[2 * u]};
                                    const double dy[2] = {g.cent[2 * pm[0] + 1] - g.vp[2 * u + 1], g.cent[2 * pm[1] + 1] - g.vm[2 * u + 1]};
                                    cd v = 0;
                                    for (int s = 0; s < 2; ++s)
                                        for (int t = 0; t < 2; ++t) {
                                            const size_t q = (size_t)qm[s] * nc + qn[t];
                                            v += am_[s] * an[t] * (jw * (dx[s] * ex[t] + dy[s] * ey[t]) * GA[q] + 4.0 * GQ[q] * inv_jw);
                                        }
                                    col[i] = v;
                                } else {
                                    col[i] = Bp[(size_t)c * Nv + (u - Ne)];
                                }
                            }
                        } else {
                            for (int i = 0; i < m; ++i) {
                                const int u = rows[i];
                                col[i] = u < Ne ? Bp[(size_t)u * Nv + (c - Ne)] : Cp[(size_t)(u - Ne) * Nv + (c - Ne)];
                            }
                        }
                    }
                    for (int i = 0; i < m; ++i) {           // the near correction and the extra blocks
                        const int u = rows[i];
                        if (u >= Ne) continue;
                        for (i64 q = np_[u]; q < np_[u + 1]; ++q)
                            if (tof[ni[q]] == J) M[i + (size_t)lof[ni[q]] * m] += nv[q];
                    }
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i) M[i + (size_t)j * m] *= ds[rows[i]] * ds[cols[j]];
                    for (int t : tr) mark_r[t] = -1;
                    for (int t : tc) mark_c[t] = -1;
                    Tile<Sc> &T = T_[(size_t)I * nb + J];
                    T.m = m; T.n = n;
                    bool low = false;
                    if (fa[(size_t)I * nb + J]) {
                        const int rmax = std::min(max_rank > 0 ? max_rank : std::min(m, n) / 2,
                                                  (int)(0.6 * (double)m * n / (m + n)));
                        Mc = M;
                        int k = 0;
                        if (aca_full(Mc, m, n, atol, rmax, U, V, k)) {
                            recompress(T, m, n, U, V, k, atol, rmax);
                            low = true;
                        }
                    }
                    if (!low) {
                        T.kind = 'D'; T.D.resize((size_t)m * n);
                        for (size_t q = 0; q < M.size(); ++q) T.D[q] = Sc(M[q]);
                    }
                }
            }
            t_assemble_ = blrk::wall() - t0;
            for (const auto &T : T_) {
                if (T.kind == 'L') { ++lowrank0_; rank0_ += T.r; }
                else ++dense0_;
            }
            bytes0_ = bytes();
            t0 = blrk::wall();
            factor(max_rank);
            t_factor_ = blrk::wall() - t0;
        }
    }

    // the approximate inverse: R (N, or N x m) in the original numbering
    py::array_t<cd> solve(py::array_t<cd, py::array::c_style | py::array::forcecast> R) {
        using namespace blrk;
        const bool vec = R.ndim() == 1;
        const int nr = vec ? 1 : (int)R.shape(1);
        if (R.shape(0) != N_) throw std::invalid_argument("BLRFactor.solve: R must have N rows");
        const cd *r = R.data();
        py::array_t<cd> out = vec ? py::array_t<cd>(N_) : py::array_t<cd>({N_, nr});
        cd *o = out.mutable_data();
        const int nb = nb_;
        std::vector<std::vector<Sc>> y(nb);
        {
            py::gil_scoped_release nogil;
            for (int k = 0; k < nb; ++k) {
                const int m = start_[k + 1] - start_[k];
                y[k].resize((size_t)m * nr);
                for (int i = 0; i < m; ++i)
                    for (int c = 0; c < nr; ++c) y[k][i + (size_t)c * m] = Sc(r[(size_t)order_[start_[k] + i] * nr + c]);
            }
            for (int k = 0; k < nb; ++k) {                  // L y = P b
                const int m = start_[k + 1] - start_[k];
                swaps(y[k].data(), m, nr, piv_[k].data(), m);
                subtract(k, 0, k, y, nr);
                trsm('L', 'L', 'N', 'U', m, nr, LU_[k].data(), m, y[k].data(), m);
            }
            for (int k = nb - 1; k >= 0; --k) {             // U z = y
                const int m = start_[k + 1] - start_[k];
                subtract(k, k + 1, nb, y, nr);
                trsm('L', 'U', 'N', 'N', m, nr, LU_[k].data(), m, y[k].data(), m);
            }
            for (int k = 0; k < nb; ++k) {
                const int m = start_[k + 1] - start_[k];
                for (int i = 0; i < m; ++i)
                    for (int c = 0; c < nr; ++c) o[(size_t)order_[start_[k] + i] * nr + c] = cd(y[k][i + (size_t)c * m]);
            }
        }
        return out;
    }

    py::dict stats() const {
        py::dict d;
        int dense = 0, low = 0; long rank = 0;
        for (const auto &T : T_) { if (T.kind == 'L') { ++low; rank += T.r; } else if (T.kind == 'D') ++dense; }
        d["tiles"] = nb_; d["dense"] = dense; d["lowrank"] = low; d["rank_sum"] = rank;
        d["dense_assembled"] = dense0_; d["lowrank_assembled"] = lowrank0_; d["rank_sum_assembled"] = rank0_;
        d["bytes"] = (double)bytes(); d["bytes_assembled"] = (double)bytes0_;
        d["seconds_assemble"] = t_assemble_; d["seconds_factor"] = t_factor_;
        d["singular"] = singular_;
        return d;
    }

private:
    size_t bytes() const {
        size_t b = 0;
        for (const auto &T : T_) b += T.bytes();
        for (const auto &l : LU_) b += l.size() * sizeof(Sc);
        return b;
    }

    // y[k] -= sum over j in [j0, j1) of T[k][j] y[j]: the tiles' products on the threads, summed into y[k]
    void subtract(int k, int j0, int j1, std::vector<std::vector<Sc>> &y, int nr) const {
        using namespace blrk;
        if (j1 <= j0) return;
        const int m = start_[k + 1] - start_[k], nb = nb_;
#pragma omp parallel
        {
            std::vector<Sc> acc((size_t)m * nr, Sc(0)), tmp;
            bool any = false;
#pragma omp for schedule(dynamic, 1) nowait
            for (int j = j0; j < j1; ++j) {
                const Tile<Sc> &T = T_[(size_t)k * nb + j];
                const int n = T.n;
                if (T.kind == 'D') {
                    gemm('N', 'N', m, nr, n, Sc(1), T.D.data(), m, y[j].data(), n, Sc(1), acc.data(), m);
                    any = true;
                } else if (T.kind == 'L' && T.r) {
                    tmp.assign((size_t)T.r * nr, Sc(0));
                    gemm('N', 'N', T.r, nr, n, Sc(1), T.V.data(), T.r, y[j].data(), n, Sc(0), tmp.data(), T.r);
                    gemm('N', 'N', m, nr, T.r, Sc(1), T.U.data(), m, tmp.data(), T.r, Sc(1), acc.data(), m);
                    any = true;
                }
            }
            if (any) {
#pragma omp critical
                for (size_t q = 0; q < acc.size(); ++q) y[k][q] -= acc[q];
            }
        }
    }

    // C -= A B (the Schur update of tile (i, j) by step k)
    void schur(blrk::Tile<Sc> &c, const blrk::Tile<Sc> &a, const blrk::Tile<Sc> &b, int max_rank) {
        using namespace blrk;
        const int mi = a.m, mk = a.n, nj = b.n;
        if ((a.kind == 'L' && a.r == 0) || (b.kind == 'L' && b.r == 0)) return;
        if (a.kind == 'D' && b.kind == 'D') {
            c.dense();
            gemm('N', 'N', mi, nj, mk, Sc(-1), a.D.data(), mi, b.D.data(), mk, Sc(1), c.D.data(), mi);
            return;
        }
        std::vector<Sc> Uw, Vw;                         // the update U_u V_u (rank ru), where computed
        const Sc *Uu, *Vu;
        int ru;
        if (a.kind == 'L' && b.kind == 'D') {
            ru = a.r; Uu = a.U.data();
            Vw.assign((size_t)ru * nj, Sc(0));
            gemm('N', 'N', ru, nj, mk, Sc(1), a.V.data(), ru, b.D.data(), mk, Sc(0), Vw.data(), ru);
            Vu = Vw.data();
        } else if (a.kind == 'D' && b.kind == 'L') {
            ru = b.r; Vu = b.V.data();
            Uw.assign((size_t)mi * ru, Sc(0));
            gemm('N', 'N', mi, ru, mk, Sc(1), a.D.data(), mi, b.U.data(), mk, Sc(0), Uw.data(), mi);
            Uu = Uw.data();
        } else {
            std::vector<Sc> t((size_t)a.r * b.r);        // V_a U_b
            gemm('N', 'N', a.r, b.r, mk, Sc(1), a.V.data(), a.r, b.U.data(), mk, Sc(0), t.data(), a.r);
            if (a.r <= b.r) {
                ru = a.r; Uu = a.U.data();
                Vw.assign((size_t)ru * nj, Sc(0));
                gemm('N', 'N', ru, nj, b.r, Sc(1), t.data(), a.r, b.V.data(), b.r, Sc(0), Vw.data(), ru);
                Vu = Vw.data();
            } else {
                ru = b.r; Vu = b.V.data();
                Uw.assign((size_t)mi * ru, Sc(0));
                gemm('N', 'N', mi, ru, a.r, Sc(1), a.U.data(), mi, t.data(), a.r, Sc(0), Uw.data(), mi);
                Uu = Uw.data();
            }
        }
        if (c.kind == 'D') {
            gemm('N', 'N', mi, nj, ru, Sc(-1), Uu, mi, Vu, ru, Sc(1), c.D.data(), mi);
            return;
        }
        const int rc = c.r, r2 = rc + ru;               // [U_c, -U_u] [V_c; V_u], recompressed
        std::vector<Sc> U2((size_t)mi * r2), V2((size_t)r2 * nj);
        std::copy(c.U.begin(), c.U.end(), U2.begin());
        for (size_t q = 0; q < (size_t)mi * ru; ++q) U2[(size_t)mi * rc + q] = -Uu[q];
        for (int j = 0; j < nj; ++j) {
            for (int q = 0; q < rc; ++q) V2[q + (size_t)j * r2] = c.V[q + (size_t)j * rc];
            for (int q = 0; q < ru; ++q) V2[rc + q + (size_t)j * r2] = Vu[q + (size_t)j * ru];
        }
        const int rmax = std::min(max_rank > 0 ? max_rank : std::min(mi, nj) / 2,
                                  (int)(0.6 * (double)mi * nj / (mi + nj)));
        recompress(c, mi, nj, U2, V2, r2, atol_, rmax);
    }

    // the right-looking tile LU: the diagonal tile factored (its pivots kept), its row's tiles permuted and solved
    // with L, its column's with U, the trailing tiles updated - each step's tiles on the threads
    void factor(int max_rank) {
        using namespace blrk;
        const int nb = nb_;
        LU_.resize(nb); piv_.resize(nb);
        for (int k = 0; k < nb; ++k) {
            Tile<Sc> &dk = T_[(size_t)k * nb + k];
            dk.dense();
            const int mk = dk.m;
            LU_[k] = std::move(dk.D);
            dk.D.clear(); dk.kind = 'N';
            piv_[k].resize(mk);
            int info = 0, mm = mk;
            lapack<Sc>().getrf(&mm, &mm, LU_[k].data(), &mm, piv_[k].data(), &info);
            if (info > 0) ++singular_;
            const Sc *L = LU_[k].data();
            const int *pv = piv_[k].data();
#pragma omp parallel for schedule(dynamic, 1)
            for (int j = 0; j < nb; ++j) {              // the row: P, then L^-1 (right of the diagonal)
                if (j == k) continue;
                Tile<Sc> &t = T_[(size_t)k * nb + j];
                if (t.kind == 'D') {
                    swaps(t.D.data(), mk, t.n, pv, mk);
                    if (j > k) trsm('L', 'L', 'N', 'U', mk, t.n, L, mk, t.D.data(), mk);
                } else if (t.kind == 'L' && t.r) {
                    swaps(t.U.data(), mk, t.r, pv, mk);
                    if (j > k) trsm('L', 'L', 'N', 'U', mk, t.r, L, mk, t.U.data(), mk);
                }
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = k + 1; i < nb; ++i) {          // the column: U^-1 from the right
                Tile<Sc> &t = T_[(size_t)i * nb + k];
                if (t.kind == 'D') trsm('R', 'U', 'N', 'N', t.m, mk, L, mk, t.D.data(), t.m);
                else if (t.kind == 'L' && t.r) trsm('R', 'U', 'N', 'N', t.r, mk, L, mk, t.V.data(), t.r);
            }
            const int rest = nb - k - 1;
#pragma omp parallel for schedule(dynamic, 1)
            for (i64 p = 0; p < (i64)rest * rest; ++p) {
                const int i = k + 1 + (int)(p / rest), j = k + 1 + (int)(p % rest);
                schur(T_[(size_t)i * nb + j], T_[(size_t)i * nb + k], T_[(size_t)k * nb + j], max_rank);
            }
        }
    }

    int N_ = 0, nb_ = 0; double atol_ = 0, t_assemble_ = 0, t_factor_ = 0;
    std::vector<int> order_, start_;
    std::vector<blrk::Tile<Sc>> T_;
    std::vector<std::vector<Sc>> LU_;
    std::vector<std::vector<int>> piv_;
    int dense0_ = 0, lowrank0_ = 0, singular_ = 0; long rank0_ = 0; size_t bytes0_ = 0;
};

int num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ---------------------------------------------------------------------------------------------
// near triangle pairs: rho^2 + dz^2 < (fac * max(size_p, size_q))^2, sorted by (p, q) - the set and the
// order of fast.py's KD-tree ball queries, symmetrised with np.unique and filtered
// ---------------------------------------------------------------------------------------------
py::tuple near_pairs(arr<double> cent, arr<double> size, arr<double> z, double fac) {
    const int Nt = (int)size.shape(0);
    const double *c = cent.data(), *sz = size.data(), *zz = z.data();
    double smax = 0, x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (int t = 0; t < Nt; ++t) {
        smax = std::max(smax, sz[t]);
        x0 = std::min(x0, c[2 * t]); x1 = std::max(x1, c[2 * t]);
        y0 = std::min(y0, c[2 * t + 1]); y1 = std::max(y1, c[2 * t + 1]);
    }
    double cell = std::max(fac * smax, 1e-12);
    i64 nx = std::max(i64(1), (i64)((x1 - x0) / cell) + 1), ny = std::max(i64(1), (i64)((y1 - y0) / cell) + 1);
    while (nx * ny > i64(4) * Nt + 16) { cell *= 1.5; nx = (i64)((x1 - x0) / cell) + 1; ny = (i64)((y1 - y0) / cell) + 1; }
    std::vector<i64> cellof(Nt), start(nx * ny + 1, 0), order(Nt);
    for (int t = 0; t < Nt; ++t) {
        i64 i = std::min(nx - 1, (i64)((c[2 * t] - x0) / cell)), j = std::min(ny - 1, (i64)((c[2 * t + 1] - y0) / cell));
        cellof[t] = i * ny + j; start[cellof[t] + 1]++;
    }
    for (i64 k = 0; k < nx * ny; ++k) start[k + 1] += start[k];
    { std::vector<i64> fill(start.begin(), start.end() - 1);
      for (int t = 0; t < Nt; ++t) order[fill[cellof[t]]++] = t; }
    const i64 reach = (i64)std::ceil(fac * smax / cell);
    std::vector<std::vector<int>> nb(Nt);
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(dynamic, 64)
        for (int p = 0; p < Nt; ++p) {
            i64 ci = cellof[p] / ny, cj = cellof[p] % ny;
            std::vector<int> &out = nb[p];
            for (i64 i = std::max(i64(0), ci - reach); i <= std::min(nx - 1, ci + reach); ++i)
                for (i64 j = std::max(i64(0), cj - reach); j <= std::min(ny - 1, cj + reach); ++j)
                    for (i64 k = start[i * ny + j]; k < start[i * ny + j + 1]; ++k) {
                        int q = (int)order[k];
                        double dx = c[2 * p] - c[2 * q], dy = c[2 * p + 1] - c[2 * q + 1], dz = zz[p] - zz[q];
                        double R = fac * std::max(sz[p], sz[q]);
                        if (dx * dx + dy * dy + dz * dz < R * R) out.push_back(q);
                    }
            std::sort(out.begin(), out.end());
        }
    }
    size_t n = 0;
    for (auto &v : nb) n += v.size();
    py::array_t<int64_t> P(n), Q(n);
    int64_t *pp = P.mutable_data(), *qq = Q.mutable_data();
    size_t k = 0;
    for (int p = 0; p < Nt; ++p)
        for (int q : nb[p]) { pp[k] = p; qq[k] = q; ++k; }
    return py::make_tuple(P, Q);
}

// ---------------------------------------------------------------------------------------------
// out[s, i] = sum_k J0(rho_i kr_k) W[s, k] for real kr (glibc's j0), rho-parallel
// ---------------------------------------------------------------------------------------------
py::array_t<cd> hankel_real(arr<double> rho, arr<double> kr, arr<cd> W) {
    const int nr = (int)rho.shape(0), nk = (int)kr.shape(0), ns = (int)W.shape(0);
    if ((int)W.shape(1) != nk) throw std::invalid_argument("hankel_real: W must be (nspec, len(kr))");
    py::array_t<cd> out({ns, nr});
    cd *o = out.mutable_data();
    const double *r = rho.data(), *k = kr.data();
    const cd *w = W.data();
    {
        py::gil_scoped_release nogil;
#pragma omp parallel
        {
            std::vector<cd> acc(ns);
#pragma omp for schedule(static)
            for (int i = 0; i < nr; ++i) {
                std::fill(acc.begin(), acc.end(), cd(0));
                for (int m = 0; m < nk; ++m) {
                    double j = bessel_j0(r[i] * k[m]);
                    for (int s = 0; s < ns; ++s) acc[s] += j * w[(size_t)s * nk + m];
                }
                for (int s = 0; s < ns; ++s) o[(size_t)s * nr + i] = acc[s];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Z <- diag(d) Z diag(d) in place, and its complex64 copy (one pass over the matrix, row-parallel)
// ---------------------------------------------------------------------------------------------
py::array_t<cf> scale_c64(py::array_t<cd, py::array::c_style> Z, arr<double> d) {
    auto b = Z.request();
    const size_t n = b.shape[0], m = b.shape[1];
    if ((size_t)d.shape(0) != n || n != m) throw std::invalid_argument("scale_c64: Z square, d its size");
    cd *z = (cd *)b.ptr;
    const double *dd = d.data();
    py::array_t<cf> out({n, m});
    cf *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static)
        for (i64 i = 0; i < (i64)n; ++i) {
            cd *zr = z + (size_t)i * m; cf *orow = o + (size_t)i * m;
            for (size_t j = 0; j < m; ++j) { zr[j] *= dd[i] * dd[j]; orow[j] = cf(zr[j]); }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// G_q (centroid rule, singular parts added) x area_t x area_f between every triangle t and fan triangle f
// ---------------------------------------------------------------------------------------------
py::array_t<cd> fan_q(py::dict geom, py::dict tabs, arr<int> F) {
    MeshHold mh(geom); TabHold th(tabs);
    const Mesh &g = mh.m; const Tables &tb = th.t;
    const int Nt = g.Nt, nf = (int)F.shape(0);
    const int *f = F.data();
    py::array_t<cd> out({Nt, nf});
    cd *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static)
        for (int t = 0; t < Nt; ++t)
            for (int j = 0; j < nf; ++j) {
                cd ga, gq;
                tb.kern(g.lt[t], g.lt[f[j]], dist(g.cent, t, f[j]), ga, gq);
                o[(size_t)t * nf + j] = gq * (g.area[t] * g.area[f[j]]);
            }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// T^T Z T for T = (identity on `keep` in its first nk columns, rows `elim` = C): quads.project without the
// N x Nr temporaries. Z T is formed in place, row by row (Z[r, :Nr]; keep is ascending so keep[j] >= j is read
// before it is written, the eliminated entries are copied out first), then T^T (Z T) row by row into the
// Nr x Nr result: row j = (Z T)[keep[j]] + sum_i C[i, j] (Z T)[elim[i]] (Ct: C^T as CSR). Z is overwritten.
// ---------------------------------------------------------------------------------------------
template <class C>
py::array_t<C> project_t(py::array_t<C, py::array::c_style> Z, arr<int> keep, arr<int> elim,
                         arr<int> cp, arr<int> ci, arr<double> cv, arr<int> tp, arr<int> ti, arr<double> tv, int Nr) {
    auto b = Z.request();
    const size_t N = b.shape[0];
    const int nk = (int)keep.shape(0), ne = (int)elim.shape(0);
    if ((size_t)b.shape[1] != N || (size_t)(nk + ne) != N || Nr < nk || (int)cp.shape(0) != ne + 1 ||
        (int)tp.shape(0) != Nr + 1)
        throw std::invalid_argument("project: Z square, keep + elim its size, C (ne x Nr) and C^T as CSR");
    C *z = (C *)b.ptr;
    const int *k = keep.data(), *e = elim.data(), *cpp = cp.data(), *cii = ci.data(), *tpp = tp.data(), *tii = ti.data();
    const double *cvv = cv.data(), *tvv = tv.data();
    py::array_t<C> out({(size_t)Nr, (size_t)Nr});
    C *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
#pragma omp parallel
        {
            std::vector<C> ze(ne);
#pragma omp for schedule(static)
            for (i64 r = 0; r < (i64)N; ++r) {
                C *zr = z + (size_t)r * N;
                for (int i = 0; i < ne; ++i) ze[i] = zr[e[i]];
                for (int j = 0; j < nk; ++j) zr[j] = zr[k[j]];
                for (int j = nk; j < Nr; ++j) zr[j] = C(0);
                for (int i = 0; i < ne; ++i)
                    for (int p = cpp[i]; p < cpp[i + 1]; ++p) zr[cii[p]] += (typename C::value_type)cvv[p] * ze[i];
            }
#pragma omp for schedule(static)
            for (i64 j = 0; j < (i64)Nr; ++j) {
                C *oj = o + (size_t)j * Nr;
                if (j < nk) std::copy(z + (size_t)k[j] * N, z + (size_t)k[j] * N + Nr, oj);
                else std::fill(oj, oj + Nr, C(0));
                for (int p = tpp[j]; p < tpp[j + 1]; ++p) {
                    const C *zi = z + (size_t)e[tii[p]] * N;
                    const typename C::value_type c = (typename C::value_type)tvv[p];
                    for (int m = 0; m < Nr; ++m) oj[m] += c * zi[m];
                }
            }
        }
    }
    return out;
}

#include "static_kernels.hpp"

PYBIND11_MODULE(_core, m) {
    m.doc() = "compiled kernels for mom25d";
    m.def("static_pairs", &static_pairs, "quasi-static pair batches, host float64, explicit OpenMP worker count");
    m.def("static_thread_count", &static_thread_count);
#ifdef _OPENMP
    m.attr("static_openmp") = true;
#else
    m.attr("static_openmp") = false;
#endif
    m.def("fill_far", &fill_far_t<cd>, "far-field entries into a complex128 Z");
    m.def("fill_far_c64", &fill_far_t<cf>, "far-field entries into a complex64 Z");
    m.def("far_block", &far_block);
    m.def("matvec_far", &matvec_far);
    m.def("matvec_far_multi", &matvec_far_multi, "Y = Z_far X for several right-hand sides");
    m.def("offplane_pairs", &offplane_pairs);
    m.def("near_into_dense", &near_into_dense);
    m.def("near_matvec", &near_matvec);
    m.def("near_coo", &near_coo);
    m.def("near_q", &near_q, "the near pairs' G_q moment corrections alone");
    m.def("tree_cols", &tree_cols, "X <- X K, the loop/tree basis (looptree.py)");
    m.def("tree_rows", &tree_rows, "X <- K^T X, the loop/tree basis (looptree.py)");
    m.def("charge_into", &charge_into, "the loop/tree basis's charge block into Z[tree, tree]");
    py::class_<BLRFactorT<cf>>(m, "BLRFactor", "block low-rank LU of the equilibrated matrix (blr.py's, compiled), complex64")
        .def(py::init<arr<int>, arr<int>, arr<uint8_t>, py::dict, py::dict, double, arr<i64>, arr<int>, arr<cd>,
                      arr<cd>, arr<cd>, arr<double>, double, int>())
        .def("solve", &BLRFactorT<cf>::solve, "the approximate inverse applied to R (N or N x m)")
        .def("stats", &BLRFactorT<cf>::stats);
    py::class_<BLRFactorT<cd>>(m, "BLRFactor128", "the same in complex128: where the matrix's two parts are too "
                                                  "far apart for a complex64 factor (low frequency)")
        .def(py::init<arr<int>, arr<int>, arr<uint8_t>, py::dict, py::dict, double, arr<i64>, arr<int>, arr<cd>,
                      arr<cd>, arr<cd>, arr<double>, double, int>())
        .def("solve", &BLRFactorT<cd>::solve, "the approximate inverse applied to R (N or N x m)")
        .def("stats", &BLRFactorT<cd>::stats);
    m.def("num_threads", &num_threads);
    m.def("near_pairs", &near_pairs, "near triangle pairs (sorted by p, q)");
    m.def("hankel_real", &hankel_real, "sum_k J0(rho kr_k) W[s, k] for real kr");
    m.def("scale_c64", &scale_c64, "Z <- D Z D in place, and its complex64 copy");
    m.def("fan_q", &fan_q, "G_q moments between every triangle and the fan triangles");
    m.def("project", &project_t<cd>, "T^T Z T (quad cells, line functions), complex128, Z overwritten");
    m.def("project_c64", &project_t<cf>, "T^T Z T (quad cells, line functions), complex64, Z overwritten");
}
