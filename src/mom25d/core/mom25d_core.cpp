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
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using cd = std::complex<double>;
using cf = std::complex<float>;
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
#pragma omp parallel for schedule(dynamic, 4) if (nr * (long)nc > 4096)
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
// analytic 1/sqrt(rho^2 + d^2) moments: inner integral analytic over q, outer 6-point rule over p
// ---------------------------------------------------------------------------------------------
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
    py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double V[6];
        for (int k = 0; k < 3; ++k) { V[2 * k] = nd[2 * tr[3 * qq[i] + k]]; V[2 * k + 1] = nd[2 * tr[3 * qq[i] + k] + 1]; }
        offplane_one(V, p6 + 12 * (size_t)pp[i], w6 + 6 * (size_t)pp[i], d, a[i], b + 2 * i, c + 2 * i, e[i]);
    }
    return py::make_tuple(m0, mro, mrs, mrr);
}

// ---------------------------------------------------------------------------------------------
// near-field correction: accurate - centroid, per near triangle pair
//   mode 0: atomically add into dense Z (any T);  mode 1: y += dZ x  (x given);  mode 2: COO output
// ---------------------------------------------------------------------------------------------
struct NearOut { int mode; void *Z; size_t ld; bool single; const cd *x; cd *y;
                 int *ri, *ci; cd *vv; cd *qcorr; };

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
        // analytic parts
        double V[6];
        for (int k = 0; k < 3; ++k) { V[2 * k] = nodes[2 * tris[3 * q + k]]; V[2 * k + 1] = nodes[2 * tris[3 * q + k] + 1]; }
        auto add = [&](double dd, cd cA, cd cq) {
            double m0, mro[2], mrs[2], mrr;
            offplane_one(V, P6 + 12 * (size_t)p, W6 + 6 * (size_t)p, dd, m0, mro, mrs, mrr);
            M0 += cA * m0; Mox += cA * mro[0]; Moy += cA * mro[1]; Msx += cA * mrs[0]; Msy += cA * mrs[1];
            Mrr += cA * mrr; Mq += cq * m0;
        };
        if (tb.kind[t] == 0) {
            add(0.0, tb.cA[t], tb.cq[t]);
            if (tb.dimg[t] > 0) add(tb.dimg[t], tb.cAi[t], tb.cqi[t]);
        } else if (tb.kind[t] == 1) add(tb.d[t], tb.cA[t], tb.cq[t]);
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
    {
        py::gil_scoped_release nogil;
        near_core(mh.m, th.t, nodes.data(), tris.data(), P3.data(), W3.data(), P6.data(), W6.data(),
                  P.data(), Q.data(), n, w, o);
    }
    return py::make_tuple(ri, ci, vv, qc);
}

int num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "compiled kernels for mom25d";
    m.def("fill_far", &fill_far_t<cd>, "far-field entries into a complex128 Z");
    m.def("fill_far_c64", &fill_far_t<cf>, "far-field entries into a complex64 Z");
    m.def("far_block", &far_block);
    m.def("matvec_far", &matvec_far);
    m.def("matvec_far_multi", &matvec_far_multi, "Y = Z_far X for several right-hand sides");
    m.def("offplane_pairs", &offplane_pairs);
    m.def("near_into_dense", &near_into_dense);
    m.def("near_matvec", &near_matvec);
    m.def("near_coo", &near_coo);
    m.def("num_threads", &num_threads);
}
