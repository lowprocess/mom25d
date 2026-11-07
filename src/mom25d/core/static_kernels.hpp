// Quasi-static pair batches: GIL-free OpenMP, bounded outputs and no global
// omp_set_num_threads calls. A pair has one owner and a deterministic sum.
py::array_t<double> static_pairs(py::dict geometry, arr<int> P, arr<int> Q,
                                arr<double> D, arr<int> kind, bool scalar_only, int workers) {
    arr<double> V(geometry["V"]), L(geometry["L"]), tau(geometry["tau"]), un(geometry["un"]);
    arr<double> area(geometry["area"]), P3(geometry["P3"]);
    std::vector<arr<double>> points, weights;
    const int nt = (int)area.size(), n = (int)P.size(), width = scalar_only ? 1 : 6;
    if (workers < 1 || P.ndim() != 1 || Q.ndim() != 1 || D.ndim() != 1 || kind.ndim() != 1 ||
        Q.size() != n || D.size() != n || kind.size() != n ||
        V.ndim() != 3 || V.shape(0) != nt || V.shape(1) != 3 || V.shape(2) != 2 ||
        L.ndim() != 2 || L.shape(0) != nt || L.shape(1) != 3 ||
        tau.ndim() != 3 || tau.shape(0) != nt || tau.shape(1) != 3 || tau.shape(2) != 2 ||
        un.ndim() != 3 || un.shape(0) != nt || un.shape(1) != 3 || un.shape(2) != 2 ||
        P3.ndim() != 3 || P3.shape(0) != nt || P3.shape(1) != 3 || P3.shape(2) != 2)
        throw std::invalid_argument("static_pairs: invalid geometry, pair arrays or worker count");
    for (int level = 0; level < 3; ++level) {
        points.emplace_back(geometry[py::str("P" + std::to_string(level))]);
        weights.emplace_back(geometry[py::str("W" + std::to_string(level))]);
        const auto &p = points.back(); const auto &w = weights.back();
        if (p.ndim() != 3 || p.shape(0) != nt || p.shape(2) != 2 ||
            w.ndim() != 1 || w.shape(0) != p.shape(1))
            throw std::invalid_argument("static_pairs: invalid observation quadrature");
    }
    const int *pp = P.data(), *qq = Q.data(), *kk = kind.data();
    const double *dd = D.data();
    for (int i = 0; i < n; ++i)
        if (pp[i] < 0 || pp[i] >= nt || qq[i] < 0 || qq[i] >= nt ||
            kk[i] < -1 || kk[i] > 2 || !std::isfinite(dd[i]))
            throw std::invalid_argument("static_pairs: invalid triangle index, kind or offset");
    py::array_t<double> out({n, width});
    double *result = out.mutable_data();
    const double *v = V.data(), *length = L.data(), *tangent = tau.data(), *normal = un.data();
    const double *a = area.data(), *q3 = P3.data();
    const double *qp[3], *qw[3]; int nq[3];
    for (int k = 0; k < 3; ++k) {
        qp[k] = points[k].data(); qw[k] = weights[k].data(); nq[k] = (int)weights[k].size();
    }
    {
        py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(guided) if(workers > 1)
        for (int i = 0; i < n; ++i) {
            const int p = pp[i], q = qq[i], k = kk[i];
            const double d = dd[i], ad = std::fabs(d);
            double m[6] = {};
            if (k < 0) {
                const double w = a[p] * a[q] / 9;
                for (int ip = 0; ip < 3; ++ip) {
                    const double x = q3[6*(size_t)p+2*ip], y = q3[6*(size_t)p+2*ip+1];
                    for (int iq = 0; iq < 3; ++iq) {
                        const double sx = q3[6*(size_t)q+2*iq], sy = q3[6*(size_t)q+2*iq+1];
                        const double dx = x-sx, dy = y-sy;
                        const double g = w / std::sqrt(dx*dx+dy*dy+d*d);
                        m[0] += g;
                        if (!scalar_only) {
                            m[1] += g*x; m[2] += g*y; m[3] += g*sx; m[4] += g*sy;
                            m[5] += g*(x*sx+y*sy);
                        }
                    }
                }
            } else {
                for (int j = 0; j < nq[k]; ++j) {
                    const double x = qp[k][2*((size_t)p*nq[k]+j)], y = qp[k][2*((size_t)p*nq[k]+j)+1];
                    double I1 = 0, IRx = 0, IRy = 0;
                    for (int e = 0; e < 3; ++e) {
                        const size_t ve = 6*(size_t)q+2*e;
                        const double vx = v[ve]-x, vy = v[ve+1]-y;
                        const double sm = vx*tangent[ve]+vy*tangent[ve+1];
                        const double sp = sm+length[3*(size_t)q+e];
                        const double t0 = vx*normal[ve]+vy*normal[ve+1];
                        const double r02 = t0*t0+d*d, r0 = std::sqrt(r02);
                        const double rm = std::sqrt(r02+sm*sm), rp = std::sqrt(r02+sp*sp);
                        const double lg = r0 > 1e-14*length[3*(size_t)q+e]
                                          ? std::asinh(sp/r0)-std::asinh(sm/r0) : 0;
                        double inner = t0*lg;
                        if (ad > 0)
                            inner -= ad*(std::atan2(t0*sp, r02+ad*rp)-std::atan2(t0*sm, r02+ad*rm));
                        I1 += inner;
                        if (!scalar_only) {
                            const double er = .5*(r02*lg+sp*rp-sm*rm);
                            IRx += er*normal[ve]; IRy += er*normal[ve+1];
                        }
                    }
                    const double w = a[p]*qw[k][j];
                    m[0] += w*I1;
                    if (!scalar_only) {
                        const double sx = IRx+x*I1, sy = IRy+y*I1;
                        m[1] += w*x*I1; m[2] += w*y*I1;
                        m[3] += w*sx; m[4] += w*sy; m[5] += w*(x*sx+y*sy);
                    }
                }
            }
            for (int j = 0; j < width; ++j) result[(size_t)i*width+j] = m[j];
        }
    }
    return out;
}

int static_thread_count(int workers) {
    if (workers < 1) throw std::invalid_argument("workers must be positive");
    int count = 1;
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel num_threads(workers)
    {
#pragma omp single
        count = omp_get_num_threads();
    }
#endif
    return count;
}
