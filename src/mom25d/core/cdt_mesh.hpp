// cdt_mesh.hpp - the meshers' constrained Delaunay triangulation (cdtmesh.py, geo.mesh_polygons): mesh2d.hpp, the
// meshers' own triangulator. The same file in mom25d's and pcbqs's cores.
//
//   cdt_triangulate(V, E, min_dist, min_angle, max_steiner) -> (vertices, triangles)
//
// V (n, 2) points, E (m, 2) segments between them (the constraints). The input points keep their indices; points
// added (where crossing segments are split, or by the refinement) come after them. Triangles counter-clockwise.
// The triangles outside the outline (reached from the bounding triangle without crossing a segment) are left out, as
// Triangle's -p does: kept, the hull's slivers along a straight outline (points a hair off its line) have their
// centroids on the copper's edge (pcbqs's wpthb PDN: a zero-area triangle). With min_angle (degrees) the rest is
// refined until no triangle has a smaller angle (Ruppert; segments split where encroached; at most max_steiner points
// added, 0: no limit; the input's own sharp corners left as they are). min_dist: a point closer than this to a
// segment is on it.
// Duplicate points are an error (the caller merges them).
#pragma once
#include "mesh2d.hpp"

static py::tuple cdt_triangulate(arr<double> V, arr<int> E, double min_dist, double min_angle, long long max_steiner) {
    if (V.ndim() != 2 || V.shape(1) != 2) throw std::invalid_argument("cdt_triangulate: V must be (n, 2)");
    if (E.ndim() != 2 || E.shape(1) != 2) throw std::invalid_argument("cdt_triangulate: E must be (m, 2)");
    const size_t n = (size_t)V.shape(0), m = (size_t)E.shape(0);
    const double *v = V.data();
    const int *e = E.data();
    for (size_t k = 0; k < m; ++k) {
        const int a = e[2 * k], b = e[2 * k + 1];
        if (a < 0 || b < 0 || (size_t)a >= n || (size_t)b >= n)
            throw std::invalid_argument("cdt_triangulate: segment index out of range");
    }
    std::vector<double> ov;
    std::vector<int> ot;
    if (n >= 3) {
        py::gil_scoped_release nogil;
        mesh2d::CDT cdt(v, n, min_dist);
        for (size_t k = 0; k < m; ++k)
            if (e[2 * k] != e[2 * k + 1]) cdt.insert_segment(e[2 * k] + 3, e[2 * k + 1] + 3);
        cdt.mark_outer();
        if (min_angle > 0) {
            double x0 = v[0], x1 = v[0], y0 = v[1], y1 = v[1];
            for (size_t i = 0; i < n; ++i)
                x0 = std::min(x0, v[2 * i]), x1 = std::max(x1, v[2 * i]), y0 = std::min(y0, v[2 * i + 1]),
                y1 = std::max(y1, v[2 * i + 1]);
            const long long budget = max_steiner > 0 ? max_steiner : (long long)(10 * n + 1000000);
            cdt.refine(min_angle * 3.14159265358979323846 / 180.0, budget, 1e-9 * std::max(x1 - x0, y1 - y0));
        }
        const size_t nv = cdt.size() - 3;
        ov.resize(2 * nv);
        for (size_t i = 0; i < nv; ++i) ov[2 * i] = cdt.X[i + 3], ov[2 * i + 1] = cdt.Y[i + 3];
        for (const auto &t : cdt.T)
            if (!t.outer)
                for (int k = 0; k < 3; ++k) ot.push_back(t.v[k] - 3);
    } else {
        ov.assign(v, v + 2 * n);
    }
    py::array_t<double> outv({(py::ssize_t)(ov.size() / 2), (py::ssize_t)2});
    std::copy(ov.begin(), ov.end(), outv.mutable_data());
    py::array_t<int> outt({(py::ssize_t)(ot.size() / 3), (py::ssize_t)3});
    std::copy(ot.begin(), ot.end(), outt.mutable_data());
    return py::make_tuple(outv, outt);
}
