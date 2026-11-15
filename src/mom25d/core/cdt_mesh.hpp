// cdt_mesh.hpp - the meshers' constrained Delaunay triangulation (cdtmesh.py, geo.mesh_polygons): CDT
// (artem-ogre/CDT 2.0.1, vendored in core/CDT, MPL-2.0). The same file in mom25d's and pcbqs's cores.
//
//   cdt_triangulate(V, E, min_dist, min_angle, max_steiner) -> (vertices, triangles)
//
// V (n, 2) points, E (m, 2) segments between them (the constraints). The input points keep their indices; points
// added (where crossing segments are split, or by the refinement) come after them. Triangles counter-clockwise.
// The triangles outside the outline (reached from the convex hull without crossing a segment) are left out, as
// Triangle's -p does: kept, the hull's slivers along a straight outline (points a hair off its line) have their
// centroids on the copper's edge (pcbqs's wpthb PDN: a zero-area triangle). With min_angle (degrees) the rest is
// refined until no triangle has a smaller angle (Ruppert; segments split where encroached; at most max_steiner points
// added, 0: no limit; the input's own sharp corners left as they are). min_dist: a point closer than this to a
// segment is on it.
// Duplicate points are an error (the caller merges them).
#pragma once
#include "CDT/CDT.h"

static py::tuple cdt_triangulate(arr<double> V, arr<int> E, double min_dist, double min_angle, long long max_steiner) {
    if (V.ndim() != 2 || V.shape(1) != 2) throw std::invalid_argument("cdt_triangulate: V must be (n, 2)");
    if (E.ndim() != 2 || E.shape(1) != 2) throw std::invalid_argument("cdt_triangulate: E must be (m, 2)");
    const size_t n = (size_t)V.shape(0), m = (size_t)E.shape(0);
    const double *v = V.data();
    const int *e = E.data();
    std::vector<CDT::V2d<double>> pts(n);
    for (size_t i = 0; i < n; ++i) pts[i] = CDT::V2d<double>(v[2 * i], v[2 * i + 1]);
    std::vector<CDT::Edge> edges;
    edges.reserve(m);
    for (size_t k = 0; k < m; ++k) {
        const int a = e[2 * k], b = e[2 * k + 1];
        if (a < 0 || b < 0 || (size_t)a >= n || (size_t)b >= n)
            throw std::invalid_argument("cdt_triangulate: segment index out of range");
        if (a != b) edges.emplace_back(CDT::VertInd(a), CDT::VertInd(b));
    }
    std::vector<double> ov;
    std::vector<int> ot;
    {
        py::gil_scoped_release nogil;
        CDT::Triangulation<double> cdt(CDT::VertexInsertionOrder::Auto, CDT::IntersectingConstraintEdges::TryResolve,
                                       min_dist);
        cdt.insertVertices(pts);
        cdt.insertEdges(edges);
        if (min_angle > 0) {
            CDT::TriIndUSet outer = cdt.collectOuterTriangles();
            const long long budget = max_steiner > 0 ? max_steiner : (long long)(10 * n + 1000000);
            cdt.refineTriangles(CDT::VertInd((unsigned long long)budget), CDT::RefinementCriterion::SmallestAngle,
                                CDT::degToRad(min_angle), &outer);
            cdt.finalizeTriangulation(outer);
        } else {
            cdt.eraseOuterTriangles();
        }
        ov.resize(2 * cdt.vertices.size());
        for (size_t i = 0; i < cdt.vertices.size(); ++i) {
            ov[2 * i] = cdt.vertices[i].x;
            ov[2 * i + 1] = cdt.vertices[i].y;
        }
        ot.resize(3 * cdt.triangles.size());
        for (size_t t = 0; t < cdt.triangles.size(); ++t)
            for (int k = 0; k < 3; ++k) ot[3 * t + k] = (int)cdt.triangles[t].vertices[k];
    }
    py::array_t<double> outv({(py::ssize_t)(ov.size() / 2), (py::ssize_t)2});
    std::copy(ov.begin(), ov.end(), outv.mutable_data());
    py::array_t<int> outt({(py::ssize_t)(ot.size() / 3), (py::ssize_t)3});
    std::copy(ot.begin(), ot.end(), outt.mutable_data());
    return py::make_tuple(outv, outt);
}
