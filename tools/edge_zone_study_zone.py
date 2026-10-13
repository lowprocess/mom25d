import sys; sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from functools import partial
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import affinity
import mom25d.sim, mom25d.geo as geo
from edge_zone_study_meshes import D, make
RF = unary_union([shape(c['polygon']) for l in D['copper'].values() for c in l if c['net'] == 'RF'])
def zoned(buf_mm, **kw):
    z = affinity.scale(RF.buffer(buf_mm), 1e-3, 1e-3, origin=(0, 0))
    mom25d.sim.mesh_polygons = partial(geo.mesh_polygons, edge_zone=z)
    try:
        return make(**kw)
    finally:
        mom25d.sim.mesh_polygons = geo.mesh_polygons
ZONES = {'zone 1.0 mm': (1.0, None), 'zone 0.5 mm': (0.5, None)}
if __name__ == '__main__':
    for n, (b, f) in ZONES.items():
        s = zoned(b); print(f"{n:24s} N={s.Ne + s.Nv:6d} per layer", {int(k): int((s.edge_layer == k).sum()) for k in s.layers})
