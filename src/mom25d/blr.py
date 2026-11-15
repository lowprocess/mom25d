"""The block low-rank solver's tiling: the unknowns clustered into tiles of about `tile` (cluster_tiles). The
factor itself is the compiled core's (core/blr.hpp, fast._CompiledBLR); its Python one, reached only with an
older core, is retired (R-F3)."""
import numpy as np


def cluster_tiles(pts, tile):
    """Geometric bisection (fast.Cluster): list of index arrays (leaves of <= tile points) and their bounding boxes."""
    from .fast import Cluster
    leaves = lambda c: [c] if not c.kids else leaves(c.kids[0]) + leaves(c.kids[1])
    tiles = leaves(Cluster(np.arange(len(pts)), pts, tile))
    return [c.idx for c in tiles], [(c.lo, c.hi) for c in tiles]
