"""The kicad2ems plugin's mesh file: a preview writes mom25d's triangles (<name>_mesh_mom25d.npz) for kicad2ems's
3D view, on the copper layers, each edge drawn once (from kicad2ems's tests of kicad2ems.meshes)."""
import os
import numpy as np
import pytest
from kicad2ems import meshes
from k2e_fixtures import ROOT

pytest.importorskip('triangle')


def test_preview_writes_the_triangles(run_cli, tmp_path):
    from kicad2ems.board import Board
    out = tmp_path / 'out'
    code, so, se = run_cli(os.path.join(ROOT, 'examples', 'rf_interdig.toml'), '--preview', '--solvers', 'mom25d',
                           '--points', '21', '--no-report', '--out', out)
    assert code == 0, so + se
    found = meshes.find(str(out))
    assert 'mom25d' in found
    m = Board.load(str(out / 'rf_interdig_board.json'))
    planes = sorted(set(m.cu_z.values()))                           # the copper layers' heights
    d = np.load(found['mom25d'])
    assert len(d['tris']) > 100 and set(np.unique(d['nodes'][:, 2])) <= set(planes)     # on copper layers
    edges = meshes.segments(found['mom25d'])
    assert len(edges) // 2 < 3 * len(d['tris'])                                          # each edge once
    assert 'triangles' in meshes.counts(found['mom25d'])
