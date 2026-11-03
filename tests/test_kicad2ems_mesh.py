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


def test_the_run_writes_its_quad_cells_beside_the_preview(run_cli, tmp_path):
    """With --mom-quads on the preview meshes what the run solves on (its structured runs), and the run writes that
    mesh again as mom25d-run with the diagonals it merged into quad cells (`hidden`), which segments() leaves out."""
    out = tmp_path / 'out'
    args = (os.path.join(ROOT, 'examples', 'rf_interdig.toml'), '--solvers', 'mom25d', '--mom-quads', 'on',
            '--mom-port', 'lumped', '--points', '21', '--no-report', '--out', out)
    code, so, se = run_cli(*args, '--preview')
    assert code == 0, so + se
    code, so, se = run_cli(*args)
    assert code == 0, so + se
    found = meshes.find(str(out))
    assert {'mom25d', 'mom25d-run'} <= set(found)
    pre, run = np.load(found['mom25d']), np.load(found['mom25d-run'])
    assert np.array_equal(pre['tris'], run['tris']) and np.allclose(pre['nodes'], run['nodes'])    # the same mesh
    hidden = run['hidden']
    assert len(hidden) > 50 and len(pre['hidden']) == 0                                # quads only in the run's
    tris = run['tris']
    edges = {tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    assert all(tuple(h) in edges for h in hidden)                                      # each one a triangle edge
    assert len(meshes.segments(found['mom25d'])) - len(meshes.segments(found['mom25d-run'])) == 2 * len(hidden)
    assert 'quad cells' in meshes.counts(found['mom25d-run'])
