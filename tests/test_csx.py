"""AppCSXCAD / openEMS geometry files as mom25d models: stack and ground, metal, vias, vertical metal, lumped
parts and ports (with and without the <mom25d> hints), a solve; the gmsh mesher's contract."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from mom25d.csx import CSXModel                                   # noqa: E402

W = 0.38                                                           # ~50 ohm on 0.2 mm FR-4


def _box(prims, p1, p2, prio=10):
    b = ET.SubElement(prims, 'Box', Priority=str(prio))
    ET.SubElement(b, 'P1', X=str(p1[0]), Y=str(p1[1]), Z=str(p1[2]))
    ET.SubElement(b, 'P2', X=str(p2[0]), Y=str(p2[1]), Z=str(p2[2]))


def _poly(prims, pts, z, tag='Polygon', length=None, prio=10):
    kw = dict(Priority=str(prio), Elevation=str(z), NormDir='2', QtyVertices=str(len(pts)))
    if length is not None:
        kw['Length'] = str(length)
    p = ET.SubElement(prims, tag, **kw)
    for x, y in pts:
        ET.SubElement(p, 'Vertex', X1=str(x), X2=str(y))


def write_csx(path, hints=False, part=False, via=False, wall=False, line=10.0, sheet=False):
    """A microstrip line along x (y = 0) on 0.2 mm FR-4 over a B.Cu plane, mm, ports at both ends. sheet: the
    line is a conducting sheet, written as CSXCAD writes one (Conductivity / Thickness on the element)."""
    root = ET.Element('ContinuousStructure', CoordSystem='0')
    ET.SubElement(root, 'RectilinearGrid', DeltaUnit='0.001', CoordSystem='0')
    props = ET.SubElement(root, 'Properties')
    board = [(-2, -4), (line + 2, -4), (line + 2, 4), (-2, 4)]
    mat = ET.SubElement(props, 'Material', ID='0', Name='core')
    _poly(ET.SubElement(mat, 'Primitives'), board, 0.0, 'LinPoly', 0.2, prio=1)
    ET.SubElement(mat, 'Property', Epsilon='4.4,1,1', Kappa=f"{2 * np.pi * 3e9 * 8.8541878128e-12 * 4.4 * 0.02},0,0")
    gnd = ET.SubElement(props, 'Metal', ID='1', Name='cu_B.Cu')
    _poly(ET.SubElement(gnd, 'Primitives'), board, 0.0)
    top = ET.SubElement(props, 'ConductingSheet', ID='2', Name='cu_F.Cu', Conductivity='5.800000e+07',
                        Thickness='3.500000e-05') if sheet else ET.SubElement(props, 'Metal', ID='2', Name='cu_F.Cu')
    tp = ET.SubElement(top, 'Primitives')
    if part:                                                       # a 1.0 mm gap at the middle, bridged by a part
        _poly(tp, [(0, -W / 2), (line / 2 - 0.5, -W / 2), (line / 2 - 0.5, W / 2), (0, W / 2)], 0.2)
        _poly(tp, [(line / 2 + 0.5, -W / 2), (line, -W / 2), (line, W / 2), (line / 2 + 0.5, W / 2)], 0.2)
        el = ET.SubElement(props, 'LumpedElement', ID='9', Name='part_R1', Direction='0', Caps='1', R='50', LEtype='1')
        _box(ET.SubElement(el, 'Primitives'), (line / 2 - 0.5, -W / 2, 0.2), (line / 2 + 0.5, W / 2, 0.2), 12)
    else:
        _poly(tp, [(0, -W / 2), (line, -W / 2), (line, W / 2), (0, W / 2)], 0.2)
    if via or wall:
        _poly(tp, [(2, 1.5), (3, 1.5), (3, 2.5), (2, 2.5)], 0.2)    # a pad off the line
    vm = ET.SubElement(props, 'Metal', ID='3', Name='vias')
    vp = ET.SubElement(vm, 'Primitives')
    if via:
        c = ET.SubElement(vp, 'Cylinder', Priority='11', Radius='0.15')
        ET.SubElement(c, 'P1', X='2.5', Y='2', Z='0')
        ET.SubElement(c, 'P2', X='2.5', Y='2', Z='0.2')
    if wall:
        _box(vp, (2, 2, 0), (3, 2, 0.2), 11)                        # vertical sheet along x under the pad
    for n, x in ((1, 0.05), (2, line - 0.05)):
        el = ET.SubElement(props, 'LumpedElement', ID=str(10 + n), Name=f'port_resist_{n}', Direction='2', Caps='1',
                           R='50', LEtype='0')
        _box(ET.SubElement(el, 'Primitives'), (x, -W / 2, 0), (x, W / 2, 0.2), 12)
    if hints:
        h = ET.SubElement(root, 'mom25d', version='1', ground='B.Cu', f_loss='3e9')
        ET.SubElement(h, 'Layer', name='F.Cu', z='0.2')
        ET.SubElement(h, 'Layer', name='B.Cu', z='0')
        ET.SubElement(h, 'Dielectric', name='core', z0='0', z1='0.2', er='4.4', tand='0.02')
        for n, x, d in ((1, 0.05, 1), (2, line - 0.05, -1)):
            ET.SubElement(h, 'Port', number=str(n), name=f"IN{n}", layer='F.Cu', ref_layer='B.Cu', x=str(x), y='0',
                          dx=str(d), dy='0', width=str(W))
    ET.ElementTree(root).write(path)
    return str(path)


def test_stack_ground_ports_inferred(tmp_path):
    m = CSXModel(write_csx(tmp_path / 'a.xml'), f_loss=3e9, log=lambda *a: None)
    assert m.unit == 1e-3 and m.ground == 'B.Cu' and m.iface == {'F.Cu': 1}
    assert len(m.stack.layers) == 1 and m.stack.layers[0].h == pytest.approx(0.2e-3)
    assert m.stack.layers[0].er == 4.4 and m.stack.layers[0].tand == pytest.approx(0.02, rel=1e-6)
    p1, p2 = m.port_fps[1], m.port_fps[2]
    assert (p1['name'], p2['name']) == ('P1', 'P2')                # no hints: P<n>
    assert p1['rot'] == pytest.approx(0) and abs(p2['rot']) == pytest.approx(180)  # into the line from both ends
    assert p1['props']['mom.layer'] == 'F.Cu' and p1['props']['mom.kref'] == 0 and m.port_z0 == {'P1': 50, 'P2': 50}
    assert m.shapes[1][0].area == pytest.approx(10 * W * 1e-6, rel=1e-6)
    assert not m.vias and not m.lumped


def test_hints_vias_walls_parts(tmp_path):
    m = CSXModel(write_csx(tmp_path / 'b.xml', hints=True, via=True), log=lambda *a: None)
    assert [m.port_fps[k]['name'] for k in (1, 2)] == ['IN1', 'IN2']
    assert m.stack.layers[0].tand == 0.02 and not any('Kappa' in n for n in m.notes)
    assert len(m.vias) == 1 and (m.vias[0].k_bot, m.vias[0].k_top) == (0, 1)
    assert m.vias[0].x == pytest.approx(2.5e-3) and m.vias[0].a == pytest.approx(0.15e-3)
    w = CSXModel(write_csx(tmp_path / 'c.xml', wall=True), f_loss=3e9, wall_pitch=0.25e-3, log=lambda *a: None)
    assert len(w.vias) == 5 and any('vertical metal' in n for n in w.notes)   # 1 mm wall, 0.25 mm pitch
    p = CSXModel(write_csx(tmp_path / 'd.xml', part=True), f_loss=3e9, log=lambda *a: None)
    assert len(p.lumped) == 1 and p.lumped[0].Z(1e9) == pytest.approx(50)
    assert p.shapes[1][0].area == pytest.approx(10 * W * 1e-6, rel=1e-6)    # the strip bridges the gap


def test_conducting_sheet_is_metal_loss(tmp_path):
    """A conducting sheet's conductivity (as CSXCAD writes it: attributes of the element) is the conductor loss of
    the model's simulation, unless the call gives its own."""
    from mom25d.sim import Metal
    m = CSXModel(write_csx(tmp_path / 's.xml', hints=True, sheet=True), log=lambda *a: None)
    assert m.sigma == 5.8e7 and m.sheet_t == pytest.approx(35e-6)
    sim = m.simulation(dmax=1.5e-3, fmin=2e9, fmax=4e9, verbose=False, characterize=False)
    assert sim.metal is not None and sim.metal.sigma == 5.8e7 and sim.metal.t == pytest.approx(35e-6)
    sim = m.simulation(dmax=1.5e-3, fmin=2e9, fmax=4e9, verbose=False, characterize=False, metal=None)
    assert sim.metal is None
    assert CSXModel(write_csx(tmp_path / 'p.xml', hints=True), log=lambda *a: None).sigma is None


def test_errors(tmp_path):
    path = tmp_path / 'e.xml'
    ET.ElementTree(ET.Element('ContinuousStructure')).write(path)
    with pytest.raises(ValueError, match='no dielectric'):
        CSXModel(str(path))
    ET.ElementTree(ET.Element('Foo')).write(path)
    with pytest.raises(ValueError, match='ContinuousStructure'):
        CSXModel(str(path))
    with pytest.raises(ValueError, match='no copper layer'):
        CSXModel(write_csx(tmp_path / 'f.xml'), ground='In7.Cu')


@pytest.mark.parametrize('port', ['via', 'auto'])
def test_solve_line(tmp_path, port):
    from mom25d.run import solve_band
    m = CSXModel(write_csx(tmp_path / 'g.xml', hints=True, line=16.0), port=port, log=lambda *a: None)
    sim = m.simulation(dmax=0.8e-3, fmin=2e9, fmax=6e9, verbose=False, wave_down_to=2e9)
    kinds = {type(p).__name__ for p in sim.ports}
    assert kinds == ({'ViaPort'} if port == 'via' else {'LinePort'})
    f, S, info = solve_band(sim, 2e9, 6e9, linear=3, zref=m.port_z0, log=lambda *a: None)
    assert info['names'] == ['IN1', 'IN2'] and S.shape == (3, 2, 2)
    assert np.abs(S[:, 0, 1] - S[:, 1, 0]).max() < (1e-3 if port == 'via' else 5e-3)   # reciprocal (wave: a fit)
    assert (np.abs(S[:, 1, 0]) > (0.85 if port == 'via' else 0.95)).all()      # a matched line passes
    assert (np.linalg.svd(S, compute_uv=False).max(1) < 1 + 1e-6).all()       # passive


@pytest.mark.parametrize('mesher', ['gmsh-layers', 'gmsh-rings', 'gmsh-graded', 'distmesh', 'distmesh-rings'])
def test_mesher_contract(mesher):
    from functools import partial
    from shapely.geometry import box
    if mesher.startswith('gmsh'):
        pytest.importorskip('gmsh')
        from mom25d.mesh_gmsh import mesh_polygons_gmsh
        fn = partial(mesh_polygons_gmsh, mode=mesher.split('-')[1])
    else:
        from mom25d.mesh_distmesh import mesh_polygons_distmesh
        fn = partial(mesh_polygons_distmesh, rings=mesher.endswith('rings'))
    g = box(0, 0, 4e-3, 0.5e-3).union(box(1.5e-3, 0, 2.5e-3, 2e-3))
    lines = [((1e-3, 0), (1e-3, 0.5e-3))]                          # a port cut across the strip
    for _ in (0,):
        nd, tr, vids = fn(g, 0.4e-3, 0.5e-3, lines=lines, vias=[(2e-3, 1.5e-3, 0.1e-3)])
        V = nd[tr]
        area = 0.5 * ((V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0]))
        assert (area > 0).all() and area.sum() == pytest.approx(g.area, rel=1e-9)   # CCW, covers the metal
        assert np.hypot(*(nd[vids[0]] - [2e-3, 1.5e-3])) < 1e-12                  # the via vertex
        on_cut = np.isclose(nd[:, 0], 1e-3, atol=1e-12)
        assert on_cut.sum() >= 2                                                   # the cut is made of mesh edges
        e = {tuple(sorted(p)) for t in tr for p in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        cut_edges = [(a, b) for a, b in e if on_cut[a] and on_cut[b]]
        assert sum(abs(nd[a, 1] - nd[b, 1]) for a, b in cut_edges) == pytest.approx(0.5e-3, rel=1e-9)
