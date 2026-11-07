"""CLI contracts, converter ambiguity, and real solver/dump round trip."""
import json
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import box, mapping
from mom25d.mqs_cli import parser, validate, select, discover, run


def board():
    def fp(ref,value,pads):
        return dict(ref=ref,value=value,props={},pads=[dict(name=str(i),net=net,layers=['F.Cu'],
                     geom=mapping(poly),xy=list(poly.centroid.coords)[0]) for i,(net,poly) in enumerate(pads)])
    vin,ground=box(0,0,4,1),box(0,2,4,3)
    return dict(outline=mapping(box(-1,-1,5,4)),stack=dict(copper=[dict(name='F.Cu',z=0,t=.035)],layers=[],thickness=0),
        copper={'F.Cu':[dict(net='VIN',polygon=mapping(vin)),dict(net='GND',polygon=mapping(ground))]},vias=[],
        prims=dict(footprints=[fp('U1','buck',[('VIN',box(0,0,1,1)),('GND',box(0,2,1,3)),('SW',box(0,4,1,5))]),
                              fp('C1','10u',[('VIN',box(3,0,4,1)),('GND',box(3,2,4,3))]),
                              fp('L1','1uH',[('SW',box(1,4,2,5)),('OUT',box(3,4,4,5))])]))


def test_describe_is_clean_json():
    p=subprocess.run([sys.executable,'-m','mom25d.mqs','--describe','--json'],capture_output=True,text=True,check=True)
    d=json.loads(p.stdout)
    assert d['format']==1 and any(o['key']=='fill_backend' for o in d['options'])
    assert not p.stderr


def test_discovery_and_explicit_override():
    b=board();a=parser().parse_args([])
    assert discover(b)[0]['switch']=='SW'
    selected,terms=select(b,a)
    assert selected['converter']=='U1' and selected['vin']=='VIN' and selected['capacitors']==['C1']
    assert len(terms)==4
    # Ambiguous extra regulator requires selection, never chooses the first.
    other=dict(b['prims']['footprints'][0],ref='U2')
    b['prims']['footprints'].append(other)
    with pytest.raises(ValueError,match='select --converter'):
        select(b,a)
    a.converter='U1';a.vin='VIN'
    assert select(b,a)[0]['converter']=='U1'


@pytest.mark.parametrize('args',[['--fsw','0'],['--harmonics','1.5'],['--heights','nan'],['--heights','1,1'],
                                 ['--duty','1'],['--vin-volts','5'],['--scan-points','1'],['--edge-time','-1']])
def test_invalid_settings(args):
    with pytest.raises(ValueError):validate(parser().parse_args(args))


def test_solver_outputs_are_standard_dumps(tmp_path):
    b=tmp_path/'board.json';b.write_text(json.dumps(board()))
    a=parser().parse_args([str(b),'--out',str(tmp_path/'result'),'--harmonics','1,2','--scan-points','5',
                          '--heights','1,2','--dmax','1','--fine','1','--fill-backend','numpy'])
    result=run(a)
    assert result['triangles']>0 and np.array(result['peak_H_Am']).shape==(2,2)
    assert np.isfinite(result['copper_L_H']).all()
    from kicad2ems.fields.frames import find_dumps
    dumps=find_dumps(str(tmp_path/'result'))
    assert len(dumps)==3
    # Writer and frontend reader agree on units, axes and frequency/component ordering.
    files=list((tmp_path/'result'/'mqs'/'exc_converter').glob('*.npz'))
    for path in files:
        d=np.load(path)
        assert d['data'].shape==(2,len(d['x']),len(d['y']),len(d['z']),3)
        assert np.isfinite(d['data']).all()
    assert (tmp_path/'result'/'mqs.json').exists()

    # Reusing an output folder cannot expose stale planes/volume to the GUI.
    a.heights='1'
    run(a)
    assert len(find_dumps(str(tmp_path/'result')))==1
