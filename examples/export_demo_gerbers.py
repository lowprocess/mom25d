"""Export the demo board to Gerber X2 + Excellon + KiCad job file (tests the Gerber import path)."""
import json, os, sys
from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mom25d.kicad import KiCadBoard, _items, _xy, _val, _layers
from mom25d.gerber import write_gerber

def export(pcb, outdir):
    os.makedirs(outdir, exist_ok=True)
    b = KiCadBoard(pcb)
    base = os.path.splitext(os.path.basename(pcb))[0]
    files = []
    for i, lay in enumerate(b.copper_order):
        tracks = [(*_xy(s, 'start'), *_xy(s, 'end'), _val(s, 'width', 0.2, float))
                  for s in _items(b.root, 'segment') if lay in _layers(s)]
        tracks = [(x0, -y0, x1, -y1, w) for x0, y0, x1, y1, w in tracks]
        flashes = [('C', v['x'], -v['y'], v['size'], 0) for v in b.vias if lay in v['layers'] and v['size'] > 0]
        if lay == 'F.Cu':                                    # one RoundRect pad to exercise macros
            flashes.append(('RR', 3.0, -8.0, 0.4, 0.1))
        zones = [g for (l, n, g) in b.items if l == lay and g.area > 20]       # the plane polygons
        reg = affinity.scale(unary_union(zones), 1, -1, origin=(0, 0)) if zones else box(0, 0, 0, 0)
        func = 'Copper,L%d,%s' % (i + 1, 'Top' if i == 0 else ('Bot' if i == len(b.copper_order) - 1 else 'Inr'))
        fn = f"{base}-{lay.replace('.', '_')}.gbr"
        write_gerber(os.path.join(outdir, fn), reg, func, tracks, flashes)
        files.append(dict(Path=fn, FileFunction=func, FilePolarity='Positive'))
    stack = []
    for it in b.stack_items:
        if it['kind'] == 'copper':
            stack.append(dict(Type='Copper', Thickness=it['t'] * 1e3, Name=it['name']))
        elif it['kind'] == 'mask':
            stack.append(dict(Type='SolderMask', Thickness=it['t'] * 1e3, Name=it['name'], DielectricConstant=it['er'], LossTangent=it['tand']))
        else:
            stack.append(dict(Type='Dielectric', Thickness=it['t'] * 1e3, Name=it['name'], Material='FR4', DielectricConstant=it['er'], LossTangent=it['tand']))
    json.dump(dict(Header=dict(GenerationSoftware=dict(Application='mom25d')), FilesAttributes=files,
                   MaterialStackup=stack), open(os.path.join(outdir, base + '-job.gbrjob'), 'w'), indent=1)
    # drill files: through (1..n) and blind spans
    groups = {}
    for v in b.vias:
        i0, i1 = b.copper_order.index(v['layers'][0]) + 1, b.copper_order.index(v['layers'][-1]) + 1
        groups.setdefault((i0, i1), []).append(v)
    for (i0, i1), vs in groups.items():
        kind = 'PTH' if (i0, i1) == (1, len(b.copper_order)) else 'Blind'
        fn = f"{base}-{i0}-{i1}.drl"
        L = ['M48', f'; #@! TF.FileFunction,Plated,{i0},{i1},{kind}', 'METRIC']
        tools = sorted({v['drill'] for v in vs})
        L += [f'T{k + 1}C{d:.3f}' for k, d in enumerate(tools)] + ['%', 'G90', 'G05']
        for k, d in enumerate(tools):
            L.append(f'T{k + 1}')
            L += [f"X{v['x']:.4f}Y{-v['y']:.4f}" for v in vs if v['drill'] == d]
        L.append('M30')
        open(os.path.join(outdir, fn), 'w').write("\n".join(L) + "\n")
    return os.path.join(outdir, base + '-job.gbrjob')

if __name__ == '__main__':
    print(export(sys.argv[1], sys.argv[2]))
