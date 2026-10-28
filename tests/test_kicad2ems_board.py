"""The boards mom25d's own tests read (tests/boards, test_board.py) are what kicad2ems writes now: each rebuilt
from kicad2ems's generated example with the request it records gives the same board file. So test_board.py's
checks of mom25d.board hold for kicad2ems's boards, without running them twice."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from kicad2ems.api.frontend import Board, KiCadBoard
from k2e_fixtures import GEN                                        # kicad2ems's examples/make_demo_board.py

BOARDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boards')


def _differences(a, b, path=''):
    """Where two board dicts differ: keys, lengths, values (floats to 1e-9)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [] if set(a) == set(b) else [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return out + [d for k in sorted(set(a) & set(b)) for d in _differences(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items, now {len(b)}"]
        return [d for i, (u, v) in enumerate(zip(a, b)) for d in _differences(u, v, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        ok = a is not None and b is not None and np.isclose(a, b, rtol=1e-9, atol=1e-9)
        return [] if ok else [f"{path}: {a} -> {b}"]
    return [] if a == b else [f"{path}: {a!r} -> {b!r}"]


@pytest.mark.parametrize('name', sorted(f[:-5] for f in os.listdir(BOARDS) if f.endswith('.json')))
def test_board_files_are_what_kicad2ems_writes(name, tmp_path):
    with open(os.path.join(BOARDS, name + '.json')) as fh:
        frozen = json.load(fh)
    base = str(tmp_path / name)
    subprocess.check_call([sys.executable, GEN, base] + ([] if name == 'demo' else [name]), stdout=subprocess.DEVNULL)
    r = frozen['request']
    now = Board(KiCadBoard(base + '.kicad_pcb'), nets=r['nets'], pos_file=base + '-top.pos' if r['pos'] else None,
                include_mask=r['mask'], region=r['region'], ground=r['ground'], parts=r['parts'], ports=r['ports'],
                cut_ports=r['cut_ports'], edge_walls=r['edge_walls'], edge_keepout=r['edge_keepout'],
                log=lambda *a: None).to_dict()
    now = json.loads(json.dumps(now, default=list))
    for d in (frozen, now):                                         # where the files were: not the board
        d.pop('source')
        d['request'].pop('pos')
    diffs = _differences(frozen, now)
    assert not diffs, (f"tests/boards/{name}.json is not what kicad2ems writes now (write it again from "
                       f"make_demo_board.py): " + '; '.join(diffs[:5]))
