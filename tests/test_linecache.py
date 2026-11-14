"""The wave ports' line characterisations kept on disk by their cross-section (mom25d.linecache)."""
import numpy as np

from mom25d import linecache
from mom25d.multilayer import Layer, LayerStack


def test_a_line_solved_once_is_kept_for_the_next_run(tmp_path, monkeypatch):
    """The same stack and cross-section: kept (a board run again had solved its feeds' lines again, 14 s of the
    launch board's 15 s set-up); another stack or another width: solved; MOM25D_CACHE=off: nothing kept."""
    monkeypatch.setenv('MOM25D_CACHE', str(tmp_path / 'cache'))
    calls = []

    def solve(scale=1.0):
        calls.append(scale)
        return np.array([1j * 100.0, 1j * 210.0]) * scale, np.array([50.0 + 0.1j, 49.9 + 0.1j])
    st = LayerStack([Layer(0.8e-3, 4.3, 0.02)])
    args = dict(layer=1, w=1.5e-3, fs=np.array([3e9, 6e9]), planes=[], dmax=0.3e-3)
    a, kept = linecache.cached('xs', st, args, solve)
    b, kept2 = linecache.cached('xs', st, args, solve)
    assert not kept and kept2 and len(calls) == 1 and all(np.array_equal(x, y) for x, y in zip(a, b))
    other = LayerStack([Layer(0.8e-3, 4.4, 0.02)])
    linecache.cached('xs', other, args, lambda: solve(2.0))
    linecache.cached('xs', st, dict(args, w=1.6e-3), lambda: solve(3.0))
    linecache.cached('pair', st, args, lambda: solve(4.0))
    assert calls == [1.0, 2.0, 3.0, 4.0] and len(list((tmp_path / 'cache').glob('line_*.npz'))) == 4
    for f in (tmp_path / 'cache').glob('line_*.npz'):
        f.write_bytes(b'not a cache')                               # unreadable: solved again, and kept again
    _, kept = linecache.cached('xs', st, args, solve)
    assert not kept and len(calls) == 5 and linecache.cached('xs', st, args, solve)[1]
    monkeypatch.setenv('MOM25D_CACHE', 'off')
    _, kept = linecache.cached('xs', st, args, solve)
    assert not kept and len(calls) == 6
