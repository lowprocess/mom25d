"""
mom25d.linecache - the wave ports' line characterisations (sim.line_char_xs, sim.line_char_pair) kept on disk by
their cross-section, so that a board run again does not solve its feeds' lines again: on tests/boards/launch.json
they are 14.4 s of a 14.6 s set-up, at every run (R-D10).

    MOM25D_CACHE    the folder (default: $XDG_CACHE_HOME/mom25d, else ~/.cache/mom25d); 'off': nothing kept

    out, kept = cached('xs', stack, dict(layer=.., w=.., ...), lambda: line_char_xs(...))

An entry's key is the stack (each layer's h, er, tan d and fref; the covers), what the call was given and a digest
of mom25d's own code (its modules and compiled core): a change to any of them solves the line again. A cache that
cannot be read or written is no cache (the line is solved).
"""
import hashlib
import json
import os
import pathlib

import numpy as np

VERSION = 1
_CODE = None


def folder():
    """Where the lines are kept, or None (MOM25D_CACHE=off)."""
    v = os.environ.get('MOM25D_CACHE', '').strip()
    if v.lower() in ('off', '0', 'no', 'none', 'false'):
        return None
    if v:
        return pathlib.Path(v)
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return pathlib.Path(base) / 'mom25d'


def code():
    """A digest of mom25d's own modules and compiled core: what solved a kept line."""
    global _CODE
    if _CODE is None:
        h = hashlib.sha256()
        here = pathlib.Path(__file__).parent
        for p in sorted(here.glob('*.py')) + sorted(here.glob('_core*')):
            h.update(p.name.encode())
            h.update(p.read_bytes())
        _CODE = h.hexdigest()[:16]
    return _CODE


def _stack(stack):
    return dict(layers=[[l.h, l.er, l.tand, l.fref] for l in stack.layers], top=stack.top, top_er=stack.top_er,
                bottom=stack.bottom, bottom_er=stack.bottom_er, pec=list(stack.pec))


def _plain(o):
    return o.tolist() if hasattr(o, 'tolist') else str(o)


def cached(kind, stack, args, solve):
    """(solve()'s tuple of arrays, whether it was kept from before): kept under (kind, the stack, args)."""
    d = folder()
    if d is None:
        return solve(), False
    key = json.dumps(dict(version=VERSION, code=code(), kind=kind, stack=_stack(stack), args=args), sort_keys=True,
                     default=_plain)
    path = d / f"line_{hashlib.sha256(key.encode()).hexdigest()[:32]}.npz"
    try:
        with np.load(path) as z:
            if str(z['key']) == key:
                return tuple(z[f'a{i}'] for i in range(int(z['n']))), True
    except (OSError, KeyError, ValueError):
        pass
    out = solve()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{path.stem}.{os.getpid()}.tmp.npz"          # written whole, then put in place
        np.savez(tmp, key=np.array(key), n=len(out), **{f'a{i}': np.asarray(a) for i, a in enumerate(out)})
        os.replace(tmp, path)
    except OSError:
        pass
    return out, False
