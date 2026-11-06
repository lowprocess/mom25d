"""
A power rail as the board has it: the nets it spans (the rail tree), the parts on it and what each one is.

    r = Rail(board, '+3V3', ground=('GND',), loads={'U21': 2.0, 'U14': 1.0})
    r.nets        # {'+3V3', '/DCDC/3V3_OUT'}: joined through the series parts in r.series
    r.series      # [Series(part, a, b, kind, R, z100)]: sense / 0 ohm resistors, beads, jumpers
    r.source      # the terminal the rail is fed from (a power inductor's output pad, or the part named source=)
    r.ground_source, r.decaps, r.loads

The tree is grown from `net` through two-pad parts between two non-ground nets that are, by their reference and value,
  resistors up to max_ohms (default 1 ohm: current-sense and 0 ohm links; a feedback divider's 10 k is not),
  ferrite beads (FB*, or an L* whose value reads like one: 600R@100MHz, BLM..., bead), jumpers and net ties (JP*, NT*),
and any part named in through=; never one in stop=. A power inductor (an L* in henries) is where it ends: the
regulator's output, and the default source (its pad on the tree, the regulator's ground pads the return's source).

The board is board.json's dict with its footprints (prims.footprints: ref, value, props, lib, side, dnp, pads with
name, net, layers, geom). A terminal is a part's pads on some nets: [(layer, shapely polygon)] per pad, a
through-hole pad on every copper layer of the stack.
"""
import re
from dataclasses import dataclass, field

from shapely.geometry import shape

_SI = {'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'μ': 1e-6, 'm': 1e-3, 'k': 1e3, 'K': 1e3, 'M': 1e6,
       'G': 1e9, 'R': 1.0, 'r': 1.0, 'E': 1.0}
_NUM = re.compile(r'^(\d*)(?:[.,](\d+))?([fpnuµμmkKMGRrE]?)(\d*)\s*(F|H|Ω|ohms?|OHMS?|R)?$')
BEAD = re.compile(r'(\d+\s*(R|Ω|ohm)\s*@|bead|ferrite|^BLM|^MPZ|^FB)', re.I)
DEFAULT_BEAD_DCR, DEFAULT_BEAD_Z100 = 0.05, 600.0


def parse_value(text):
    """'0R001' 1e-3, '4k7' 4700, '1u8' 1.8e-6, '2.2uH' 2.2e-6, '100n' 1e-7, '10m' 0.01; None if not a value."""
    m = _NUM.match(str(text).strip())
    if not m or not (m.group(1) or m.group(2) or m.group(4)):
        return None
    whole, frac, pre, frac2, _ = m.groups()
    digits = f"{whole or '0'}.{frac or frac2 or '0'}"
    if frac and frac2:
        return None
    return float(digits) * _SI.get(pre, 1.0)


def part_value(fp):
    """The first value among a footprint's Val / Value fields and their '_'-separated pieces ('R_0R001_0805'), or
    None. fp: a footprint dict or a Part."""
    props, value = _pv(fp)
    for key in ('Val', 'Value', 'value'):
        v = props.get(key) if key != 'value' else value
        if not v:
            continue
        for tok in [v] + re.split(r'[_\s]+', str(v)):
            x = parse_value(tok)
            if x is not None:
                return x
    return None


def _pv(fp):
    return (fp['props'], fp.get('value')) if isinstance(fp, dict) else (fp.props, fp.value)


def field_value(fp, *keys):
    """The value of the first of a part's fields `keys` (any case, or sim.<key>) that has one, or None."""
    props, _ = _pv(fp)
    for k in keys:
        for kk in (k, k.upper(), k.lower(), 'sim.' + k.lower()):
            if props.get(kk) not in (None, ''):
                x = parse_value(re.sub(r'(ohms?|Ω|H|F)$', '', str(props[kk]).strip()))
                if x is not None:
                    return x
    return None


@dataclass
class Pad:
    part: str
    name: str
    net: str
    layers: list
    geom: object                # shapely, mm
    xy: tuple


@dataclass
class Part:
    ref: str
    value: str
    props: dict
    lib: str
    side: str
    pads: list
    raw: dict = field(repr=False, default=None)

    def nets(self):
        return {p.net for p in self.pads if p.net}

    def on(self, nets):
        return [p for p in self.pads if p.net in nets]


@dataclass
class Series:
    part: Part
    a: str                      # the net nearer the rail's net
    b: str
    kind: str                   # 'R', 'bead', 'jumper', 'through'
    R: float                    # DC resistance, ohms
    z100: float = None          # a bead's impedance at 100 MHz
    defaulted: bool = False


def parts_of(board):
    out = []
    for fp in board['prims']['footprints']:
        if fp.get('dnp'):
            continue
        pads = [Pad(fp['ref'], p.get('name'), p.get('net'), p.get('layers', []), shape(p['geom']), tuple(p['xy']))
                for p in fp.get('pads', []) if p.get('geom')]
        out.append(Part(fp['ref'], fp.get('value', ''), fp.get('props', {}), fp.get('lib', ''), fp.get('side', 'F'),
                        pads, fp))
    return out


def classify(p, max_ohms=1.0):
    """(kind, R ohms, z100, defaulted) of a two-net part that can join a rail tree, ('inductor', henries, ...) for a
    power inductor, or None."""
    ref, val = p.ref.upper(), ' '.join(str(p.props.get(k, '')) for k in ('Val', 'Value')) + ' ' + str(p.value)
    if ref.startswith(('JP', 'NT')):
        return ('jumper', 1e-5, None, False)
    if ref.startswith('FB') or (ref.startswith('L') and BEAD.search(val.strip())):
        dcr = field_value(p, 'DCR', 'dcr')
        z = re.search(r'(\d+)\s*(R|Ω|ohm)\s*@', val, re.I)
        return ('bead', dcr or DEFAULT_BEAD_DCR, float(z.group(1)) if z else DEFAULT_BEAD_Z100, dcr is None)
    if re.match(r'L\d', ref):
        h = part_value(p)
        return ('inductor', h, None, False) if h else None
    if re.match(r'R\d', ref):
        r = part_value(p)
        if r is not None and r <= max_ohms:
            return ('R', max(r, 1e-5), None, False)
    return None


@dataclass
class Rail:
    board: dict
    net: str
    ground: tuple = ('GND',)
    loads: dict = field(default_factory=dict)       # {ref: amps}
    source: str = None                              # a part's ref (default: the power inductor feeding the tree)
    ground_source: str = None
    max_ohms: float = 1.0
    through: tuple = ()
    stop: tuple = ()

    def __post_init__(self):
        self.parts = {p.ref: p for p in parts_of(self.board)}
        g = set(self.ground)
        self.nets, self.series, self.inductors, self.notes = {self.net}, [], [], []
        todo = [self.net]
        two = [p for p in self.parts.values() if len(p.nets()) == 2 and not (p.nets() & g)]
        while todo:
            n = todo.pop()
            for p in two:
                if n not in p.nets() or p.ref in self.stop or any(s.part is p for s in self.series):
                    continue
                other = next(x for x in p.nets() if x != n)
                c = ('through', field_value(p, 'DCR', 'dcr') or 1e-5, None, False) if p.ref in self.through else \
                    classify(p, self.max_ohms)
                if c is None:
                    continue
                if c[0] == 'inductor':
                    if not any(q is p for q, _ in self.inductors):
                        self.inductors.append((p, n))
                    continue
                self.series.append(Series(p, n, other, c[0], c[1], c[2], c[3]))
                if other not in self.nets:
                    self.nets.add(other)
                    todo.append(other)
        # the source: the named part's pads on the tree, else the power inductors' tree-side pads
        if self.source:
            src = [self.parts[self.source]]
        else:
            src = [p for p, _ in self.inductors]
            if not src:
                raise ValueError(f"{self.net}: no power inductor feeds the rail tree ({', '.join(sorted(self.nets))}); "
                                 f"name its source (source = <ref>)")
            if len(src) > 1:
                self.notes.append(f"{len(src)} power inductors feed the rail: {', '.join(p.ref for p in src)} (all sources)")
        self.source_parts = src
        self.source_pads = [q for p in src for q in p.on(self.nets)]
        if not self.source_pads:
            raise ValueError(f"source {', '.join(p.ref for p in src)} has no pad on the rail "
                             f"({', '.join(sorted(self.nets))})")
        # the return's source: the named part's ground pads, else those of the parts on each inductor's switch node
        if self.ground_source:
            gp = self.parts[self.ground_source].on(g)
        else:
            gp = []
            for p in src:
                sw = [x for x in p.nets() if x not in self.nets]
                for q in self.parts.values():
                    if q is not p and sw and sw[0] in q.nets():
                        gp += q.on(g)
        if not gp:
            raise ValueError(f"{self.net}: no ground pads for the return's source: name ground_source = <ref>")
        self.ground_source_pads = gp
        # decaps: two-pad capacitors from the tree to ground
        self.decaps = []
        for p in self.parts.values():
            ns = p.nets()
            if re.match(r'C\d', p.ref.upper()) and len(p.pads) == 2 and ns & self.nets and ns & g and len(ns) == 2:
                self.decaps.append(p)
        missing = [r for r in self.loads if r not in self.parts]
        if missing:
            raise ValueError(f"loads not on the board: {', '.join(missing)}")
        for r in self.loads:
            if not self.parts[r].on(self.nets):
                raise ValueError(f"load {r} has no pad on the rail ({', '.join(sorted(self.nets))})")

    def describe(self):
        lines = [f"rail {self.net}: nets {', '.join(sorted(self.nets))}"]
        for s in self.series:
            lines.append(f"  {s.part.ref} {s.kind} {s.R:.3g} ohm{' (default DCR)' if s.defaulted else ''}: {s.a} -> {s.b}")
        lines.append(f"  source {', '.join(p.ref for p in self.source_parts)} ({len(self.source_pads)} pads), return "
                     f"source {len(self.ground_source_pads)} ground pads; {len(self.decaps)} decaps, "
                     f"{len(self.loads)} loads")
        return lines + [f"  note: {n}" for n in self.notes]


def terminal(pads, layers):
    """[(layer, polygon)] of pads: an SMD pad on its layer, a through-hole one (*.Cu, or several layers) on each of
    `layers` (the stack's copper layers, names)."""
    out = []
    for p in pads:
        ls = layers if any(l in ('*.Cu', '*.Cu*') for l in p.layers) else [l for l in p.layers if l in layers]
        out += [(l, p.geom) for l in ls]
    return out
