"""Parity of the kicad2ems plugin: mom25d's planar model read back (mom25d.board.BoardModel) against kicad2ems's
board, minus what the plugin declares (the outline, copper below the ground, vias, edge walls as via rows, parts as
strips). Every difference must be declared; a via moved after the declaration fails. From kicad2ems's tests,
which check its FDTD model the same way."""
import pytest
from kicad2ems.solvers import registry
from kicad2ems.parity import check
from k2e_fixtures import prepared_job

pytest.importorskip('triangle')

CASES = dict(transition=('demo', ['--nets', 'RF,GND', '--ground', 'B.Cu']),
             lowpass=('lowpass', []),
             launch=('launch', ['--edge-walls', 'all']))


@pytest.mark.parametrize('case', list(CASES))
def test_every_difference_is_declared(case, boards, tmp_path):
    name, args = CASES[case]
    j, prep = prepared_job(tmp_path, boards[name], 'mom25d', *args)
    sv = registry()['mom25d']
    assert check(j.m, sv.readback(j, prep), sv.deviations(j, prep)) == []


def test_undeclared_changes_fail(boards, tmp_path, monkeypatch):
    from mom25d.board import BoardModel
    name, args = CASES['transition']
    j, prep = prepared_job(tmp_path, boards[name], 'mom25d', *args)
    real = BoardModel._via
    monkeypatch.setattr(BoardModel, '_via', lambda self, x, *a, **k: real(self, x + 0.1 if x > 0 else x, *a, **k))
    sv = registry()['mom25d']
    problems = check(j.m, sv.readback(j, prep), sv.deviations(j, prep))
    assert problems and all(p.startswith('via:') for p in problems)


def test_a_dropped_declaration_fails(boards, tmp_path):
    j, prep = prepared_job(tmp_path, boards['lowpass'], 'mom25d')
    sv = registry()['mom25d']
    devs = [d for d in sv.deviations(j, prep) if not d.item.startswith('part:')]       # parts left unsaid
    problems = check(j.m, sv.readback(j, prep), devs)
    assert any(p.startswith(('part', 'copper on F.Cu')) for p in problems), problems
