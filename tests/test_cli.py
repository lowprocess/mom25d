"""The command line: a --config file's values under the typed options, paths relative to the file, typos named,
clean one-line errors."""
import os
import pytest
from mom25d.cli import parse, main


def test_config_under_the_command_line(tmp_path, monkeypatch):
    d = tmp_path / 'proj'
    d.mkdir()
    (d / 'b.kicad_pcb').write_text('')
    (d / 'b-top.pos').write_text('')
    cfg = d / 'b.toml'
    cfg.write_text('board = "b.kicad_pcb"\npos = "b-top.pos"\nout = "out"\nfmin = 1.0\nfmax = 8.0\nnets = ["RF", "GND"]\n'
                   'no-mask = true\n[dmax_layer]\n"In1.Cu" = 0.8\n')
    monkeypatch.chdir(tmp_path)                                    # somewhere else
    _, a = parse(['--config', str(cfg), '--fmax', '3'])
    assert a.fmax == 3.0 and a.fmin == 1.0                         # the command line wins
    assert a.board == str(d / 'b.kicad_pcb') and a.pos == str(d / 'b-top.pos') and a.out == str(d / 'out')
    assert a.nets == 'RF,GND' and a.no_mask and a.dmax_layer == ['In1.Cu=0.8']
    _, a = parse(['other.kicad_pcb', '--config', str(cfg)])        # a board typed on the command line wins
    assert a.board == 'other.kicad_pcb'


def test_config_paths_for_the_repository_root_still_work(tmp_path, monkeypatch, capsys):
    (tmp_path / 'examples').mkdir()
    (tmp_path / 'examples' / 'b-top.pos').write_text('')
    cfg = tmp_path / 'examples' / 'old.toml'
    cfg.write_text('pos = "examples/b-top.pos"\n')
    monkeypatch.chdir(tmp_path)
    _, a = parse(['b.kicad_pcb', '--config', str(cfg)])
    assert a.pos == 'examples/b-top.pos' and 'relative to the working directory' in capsys.readouterr().out


def test_config_typos_and_bad_files(tmp_path, capsys):
    cfg = tmp_path / 'c.toml'
    cfg.write_text('fmx = 6\nncros = 3\n')
    with pytest.raises(SystemExit) as e:
        parse(['b.kicad_pcb', '--config', str(cfg)])
    err = capsys.readouterr().err
    assert e.value.code == 2 and 'fmx (did you mean fmax?)' in err and 'ncros (did you mean ncross?)' in err
    with pytest.raises(SystemExit):
        parse(['b.kicad_pcb', '--config', str(tmp_path / 'missing.toml')])
    assert 'missing.toml' in capsys.readouterr().err
    assert main([str(tmp_path / 'nope.kicad_pcb')]) == 2
    assert 'nope.kicad_pcb' in capsys.readouterr().err
