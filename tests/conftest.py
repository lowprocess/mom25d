"""The kicad2ems plugin's tests (test_kicad2ems*.py) use kicad2ems's fixtures and example boards, from a kicad2ems
checkout: KICAD2EMS_SRC, default ../kicad2ems next to this repository. Without one (or without kicad2ems
installed) they are left out and mom25d's own tests run alone."""
import importlib.util, os, sys

K2E = os.environ.get('KICAD2EMS_SRC') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'kicad2ems')
if importlib.util.find_spec('kicad2ems') and os.path.exists(os.path.join(K2E, 'tests', 'k2e_fixtures.py')):
    sys.path.insert(0, os.path.join(K2E, 'tests'))
    from k2e_fixtures import *                                      # noqa: E402,F401,F403
else:
    collect_ignore_glob = ['test_kicad2ems*.py']
