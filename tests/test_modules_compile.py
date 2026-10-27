import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', sorted(ROOT.glob('*.py')), ids=lambda p: p.name)
def test_app_module_compiles(path):
    # The GUI is not imported by the engine tests; catch syntax errors here.
    py_compile.compile(str(path), doraise=True)
