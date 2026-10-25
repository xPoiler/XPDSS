from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import pytest


@pytest.fixture(autouse=True)
def _no_output_headroom(monkeypatch, request):
    """DSP tests check absolute levels: run them without the output headroom
    (tests that need it opt in with @pytest.mark.headroom)."""
    if request.node.get_closest_marker("headroom"):
        return
    import audio_analyzer
    original = audio_analyzer.AudioBypass.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.output_headroom_db = 0.0

    monkeypatch.setattr(audio_analyzer.AudioBypass, "__init__", init)


def pytest_configure(config):
    config.addinivalue_line("markers", "headroom: keep the default output headroom")
