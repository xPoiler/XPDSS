from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import pytest


@pytest.fixture(autouse=True)
def _no_system_tray(monkeypatch):
    """Engine tests do not own a Windows tray event loop or icon lifecycle."""
    import audio_analyzer
    monkeypatch.setattr(audio_analyzer.AudioBypass, "_setup_tray", lambda self: None)


@pytest.fixture(autouse=True)
def _no_automatic_live_model_load(monkeypatch):
    """DSP tests use deterministic models; real model timing has its own tool."""
    import ai_live
    monkeypatch.setattr(ai_live, 'models_present', lambda *args: False)


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
