import numpy as np
import pytest
from scipy.signal import butter, sosfilt

import audio_analyzer as aa
import ai_separation as A

SR = 48000
FAKE = lambda seg: np.stack([seg * g for g in (0.1, 0.2, 0.3, 0.4)])   # drums, bass, other, vocals


def _signal(seconds, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((SR * seconds, 2)) * 0.1
    return sosfilt(butter(8, 18000, fs=SR, output='sos'), x, axis=0).astype(np.float32)


def test_streaming_stems_are_time_aligned_and_seamless():
    s = A.StreamingSeparator(SR, run_fn=FAKE)
    s.warm_up(2)
    x = _signal(10)
    out = np.zeros((4, 2, len(x)), np.float32)
    fed = 0
    for i in range(0, len(x), 480):
        s.feed(x[i:i + 480]); fed += 480
        while s.step():
            pass
        start = fed - 480 - s.latency
        if start >= 0:
            y = s.pull(start, 480)
            assert y is not None
            out[:, :, start:start + 480] = y
    a, b = SR * 2, SR * 7
    tot, ref = out.sum(0)[:, a:b], x[a:b].T
    assert 10 * np.log10((ref ** 2).sum() / ((tot - ref) ** 2).sum()) > 45
    assert s.misses == 0


def test_pull_reports_missing_regions():
    s = A.StreamingSeparator(SR, run_fn=FAKE)
    s.feed(_signal(1))
    assert s.pull(0, 480) is None and s.misses == 1


def test_controller_without_model_reports_missing(tmp_path):
    ctl = A.AIController(model_path=str(tmp_path / "none.onnx"))
    ctl.ensure(True, SR)
    assert ctl.state == "missing" and "not downloaded" in ctl.status_text


def _ai_engine(tmp_path, sep):
    app = aa.AudioBypass("a", "b", simulation=True, config_file=str(tmp_path / "c.json"))
    app.current_channels = 6
    app.upmix_enabled = app.spectral_upmix = app.center_extraction = True
    app.ai_music_mode = True
    ctl = app._ai_controller()
    ctl.separator, ctl.state, ctl.provider = sep, "ready", "CPUExecutionProvider"
    return app


def test_ai_route_places_stems_and_falls_back_without_gaps(tmp_path):
    sep = A.StreamingSeparator(SR, run_fn=FAKE)
    sep.compute_s = 0.0
    sep.latency = sep.latency_for(0.0)
    app = _ai_engine(tmp_path, sep)
    t = np.arange(SR * 6) / SR
    x = np.zeros((len(t), 8), np.float32)
    x[:, 0] = x[:, 1] = 0.2 * np.sin(2 * np.pi * 300 * t)       # centred source
    outs = []
    for bi, i in enumerate(range(0, len(x), 480)):
        outs.append(app._apply_upmix(x[i:i + 480].copy(), out_ch=6).copy())
        if not (300 <= bi < 360):          # the "GPU" stalls for 0.6 s: fallback must cover it
            while sep.step():
                pass
    out = np.concatenate(outs)
    assert app._route_mode == 'ai'
    lat = sep.latency + 1024 + 72
    body = out[lat + SR // 2:]
    # never silent, never a jump: the fallback is time-aligned
    w = 480
    rms = np.sqrt((body[:len(body) // w * w, :3] ** 2).reshape(-1, w, 3).mean(axis=(1, 2)))
    assert rms.min() > 0.02
    d = np.abs(np.diff(body[:, 2]))
    assert d.max() < 0.2 * 2 * np.pi * 300 / SR * np.sqrt(2) * 3
    assert sep.misses > 0
    # the centred "vocal" share (fake vocals = 0.4 of the mix) is in the centre
    seg = out[lat + SR:lat + SR + SR]
    assert (seg[:, 2] ** 2).sum() > 0.2 * (seg[:, [0, 1, 2]] ** 2).sum()


def test_controller_falls_back_when_a_provider_fails_at_runtime(tmp_path, monkeypatch):
    """DirectML failed on a ConvTranspose layer on an RTX 3060 Ti: a provider
    that fails its test run must hand over to the next one, not stop."""
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(A, "model_ok", lambda p: True)
    monkeypatch.setattr(A, "available_chain", lambda: [("DirectML", "Dml", {}), ("CPU", "Cpu", {})])

    class Sess:
        def __init__(self, name): self.name = name

    monkeypatch.setattr(A, "make_session", lambda path, provider, options: Sess(provider))

    def run(self, mix44):
        if self.session.name == "Dml":
            raise RuntimeError("Non-zero status code returned while running ConvTranspose node")
        return np.stack([mix44 * g for g in (0.1, 0.2, 0.3, 0.4)])

    monkeypatch.setattr(A.StreamingSeparator, "_run_model", run)
    ctl = A.AIController(model_path=str(model))
    ctl._load(SR)
    try:
        assert ctl.state == "ready" and ctl.provider == "CPU"
        assert ctl.failures and "ConvTranspose" in ctl.failures[0]
    finally:
        ctl._unload()
