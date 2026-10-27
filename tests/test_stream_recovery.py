"""Exercise startup failures without opening the machine's audio devices."""
import types
import threading

import numpy as np
import pytest

import audio_analyzer as aa


@pytest.fixture
def hardware(tmp_path, monkeypatch):
    app = aa.AudioBypass("CABLE Output", "AVR", simulation=True,
                         config_file=str(tmp_path / "config.json"))
    app.simulation = False
    app.bypass_all = True
    state = types.SimpleNamespace(failure=None, created=[], on_open=lambda: None)

    class Stream:
        def __init__(self, kind, **kwargs):
            state.on_open()
            if state.failure == (kind, "open"):
                raise RuntimeError("Device is not ready")
            self.kind = kind
            self.active = False
            self.closed = False
            self.channels = kwargs["channels"]
            state.created.append(self)

        def start(self):
            if state.failure == (self.kind, "start"):
                raise RuntimeError("Driver is still starting")
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False
            self.closed = True

    monkeypatch.setattr(aa, "sd", types.SimpleNamespace(
        InputStream=lambda **kw: Stream("input", **kw),
        OutputStream=lambda **kw: Stream("output", **kw),
        WasapiSettings=lambda **kw: None,
    ))
    return app, state


@pytest.mark.parametrize("kind", ["input", "output"])
@pytest.mark.parametrize("stage", ["open", "start"])
def test_startup_failure_recovers_when_driver_becomes_ready(hardware, kind, stage):
    app, state = hardware
    state.failure = (kind, stage)
    app.open_input_stream()
    app.open_output_stream(8)
    assert getattr(app, kind + "_stream") is None
    # A health tick while the driver is still unavailable must remain safe.
    app._check_stream_health()
    state.failure = None
    app._check_stream_health()
    assert app.input_stream is not None and app.input_stream.active
    assert app.output_stream is not None and app.output_stream.active
    assert app.current_channels == 8
    # Verify the recovered pipeline actually passes audio.
    app.input_callback(np.full((1920, 8), 0.1, np.float32), 1920, None, None)
    out = np.zeros((480, 8), np.float32)
    app.output_callback(out, 480, None, None)
    assert np.max(out) > 0.05


@pytest.mark.parametrize("kind", ["input", "output"])
def test_failed_start_closes_stream_before_retry(hardware, kind):
    app, state = hardware
    state.failure = (kind, "start")
    if kind == "input":
        app.open_input_stream()
    else:
        app.open_output_stream(8)
    assert state.created
    assert all(stream.closed for stream in state.created)


def test_health_check_preserves_working_stereo_fallback(hardware):
    app, state = hardware
    app.open_input_stream()
    app.open_output_stream(2)
    app.failed_channel_counts.add(8)
    streams = list(state.created)
    app._check_stream_health()
    assert state.created == streams
    assert app.current_channels == 2
    assert 8 in app.failed_channel_counts


def test_health_check_does_not_open_streams_after_shutdown(hardware):
    app, state = hardware
    app.stop_event.set()
    app._check_stream_health()
    assert state.created == []


@pytest.mark.parametrize("com_result", [0, 1, -2147417850])
def test_audio_worker_initializes_com_for_its_entire_lifetime(hardware, monkeypatch, com_result):
    app, state = hardware
    app.tray_icon = None
    calls = []

    def initialize(reserved, flags):
        calls.append(("initialize", threading.get_ident()))
        assert reserved is None and flags == 0  # multithreaded apartment
        return com_result

    def uninitialize():
        assert all(stream.closed for stream in state.created)
        calls.append(("uninitialize", threading.get_ident()))

    monkeypatch.setattr(aa, "IS_WINDOWS", True)
    monkeypatch.setattr(aa.ctypes, "windll", types.SimpleNamespace(ole32=types.SimpleNamespace(
        CoInitializeEx=initialize,
        CoUninitialize=uninitialize,
    )), raising=False)
    observed = []
    state.on_open = lambda: observed.append(list(calls))

    def finish_tick(_):
        observed.append(list(calls))
        app.stop()

    monkeypatch.setattr(aa.time, "sleep", finish_tick)
    worker = threading.Thread(target=app.run)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    # COM is ready before either stream opens and remains ready in the loop.
    assert observed == [[("initialize", worker.ident)]] * 3
    # RPC_E_CHANGED_MODE means an apartment already exists: do not uninitialize it.
    expected = [("initialize", worker.ident)]
    if com_result in (0, 1):
        expected.append(("uninitialize", worker.ident))
    assert calls == expected


def test_worker_keeps_com_until_external_shutdown_closes_streams(hardware, monkeypatch):
    app, state = hardware
    saving = threading.Event()
    finish_save = threading.Event()
    loop_exited = threading.Event()
    com_released = threading.Event()

    def save():
        saving.set()
        assert finish_save.wait(5)

    def loop():
        assert app.stop_event.wait(5)
        loop_exited.set()

    monkeypatch.setattr(app, "save_config", save)
    monkeypatch.setattr(app, "_run_audio_loop", loop)
    monkeypatch.setattr(aa, "IS_WINDOWS", True)
    monkeypatch.setattr(aa.ctypes, "windll", types.SimpleNamespace(ole32=types.SimpleNamespace(
        CoInitializeEx=lambda *args: 0,
        CoUninitialize=com_released.set,
    )), raising=False)
    app.open_input_stream()
    app.open_output_stream(8)
    worker = threading.Thread(target=app.run)
    stopper = threading.Thread(target=app.stop)
    worker.start()
    stopper.start()
    try:
        assert saving.wait(5) and loop_exited.wait(5)
        assert not com_released.wait(0.1)
    finally:
        finish_save.set()
        stopper.join(5)
        worker.join(5)
    assert not stopper.is_alive() and not worker.is_alive()
    assert all(stream.closed for stream in state.created)
    assert com_released.is_set()
