import threading
import time
from pathlib import Path

import numpy as np
import pytest

import ai_live as L
import audio_analyzer as aa
from xpdss_state import PROFILE_DEFAULTS


class DelayedModel:
    """Model contract: each hop emits the preceding hop, in app stem order."""
    def __init__(self):
        self.reset()
        self.closed = False

    def reset(self):
        self.previous = np.zeros((128, 2), np.float32)

    def process(self, block):
        out = np.stack([self.previous * g for g in (0.1, 0.2, 0.3, 0.4)])
        self.previous = block.copy()
        return out

    def close(self):
        self.closed = True


@pytest.mark.parametrize('rates', [(48000, 44100), (44100, 48000), (96000, 44100)])
def test_resampler_is_independent_of_callback_boundaries(rates):
    x = np.random.default_rng(2).normal(0, 0.1, (6000, 2)).astype(np.float32)
    whole = L.StreamingResampler(*rates).process(x)
    converter = L.StreamingResampler(*rates)
    chunks = []
    pos = 0
    for size in (127, 481, 17, 1024, 256, 4095):
        chunks.append(converter.process(x[pos:pos + size]))
        pos += size
    np.testing.assert_allclose(np.concatenate(chunks), whole, atol=1e-7)


@pytest.mark.parametrize('sr', [44100, 48000, 96000])
def test_live_stems_align_to_host_audio_without_chunk_seams(sr):
    sep = L.LiveSeparator(sr, DelayedModel())
    t = np.arange(sr // 2) / sr
    x = np.stack([np.sin(2*np.pi*300*t), np.sin(2*np.pi*700*t)], 1).astype(np.float32)*0.1
    chunks = []
    for start in range(0, len(x)-480, 480):
        sep.feed(x[start:start+480])
        sep.step()
        at = start - sep.latency
        if at >= 2000:
            stems = sep.pull(at, 480)
            assert stems is not None
            reference = x[at:at+480].T
            assert np.max(np.abs(stems.sum(0)-reference)) < 0.005
            chunks.append(stems)
    assert chunks and sep.misses == 0


def test_live_queue_is_bounded_and_does_not_replay_stems_after_gap():
    sep = L.LiveSeparator(48000, DelayedModel())
    for _ in range(50):
        sep.feed(np.ones((480, 2), np.float32))
    assert sep.pending_samples <= sep.max_pending
    assert sep.resets > 0
    sep.step()
    assert sep.pull(0, 480) is None


def test_model_failure_leaves_fallback_available():
    model = DelayedModel()
    def fail(_):
        raise RuntimeError('inference failed')
    model.process = fail
    sep = L.LiveSeparator(48000, model)
    sep.feed(np.ones((480, 2), np.float32))
    sep.start()
    try:
        deadline = time.monotonic() + 2
        while not sep.error and time.monotonic() < deadline:
            time.sleep(0.005)
        assert 'inference failed' in sep.error
        assert sep.pull(0, 480) is None
    finally:
        sep.stop()
    assert model.closed


def test_disabling_during_load_never_publishes_or_keeps_model(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    model = DelayedModel()
    def load(*args):
        entered.set()
        assert release.wait(3)
        return model
    monkeypatch.setattr(L, 'models_present', lambda *args: True)
    monkeypatch.setattr(L, 'load_model', load)
    ctl = L.LiveController(model_dir=tmp_path)
    ctl.ensure(True, 48000)
    assert entered.wait(2)
    ctl.ensure(False, 48000)
    release.set()
    ctl.wait_idle(3)
    assert ctl.state == 'off' and ctl.separator is None and model.closed


def test_stream_recovers_continuously_after_queue_reset():
    sep = L.LiveSeparator(48000, DelayedModel())
    for _ in range(50):
        sep.feed(np.zeros((480, 2), np.float32))
    while sep.step():
        pass
    for _ in range(30):
        start = sep.fed
        sep.feed(np.full((480, 2), .1, np.float32))
        sep.step()
        if start > 27000:
            stems = sep.pull(start-sep.latency, 480)
            assert stems is not None
            np.testing.assert_allclose(stems.sum(0), .1, atol=.001)


def engine(tmp_path):
    app = aa.AudioBypass('a', 'b', simulation=True, config_file=str(tmp_path/'c.json'))
    app.upmix_enabled = app.center_extraction = True
    app.current_channels = 6
    return app


def test_live_defaults_off_and_never_creates_controller_when_off(tmp_path):
    assert PROFILE_DEFAULTS['ai_live_mode'] is False
    app = engine(tmp_path)
    app._apply_upmix(np.zeros((480, 8), np.float32), out_ch=6)
    assert not hasattr(app, '_live')


def test_live_input_reserve_restores_previous_behavior_when_disabled(tmp_path):
    app = engine(tmp_path)
    app.target_latency_samples = 2880  # previous ordinary adaptive reserve: 60 ms
    app.ai_live_mode = True
    app._read_input_block(480)
    assert app.target_latency_samples == 816
    app.ai_live_mode = False
    app._read_input_block(480)
    assert app.target_latency_samples == 2880
    assert app._target_max_samples == 3840


def test_live_initial_priming_does_not_round_reserve_up_to_callback_size(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    app.input_callback(np.full((1440, 8), .1, np.float32), 1440, None, None)
    assert app._read_input_block(480) is not None
    assert app.available_samples == 816


def test_live_entry_trims_an_already_primed_adaptive_reserve(tmp_path):
    app = engine(tmp_path)
    app.target_latency_samples = 1920
    app._in_primed = True
    app.input_callback(np.full((2400, 8), .1, np.float32), 2400, None, None)
    app.ai_live_mode = True
    app._read_input_block(480)
    assert app.available_samples == 816


def test_capture_gap_with_same_separator_never_misaligns_stems(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    sep = L.LiveSeparator(48000, DelayedModel())
    ctl = app._live_controller()
    ctl.separator, ctl.state = sep, 'ready'
    ctl.ensure = lambda *args: None
    for i in range(50):
        if i == 20:
            app.ai_live_mode = False
        else:
            app.ai_live_mode = True
        block = np.zeros((480, 8), np.float32)
        block[:, :2] = (np.arange(i*480, (i+1)*480)/48000)[:, None]
        app.input_callback(block, 480, None, None)
        while sep.step():
            pass
        if i == 20:
            continue  # capture can continue while an output stream restarts
        source = app._read_input_block(480)
        if i > 25:
            assert app._live_stems is not None
            assert np.max(np.abs(app._live_stems.sum(0)-source[:, :2].T)) < .002
    assert sep.resets == 1


def test_live_never_increases_its_reserve_on_underrun(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    app._read_input_block(480)
    for _ in range(4):
        app._in_primed = True
        app.available_samples = 0
        app._read_input_block(480)
    assert app.target_latency_samples <= 816


@pytest.mark.parametrize('entering', [True, False])
def test_classic_live_transition_advances_classic_dsp_only_once(tmp_path, entering):
    app = engine(tmp_path)
    block = np.full((480, 8), .03, np.float32)
    app.ai_live_mode = not entering
    app._apply_upmix(block.copy(), 6)
    before = app.widening_write_ptr
    app.ai_live_mode = entering
    app._apply_upmix(block.copy(), 6)
    assert app.widening_write_ptr - before == 480


def test_live_preserves_native_multichannel_and_saved_music_setting(tmp_path, monkeypatch):
    app = engine(tmp_path)
    app.ai_live_mode = app.ai_music_mode = True
    app.is_multichannel_detected = True
    app._apply_upmix(np.zeros((480, 8), np.float32), out_ch=6)
    assert app._route_mode == 'pass'
    assert app.ai_music_mode is True
    assert not hasattr(app, '_live')


def test_live_places_current_stems_and_falls_back_without_replaying_old_audio(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    sep = L.LiveSeparator(48000, DelayedModel())
    # Drive inference deterministically while exercising the real renderer.
    ctl = app._live_controller()
    ctl.separator, ctl.state = sep, 'ready'
    ctl.ensure = lambda *args: None
    x = np.zeros((48000, 8), np.float32)
    x[:24000, :2] = .1
    outputs = []
    for start in range(0, len(x), 480):
        app.input_callback(x[start:start+480].copy(), 480, None, None)
        if start < 24000:
            while sep.step():
                pass
        source = app._read_input_block(480)
        if source is not None:
            outputs.append(app._apply_upmix(source.copy(), out_ch=6).copy())
    out = np.concatenate(outputs)
    assert app._route_mode == 'live'
    assert np.mean(out[6000:12000, 2]) > .02
    assert np.max(np.abs(out[30000:])) < 1e-5
    assert sep.misses > 0


def test_live_stems_share_input_buffer_delay_and_follow_clock_drift(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    sep = L.LiveSeparator(48000, DelayedModel())
    ctl = app._live_controller()
    ctl.separator, ctl.state = sep, 'ready'
    ctl.ensure = lambda *args: None
    t = np.arange(48000)/48000
    x = np.zeros((len(t),8), np.float32)
    x[:, 0] = .1*np.sin(2*np.pi*300*t)
    x[:, 1] = .1*np.sin(2*np.pi*700*t)
    checked = 0
    for at in range(0,len(x),480):
        app.input_callback(x[at:at+480].copy(),480,None,None)
        while sep.step():
            pass
        src = app._read_input_block(480)
        if src is not None and at > 3000:
            assert app._live_stems is not None
            assert np.max(np.abs(app._live_stems.sum(0)-src[:,:2].T)) < .005
            app._apply_upmix(src.copy(),6)
            np.testing.assert_array_equal(app._live_delayed_src,src)
            checked += 1
    assert checked > 50


@pytest.mark.parametrize('mode', ['classic', 'spectral', 'bypass'])
def test_live_off_matches_existing_dsp_bit_for_bit(tmp_path, mode):
    # Reference captured from the pre-AI-Live audio_analyzer.py, not calculated
    # with the implementation under test. Covers original callback output.
    app = engine(tmp_path)
    app.ai_live_mode = False
    app.spectral_upmix = mode == 'spectral'
    app.bypass_all = mode == 'bypass'
    rng = np.random.default_rng(517)
    source = rng.normal(0,.03,(4800,8)).astype(np.float32)
    source[:, 2:] = 0
    actual = np.concatenate([app._apply_upmix(source[i:i+480].copy(),6).copy()
                             for i in range(0,len(source),480)])
    with np.load(Path(__file__).with_name('ai_live_off_reference.npz')) as golden:
        np.testing.assert_array_equal(actual, golden[mode])


def test_device_change_during_load_discards_old_model(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    model = DelayedModel()
    def load(*args):
        entered.set()
        assert release.wait(3)
        return model
    monkeypatch.setattr(L, 'models_present', lambda *args: True)
    monkeypatch.setattr(L, 'load_model', load)
    ctl = L.LiveController(model_dir=tmp_path)
    ctl.ensure(True, 48000)
    assert entered.wait(2)
    ctl.invalidate()
    release.set()
    ctl.wait_idle(3)
    assert ctl.separator is None and model.closed


def test_bad_model_download_never_replaces_existing_file(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(L.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'not a model'))
    original = L.model_path(tmp_path, 'drums')
    original.write_bytes(b'existing')
    with pytest.raises(ValueError, match='checksum'):
        L.download_models(tmp_path)
    assert original.read_bytes() == b'existing'
    assert not original.with_suffix('.part').exists()


def test_live_pull_can_receive_inflight_result_within_budget():
    sep = L.LiveSeparator(48000, DelayedModel())
    sep.feed(np.full((960, 2), .1, np.float32))
    worker = threading.Thread(target=lambda: (time.sleep(.005), sep.step()))
    worker.start()
    try:
        stems = sep.pull(0, 480, wait_s=.1)
        assert stems is not None
        assert sep.misses == 0
    finally:
        worker.join()


def test_live_pull_deadline_does_not_wait_for_future_capture():
    sep = L.LiveSeparator(48000, DelayedModel())
    start = time.perf_counter()
    assert sep.pull(0, 480, wait_s=.002) is None
    assert time.perf_counter()-start < .1


def test_waiting_for_live_stems_does_not_block_capture(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    sep = L.LiveSeparator(48000, DelayedModel())
    ctl = app._live_controller()
    ctl.separator, ctl.state = sep, 'ready'
    app.input_callback(np.full((1440, 8), .1, np.float32), 1440, None, None)
    def pull(at, n, wait_s=0):
        acquired = app.buffer_lock.acquire(blocking=False)
        assert acquired, 'Waiting inside the capture lock can prevent the missing input arriving'
        app.buffer_lock.release()
        return None
    sep.pull_available = pull
    assert app._read_input_block(480) is not None


def test_live_missed_block_does_not_make_an_abrupt_speaker_level_step(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    src = np.zeros((480, 8), np.float32)
    src[:, :2] = .1
    out = np.zeros((480, 6), np.float32)
    for _ in range(20):
        app._live_stems = np.stack([src[:, :2].T*g for g in (.1, .2, .3, .4)])
        app._route_live(src, out, 6)
    previous = out[-1].copy()
    app._live_stems = None
    app._route_live(src, out, 6)
    assert np.max(np.abs(out[0]-previous)) < .001
    assert np.max(np.abs(np.diff(out[:150], axis=0))) < .002


def test_live_declick_tail_ends_without_replaying_old_audio(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    src = np.zeros((48, 8), np.float32)
    src[:, :2] = .1
    out = np.zeros((48, 6), np.float32)
    for _ in range(10):
        app._live_stems = np.stack([src[:, :2].T*g for g in (.1,.2,.3,.4)])
        app._route_live(src, out, 6)
    # Isolate the route transition from the existing Classic effect tails.
    app._route_upmix = lambda src, dst, ch: dst.fill(0)
    src.fill(0)
    blocks = []
    for _ in range(10):
        app._live_stems = None
        app._route_live(src, out, 6)
        blocks.append(out.copy())
    assert np.max(np.abs(np.concatenate(blocks)[192:])) == 0


def test_partial_live_result_keeps_only_correctly_tagged_samples():
    sep = L.LiveSeparator(48000, DelayedModel())
    sep.feed(np.full((480, 2), .1, np.float32))
    sep.step()
    stems, valid = sep.pull_available(0, 480)
    assert 100 < valid.sum() < 480
    assert np.max(np.abs(stems[:, :, ~valid])) == 0
    assert np.max(np.abs(stems.sum(0)[:, valid][:, 40:]-.1)) < .002
    assert 0 < sep.recent_coverage() < 100
    # Wrapped slots contain old audio but must never be accepted for new tags.
    assert sep.pull_available(sep._capacity, 480) is None


def test_partial_stem_route_fades_before_missing_tail(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    src = np.zeros((480, 8), np.float32)
    src[:, :2] = .1
    out = np.zeros((480, 6), np.float32)
    for _ in range(10):
        app._live_stems = np.stack([src[:, :2].T*g for g in (.1,.2,.3,.4)])
        app._route_live(src, out, 6)
    app._live_stems = np.stack([src[:, :2].T*g for g in (.1,.2,.3,.4)])
    app._live_stems[:, :, 400:] = 0
    app._live_valid = np.arange(480) < 400
    app._route_live(src, out, 6)
    assert out[100, 2] > .05  # keep AI placement where results exist
    assert np.max(np.abs(np.diff(out[:, 2]))) < .002
    assert np.max(np.abs(out[400:, 2])) < .001


def test_short_partial_prefix_does_not_jump_at_block_boundary(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    src = np.zeros((480, 8), np.float32)
    src[:, :2] = .1
    out = np.zeros((480, 6), np.float32)
    for _ in range(10):
        app._live_stems = np.stack([src[:, :2].T*g for g in (.1,.2,.3,.4)])
        app._route_live(src, out, 6)
    previous = out[-1].copy()
    app._live_stems = np.stack([src[:, :2].T*g for g in (.1,.2,.3,.4)])
    app._live_stems[:, :, 64:] = 0
    app._live_valid = np.arange(480) < 64
    app._route_live(src, out, 6)
    assert np.max(np.abs(out[0]-previous)) < .001
    assert np.max(np.abs(np.diff(out[:, 2]))) < .002
