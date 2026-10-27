import time
from types import SimpleNamespace

import numpy as np
import pytest

import ai_live as live
from ai_enhance import EnhanceGuidance, EnhanceProcessor, analysis_fold
from test_ai_live import engine, DelayedModel
from xpdss_state import PROFILE_DEFAULTS, build_profile_from_object


def test_default_off_has_no_enhancer_or_model_and_profile_roundtrips(tmp_path):
    app = engine(tmp_path)
    assert PROFILE_DEFAULTS['ai_enhance'] is False
    app._apply_upmix(np.zeros((480, 8), np.float32), 6)
    assert not hasattr(app, '_live')
    assert not hasattr(app, '_enhance_processor')
    app.ai_enhance = True
    saved = build_profile_from_object(app)
    app.ai_enhance = False
    app._apply_profile_dict(saved)
    assert app.ai_enhance is True


def test_fold_includes_center_and_surround_but_never_lfe():
    x = np.zeros((480, 8), np.float32)
    x[:, 3] = 1
    assert not analysis_fold(x).any()
    x[:, 2] = .1
    assert np.all(analysis_fold(x) > 0)
    x[:, 6] = .2
    y = analysis_fold(x)
    assert np.all(y[:, 0] > y[:, 1])


@pytest.mark.parametrize('native', [False, True])
def test_enable_off_restores_route_buffer_and_existing_output(tmp_path, native):
    baseline = engine(tmp_path / 'base')
    app = engine(tmp_path / 'enhanced')
    for obj in (baseline, app):
        obj.is_multichannel_detected = native
        obj.ai_live_mode = not native
        sep = live.LiveSeparator(48000, DelayedModel())
        ctl = obj._live_controller()
        ctl.separator, ctl.state = sep, 'ready'
        ctl.ensure = lambda *a: None
    app.ai_enhance = True
    source = np.random.default_rng(8).normal(0, .02, (480, 8)).astype(np.float32)
    reserve = app.target_latency_samples
    for _ in range(8):
        baseline._apply_upmix(source.copy(), 6)
        app._apply_upmix(source.copy(), 6)
        app._enhance_guidance.snapshot = (time.monotonic(), (-1., -2., 1.5, -2., 1.))
    assert app._route_mode == ('pass' if native else 'live')
    assert app.target_latency_samples == reserve
    app.ai_enhance = False
    for _ in range(12):
        expected = baseline._apply_upmix(source.copy(), 6).copy()
        actual = app._apply_upmix(source.copy(), 6).copy()
    np.testing.assert_array_equal(actual, expected)
    assert app._enhance_processor is None
    assert app._live.separator.enhance_guidance is None
    assert app._route_mode == baseline._route_mode


def test_native_capture_analyzes_center_without_changing_source(tmp_path):
    app = engine(tmp_path)
    app.ai_enhance = True
    app.is_multichannel_detected = True
    sep = live.LiveSeparator(48000, DelayedModel())
    ctl = app._live_controller()
    ctl.separator, ctl.state = sep, 'ready'
    ctl.ensure = lambda *a: None
    app._apply_upmix(np.zeros((480, 8), np.float32), 6)
    source = np.zeros((480, 8), np.float32)
    source[:, 2] = .1
    original = source.copy()
    for _ in range(10):
        app.input_callback(source.copy(), 480, None, None)
        sep.step()
    np.testing.assert_array_equal(app.buffer[:480], original)
    assert app._enhance_guidance.updates > 0
    assert np.max(sep.model.previous) > 0
    assert app._live_wait_budget(480) == 0


def test_linked_channels_lfe_silence_and_stale_neutral():
    p = EnhanceProcessor(48000, 6)
    g = SimpleNamespace(snapshot=(time.monotonic(), (-2., -3., 2., -3., 1.5)), error='')
    x = np.zeros((480, 6), np.float32)
    x[:, 0] = .1*np.sin(np.arange(480)*.15)
    x[:, 1] = x[:, 0]*.4
    x[:, 3] = .03
    for _ in range(20):
        y = p.process(x, g)
    np.testing.assert_array_equal(y[:, 3], x[:, 3])
    np.testing.assert_allclose(y[:, 1], y[:, 0]*.4, atol=1e-8)
    assert not y[:, 2].any() and not y[:, 4:].any()
    assert np.isfinite(y).all() and not np.array_equal(y[:, 0], x[:, 0])
    g.snapshot = (0., g.snapshot[1])
    for _ in range(320):
        p.process(np.zeros_like(x), g)
    assert np.max(abs(p.gain)) < 1e-5
    for _ in range(5):
        p.process(x, g, enabled=False)
    np.testing.assert_array_equal(p.process(x, g, enabled=False), x)


def test_no_future_dependency_and_no_queued_delay():
    g = SimpleNamespace(snapshot=(time.monotonic(), (-1., -2., 1.5, -2., 1.)), error='')
    x = np.random.default_rng(9).normal(0, .03, (480, 2)).astype(np.float32)
    changed = x.copy()
    changed[300:] *= 10
    a, b = EnhanceProcessor(48000, 2), EnhanceProcessor(48000, 2)
    np.testing.assert_array_equal(a.process(x, g)[:300], b.process(changed, g)[:300])
    impulse = np.zeros_like(x)
    impulse[0] = .1
    y = EnhanceProcessor(48000, 2).process(impulse, g)
    assert np.argmax(abs(y[:, 0])) == 0 and a.queued_delay_samples == 0


def test_guidance_failure_never_stops_live_separation():
    class BrokenGuidance:
        error = ''
        def reset(self):
            pass
        def observe(self, stems):
            raise RuntimeError('test enhancement failure')
    sep = live.LiveSeparator(48000, DelayedModel())
    sep.enhance_guidance = BrokenGuidance()
    sep.feed(np.full((480, 2), .1, np.float32))
    sep.step()
    assert not sep.error
    assert 'test enhancement failure' in sep.enhance_guidance.error


def test_master_bypass_stops_enhancement(tmp_path):
    app = engine(tmp_path)
    app.ai_enhance = True
    app.bypass_all = True
    x = np.zeros((480, 8), np.float32)
    x[:, :2] = .1
    y = app._apply_upmix(x, 6)
    np.testing.assert_array_equal(y[:, :2], x[:, :2])
    assert not app._enhance_requested() and app._enhance_processor is None


@pytest.mark.parametrize('native', [False, True])
def test_enhance_never_changes_spectral_placement_or_buffer(tmp_path, native):
    app = engine(tmp_path)
    app.is_multichannel_detected = native
    app.spectral_upmix = True
    app.target_latency_samples = 2880
    x = np.random.default_rng(41).normal(0, .02, (480, 8)).astype(np.float32)
    if not native:
        x[:, 2:] = 0
    # The model is unavailable in this test, leaving tonal controls neutral
    # so output equality isolates unwanted changes to spatial processing.
    baseline = engine(tmp_path / 'baseline')
    baseline.is_multichannel_detected = native
    baseline.spectral_upmix = True
    baseline.target_latency_samples = 2880
    for enabled in (False, True, True, False):
        app.ai_enhance = enabled
        app._read_input_block(480)
        result = app._apply_upmix(x.copy(), 6).copy()
        expected = baseline._apply_upmix(x.copy(), 6).copy()
        assert app._route_mode == ('pass_d' if native else 'spectral')
        assert getattr(app, '_route_fade_from', None) is None
        assert app.target_latency_samples == 2880
        assert app.spectral_upmix is True
        np.testing.assert_array_equal(result, expected)


def test_enhance_does_not_pause_selected_music_route(tmp_path):
    app = engine(tmp_path)
    app.ai_music_mode = True
    ctl = app._ai_controller()
    ctl.state, ctl.separator = 'ready', object()
    desired = []
    ctl.ensure = lambda enabled, rate: desired.append(enabled)
    routes = []
    app._route = lambda mode, src, out, channels, **kw: routes.append(mode)
    for enabled in (False, True, False):
        app.ai_enhance = enabled
        app._apply_upmix(np.zeros((480, 8), np.float32), 6)
    assert desired == [True, True, True]
    assert routes == ['ai', 'ai', 'ai']


def _guided(x):
    from ai_enhance import BANDS
    g = EnhanceGuidance()
    # Fake separator output: all energy in the "other" stem, 16 hops per update.
    for at in range(0, len(x) - 128, 128):
        stems = np.zeros((4, 128, 2), np.float32)
        stems[2] = x[at:at+128]
        g.observe(stems)
    return np.array(g.snapshot[1])


def test_guidance_detects_added_mud_and_harshness():
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(3)
    # Pink-ish noise shaped to the reference balance is left nearly alone.
    white = rng.normal(0, .05, (44100*4, 2))
    spec = np.fft.rfft(white, axis=0)
    hz = np.fft.rfftfreq(len(white), 1/44100)
    tilt = np.where(hz < 800, 1., (np.maximum(hz, 1)/800) ** -.58)[:, None]
    base = np.fft.irfft(spec * tilt / np.sqrt(np.maximum(hz, 20)/800)[:, None], n=len(white), axis=0)
    base = (base * .1 / np.sqrt(np.mean(base**2))).astype(np.float32)
    neutral = _guided(base)
    assert np.max(abs(neutral[:4])) < 1, neutral
    for band, index in (((200, 450), 1), ((4500, 7500), 3)):
        boosted = base + (10**(8/20)-1)*sosfilt(butter(2, band, btype='bandpass', fs=44100, output='sos'), base, axis=0)
        controls = _guided(boosted.astype(np.float32))
        assert controls[index] < neutral[index] - 1.5, (band, controls, neutral)


def test_stem_enhancer_is_causal_loudness_matched_and_fades_out():
    from ai_enhance import StemEnhancer
    rng = np.random.default_rng(5)
    stems = rng.normal(0, .05, (4, 2, 480)).astype(np.float32)
    e = StemEnhancer(48000)
    for _ in range(400):
        y = e.process(stems, 1.)
    assert y.shape == stems.shape and np.isfinite(y).all() and not np.allclose(y, stems)
    ratio = np.sqrt(np.mean(y.sum(0)**2) / np.mean(stems.sum(0)**2))
    assert abs(20*np.log10(ratio)) < 1.5
    changed = stems.copy()
    changed[..., 300:] *= 10
    a, b = StemEnhancer(48000), StemEnhancer(48000)
    np.testing.assert_array_equal(a.process(stems, 1.)[..., :300], b.process(changed, 1.)[..., :300])
    for _ in range(5):
        e.process(stems, 1., enabled=False)
    assert e.wet == 0 and e.process(stems, 1., enabled=False) is stems


def test_live_route_uses_stem_enhancer_and_skips_mix_eq(tmp_path):
    app = engine(tmp_path)
    app.ai_live_mode = True
    app.ai_enhance = True
    sep = live.LiveSeparator(48000, DelayedModel())
    ctl = app._live_controller()
    ctl.separator, ctl.state = sep, 'ready'
    ctl.ensure = lambda *a: None
    x = np.random.default_rng(6).normal(0, .02, (480, 8)).astype(np.float32)
    x[:, 2:] = 0
    for _ in range(6):
        app._live_stems = np.random.default_rng(7).normal(0, .02, (4, 2, 480)).astype(np.float32)
        app._apply_upmix(x.copy(), 6)
    assert app._route_mode == 'live'
    assert app._stem_enhancer is not None and app._stem_enhancer.wet == 1
    assert getattr(app, '_enhance_processor', None) is None
