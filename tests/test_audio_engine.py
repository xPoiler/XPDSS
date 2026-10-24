import json
import types

import numpy as np
import pytest

import audio_analyzer as aa
from xpdss_state import PROFILE_DEFAULTS

SR = 48000


@pytest.fixture
def engine(tmp_path):
    app = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=str(tmp_path / "config.json"))
    app.current_channels = 8
    return app


def noise(frames, ch, scale=0.2, seed=0):
    return (np.random.default_rng(seed).standard_normal((frames, ch)) * scale).astype(np.float32)


def push_input(app, block):
    app.input_callback(block.copy(), block.shape[0], None, None)


# ── Real-time robustness ────────────────────────────────────────────────

def test_limiter_survives_channel_and_block_size_changes(engine):
    for shape in [(480, 8), (480, 2), (441, 8), (1024, 6), (480, 8)]:
        out = engine._apply_true_peak_limiter(noise(*shape, scale=2.0))
        assert out.shape == shape
        assert np.max(np.abs(out)) <= 1.0


def test_limiter_is_transparent_below_threshold_and_limits_peaks(engine):
    quiet = noise(4800, 2, scale=0.1)
    out = np.concatenate([engine._apply_true_peak_limiter(quiet[i:i + 480]) for i in range(0, 4800, 480)])
    la = int(0.0015 * SR)
    np.testing.assert_allclose(out[la:], quiet[:-la], atol=1e-6)  # pure look-ahead delay

    loud = np.ones((480, 2), dtype=np.float32) * 1.5
    for _ in range(5):
        out = engine._apply_true_peak_limiter(loud)
    assert np.max(np.abs(out)) <= 0.9801


def test_output_callback_never_raises(engine, monkeypatch):
    push_input(engine, noise(960, 8))
    monkeypatch.setattr(engine, "_apply_upmix", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    out = np.ones((480, 8), dtype=np.float32)
    engine.output_callback(out, 480, None, None)
    assert not out.any()
    assert engine._callback_error_count == 1


def test_head_tracking_survives_enabling_gaming_mid_session(engine):
    engine.head_tracking = True
    engine.head_tracker = types.SimpleNamespace(pos_x=0.1, pos_y=0.0, pos_z=1.0, rot_yaw=0.1)
    block = noise(480, 8)
    engine._apply_head_tracking_panning(block, 480, 8)
    engine.gaming_panning = True
    engine._apply_head_tracking_panning(block, 480, 8)
    assert np.all(np.isfinite(block))


def test_spectral_tilt_survives_block_size_change(engine):
    engine.spectral_tilt = True
    for frames in (480, 512, 441, 480):
        block = noise(frames, 8)
        engine._apply_spectral_tilt(block, frames, 8)
        assert np.all(np.isfinite(block))


def test_input_overflow_keeps_ring_buffer_consistent(engine):
    n = 0

    def ramp_block():
        nonlocal n
        blk = np.arange(n, n + 480, dtype=np.float32)[:, None].repeat(engine.max_input_ch, 1)
        n += 480
        return blk

    for _ in range(30):  # output stalled for ~300 ms: the ring buffer overflows
        push_input(engine, ramp_block())
    for _ in range(10):
        push_input(engine, ramp_block())
        src = engine._read_input_block(480)
        assert src is not None
        latency = (n - 1) - src[-1, 0]
        assert latency <= engine.target_latency_samples + 2048
        # pointers and counter agree
        assert (engine.write_ptr - engine.read_ptr) % engine.ring_buffer_size == engine.available_samples


def test_drift_compensation_consumes_one_extra_sample_when_buffer_grows(engine):
    push_input(engine, noise(480 * 3, 8))
    before = engine.available_samples
    engine._drift_fill_ema = engine.target_latency_samples + 1000
    src = engine._read_input_block(480)
    assert src.shape[0] == 480
    assert before - engine.available_samples == 481


def test_src_buffer_follows_input_channel_count(engine):
    engine.max_input_ch = 12
    engine.buffer = np.zeros((engine.ring_buffer_size, 12), dtype=np.float32)
    push_input(engine, noise(960, 12))
    assert engine._read_input_block(480).shape == (480, 12)


# ── Whole pipeline stress test ──────────────────────────────────────────

ALL_TOGGLES = [k for k, v in PROFILE_DEFAULTS.items() if isinstance(v, bool)
               and k not in ("head_tracking_preview", "manual_upmix", "independent_panning")]


@pytest.mark.parametrize("mode", ["observe", "full"])
def test_everything_enabled_with_changing_layouts(engine, mode):
    for key in ALL_TOGGLES:
        setattr(engine, key, True)
    engine.headphone_tracking = False
    engine.adaptive_mode = mode
    engine.head_tracker = types.SimpleNamespace(pos_x=0.05, pos_y=0.0, pos_z=1.1, rot_yaw=0.05)
    i = 0
    for out_ch in (8, 6, 2, 8):
        for frames in (480, 441, 512, 256, 1024):
            engine.current_channels = out_ch
            src = noise(frames, 8, scale=0.5, seed=i)
            out = engine._apply_upmix(src, out_ch=out_ch)
            i += 1
            assert out.shape == (frames, out_ch)
            assert np.all(np.isfinite(out))
            assert np.max(np.abs(out)) <= 1.0
    assert engine._callback_error_count == 0


# ── DSP behaviour ───────────────────────────────────────────────────────

def test_hp_externalization_keeps_mono_bass_level(engine):
    t = np.arange(9600) / SR
    s = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    block = np.stack([s, s], 1).copy()
    engine._apply_hp_externalization(block)
    gain_db = 20 * np.log10(np.std(block[4800:, 0]) / np.std(s[4800:]))
    assert -1.5 < gain_db < 2.0


def test_passthrough_lfe_redirect_is_bounded_and_low_passed(engine):
    engine.is_multichannel_detected = True
    t = np.arange(4800) / SR
    hf = (0.8 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    src = np.repeat(hf[:, None], 8, axis=1)
    src[:, 3] = 0.0
    lfe = np.concatenate([engine._apply_upmix(src[i:i + 480], out_ch=8)[:, 3].copy() for i in range(0, 4800, 480)])
    assert np.max(np.abs(lfe[2400:])) < 0.01  # 3 kHz content does not reach the sub


def test_dynamic_eq_boosts_bass_at_low_level(engine):
    engine.dynamic_eq = True
    t = np.arange(48000) / SR
    s = (0.003 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)  # ~ -53 dBFS
    block = np.stack([s, s], 1).copy()
    engine._program_level_db = -50.0
    engine._apply_loudness_comp(block, engine._program_level_db)
    assert np.std(block[24000:, 0]) > 1.5 * np.std(s[24000:])


def test_adaptive_full_mode_respects_disabled_modules(engine):
    engine.adaptive_mode = "full"
    engine.virtual_room = engine.air_exciter = engine.clarity_dsp = False
    block = noise(480, 2)
    ref = block.copy()
    engine._apply_virtual_room(block)
    engine._apply_air_exciter(block)
    engine._apply_clarity_dsp(block)
    np.testing.assert_array_equal(block, ref)


def test_xtc_matches_reference_recursion(engine):
    engine.transaural_delay = 7
    engine.transaural_gain = 0.6
    engine.transaural_feedback = 0.3
    x = noise(4800, 2, seed=3)
    out = x.copy()
    for i in range(0, 4800, 480):
        engine._apply_transaural_widener(out[i:i + 480], 480)

    from scipy.signal import sosfilt
    L, R = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    Llp, Rlp = sosfilt(engine._trans_lp_sos, L), sosfilt(engine._trans_lp_sos, R)
    d, ff, fb = 7, 0.6, 0.3
    oL, oR = np.zeros(4800), np.zeros(4800)
    for n in range(4800):
        g = lambda arr, k: arr[k] if k >= 0 else 0.0
        oL[n] = L[n] - ff * g(Rlp, n - d) - fb * g(oR, n - d)
        oR[n] = R[n] - ff * g(Llp, n - d) - fb * g(oL, n - d)
    np.testing.assert_allclose(out[:, 0], oL, atol=1e-5)
    np.testing.assert_allclose(out[:, 1], oR, atol=1e-5)


def test_secondary_delay_is_exact(engine):
    engine.secondary_output_delay_ms = 10.0
    x = np.arange(2000, dtype=np.float32)
    out = np.concatenate([engine._apply_secondary_output_delay(x[i:i + 480]) for i in range(0, 1920, 480)])
    d = 480
    np.testing.assert_array_equal(out[d:], x[:1920 - d])
    assert not out[:d].any()


def test_air_exciter_shaper_is_continuous(engine):
    engine.air_exciter = True
    t = np.arange(4800) / SR
    sweep = (np.linspace(0, 1.2, 4800) * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    block = np.stack([sweep, sweep], 1).copy()
    engine._apply_air_exciter(block)
    assert np.all(np.isfinite(block))
    assert np.max(np.abs(np.diff(block[:, 0]))) < 2.5


# ── State / control plumbing ────────────────────────────────────────────

def test_config_round_trip_is_atomic_and_ignores_unknown_keys(engine, tmp_path):
    engine.clarity_dsp = True
    engine.save_config()
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["speakers"]["clarity_dsp"] is True
    assert not (tmp_path / "config.json.tmp").exists()

    data["speakers"]["sample_rate"] = 1  # junk key must not reach the engine
    (tmp_path / "config.json").write_text(json.dumps(data))
    engine.load_config()
    assert engine.sample_rate == SR
    assert engine.clarity_dsp is True


def test_corrupt_config_is_preserved_not_overwritten(engine, tmp_path):
    (tmp_path / "config.json").write_text("{ not json")
    engine.load_config()
    assert (tmp_path / "config.json.corrupt").exists()


def test_switch_mode_starts_and_stops_tracker(engine, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "start_head_tracker", lambda: calls.append("start"))
    monkeypatch.setattr(engine, "stop_head_tracker", lambda: calls.append("stop"))
    engine.headphones_config = dict(PROFILE_DEFAULTS, headphone_tracking=True)
    engine.switch_mode("headphones")
    engine.switch_mode("speakers")
    assert calls == ["start", "stop"]


def test_find_device_never_falls_back_to_virtual_cable(engine, monkeypatch):
    devices = [
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 8, "default_samplerate": 48000},
        {"name": "Speakers (Realtek)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
    ]
    fake_sd = types.SimpleNamespace(query_devices=lambda: devices, query_hostapis=lambda: [{"name": "Windows WASAPI"}])
    monkeypatch.setattr(aa, "sd", fake_sd)
    assert engine._find_device("missing AVR", is_input=False)[0] == 1
    with pytest.raises(ValueError):
        engine._find_device("missing shaker", is_input=False, allow_fallback=False)


def test_calibration_uses_real_sample_rate():
    import calibration_helper as ch
    sr = 96000
    sweep = ch.generate_log_sweep(1.0, sr)

    def record(direct, cross, itd):
        h = np.zeros(4000)
        h[200] = direct
        h[200 + itd] = cross
        return np.convolve(sweep, h)[:len(sweep) + 4000]

    itd = 60  # 0.625 ms at 96 kHz: beyond the old hard-coded 40-sample window
    L_FL, R_FL = record(1.0, 0.0, itd), record(0.0, 0.5, itd)
    R_FR, L_FR = record(1.0, 0.0, itd), record(0.0, 0.5, itd)
    est_itd, k, _ = ch.calculate_transaural_params(L_FL, R_FL, L_FR, R_FR, sweep, sample_rate=sr)
    assert abs(est_itd - itd) <= 1
    assert 0.3 < k < 0.7
