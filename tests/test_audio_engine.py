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


@pytest.mark.parametrize("adaptive", [False, True])
def test_everything_enabled_with_changing_layouts(engine, adaptive):
    for key in ALL_TOGGLES:
        setattr(engine, key, True)
    engine.headphone_tracking = False
    engine.set_adaptive_enabled(adaptive)
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


def test_lfe_redirect_is_the_original_full_band_formula(engine):
    """Sub feed = content LFE + all mains * 0.316 (passthrough) or (L+R) * 0.316
    (upmix), full band and phase-exact: band-limiting and alignment are done
    downstream by the user's room correction."""
    la = int(0.0015 * SR)                                         # limiter look-ahead
    x = (np.random.default_rng(4).standard_normal((SR, 8)) * 0.05).astype(np.float32)
    engine.is_multichannel_detected = True
    out = np.concatenate([engine._apply_upmix(x[i:i + 480].copy(), out_ch=8).copy() for i in range(0, SR, 480)])
    mains = [0, 1, 2, 4, 5, 6, 7]
    np.testing.assert_allclose(out[la:, 3], (x[:, 3] + x[:, mains].sum(axis=1) * 0.316)[:-la], atol=1e-6)

    engine.is_multichannel_detected = False
    engine.upmix_enabled = True
    x[:, 2:] = 0.0
    out = np.concatenate([engine._apply_upmix(x[i:i + 480].copy(), out_ch=8).copy() for i in range(0, SR, 480)])
    tail = slice(SR // 2, SR)
    np.testing.assert_allclose(out[la:, 3][tail], ((x[:, 0] + x[:, 1]) * 0.316)[:-la][tail], atol=1e-6)

def test_dynamic_eq_boosts_bass_at_low_level(engine):
    engine.dynamic_eq = True
    t = np.arange(48000) / SR
    s = (0.003 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)  # ~ -53 dBFS
    block = np.stack([s, s], 1).copy()
    engine._program_level_db = -50.0
    engine._apply_loudness_comp(block, engine._program_level_db)
    assert np.std(block[24000:, 0]) > 1.5 * np.std(s[24000:])


def test_adaptive_respects_disabled_modules(engine):
    engine.set_adaptive_enabled(True)
    engine.adaptive_values = {"virtual_room_mix": 0.3, "air_amount": 0.4, "clarity_amount": 5.0}
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


# ── Adaptive renderer (single switch) ───────────────────────────────────

def music(seconds, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    x = np.zeros((n, 8), dtype=np.float32)
    t = np.arange(n) / SR
    x[:, 0] = 0.2 * np.sin(2 * np.pi * 330 * t) + rng.standard_normal(n) * 0.05
    x[:, 1] = 0.2 * np.sin(2 * np.pi * 330 * t + 0.3) + rng.standard_normal(n) * 0.05
    return x


def run_analysis(app, x, stride=aa.ANALYSIS_STRIDE):
    for i in range(0, x.shape[0] - 480 + 1, 480 * stride):
        app._adaptive_step(x[i:i + 480], 480 * stride / SR)


def test_adaptive_starts_from_manual_values_and_moves_them(engine):
    manual = engine._manual_adaptive_values()
    engine.set_adaptive_enabled(True)
    assert engine.adaptive_values == pytest.approx(manual)
    run_analysis(engine, music(20))
    moved = {k for k in manual if abs(engine._live_param(k) - manual[k]) > 0.05 * max(1.0, manual[k])}
    assert moved, "adaptive should have moved at least one slider"
    # the user's own values are untouched (and are what gets saved)
    assert engine._manual_adaptive_values() == manual


def test_adaptive_never_touches_toggles_or_sound_locations(engine):
    keys = ["width_factor", "center_spread", "side_gain_db", "gui_panning_x", "gui_panning_y",
            "transaural_gain", "transaural_delay", "sub_gain"] + [k for k, v in PROFILE_DEFAULTS.items() if isinstance(v, bool)]
    before = {k: getattr(engine, k) for k in keys}
    engine.set_adaptive_enabled(True)
    run_analysis(engine, music(15))
    assert {k: getattr(engine, k) for k in keys} == before


def test_dragging_a_slider_pins_it_until_adaptive_is_toggled(engine):
    engine.set_adaptive_enabled(True)
    run_analysis(engine, music(10))
    engine.set_manual_param("clarity_amount", 2.0)
    run_analysis(engine, music(10, seed=1))
    assert engine._live_param("clarity_amount") == 2.0
    assert not engine.is_adaptive_driven("clarity_amount")
    engine.set_adaptive_enabled(False)
    engine.set_adaptive_enabled(True)
    assert engine.is_adaptive_driven("clarity_amount")


def test_switching_off_glides_back_to_manual_values(engine):
    manual = engine._manual_adaptive_values()
    engine.set_adaptive_enabled(True)
    run_analysis(engine, music(20))
    engine.set_adaptive_enabled(False)
    run_analysis(engine, music(0.3))
    mid_glide = engine.adaptive_values
    run_analysis(engine, music(5))
    assert engine.adaptive_values == {}
    assert {k: engine._live_param(k) for k in manual} == manual
    assert mid_glide, "values should glide back, not jump"


def test_silence_holds_adaptive_values(engine):
    engine.set_adaptive_enabled(True)
    run_analysis(engine, music(15))
    held = dict(engine.adaptive_values)
    run_analysis(engine, np.zeros((SR * 5, 8), dtype=np.float32))
    assert engine.adaptive_values == held


def test_adaptive_switch_is_saved_but_live_values_are_not(engine, tmp_path):
    engine.clarity_amount = 3.3
    engine.set_adaptive_enabled(True)
    run_analysis(engine, music(15))
    engine.save_config()
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["adaptive_enabled"] is True
    assert data["speakers"]["clarity_amount"] == 3.3
    assert "adaptive_mode" not in data and "adaptive_profile" not in data


def test_effects_use_the_live_adaptive_value(engine):
    engine.virtual_room = True
    x = noise(480, 2)
    engine.virtual_room_mix = 0.0
    a = x.copy(); engine._apply_virtual_room(a)
    np.testing.assert_allclose(a, x, atol=1e-7)          # manual mix 0 -> dry
    engine.set_adaptive_enabled(True)
    engine.adaptive_values = {"virtual_room_mix": 0.3}
    engine._room_delay_buf = None
    b = noise(480, 2, seed=1); ref = b.copy()
    engine._apply_virtual_room(b)
    engine._apply_virtual_room(b)
    assert np.max(np.abs(b - ref)) > 1e-3                 # adaptive mix 0.3 -> reflections



def test_lfe_peaks_are_limited_without_ducking_the_mains(engine):
    t = np.arange(4800) / SR
    block = np.zeros((4800, 8), dtype=np.float32)
    block[:, 0] = 0.3 * np.sin(2 * np.pi * 1000 * t)          # dialogue level
    block[:, 3] = 1.8 * np.sin(2 * np.pi * 40 * t)            # sub overload
    out = np.concatenate([engine._apply_true_peak_limiter(block[i:i + 480]) for i in range(0, 4800, 480)])
    la = int(0.0015 * SR)
    np.testing.assert_allclose(out[la:, 0], block[:-la, 0], atol=1e-6)   # mains untouched
    assert np.max(np.abs(out[:, 3])) <= 0.9801                            # sub limited


def _limit_lfe(engine, sig):
    x = np.zeros((len(sig), 8), dtype=np.float32)
    x[:, 3] = sig
    out = np.concatenate([engine._apply_true_peak_limiter(x[i:i + 480]) for i in range(0, len(x), 480)])[:, 3]
    la = int(0.0015 * SR)
    return out[la:], sig[:-la]


def test_sub_limiter_does_not_distort_sustained_overload(engine):
    t = np.arange(SR * 2) / SR
    y, _ = _limit_lfe(engine, (1.8 * np.sin(2 * np.pi * 40 * t)).astype(np.float32))
    seg = y[SR // 2:SR // 2 + SR] * np.hanning(SR)
    sp = np.abs(np.fft.rfft(seg))
    fr = np.fft.rfftfreq(SR, 1 / SR)
    peak = lambda f: sp[(fr > f - 2) & (fr < f + 2)].max()
    harmonics = np.sqrt(sum(peak(40 * k) ** 2 for k in range(2, 8)))
    assert 20 * np.log10(harmonics / peak(40)) < -80
    assert np.max(np.abs(y)) <= 0.9801


def test_sub_limiter_does_not_pump(engine):
    t = np.arange(SR * 3) / SR
    sig = 0.45 * np.sin(2 * np.pi * 50 * t)
    for k in range(12):                                   # kicks overloading the sub, 4 per second
        s = int(k * 0.25 * SR)
        sig[s:s + 2400] += 1.2 * np.sin(2 * np.pi * 55 * t[s:s + 2400])
    y, x = _limit_lfe(engine, sig.astype(np.float32))
    for k in range(2, 11):                                # normal bass between kicks is untouched
        seg = slice(int(k * 0.25 * SR) + int(0.10 * SR), int(k * 0.25 * SR) + int(0.20 * SR))
        assert np.max(np.abs(y[seg])) == pytest.approx(np.max(np.abs(x[seg])), rel=0.01)
