"""One behavioural test per DSP effect, each run in isolation with a signal
chosen to show whether the effect does its job."""
import types

import numpy as np
import pytest

import audio_analyzer as aa

SR = 48000
BLOCK = 480


@pytest.fixture
def engine(tmp_path):
    app = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=str(tmp_path / "config.json"))
    app.current_channels = 8
    app.is_multichannel_detected = False
    return app


def tone(freq, seconds=1.0, amp=0.3, phase=0.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def level_db(sig, freq, bw=0.1):
    """Level of the spectral peak around `freq` (dB, Hann-windowed)."""
    sig = np.asarray(sig, dtype=np.float64)
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) * 4.0 / len(sig)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / SR)
    mask = (freqs >= freq * (1 - bw)) & (freqs <= freq * (1 + bw))
    return 20 * np.log10(np.max(spec[mask]) + 1e-12)


def rms_db(sig):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(sig, dtype=np.float64) ** 2)) + 1e-12)


def blockwise(fn, block, size=BLOCK):
    """Run an in-place effect over `block` in chunks of `size` frames."""
    out = block.copy()
    for i in range(0, out.shape[0], size):
        fn(out[i:i + size])
    return out


def stereo(sig_l, sig_r=None):
    return np.stack([sig_l, sig_l if sig_r is None else sig_r], axis=1).copy()


def upmix(engine, src, size=BLOCK):
    """Run the full upmix over `src` (frames, 8) block by block."""
    out = []
    for i in range(0, src.shape[0], size):
        out.append(engine._apply_upmix(src[i:i + size].copy(), out_ch=8).copy())
    return np.concatenate(out)


def src8(left, right):
    s = np.zeros((len(left), 8), dtype=np.float32)
    s[:, 0], s[:, 1] = left, right
    return s


# ── Block-size invariance for stateful (time-invariant) effects ─────────
# A stateful filter must give the same output whatever the block size; a
# per-block reset or discontinuity shows up as a difference here.

INVARIANT_EFFECTS = {
    "bass_exciter": lambda e, b: e._apply_harmonic_exciter(b),
    "freq_dep_width": lambda e, b: e._apply_freq_dep_width(b),
    "hp_externalization": lambda e, b: e._apply_hp_externalization(b),
    "air_exciter": lambda e, b: e._apply_air_exciter(b),
    "virtual_room": lambda e, b: e._apply_virtual_room(b),
    "transaural": lambda e, b: e._apply_transaural_widener(b, b.shape[0]),
}


@pytest.mark.parametrize("name", sorted(INVARIANT_EFFECTS))
def test_stateful_effects_are_block_size_invariant(tmp_path, name):
    x = (np.random.default_rng(7).standard_normal((SR // 2, 2)) * 0.2).astype(np.float32)
    outs = []
    for size in (480, 441, 1024):
        e = aa.AudioBypass("a", "b", simulation=True, config_file=str(tmp_path / f"{name}{size}.json"))
        e.harmonic_exciter = e.air_exciter = e.virtual_room = True
        outs.append(blockwise(lambda b: INVARIANT_EFFECTS[name](e, b), x, size))
    np.testing.assert_allclose(outs[1], outs[0], atol=1e-5)
    np.testing.assert_allclose(outs[2], outs[0], atol=1e-5)


# ── Input restoration / dynamics ────────────────────────────────────────

def test_declip_reconstructs_clipped_peaks(engine):
    true = tone(200, 0.5, amp=1.3)
    clipped = np.clip(true, -1.0, 1.0)
    out = blockwise(engine._apply_declip, stereo(clipped))[:, 0]
    mask = np.abs(clipped) >= 0.99
    err_before = np.sqrt(np.mean((clipped[mask] - true[mask]) ** 2))
    err_after = np.sqrt(np.mean((out[mask] - true[mask]) ** 2))
    assert np.max(np.abs(out)) > 1.15          # peaks rebuilt above the clip level
    assert err_after < 0.35 * err_before       # and much closer to the original
    np.testing.assert_array_equal(out[~mask], clipped[~mask])  # clean samples untouched


def test_declip_leaves_clean_audio_alone(engine):
    x = stereo(tone(200, 0.2, amp=0.9))
    np.testing.assert_array_equal(blockwise(engine._apply_declip, x), x)


def test_loudness_norm_moves_quiet_and_loud_toward_target(engine):
    quiet = stereo(tone(1000, 8.0, amp=0.0141))  # ~ -40 dBFS RMS
    out = blockwise(engine._apply_loudness_norm, quiet)
    assert rms_db(out[-SR:]) == pytest.approx(-40 + 12, abs=1.0)  # capped at +12 dB

    engine._agc_gain_smooth = 1.0
    loud = stereo(tone(1000, 4.0, amp=0.45))   # ~ -10 dBFS RMS
    out = blockwise(engine._apply_loudness_norm, loud)
    assert rms_db(out[-SR:]) == pytest.approx(-22.0, abs=1.0)  # -12 dB max cut


def test_loudness_norm_ignores_silence(engine):
    x = stereo(tone(1000, 2.0, amp=0.0005))    # below the -55 dB gate
    np.testing.assert_array_equal(blockwise(engine._apply_loudness_norm, x), x)


def test_micro_dynamics_lifts_quiet_passages_only(engine):
    engine.micro_dyn_amount = 4.0
    loud = tone(500, 6.0, amp=0.3)
    quiet = tone(500, 1.5, amp=0.3 * 10 ** (-8 / 20))  # 8 dB below the running reference
    out = blockwise(engine._apply_micro_dynamics, stereo(np.concatenate([loud, quiet])))[:, 0]
    loud_gain = rms_db(out[4 * SR:6 * SR]) - rms_db(loud[4 * SR:])
    quiet_gain = rms_db(out[-SR // 2:]) - rms_db(quiet[-SR // 2:])
    assert abs(loud_gain) < 0.5
    assert 2.5 < quiet_gain <= 4.05


def test_transient_restore_boosts_onsets_not_sustain(engine):
    engine.transient_amount = 6.0
    x = np.zeros(SR, dtype=np.float32)
    x[SR // 2:] = tone(800, 0.5, amp=0.3)       # silence, then a sudden onset
    out = blockwise(engine._apply_transient_restore, stereo(x))[:, 0]
    onset = slice(SR // 2, SR // 2 + 240)        # first 5 ms after the onset
    sustain = slice(SR - 4800, SR)
    assert np.max(np.abs(out[onset])) > 1.25 * np.max(np.abs(x[onset]))
    assert abs(rms_db(out[sustain]) - rms_db(x[sustain])) < 0.5


def test_clarity_cuts_mud_only_when_excessive(engine):
    engine.clarity_dsp = True
    engine.clarity_amount = 6.0
    muddy = tone(350, 2.0, amp=0.3) + tone(3000, 2.0, amp=0.05)
    out = blockwise(engine._apply_clarity_dsp, stereo(muddy))[:, 0]
    tail = slice(SR, 2 * SR)
    assert level_db(out[tail], 350) < level_db(muddy[tail], 350) - 1.5
    assert abs(level_db(out[tail], 3000) - level_db(muddy[tail], 3000)) < 0.5

    engine._clarity_gain_smooth = 0.0
    bright = tone(3000, 1.0, amp=0.3)
    out = blockwise(engine._apply_clarity_dsp, stereo(bright))[:, 0]
    np.testing.assert_allclose(out, bright, atol=1e-6)


# ── Tone / enhancement ──────────────────────────────────────────────────

def test_dynamic_eq_is_flat_at_normal_level(engine):
    engine._program_level_db = -15.0
    x = stereo(tone(50, 1.0) + tone(1000, 1.0) + tone(12000, 1.0))
    out = blockwise(lambda b: engine._apply_loudness_comp(b, engine._program_level_db), x)
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_bass_exciter_adds_harmonics_of_low_bass(engine):
    engine.harmonic_exciter = True
    x = stereo(tone(50, 1.0, amp=0.4))
    out = blockwise(engine._apply_harmonic_exciter, x)[:, 0]
    tail = slice(SR // 2, SR)
    assert level_db(out[tail], 100, 0.03) > level_db(x[tail, 0], 100, 0.03) + 30
    assert level_db(out[tail], 50, 0.03) == pytest.approx(level_db(x[tail, 0], 50, 0.03), abs=1.0)
    assert level_db(out[tail], 2000) < -70      # nothing leaks into the mids


def test_air_exciter_only_touches_the_air_band(engine):
    engine.air_exciter = True
    engine.air_amount = 0.3
    x = stereo(tone(1000, 1.0) + tone(14000, 1.0, amp=0.1))
    out = blockwise(engine._apply_air_exciter, x)[:, 0]
    tail = slice(SR // 2, SR)
    assert abs(level_db(out[tail], 1000) - level_db(x[tail, 0], 1000)) < 0.2
    assert level_db(out[tail], 14000) > level_db(x[tail, 0], 14000) + 0.8


def test_freq_dep_width_monos_bass_and_widens_highs(engine):
    lo = tone(60, 1.0)
    hi = tone(10000, 1.0, amp=0.1)
    mid = tone(1000, 1.0)
    # anti-phase bass + anti-phase highs + in-phase mids
    x = stereo(lo + hi + mid, -lo - hi + mid)
    out = blockwise(engine._apply_freq_dep_width, x)
    tail = slice(SR // 2, SR)
    side = 0.5 * (out[tail, 0] - out[tail, 1])
    side_in = 0.5 * (x[tail, 0] - x[tail, 1])
    assert level_db(side, 60) < level_db(side_in, 60) - 15            # bass side removed
    assert 1.0 < level_db(side, 10000) - level_db(side_in, 10000) < 3.0  # highs wider
    assert level_db(out[tail, 0], 1000) == pytest.approx(level_db(x[tail, 0], 1000), abs=0.3)

    engine._fdw_zi = {}  # drop the tail of the previous (anti-phase) signal
    mono = stereo(lo + hi + mid)
    np.testing.assert_allclose(blockwise(engine._apply_freq_dep_width, mono), mono, atol=1e-5)


def test_spectral_tilt_keeps_all_channels_aligned(engine):
    engine.spectral_tilt = True
    sig = (np.random.default_rng(1).standard_normal(SR) * 0.1).astype(np.float32)
    x = np.repeat(sig[:, None], 8, axis=1)
    out = x.copy()
    for i in range(0, SR, BLOCK):
        engine._apply_spectral_tilt(out[i:i + BLOCK], BLOCK, 8)
    for ch in range(1, 8):
        np.testing.assert_allclose(out[:, ch], out[:, 0], atol=1e-6)
    assert abs(rms_db(out[SR // 2:, 0]) - rms_db(sig[SR // 2:])) < 3.0


# ── Spatial ─────────────────────────────────────────────────────────────

def test_hp_externalization_crossfeeds_low_frequencies(engine):
    x = stereo(tone(200, 1.0), np.zeros(SR, dtype=np.float32))  # left-only
    out = blockwise(engine._apply_hp_externalization, x)
    tail = slice(SR // 2, SR)
    leak = level_db(out[tail, 1], 200) - level_db(out[tail, 0], 200)
    assert -14 < leak < -6                      # some bass reaches the other ear


def test_surround_decorrelation_is_allpass_and_decorrelates(engine):
    imp = np.zeros(8192, dtype=np.float32)
    imp[0] = 1.0
    out_l, out_r = engine._apply_surround_decorrelation(imp, imp.copy())
    mag = np.abs(np.fft.rfft(out_l))
    assert np.max(np.abs(20 * np.log10(mag))) < 0.1  # flat magnitude response

    engine2 = aa.AudioBypass("a", "b", simulation=True, config_file=engine.config_file + "2")
    n = (np.random.default_rng(3).standard_normal(SR) * 0.2).astype(np.float32)
    l, r = engine2._apply_surround_decorrelation(n, n.copy())
    assert abs(np.corrcoef(l, r)[0, 1]) < 0.5        # identical input, decorrelated output
    assert abs(rms_db(l) - rms_db(n)) < 0.5


def test_virtual_room_adds_reflections_at_the_tap_times(engine):
    engine.virtual_room = True
    engine.virtual_room_mix = 1.0
    x = np.zeros((4800, 2), dtype=np.float32)
    x[0, 0] = 1.0                                   # impulse on the left
    out = blockwise(engine._apply_virtual_room, x)
    for ms, ch in ((4.3, 0), (7.1, 1), (11.4, 0), (17.8, 1)):
        n = int(ms * SR / 1000)
        assert np.max(np.abs(out[n:n + 40, ch])) > 0.01, (ms, ch)
    assert np.max(np.abs(out[1:150, :])) < 0.01      # nothing before the first reflection


def test_transaural_cancels_crosstalk_path(engine):
    engine.transaural_delay = 7
    engine.transaural_gain = 0.5
    engine.transaural_feedback = 0.0
    x = np.zeros((4800, 2), dtype=np.float32)
    x[100, 0] = 1.0
    out = blockwise(lambda b: engine._apply_transaural_widener(b, b.shape[0]), x)
    # left impulse produces an inverted, delayed, low-passed copy on the right
    right = out[:, 1]
    assert np.argmin(right) >= 100 + 7
    assert right.min() < -0.02
    assert out[100, 0] == pytest.approx(1.0)


# ── Upmix routing ───────────────────────────────────────────────────────

def configure_upmix(engine, **flags):
    engine.upmix_enabled = True
    engine.center_extraction = True
    for k, v in flags.items():
        setattr(engine, k, v)


def _new_engine(tmp_path, name):
    app = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=str(tmp_path / f"{name}.json"))
    app.current_channels = 8
    app.is_multichannel_detected = False
    configure_upmix(app)
    return app


def test_classic_upmix_placement(engine, tmp_path):
    """Common -> center, hard-panned -> its own front only, uncorrelated -> surrounds."""
    configure_upmix(engine)
    s = tone(1000, 0.5)
    tail = slice(SR // 2, SR)
    mono = upmix(engine, src8(np.tile(s, 2), np.tile(s, 2)))
    assert rms_db(mono[tail, 2]) > rms_db(mono[tail, 0]) + 20   # mono -> center
    for ch in (4, 5, 6, 7):
        assert rms_db(mono[tail, ch]) < -60                     # mono: no surrounds

    engine = _new_engine(tmp_path, "left")
    left = upmix(engine, src8(np.tile(s, 2), np.zeros(SR, dtype=np.float32)))
    assert rms_db(left[tail, 0]) == pytest.approx(rms_db(np.tile(s, 2)[tail]) + 20 * np.log10(engine.CLASSIC_MATCH_GAIN), abs=0.5)
    for ch in (1, 2, 4, 5, 6, 7):                               # never an inverted copy elsewhere
        assert rms_db(left[tail, ch]) < -60

    engine = _new_engine(tmp_path, "wide")
    rng = np.random.default_rng(7)
    a, b = (rng.standard_normal((2, SR)) * 0.1).astype(np.float32)
    wide = upmix(engine, src8(a, b))[SR // 4:]
    assert rms_db(wide[:, 6]) > rms_db(wide[:, 4])              # 7.1: sides (6/7) > backs (4/5)
    assert rms_db(wide[:, 6]) > rms_db(wide[:, 0]) - 6          # ambience reaches the surrounds
    assert rms_db(wide[:, 2]) < rms_db(wide[:, 0]) - 20         # no ambience in the center
    sl, sr = wide[:, 6], wide[:, 7]
    assert abs(np.dot(sl, sr) / np.sqrt(np.dot(sl, sl) * np.dot(sr, sr))) < 0.3


def test_classic_opposite_phase_needs_steering(engine, tmp_path):
    """Without steering opposite-phase content stays in L/R (as mixed); with
    cinema steering it becomes a same-polarity rear source."""
    configure_upmix(engine)
    n = (np.random.default_rng(5).standard_normal(SR) * 0.1).astype(np.float32)
    tail = slice(SR // 2, None)
    off = upmix(engine, src8(n, -n))
    for ch in (2, 4, 5, 6, 7):
        assert rms_db(off[tail, ch]) < -60
    engine = _new_engine(tmp_path, "steer")
    engine.cinema_steering = True
    on = upmix(engine, src8(n, -n))
    assert rms_db(on[tail, 0]) < rms_db(on[tail, 6]) - 15   # what remains is the bass (< 200 Hz stays in L/R)
    sl, sr = on[tail, 6], on[tail, 7]
    assert np.dot(sl, sr) / np.sqrt(np.dot(sl, sl) * np.dot(sr, sr)) > 0.9


def test_center_extraction_off_folds_center_into_fronts(engine):
    configure_upmix(engine, center_extraction=False)
    s = tone(1000, 0.5)
    out = upmix(engine, src8(s, s))
    tail = slice(SR // 4, SR // 2)
    assert rms_db(out[tail, 2]) < -60
    match_db = 20 * np.log10(engine.CLASSIC_MATCH_GAIN)  # classic/spectral level match
    assert rms_db(out[tail, 0]) == pytest.approx(rms_db(s[tail]) + match_db, abs=0.5)


def test_stage_equalizer_lowers_center_by_3db(engine):
    configure_upmix(engine)
    s = tone(1000, 0.5)
    ref = upmix(engine, src8(s, s))
    engine.stage_equalizer = True
    out = upmix(engine, src8(s, s))
    tail = slice(SR // 4, SR // 2)
    assert rms_db(out[tail, 2]) - rms_db(ref[tail, 2]) == pytest.approx(-3.0, abs=0.1)


def test_lfe_redirect_and_mute(engine):
    configure_upmix(engine)
    s = tone(60, 0.5) + tone(2000, 0.5)
    out = upmix(engine, src8(s, s))
    tail = slice(SR // 4, SR // 2)
    # full band (L+R) * 0.316: no filtering in the app, the room correction does it
    for f in (60, 2000):
        assert level_db(out[tail, 3], f) == pytest.approx(level_db(s[tail], f) + 20 * np.log10(0.632), abs=0.2)
    engine.lfe_muted = True
    la = int(0.0015 * SR)  # limiter look-ahead still holds the previous run's tail
    assert not upmix(engine, src8(s, s))[la:, 3].any()

def test_widening_mode_feeds_delayed_mid_to_surrounds(engine):
    configure_upmix(engine, center_spread=1.0, gui_panning_y=-1.0, widening_mode=True)
    x = np.zeros(4800, dtype=np.float32)
    x[960] = 0.5                                    # mono click
    out = upmix(engine, src8(x, x))
    la = int(0.0015 * SR)                           # limiter look-ahead
    delay = int(engine.widening_delay_ms * SR / 1000)
    haas = int(0.010 * SR)
    arrival = 960 + delay + haas + la
    side = out[:, 6]
    assert np.max(np.abs(side[:arrival - 2])) < 1e-6           # nothing before the delayed arrival
    assert np.sum(side[arrival - 2:] ** 2) > 1e-4              # ...then the diffused echo
    # diffused: not a single coherent click (spread over several ms)
    assert np.max(side ** 2) < 0.5 * np.sum(side ** 2)


def test_cinema_steering_routes_by_correlation(engine):
    configure_upmix(engine, cinema_steering=True)
    s = tone(1000, 2.0)
    mono = upmix(engine, src8(s, s))
    configure_upmix(engine := aa.AudioBypass("a", "b", simulation=True, config_file=engine.config_file + "x"), cinema_steering=True)
    engine.current_channels = 8
    anti = upmix(engine, src8(s, -s))
    tail = slice(SR, 2 * SR)
    assert rms_db(mono[tail, 4]) < -60                          # in-phase stays front
    assert rms_db(anti[tail, 4]) > rms_db(anti[tail, 0]) + 10   # anti-phase goes back


def test_gaming_panning_sends_anti_phase_to_rear(engine):
    configure_upmix(engine, cinema_steering=True, gaming_panning=True)
    s = tone(1000, 2.0)
    anti = upmix(engine, src8(s, -s))
    tail = slice(SR, 2 * SR)
    assert rms_db(anti[tail, 4]) > -20
    assert rms_db(anti[tail, 0]) < rms_db(anti[tail, 4]) - 10


# ── Center-channel effects ──────────────────────────────────────────────

def test_dialogue_focus_boosts_speech_band_with_a_ramp(engine):
    x = np.zeros((SR, 3), dtype=np.float32)
    x[:, 2] = tone(2500, 1.0) + tone(150, 1.0)
    out = x.copy()
    for i in range(0, SR, BLOCK):
        engine._apply_dialogue_focus(out[i:i + BLOCK], True)
    tail = slice(SR // 2, SR)
    assert level_db(out[tail, 2], 2500) - level_db(x[tail, 2], 2500) == pytest.approx(3.5, abs=0.6)
    assert abs(level_db(out[tail, 2], 150) - level_db(x[tail, 2], 150)) < 0.3
    first = np.abs(out[:BLOCK, 2] - x[:BLOCK, 2])
    assert first[:48].max() < 0.02                 # gain ramps in, no step

    off = out[-BLOCK:].copy()
    engine._apply_dialogue_focus(off, False)       # ramps back out
    engine._apply_dialogue_focus(off, False)


def test_vocal_lock_boosts_harmonic_center_more_than_noise(engine):
    t = np.arange(SR) / SR
    voice = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 12)).astype(np.float32) * 0.1
    noise = (np.random.default_rng(5).standard_normal(SR) * 0.1).astype(np.float32)

    def gain(sig):
        e = aa.AudioBypass("a", "b", simulation=True, config_file=engine.config_file + str(id(sig)))
        blk = np.zeros((SR, 3), dtype=np.float32)
        blk[:, 2] = sig
        out = blk.copy()
        for i in range(0, SR, BLOCK):
            e._apply_vocal_lock(out[i:i + BLOCK], BLOCK)
        return rms_db(out[-SR // 4:, 2]) - rms_db(sig[-SR // 4:])

    g_voice, g_noise = gain(voice), gain(noise)
    assert 1.0 < g_voice <= 3.05
    assert g_noise < g_voice - 1.0


# ── Subharmonic synth ───────────────────────────────────────────────────

def test_sub_synth_generates_half_frequency(engine):
    engine.sub_synth = True
    engine.sub_gain = 1.0
    s = tone(60, 3.0, amp=0.3)
    out = np.zeros((len(s), 8), dtype=np.float32)
    out[:, 0] = out[:, 1] = s
    for i in range(0, len(s), BLOCK):
        engine._apply_subharmonic_synth(out[i:i + BLOCK], BLOCK, 8)
    tail = slice(2 * SR, 3 * SR)
    assert level_db(out[tail, 0], 30, 0.05) > level_db(s[tail], 30, 0.05) + 30
    assert level_db(out[tail, 3], 30, 0.05) > -40   # also routed to the LFE


def test_sub_synth_ducks_for_voice(engine):
    engine.sub_synth = True
    engine.sub_gain = 1.0
    s = tone(60, 3.0, amp=0.1) + tone(200, 3.0, amp=0.4)   # voice-band energy dominates
    out = np.zeros((len(s), 8), dtype=np.float32)
    out[:, 0] = out[:, 1] = s
    for i in range(0, len(s), BLOCK):
        engine._apply_subharmonic_synth(out[i:i + BLOCK], BLOCK, 8)
    assert np.max(np.abs(out[2 * SR:, 3])) < 1e-3


# ── Head tracking ───────────────────────────────────────────────────────

def tracked(engine, **pose):
    engine.head_tracking = True
    engine.head_tracker = types.SimpleNamespace(pos_x=0.0, pos_y=0.0, pos_z=1.0, rot_yaw=0.0)
    for k, v in pose.items():
        setattr(engine.head_tracker, k, v)


def run_ht(engine, x, out_ch=8):
    out = x.copy()
    for i in range(0, x.shape[0], BLOCK):
        engine._apply_head_tracking_panning(out[i:i + BLOCK], BLOCK, out_ch)
    return out


def test_head_tracking_centered_listener_is_neutral(engine):
    tracked(engine)
    x = np.repeat(tone(1000, 0.5)[:, None], 8, axis=1)
    out = run_ht(engine, x)
    tail = slice(SR // 4, SR // 2)
    for ch in (0, 1, 2):
        assert abs(rms_db(out[tail, ch]) - rms_db(x[tail, ch])) < 0.5


def test_head_tracking_compensates_level_and_delay_when_off_center(engine):
    tracked(engine, pos_x=-0.25)                    # listener moved toward the left speaker
    x = np.repeat(tone(1000, 1.0)[:, None], 8, axis=1)
    imp = np.zeros((SR, 8), dtype=np.float32)
    imp[SR // 2, :] = 1.0
    out = run_ht(engine, x)
    tail = slice(SR // 2, SR)
    # nearer speaker attenuated by exactly the physical 1/r ratio
    fov = 2 * np.tan(np.radians(engine.camera_fov / 2))
    user_x = -0.25 * engine.speaker_distance_cm * fov
    half = engine.speaker_width_cm / 2
    d_near = np.hypot(user_x + half, engine.speaker_distance_cm)
    d_ref = np.hypot(half, engine.speaker_distance_cm)
    expected = 20 * np.log10(d_near / d_ref)
    assert rms_db(out[tail, 0]) - rms_db(x[tail, 0]) == pytest.approx(expected, abs=0.2)
    assert abs(rms_db(out[tail, 1]) - rms_db(x[tail, 1])) < 0.2              # farther one untouched
    out_i = run_ht(engine, imp + 0)
    assert np.argmax(np.abs(out_i[:, 0])) > np.argmax(np.abs(out_i[:, 1]))  # nearer speaker delayed


def test_head_shadow_eq_rolls_off_highs_when_turned_away(engine):
    tracked(engine, rot_yaw=0.9)
    engine.head_shadow_eq = True
    x = np.repeat((tone(200, 1.0) + tone(8000, 1.0, amp=0.1))[:, None], 3, axis=1)
    out = run_ht(engine, x, out_ch=3)
    tail = slice(SR // 2, SR)
    assert level_db(out[tail, 0], 8000) < level_db(x[tail, 0], 8000) - 2.0
    assert abs(level_db(out[tail, 0], 200) - level_db(x[tail, 0], 200)) < 1.0


# ── Surround passthrough ────────────────────────────────────────────────

def surround_block(seed, center=0.0, surr=0.0, lfe=0.0, extras_noise=0.0, matrix=False, frames=BLOCK):
    rng = np.random.default_rng(seed)
    x = np.zeros((frames, 8), dtype=np.float32)
    L, R = rng.standard_normal(frames) * 0.1, rng.standard_normal(frames) * 0.1
    x[:, 0], x[:, 1] = L, R
    x[:, 2] = rng.standard_normal(frames) * center
    x[:, 3] = rng.standard_normal(frames) * lfe
    for ch in (4, 5, 6, 7):
        x[:, ch] = rng.standard_normal(frames) * surr
    if matrix:
        x[:, 2] = 0.5 * (L + R) + rng.standard_normal(frames) * 0.012
        for ch in (4, 6):
            x[:, ch] = 0.5 * (L - R) + rng.standard_normal(frames) * 0.012
        for ch in (5, 7):
            x[:, ch] = -0.5 * (L - R) + rng.standard_normal(frames) * 0.012
    if extras_noise:
        for ch in (2, 4, 5, 6, 7):
            x[:, ch] += rng.standard_normal(frames) * extras_noise
    return x


@pytest.mark.parametrize("kwargs", [
    dict(center=0.1, surr=0.016),          # movie: dialog + ambience
    dict(surr=0.005),                      # surround ambience 26 dB below the fronts
    dict(surr=0.00056),                    # 45 dB below the fronts
    dict(center=0.1),                      # dialog only
])
def test_discrete_surround_is_detected(kwargs):
    hits = [aa.AudioBypass._block_has_discrete_surround(surround_block(s, **kwargs)) for s in range(20)]
    assert all(hits)


@pytest.mark.parametrize("kwargs", [
    dict(),                                # stereo on a multichannel cable
    dict(extras_noise=10 ** (-130 / 20)),  # 24-bit dither
    dict(extras_noise=10 ** (-80 / 20)),   # noise floor
    dict(lfe=0.1),                         # LFE only
    dict(matrix=True),                     # third-party matrix upmix of stereo
])
def test_stereo_content_never_triggers_passthrough(kwargs):
    hits = [aa.AudioBypass._block_has_discrete_surround(surround_block(s, **kwargs)) for s in range(20)]
    assert not any(hits)


@pytest.mark.parametrize("out_ch, expected", [
    (8, {0: {0: 0.0}, 2: {2: 0.0}, 4: {4: 0.0}, 6: {6: 0.0}}),
    (6, {0: {0: 0.0}, 2: {2: 0.0}, 4: {4: 0.0, 6: -3.0}, 5: {5: 0.0, 7: -3.0}}),
    (2, {0: {0: 0.0, 2: -3.0, 3: -10.0, 4: -3.0, 6: -3.0}, 1: {1: 0.0, 2: -3.0, 3: -10.0, 5: -3.0, 7: -3.0}}),
])
def test_passthrough_downmixes_channels_the_output_lacks(engine, out_ch, expected):
    freqs = [500, 700, 1100, 40, 1300, 1700, 1900, 2300]
    src = np.zeros((SR, 8), dtype=np.float32)
    for ch, f in enumerate(freqs):
        src[:, ch] = tone(f, 1.0, amp=0.2)
    engine.is_multichannel_detected = True
    engine.current_channels = out_ch
    out = np.concatenate([engine._apply_upmix(src[i:i + BLOCK].copy(), out_ch=out_ch).copy()
                          for i in range(0, SR, BLOCK)])
    tail = slice(SR // 2, SR)
    for o, sources in expected.items():
        for c, gain_db in sources.items():
            got = level_db(out[tail, o], freqs[c], 0.03) - level_db(src[tail, c], freqs[c], 0.03)
            assert got == pytest.approx(gain_db, abs=0.3), (o, c)


def test_mode_switch_is_crossfaded(engine):
    configure_upmix(engine)
    src = np.zeros((SR, 8), dtype=np.float32)
    src[:, 0], src[:, 1] = tone(300, 1.0), tone(300, 1.0, phase=1.0)
    src[:, 2] = tone(800, 1.0)
    src[:, 4] = src[:, 5] = tone(1200, 1.0, amp=0.2)
    outs = []
    for i in range(0, SR, BLOCK):
        engine.is_multichannel_detected = i >= SR // 2
        outs.append(engine._apply_upmix(src[i:i + BLOCK].copy(), out_ch=8).copy())
    out = np.concatenate(outs)
    b = SR // 2 + int(0.0015 * SR)
    for ch in (0, 2, 4):
        normal = max(np.abs(np.diff(out[b - 2400:b - 2, ch])).max(),
                     np.abs(np.diff(out[b + 2600:b + 7000, ch])).max())
        assert np.abs(np.diff(out[b - 2:b + 2400, ch])).max() <= normal * 1.05


def test_headphone_tracking_counter_rotation_applies_on_stereo_output(engine):
    engine.headphone_tracking = True
    engine._dynamic_pan_offset = 0.5
    engine.head_tracker = types.SimpleNamespace(pos_x=0.0, pos_y=0.0, pos_z=1.0, rot_yaw=-0.785)
    s = tone(1000, 0.5)
    src = np.zeros((len(s), 8), dtype=np.float32)
    src[:, 0] = src[:, 1] = s
    out = np.concatenate([engine._apply_upmix(src[i:i + BLOCK].copy(), out_ch=2).copy()
                          for i in range(0, len(s), BLOCK)])
    tail = slice(SR // 4, SR // 2)
    assert rms_db(out[tail, 0]) < rms_db(out[tail, 1]) - 3.0



def test_sub_level_is_consistent_between_passthrough_and_upmix(tmp_path):
    """A 5.1 mix in passthrough must send the same bass to the sub as its
    stereo downmix does through the upmix (user calibration)."""
    t = np.arange(SR * 2) / SR
    mix51 = np.zeros((SR * 2, 8), dtype=np.float32)
    for ch, ph in ((0, 0.0), (1, 0.3), (2, 0.1), (4, 0.6), (5, 0.9)):
        mix51[:, ch] = 0.15 * np.sin(2 * np.pi * 55 * t + ph)
    stereo_dm = np.zeros_like(mix51)
    stereo_dm[:, 0] = mix51[:, 0] + 0.707 * mix51[:, 2] + 0.707 * mix51[:, 4]
    stereo_dm[:, 1] = mix51[:, 1] + 0.707 * mix51[:, 2] + 0.707 * mix51[:, 5]

    levels = {}
    for name, src, detected, spectral in (("upmix", stereo_dm, False, False),
                                          ("spectral", stereo_dm, False, True),
                                          ("passthrough", mix51, True, False)):
        e = aa.AudioBypass("a", "b", simulation=True, config_file=str(tmp_path / f"{name}.json"))
        e.current_channels = 8
        e.upmix_enabled = True
        e.spectral_upmix = spectral
        e.is_multichannel_detected = detected
        out = upmix(e, src)
        levels[name] = rms_db(out[SR:, 3])
    assert levels["passthrough"] == pytest.approx(levels["upmix"], abs=1.0)
    assert levels["spectral"] == pytest.approx(levels["upmix"], abs=1.0)


def test_virtual_room_reflections_only_on_surrounds(engine):
    """Reflections come from the side/back walls: fronts, center and LFE stay dry."""
    engine.virtual_room = True
    engine.virtual_room_mix = 1.0
    rng = np.random.default_rng(3)
    x = np.zeros((BLOCK * 20, 8), dtype=np.float32)
    x[:, :3] = rng.standard_normal((x.shape[0], 3)) * 0.1
    out = []
    for i in range(0, x.shape[0], BLOCK):
        b = x[i:i + BLOCK].copy()
        engine._apply_virtual_room(b)
        out.append(b)
    out = np.concatenate(out)
    assert np.array_equal(out[:, :4], x[:, :4])
    tail = slice(BLOCK * 4, None)
    for ch in (4, 5, 6, 7):
        assert rms_db(out[tail, ch]) > -50  # input -20 dB, reflections ~-14..-22 dB under it
    # sides (early reflections) louder than backs (later ones)
    assert rms_db(out[tail, 6]) > rms_db(out[tail, 4])


@pytest.mark.headroom
@pytest.mark.parametrize("bypass", [False, True])
def test_output_headroom_lowers_every_channel_equally(tmp_path, bypass):
    app = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=str(tmp_path / "c.json"))
    app.current_channels = 8
    app.is_multichannel_detected = False
    assert app.output_headroom_db == 4.0
    configure_upmix(app)
    app.bypass_all = bypass
    s = tone(1000, 0.5)
    src = src8(s, s)
    src[:, 2:] = tone(500, 0.5)[:, None] * np.ones(6, dtype=np.float32)   # so bypass has every channel
    ref_app = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=str(tmp_path / "r.json"))
    ref_app.current_channels = 8
    ref_app.is_multichannel_detected = False
    configure_upmix(ref_app)
    ref_app.bypass_all = bypass
    ref_app.output_headroom_db = 0.0
    out, ref = upmix(app, src.copy()), upmix(ref_app, src.copy())
    tail = slice(SR // 4, SR // 2)
    for ch in range(8):
        if rms_db(ref[tail, ch]) > -60:
            assert rms_db(out[tail, ch]) - rms_db(ref[tail, ch]) == pytest.approx(-4.0, abs=0.05)


def test_output_headroom_is_saved(tmp_path):
    cfg = str(tmp_path / "c.json")
    app = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=cfg)
    app.output_headroom_db = 6.0
    app.save_config()
    again = aa.AudioBypass("CABLE Output", "AVR", simulation=True, config_file=cfg)
    again.load_config()
    assert again.output_headroom_db == 6.0


@pytest.mark.parametrize("effect", ["transient_restore", "micro_dynamics"])
def test_dynamics_are_linked_so_sounds_do_not_move(engine, effect):
    """One gain for every channel: a drum hit in the center must not turn the
    fronts up or down relative to it (per-channel gains moved the image)."""
    rng = np.random.default_rng(11)
    n = SR * 2
    x = np.zeros((n, 6), dtype=np.float32)
    x[:, 0] = x[:, 1] = rng.standard_normal(n) * 0.05                 # steady bed in the fronts
    hits = np.zeros(n, dtype=np.float32)
    for k in range(0, n - 2400, SR // 4):
        hits[k:k + 2400] = rng.standard_normal(2400) * np.exp(-np.arange(2400) / 300.0) * 0.5
    x[:, 2] = hits + rng.standard_normal(n) * 0.005                   # drum hits in the center
    x[:, 4] = x[:, 5] = rng.standard_normal(n) * 0.01                 # quiet ambience
    engine._program_level_db = -20.0
    if effect == "transient_restore":
        fx = lambda b: engine._apply_transient_restore(b)
    else:
        engine.micro_dyn_amount = 5.0
        fx = lambda b: engine._apply_micro_dynamics(b)
    out = blockwise(fx, x, 480)
    w = 480
    def gains(ch):
        o = out[SR // 2:, ch].reshape(-1, w); i = x[SR // 2:, ch].reshape(-1, w)
        return 10 * np.log10((o ** 2).sum(1) / (i ** 2).sum(1))
    g_front, g_center, g_amb = gains(0), gains(2), gains(4)
    assert np.max(np.abs(g_center - g_front)) < 0.3
    assert np.max(np.abs(g_amb - g_front)) < 0.3


def test_program_level_reads_the_input_not_the_upmixed_fronts(tmp_path):
    """Upmixing moves the center out of L/R; the level estimate (Dynamic EQ,
    transient gate...) must not drop because of it."""
    s = (np.random.default_rng(4).standard_normal(SR * 2) * 0.1).astype(np.float32)
    levels = []
    for upmix_on in (False, True):
        e = _new_engine(tmp_path, f"pl{upmix_on}")
        e.upmix_enabled = upmix_on
        upmix(e, src8(s, s))
        levels.append(e._program_level_db)
    assert abs(levels[0] - levels[1]) < 0.1


def test_switching_tilt_does_not_click(engine):
    """The tilt FIR delays by 127 samples: switching it must crossfade, not jump."""
    t = np.arange(SR) / SR
    x = np.zeros((SR, 6), dtype=np.float32)
    x[:, 0] = x[:, 1] = 0.3 * np.sin(2 * np.pi * 220 * t)
    out = x.copy()
    for bi, i in enumerate(range(0, SR, 480)):
        engine.spectral_tilt = (bi // 25) % 2 == 1          # toggles every 250 ms
        b = out[i:i + 480]
        engine._apply_post_effects(b, 6, src_block=np.pad(x[i:i + 480], ((0, 0), (0, 2))))
    # a sine through a smooth system has a bounded second difference
    d2 = np.abs(np.diff(out[:, 0], 2))
    assert d2.max() < 3 * (0.3 * (2 * np.pi * 220 / SR) ** 2) * 2.0 + 1e-3


@pytest.mark.parametrize("att, rel, n_ch", [(0.9, 0.999, 1), (0.0, 0.995, 1), (0.0, np.array([0.99, 0.9995]), 2)])
def test_envelope_follower_matches_reference(att, rel, n_ch):
    """The scalar fast path must be bit-identical to the original numpy loop."""
    def reference(abs_block, att_coeff, rel_coeff, zi, dec=8):
        n_samp, n_ch = abs_block.shape
        pad = (dec - n_samp % dec) % dec
        padded = np.concatenate([abs_block, np.zeros((pad, n_ch), dtype=abs_block.dtype)]) if pad else abs_block
        n_dec = padded.shape[0] // dec
        dec_in = padded.reshape(n_dec, dec, n_ch).max(axis=1)
        att_d, rel_d = att_coeff ** dec, rel_coeff ** dec
        dec_env = np.empty_like(dec_in)
        prev = zi.copy()
        for i in range(n_dec):
            s = dec_in[i:i + 1]
            cf = np.where(s > prev, att_d, rel_d)
            prev = cf * prev + (1.0 - cf) * s
            dec_env[i] = prev[0]
        x_dec = np.arange(n_dec) * dec + dec // 2
        env = np.empty((n_samp, n_ch), dtype=abs_block.dtype)
        for ch in range(n_ch):
            env[:, ch] = np.interp(np.arange(n_samp), x_dec, dec_env[:, ch])
        return env, prev
    rng = np.random.default_rng(9)
    zi_a = np.full((1, n_ch), 0.3)
    zi_b = zi_a.copy()
    for size in (480, 441, 7):
        x = np.abs(rng.standard_normal((size, n_ch)))
        ea, zi_a = aa.AudioBypass._asym_env_decimated(x, att, rel, zi_a)
        eb, zi_b = reference(x, att, rel, zi_b)
        assert np.array_equal(ea, eb) and np.array_equal(zi_a, zi_b)
