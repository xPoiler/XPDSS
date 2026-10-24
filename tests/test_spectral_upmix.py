import time

import numpy as np
import pytest

import audio_analyzer as aa
from spectral_upmix import (AMBIENT_L, AMBIENT_R, DIRECT_C, DIRECT_L, DIRECT_R,
                            SpectralUpmixer)

SR = 48000
BLOCK = 480
T = np.arange(SR * 3) / SR
TAIL = slice(2 * SR, 3 * SR)
rng = np.random.default_rng(0)


def tone(freq, amp=0.2):
    return amp * np.sin(2 * np.pi * freq * T)


def run(x, block=BLOCK, **kw):
    up = SpectralUpmixer(SR)
    return np.concatenate([up.process(x[i:i + block], **kw) for i in range(0, len(x), block)])


def db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def band_db(sig, freq):
    sp = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    fr = np.fft.rfftfreq(len(sig), 1 / SR)
    return 20 * np.log10(sp[(fr > freq * 0.97) & (fr < freq * 1.03)].max() + 1e-12)


def pan(sig, p):
    """Constant-power stereo pan, p in [-1 (left), +1 (right)]."""
    th = (p + 1) * np.pi / 4
    return np.stack([np.cos(th) * sig, np.sin(th) * sig], axis=1)


# ── Core processor ──────────────────────────────────────────────────────

def test_reconstruction_is_exact_and_latency_is_fixed():
    x = rng.standard_normal((SR * 2, 2)) * 0.1
    out = run(x, center_spread=1.0)
    lat = SpectralUpmixer(SR).latency
    assert lat == 1024
    for ch, (d, a) in enumerate(((DIRECT_L, AMBIENT_L), (DIRECT_R, AMBIENT_R))):
        np.testing.assert_allclose(out[lat:, d] + out[lat:, a], x[:-lat, ch], atol=1e-6)  # float32 processing


@pytest.mark.parametrize("block", [441, 512, 1024, 37])
def test_output_does_not_depend_on_block_size(block):
    x = rng.standard_normal((SR, 2)) * 0.1
    np.testing.assert_allclose(run(x, block=block), run(x), atol=1e-6)


def test_mono_goes_to_center_only():
    out = run(pan(tone(440), 0.0))
    assert db(out[TAIL, DIRECT_C]) > db(out[TAIL, DIRECT_L]) + 60
    assert db(out[TAIL, AMBIENT_L]) < -100


def test_hard_panned_source_stays_in_its_speaker():
    out = run(pan(tone(440), -1.0))
    assert db(out[TAIL, DIRECT_L]) > max(db(out[TAIL, DIRECT_C]), db(out[TAIL, DIRECT_R]), db(out[TAIL, AMBIENT_L])) + 60


def test_half_left_source_plays_between_left_and_center():
    out = run(pan(tone(440), -0.5))
    assert db(out[TAIL, DIRECT_L]) == pytest.approx(db(out[TAIL, DIRECT_C]), abs=0.5)
    assert db(out[TAIL, DIRECT_R]) < db(out[TAIL, DIRECT_L]) - 60


def test_simultaneous_sources_each_land_in_their_own_speaker():
    x = pan(tone(500), 0.0) + pan(tone(1500), -1.0) + pan(tone(3000), 1.0)
    out = run(x)
    for f, ch in ((500, DIRECT_C), (1500, DIRECT_L), (3000, DIRECT_R)):
        own = band_db(out[TAIL, ch], f)
        others = [band_db(out[TAIL, c], f) for c in (DIRECT_L, DIRECT_R, DIRECT_C, AMBIENT_L, AMBIENT_R) if c != ch]
        assert own > max(others) + 40, (f, own, others)


def test_diffuse_sound_goes_to_ambience_not_center():
    x = rng.standard_normal((SR * 3, 2)) * 0.1
    out = run(x)
    assert db(out[TAIL, AMBIENT_L]) > db(x[TAIL, 0]) - 2.0
    assert db(out[TAIL, DIRECT_C]) < db(x[TAIL, 0]) - 12.0


def test_vocal_over_ambience_is_separated():
    amb = rng.standard_normal((SR * 3, 2)) * 0.05
    out = run(pan(tone(700), 0.0) + amb)
    assert band_db(out[TAIL, DIRECT_C], 700) > band_db(out[TAIL, AMBIENT_L], 700) + 30


@pytest.mark.parametrize("freq", [40, 60, 80, 100, 120])
def test_bass_is_left_in_the_fronts(freq):
    x = pan(tone(freq), 0.0)
    out = run(x)
    lat = 1024
    ref = db(x[TAIL, 0])
    err = out[lat:, DIRECT_L] - x[:-lat, 0]
    assert db(err[2 * SR - lat:]) < ref - 40       # only window spill-over remains
    assert db(out[TAIL, DIRECT_C]) < ref - 40


def test_width_and_spread_controls():
    wide = run(pan(tone(440), -0.5), width=2.0)
    assert db(wide[TAIL, DIRECT_L]) > db(wide[TAIL, DIRECT_C]) + 40      # pushed hard left
    phantom = run(pan(tone(440), 0.0), center_spread=1.0)
    assert db(phantom[TAIL, DIRECT_C]) < -100                          # no center
    no_center = run(pan(tone(440), 0.0), center=False)
    assert db(no_center[TAIL, DIRECT_C]) < -100


def test_stationary_input_gives_stable_output():
    """No 'musical noise': the level of each stream must not wobble."""
    amb = rng.standard_normal((SR * 3, 2)) * 0.05
    out = run(pan(tone(700), 0.0) + pan(tone(1900, 0.1), -0.6) + amb)
    win = SR // 20
    for ch in (DIRECT_L, DIRECT_C, AMBIENT_L):
        seg = out[SR:3 * SR, ch].reshape(-1, win)
        levels = 10 * np.log10(np.mean(seg ** 2, axis=1))
        assert np.std(levels) < 0.6, ch


def test_transient_has_no_pre_echo():
    x = np.zeros((SR, 2))
    x[SR // 2] = 0.8                                        # centered click
    out = run(x)
    hit = SR // 2 + 1024
    total = np.sum(out[:, DIRECT_C] ** 2)
    assert np.sum(out[:hit - 64, DIRECT_C] ** 2) < 1e-3 * total


# ── Engine integration ──────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path):
    app = aa.AudioBypass("a", "b", simulation=True, config_file=str(tmp_path / "c.json"))
    app.current_channels = 8
    app.upmix_enabled = True
    app.center_extraction = True
    app.spectral_upmix = True
    return app


def render(app, src, out_ch=8, flip=None):
    outs = []
    for i in range(0, src.shape[0], BLOCK):
        if flip:
            flip(app, i)
        outs.append(app._apply_upmix(src[i:i + BLOCK].copy(), out_ch=out_ch).copy())
    return np.concatenate(outs)


def src8(stereo):
    s = np.zeros((stereo.shape[0], 8), dtype=np.float32)
    s[:, :2] = stereo
    return s


def test_engine_routes_direct_to_front_and_ambience_to_surrounds(engine):
    amb = rng.standard_normal((SR * 3, 2)) * 0.05
    out = render(engine, src8(pan(tone(700), 0.0) + amb))
    tail = slice(2 * SR, 3 * SR)
    assert band_db(out[tail, 2], 700) > band_db(out[tail, 6], 700) + 30     # vocal: center, not sides
    for ch in (4, 5, 6, 7):
        assert db(out[tail, ch]) > -40                                     # ambience reaches all surrounds
    assert db(out[tail, 6]) > db(out[tail, 4])                              # sides louder than backs


def test_engine_5_1_uses_channels_4_5_for_surrounds(engine):
    engine.current_channels = 6
    out = render(engine, src8(rng.standard_normal((SR * 2, 2)) * 0.1), out_ch=6)
    assert db(out[SR:, 4]) > -30 and db(out[SR:, 5]) > -30


def test_lfe_level_matches_classic_upmix(engine, tmp_path):
    x = src8(pan(tone(50, 0.3), 0.0) + pan(tone(70, 0.2), -0.7))
    spec = render(engine, x)
    classic = aa.AudioBypass("a", "b", simulation=True, config_file=str(tmp_path / "k.json"))
    classic.current_channels = 8
    classic.upmix_enabled = True
    ref = render(classic, x)
    lat = 1024
    np.testing.assert_allclose(spec[SR + lat:, 3], ref[SR:-lat, 3], atol=1e-4)


def test_passthrough_is_delayed_to_match_when_spectral_is_selected(engine):
    engine.is_multichannel_detected = True
    x = rng.standard_normal((SR, 8)).astype(np.float32) * 0.1
    out = render(engine, x)
    la = int(0.0015 * SR) + 1024                                          # limiter + spectral latency
    np.testing.assert_allclose(out[la:, 0], x[:-la, 0], atol=1e-6)


def _max_step_ratio(out, b, ch):
    normal = max(np.abs(np.diff(out[b - 3000:b - 2, ch])).max(),
                 np.abs(np.diff(out[b + 3000:b + 8000, ch])).max())
    return np.abs(np.diff(out[b - 2:b + 2400, ch])).max() / normal


def test_auto_switch_between_passthrough_and_spectral_is_seamless(engine):
    x = src8(pan(tone(300), -0.3) + pan(tone(900), 0.4))
    x[:, 2] = tone(1200)[:x.shape[0]]
    flip = lambda app, i: setattr(app, 'is_multichannel_detected', i >= SR * 3 // 2)
    out = render(engine, x, flip=flip)
    b = SR * 3 // 2 + int(0.0015 * SR)
    for ch in (0, 1, 2):
        assert _max_step_ratio(out, b, ch) <= 1.05


def test_ab_toggle_between_engines_is_crossfaded(engine):
    x = src8(pan(tone(300), -0.3) + pan(tone(900), 0.4) + rng.standard_normal((SR * 3, 2)) * 0.02)
    flip = lambda app, i: setattr(app, 'spectral_upmix', i < SR * 3 // 2)
    out = render(engine, x, flip=flip)
    b = SR * 3 // 2 + int(0.0015 * SR)
    for ch in (0, 1, 2):
        assert _max_step_ratio(out, b, ch) <= 1.1


def test_engine_cpu_budget(engine):
    x = src8(rng.standard_normal((BLOCK, 2)) * 0.1)
    for _ in range(20):
        engine._apply_upmix(x.copy(), out_ch=8)
    t0 = time.perf_counter()
    for _ in range(200):
        engine._apply_upmix(x.copy(), out_ch=8)
    per_block_ms = (time.perf_counter() - t0) / 200 * 1000
    assert per_block_ms < 3.0, per_block_ms   # 10 ms budget per block


# ── Front/back steering (cinema / gaming) ───────────────────────────────

from scipy.signal import hilbert  # noqa: E402

from spectral_upmix import (BACK_L, BACK_R, SIDE_L, SIDE_R, STEER_CINEMA,  # noqa: E402
                            STEER_GAMING, STEER_OFF)


def phased(sig, phase_deg, left=1.0, right=1.0):
    """Same source in L and R with a given inter-channel phase difference."""
    analytic = hilbert(sig)
    r = np.real(analytic * np.exp(1j * np.radians(phase_deg)))
    return np.stack([left * sig, right * r], axis=1)


def rear_db(out):
    return db(out[TAIL, SIDE_L] + out[TAIL, SIDE_R] + out[TAIL, BACK_L] + out[TAIL, BACK_R])


def test_steering_off_keeps_anti_phase_content_out_of_the_center():
    out = run(phased(tone(700), 180), steering=STEER_OFF)
    assert db(out[TAIL, DIRECT_C]) < db(out[TAIL, DIRECT_L]) - 40
    assert rear_db(out) < -100


def test_cinema_sends_anti_phase_content_to_the_surrounds():
    for x in (phased(tone(700), 180), phased(tone(700), 180, right=0.4)):
        out = run(x, steering=STEER_CINEMA)
        front = max(db(out[TAIL, DIRECT_L]), db(out[TAIL, DIRECT_C]), db(out[TAIL, DIRECT_R]))
        assert rear_db(out) > front + 30
    out = run(phased(tone(700), 180, right=0.4), steering=STEER_CINEMA)
    assert db(out[TAIL, SIDE_L]) > db(out[TAIL, SIDE_R]) + 3     # rear source keeps its left/right position


def test_cinema_leaves_in_phase_content_in_front():
    out = run(pan(tone(700), -0.4), steering=STEER_CINEMA)
    assert rear_db(out) < -100
    ref = run(pan(tone(700), -0.4), steering=STEER_OFF)
    np.testing.assert_allclose(out[:, DIRECT_L], ref[:, DIRECT_L], atol=1e-6)


def _rear_share(out):
    p = lambda ch: np.mean(out[TAIL, ch] ** 2)
    rear = sum(p(c) for c in (SIDE_L, SIDE_R, BACK_L, BACK_R))
    return rear / (rear + sum(p(c) for c in (DIRECT_L, DIRECT_R, DIRECT_C)))


def test_gaming_places_partly_rear_on_sides_and_fully_rear_on_backs():
    full = run(phased(tone(700), 180), steering=STEER_GAMING, split_sides_backs=True)
    part = run(phased(tone(700), 125), steering=STEER_GAMING, split_sides_backs=True)
    assert db(full[TAIL, BACK_L]) > db(full[TAIL, SIDE_L]) + 20
    assert db(part[TAIL, SIDE_L]) > db(part[TAIL, BACK_L]) + 6


def test_gaming_is_more_decisive_than_cinema():
    for deg, gaming_more_rear in ((100, False), (150, True)):
        g = _rear_share(run(phased(tone(700), deg), steering=STEER_GAMING, split_sides_backs=True))
        c = _rear_share(run(phased(tone(700), deg), steering=STEER_CINEMA, split_sides_backs=True))
        assert (g > c) == gaming_more_rear, (deg, g, c)


def test_engine_cinema_with_spectral_on_5_1(engine):
    engine.cinema_steering = True
    engine.current_channels = 6
    out = render(engine, src8(phased(tone(700), 180)), out_ch=6)
    tail = slice(2 * SR, 3 * SR)
    assert db(out[tail, 4]) > db(out[tail, 2]) + 30
    assert db(out[tail, 5]) > -40


# ── Musical noise ("low-bitrate" twinkle) ───────────────────────────────

from scipy.ndimage import uniform_filter  # noqa: E402


def _spike_energy_db(sig, ref):
    """Energy in isolated time-frequency spikes (>7 dB over their 7x7
    neighbourhood) in 1.5-16 kHz, relative to the input energy there."""
    n, h = 1024, 256
    w = np.hanning(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f > 1500) & (f < 16000)
    spec = lambda s: np.array([np.abs(np.fft.rfft(s[i:i + n] * w))[band] ** 2 for i in range(0, len(s) - n, h)])
    P = spec(sig)
    spikes = P > 5 * uniform_filter(P, size=7, mode='nearest')
    return 10 * np.log10(P[spikes].sum() / spec(ref).sum() + 1e-15)


@pytest.mark.parametrize("coherent_share, limit_db", [(0.0, -60.0), (1.0, -30.0)])
def test_noise_like_content_does_not_twinkle_in_the_center(coherent_share, limit_db):
    """Reverb / cymbal-like noise must not be scattered bin by bin into the
    center (that is what sounds like an old low-bitrate stream)."""
    r = np.random.default_rng(1)
    a, b, c = r.standard_normal((3, SR * 3)) * 0.05
    x = np.stack([a + coherent_share * c, b + coherent_share * c], axis=1) * (0.7 if coherent_share else 1.0)
    out = run(x)[SR:]
    assert _spike_energy_db(out[:, DIRECT_C], x[SR:, 0]) < limit_db


def test_close_notes_do_not_intermodulate():
    tt = np.arange(SR * 4) / SR
    a = sum(0.15 / k * np.sin(2 * np.pi * 1000 * k * tt) for k in (1, 2, 3))
    b = sum(0.15 / k * np.sin(2 * np.pi * 1020 * k * tt) for k in (1, 2, 3))
    out = run(np.stack([0.7071 * a + np.cos(np.pi / 20) * b, 0.7071 * a + np.sin(np.pi / 20) * b], axis=1))
    seg = out[2 * SR:, DIRECT_C] * np.hanning(2 * SR)
    sp = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(2 * SR, 1 / SR)
    near = np.zeros_like(fr, bool)
    for f0 in (1000, 2000, 3000, 1020, 2040, 3060):
        near |= np.abs(fr - f0) < 3
    inband = (fr > 200) & (fr < 8000)
    assert 10 * np.log10(sp[inband & ~near].sum() / sp[near].sum()) < -40
