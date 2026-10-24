import numpy as np

from adaptive_renderer import AdaptiveFeatureEngine

SR = 48000
BLOCK = 480


def feed(engine, signal, stride=1):
    """Feed a (frames, ch) signal block by block, as the analysis thread does."""
    snap = None
    for i in range(0, signal.shape[0] - BLOCK + 1, BLOCK * stride):
        snap = engine.analyze_block(signal[i:i + BLOCK], dt=BLOCK * stride / SR)
    return snap


def stereo(left, right=None):
    return np.column_stack([left, left if right is None else right]).astype(np.float32)


def tone(freq, seconds, amp=0.3, phase=0.0):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * freq * t + phase)


def test_dialogue_like_mono_scores_more_central_than_wide_stereo():
    voice = tone(300, 2, 0.4) + tone(2300, 2, 0.25)
    wide = stereo(tone(300, 2, 0.45) + tone(9500, 2, 0.12), tone(300, 2, 0.45, np.pi / 2) - tone(9500, 2, 0.12))
    mono_snap = feed(AdaptiveFeatureEngine(SR, tau=0.2), stereo(voice))
    wide_snap = feed(AdaptiveFeatureEngine(SR, tau=0.2), wide)
    assert mono_snap.stereo_correlation > 0.9
    assert mono_snap.dialogue_score > wide_snap.dialogue_score
    assert wide_snap.immersion_score > mono_snap.immersion_score


def test_loudness_range_separates_compressed_from_dynamic_content():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR * 12) * 0.1
    compressed = stereo(noise)
    # dynamic: alternate loud / quiet (-16 dB) passages every 2 s
    env = np.repeat(np.tile([1.0, 0.16], 3), SR * 2)
    dynamic = stereo(noise * env)
    c = feed(AdaptiveFeatureEngine(SR), compressed, stride=5)
    d = feed(AdaptiveFeatureEngine(SR), dynamic, stride=5)
    assert c.dynamic_range_db < 2.0 and c.compression_score > 0.9
    assert d.dynamic_range_db > 12.0 and d.compression_score < 0.1


def test_tonal_balance_scores():
    muddy = feed(AdaptiveFeatureEngine(SR, tau=0.2), stereo(tone(350, 2) + tone(3000, 2, 0.05)))
    bright = feed(AdaptiveFeatureEngine(SR, tau=0.2), stereo(tone(3000, 2) + tone(10000, 2, 0.2)))
    assert muddy.mud_score > 0.8 > bright.mud_score
    assert bright.air_score > muddy.air_score


def test_silence_is_ignored_and_holds_measurements():
    engine = AdaptiveFeatureEngine(SR, tau=0.2)
    assert feed(engine, stereo(np.zeros(SR))) is None
    snap = feed(engine, stereo(tone(350, 1)))
    held = feed(engine, stereo(np.zeros(SR)))
    assert held == snap
    assert engine.last_block_active is False


def test_measurements_change_slowly():
    engine = AdaptiveFeatureEngine(SR, tau=3.0)
    feed(engine, stereo(tone(3000, 4)))
    before = engine.latest_snapshot().mud_score
    after = feed(engine, stereo(tone(350, 0.5))).mud_score   # half a second of mud
    assert after - before < 0.5 * (1.0 - before)
