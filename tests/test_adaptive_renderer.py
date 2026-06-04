import numpy as np

from adaptive_renderer import AdaptiveFeatureEngine


def make_timebase(sample_rate=48000, seconds=0.25):
    return np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate


def test_dialogue_like_mono_signal_scores_more_central_than_wide_stereo():
    sr = 48000
    t = make_timebase(sr)
    voice = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.25 * np.sin(2 * np.pi * 2300 * t)
    mono_block = np.column_stack([voice, voice]).astype(np.float32)

    width_l = 0.45 * np.sin(2 * np.pi * 300 * t)
    width_r = 0.45 * np.sin(2 * np.pi * 300 * t + np.pi / 2)
    air = 0.12 * np.sin(2 * np.pi * 9500 * t)
    wide_block = np.column_stack([width_l + air, width_r - air]).astype(np.float32)

    engine = AdaptiveFeatureEngine(sample_rate=sr, smoothing=0.0)
    mono_snapshot = engine.analyze_block(mono_block)
    wide_snapshot = engine.analyze_block(wide_block)

    assert mono_snapshot.stereo_correlation > 0.90
    assert mono_snapshot.side_ratio < 0.20
    assert mono_snapshot.dialogue_score > wide_snapshot.dialogue_score


def test_wide_stereo_signal_scores_more_immersive_than_mono_signal():
    sr = 48000
    t = make_timebase(sr)

    mono = 0.45 * np.sin(2 * np.pi * 1000 * t)
    mono_block = np.column_stack([mono, mono]).astype(np.float32)

    left = 0.4 * np.sin(2 * np.pi * 700 * t)
    right = 0.4 * np.sin(2 * np.pi * 1700 * t + np.pi / 3)
    wide_block = np.column_stack([left, right]).astype(np.float32)

    engine = AdaptiveFeatureEngine(sample_rate=sr, smoothing=0.0)
    mono_snapshot = engine.analyze_block(mono_block)
    wide_snapshot = engine.analyze_block(wide_block)

    assert wide_snapshot.side_ratio > mono_snapshot.side_ratio
    assert wide_snapshot.immersion_score > mono_snapshot.immersion_score


def test_clipped_signal_scores_more_compressed_than_dynamic_signal():
    sr = 48000
    t = make_timebase(sr)

    dynamic = 0.15 * np.sin(2 * np.pi * 220 * t)
    dynamic += 0.75 * np.exp(-((t - 0.05) / 0.002) ** 2)
    dynamic_block = np.column_stack([dynamic, dynamic]).astype(np.float32)

    clipped = np.clip(1.8 * np.sin(2 * np.pi * 220 * t), -0.35, 0.35)
    clipped_block = np.column_stack([clipped, clipped]).astype(np.float32)

    engine = AdaptiveFeatureEngine(sample_rate=sr, smoothing=0.0)
    dynamic_snapshot = engine.analyze_block(dynamic_block)
    clipped_snapshot = engine.analyze_block(clipped_block)

    assert clipped_snapshot.compression_score > dynamic_snapshot.compression_score
