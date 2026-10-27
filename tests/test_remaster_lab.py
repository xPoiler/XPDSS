"""Integrity checks for the isolated prototypes, not production feature tests."""
import time
import numpy as np
from tools.remaster_lab import Guidance, GuidedRemaster


def test_off_is_exact_identity_after_active_processing():
    p = GuidedRemaster(channels=6)
    x = np.random.default_rng(1).normal(0,.1,(480,6)).astype(np.float32)
    p.process(x, target=(-3,1.8,-2.5))
    assert np.array_equal(p.process(x, enabled=False), x)
    p.amount = 0
    assert np.array_equal(p.process(x), x)


def test_no_queued_delay_and_no_future_dependence():
    x = np.random.default_rng(2).normal(0,.1,(960,2)).astype(np.float32)
    a, b = GuidedRemaster(), GuidedRemaster()
    xa, xb = x.copy(), x.copy()
    xb[480:] *= -1
    ya = a.process(xa[:480], target=(-3,1.8,-2.5))
    yb = b.process(xb[:480], target=(-3,1.8,-2.5))
    assert np.array_equal(ya, yb)
    p = GuidedRemaster()
    pulse = np.zeros((480,2), np.float32); pulse[137,0] = .5
    out = p.process(pulse, target=(-3,1.8,-2.5))
    assert np.argmax(abs(out[:,0])) == 137
    assert np.all(out[:137] == 0)


def test_surround_channels_do_not_leak_and_lfe_is_exact():
    for ch in range(8):
        p = GuidedRemaster(channels=8)
        x = np.zeros((480,8), np.float32)
        x[:,ch] = np.random.default_rng(ch).normal(0,.1,480)
        y = p.process(x, target=(-3,1.8,-2.5))
        assert np.all(y[:, np.arange(8) != ch] == 0)
        assert np.array_equal(y[:,3], x[:,3])


def test_linked_stereo_preserves_proportional_channels():
    p = GuidedRemaster()
    x = np.random.default_rng(3).normal(0,.05,9600).astype(np.float32)
    x = np.stack([x,x*.4],axis=1)
    y = np.concatenate([p.process(z, target=(-3,1.8,-2.5)) for z in np.split(x,20)])
    np.testing.assert_allclose(y[:,1], y[:,0]*.4, atol=2e-8)


def test_silence_finite_and_stale_guidance_returns_to_neutral():
    p = GuidedRemaster()
    g = Guidance(); g.target = (-3,1.8,-2.5); g.last_update = time.monotonic()
    for _ in range(50):
        y = p.process(np.zeros((480,2),np.float32), g)
        assert np.all(y == 0)
    assert max(abs(p.gain)) > 1
    g.last_update = 0
    for _ in range(300):
        y = p.process(np.zeros((480,2),np.float32), g)
    assert max(abs(p.gain)) < 1e-7
    assert np.isfinite(y).all()


def test_neural_stem_evidence_changes_presence_target():
    g_voice, g_other = Guidance(), Guidance()
    x = np.random.default_rng(4).normal(0,.03,(128,2)).astype(np.float32)
    for _ in range(32):
        voice = np.zeros((4,128,2),np.float32); voice[3] = x
        other = np.zeros_like(voice); other[2] = x
        g_voice.observe(voice); g_other.observe(other)
    assert g_voice.target[1] > g_other.target[1]
    assert g_voice.vocal_share > .99
