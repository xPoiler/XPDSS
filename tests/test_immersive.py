import numpy as np
import pytest

import audio_analyzer as aa
from room_expansion import RoomExpander
from spectral_upmix import DIRECT_L, SIDE_L, SpectralUpmixer

SR = 48000


def test_room_expander_is_block_size_invariant_decaying_and_decorrelated():
    imp = np.zeros(SR * 2); imp[0] = 1.0
    outs = []
    for size in (480, 1024, 137):
        r = RoomExpander(SR)
        outs.append(np.concatenate([r.process(imp[i:i + size], 4) for i in range(0, len(imp), size)]))
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])
    out = outs[0]
    onset = np.argmax(np.abs(out[:, 0]) > 1e-6) / SR
    assert 0.03 < onset < 0.08                       # pre-delay: after the direct sound
    e = (out ** 2).sum(1)
    assert e[int(1.5 * SR):].sum() < 1e-3 * e.sum()  # ~1 s tail
    c = np.corrcoef(out[SR // 10:].T)
    assert np.max(np.abs(c[np.triu_indices(4, 1)])) < 0.35


def _engine(tmp_path, **flags):
    app = aa.AudioBypass("a", "b", simulation=True, config_file=str(tmp_path / f"c{len(flags)}{id(flags)}.json"))
    app.current_channels = 6
    app.upmix_enabled = True
    app.center_extraction = True
    for k, v in flags.items():
        setattr(app, k, v)
    return app


@pytest.mark.parametrize("spectral", [False, True])
def test_immersive_adds_surround_energy_in_both_engines(tmp_path, spectral):
    # a dry, centred source: without IMMERSIVE the surrounds stay (almost)
    # silent; with it they carry the room expansion around the listener
    rng = np.random.default_rng(1)
    mono = (rng.standard_normal(SR * 3) * 0.05).astype(np.float32)
    x = np.zeros((len(mono), 8), np.float32); x[:, 0] = x[:, 1] = mono
    res = []
    for imm in (False, True):
        e = _engine(tmp_path, spectral_upmix=spectral, immersive=imm)
        out = np.concatenate([e._apply_upmix(x[i:i + 480].copy(), out_ch=6).copy() for i in range(0, len(x), 480)])
        res.append((out[SR:, 4:6] ** 2).sum())
    assert res[1] > res[0] * 100 and res[1] > 1e-3


def test_switching_immersive_is_smooth(tmp_path):
    """Every IMMERSIVE part crossfades: no step in the surrounds on a sine."""
    t = np.arange(SR * 2) / SR
    sig = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    x = np.zeros((len(sig), 8), np.float32); x[:, 0] = sig; x[:, 1] = -0.3 * sig
    e = _engine(tmp_path, spectral_upmix=False)
    outs = []
    for bi, i in enumerate(range(0, len(x), 480)):
        e.immersive = (bi // 20) % 2 == 1
        outs.append(e._apply_upmix(x[i:i + 480].copy(), out_ch=6).copy())
    out = np.concatenate(outs)[SR // 4:]
    step = np.abs(np.diff(out[:, 4]))
    assert step.max() < 8 * np.median(step[step > 0]) + 0.02


def test_hard_pans_go_to_the_side_surround():
    rng = np.random.default_rng(2)
    s = rng.standard_normal(SR * 3) * 0.1
    up = SpectralUpmixer(SR)
    x = np.stack([s, 0 * s], axis=1)
    out = np.concatenate([up.process(x[i:i + 480], hard_pan_sides=True) for i in range(0, len(x), 480)])[SR:]
    p = (out ** 2).sum(0)
    # what remains in front is the bass below the 150-250 Hz handover (by design)
    assert p[SIDE_L] > 50 * p[DIRECT_L]
    up = SpectralUpmixer(SR)
    centre = np.stack([s, s], axis=1)
    out = np.concatenate([up.process(centre[i:i + 480], hard_pan_sides=True) for i in range(0, len(x), 480)])[SR:]
    assert (out[:, SIDE_L] ** 2).sum() < 1e-6 * (out ** 2).sum()
