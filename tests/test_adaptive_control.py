import math

import numpy as np

from adaptive_control import ADAPTIVE_PARAMS, AdaptiveController, compute_targets

SLIDER_RANGES = {  # GUI slider ranges (gui_components)
    "clarity_amount": (1.0, 6.0),
    "transient_amount": (1.0, 12.0),
    "micro_dyn_amount": (1.0, 12.0),
    "air_amount": (0.0, 1.0),
    "virtual_room_mix": (0.0, 1.0),
}

BASE = dict(mud_score=0.3, air_score=0.5, compression_score=0.5, dialogue_score=0.3, immersion_score=0.5)


def targets(**overrides):
    return compute_targets({**BASE, **overrides})


def test_only_amount_sliders_are_driven():
    assert set(compute_targets(BASE)) == set(ADAPTIVE_PARAMS)
    forbidden = {"width_factor", "center_spread", "side_gain_db", "transaural_gain", "dialogue_focus"}
    assert not forbidden & set(compute_targets(BASE))


def test_targets_stay_inside_slider_ranges():
    rng = np.random.default_rng(0)
    for _ in range(500):
        scores = {k: float(v) for k, v in zip(BASE, rng.uniform(0, 1, len(BASE)))}
        for mode in ("speakers", "headphones"):
            for key, value in compute_targets(scores, mode).items():
                lo, hi = SLIDER_RANGES[key]
                assert lo <= value <= hi, (key, value)


def test_mapping_directions():
    assert targets(mud_score=0.9)["clarity_amount"] > targets(mud_score=0.1)["clarity_amount"]
    assert targets(compression_score=0.9)["transient_amount"] > targets(compression_score=0.1)["transient_amount"]
    assert targets(compression_score=0.9)["micro_dyn_amount"] < targets(compression_score=0.1)["micro_dyn_amount"]
    assert targets(air_score=0.1)["air_amount"] > targets(air_score=0.9)["air_amount"]
    assert targets(immersion_score=0.1)["virtual_room_mix"] > targets(immersion_score=0.9)["virtual_room_mix"]
    assert targets(dialogue_score=0.9)["virtual_room_mix"] < targets(dialogue_score=0.1)["virtual_room_mix"]
    assert compute_targets(BASE, "headphones")["virtual_room_mix"] < compute_targets(BASE, "speakers")["virtual_room_mix"]


def test_glides_from_start_values_with_time_constant():
    ctrl = AdaptiveController(tau=2.0)
    start = {k: 4.0 if "amount" in k and k != "air_amount" else 0.25 for k in ADAPTIVE_PARAMS}
    ctrl.reset(start)
    target = compute_targets(BASE)["clarity_amount"]
    values = ctrl.update(BASE, dt=2.0)
    expected = 4.0 + (1.0 - math.exp(-1.0)) * (target - 4.0)
    assert math.isclose(values["clarity_amount"], expected, rel_tol=1e-9)


def test_silence_holds_values():
    ctrl = AdaptiveController()
    ctrl.reset({k: 1.0 for k in ADAPTIVE_PARAMS})
    before = ctrl.values()
    assert ctrl.update(None, dt=10.0) == before


def test_glide_back_reports_arrival():
    ctrl = AdaptiveController()
    ctrl.reset({k: 2.0 for k in ADAPTIVE_PARAMS})
    manual = {k: 4.0 for k in ADAPTIVE_PARAMS}
    assert not ctrl.glide_to(manual, dt=0.05)
    for _ in range(100):
        done = ctrl.glide_to(manual, dt=0.05)
    assert done
