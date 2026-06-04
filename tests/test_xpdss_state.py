from types import SimpleNamespace

from xpdss_state import (
    ADAPTIVE_DEFAULTS,
    PROFILE_DEFAULTS,
    SECONDARY_OUTPUT_DEFAULTS,
    apply_adaptive_state_to_object,
    apply_profile_to_object,
    apply_secondary_output_state_to_object,
    build_adaptive_state_from_object,
    build_profile_from_object,
    build_secondary_output_state_from_object,
)


def make_state(**overrides):
    state = SimpleNamespace(**PROFILE_DEFAULTS)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_build_profile_from_object_keeps_recent_dsp_fields():
    state = make_state(
        clarity_dsp=True,
        clarity_amount=5.5,
        virtual_room=True,
        virtual_room_mix=0.42,
        air_exciter=True,
        air_amount=0.33,
        dialogue_focus=True,
    )

    profile = build_profile_from_object(state)

    assert profile["clarity_dsp"] is True
    assert profile["clarity_amount"] == 5.5
    assert profile["virtual_room"] is True
    assert profile["virtual_room_mix"] == 0.42
    assert profile["air_exciter"] is True
    assert profile["air_amount"] == 0.33
    assert profile["dialogue_focus"] is True


def test_apply_profile_to_object_restores_defaults_for_missing_fields():
    state = make_state(clarity_amount=6.0, air_amount=0.8, dialogue_focus=True)

    apply_profile_to_object(state, {"clarity_dsp": True, "clarity_amount": 2.25})

    assert state.clarity_dsp is True
    assert state.clarity_amount == 2.25
    assert state.air_amount == PROFILE_DEFAULTS["air_amount"]
    assert state.dialogue_focus == PROFILE_DEFAULTS["dialogue_focus"]


def test_adaptive_state_round_trip_keeps_mode_profile_and_compare_flags():
    state = make_state(
        adaptive_mode="full",
        adaptive_profile="cinema",
        adaptive_compare_bypass=True,
        adaptive_monitor_enabled=False,
        adaptive_observe_only=False,
    )

    adaptive_state = build_adaptive_state_from_object(state)

    assert adaptive_state == {
        "adaptive_monitor_enabled": False,
        "adaptive_observe_only": False,
        "adaptive_mode": "full",
        "adaptive_profile": "cinema",
        "adaptive_compare_bypass": True,
    }

    target = make_state()
    apply_adaptive_state_to_object(target, {"adaptive_mode": "assist", "adaptive_profile": "night"})

    assert target.adaptive_mode == "assist"
    assert target.adaptive_profile == "night"
    assert target.adaptive_compare_bypass == ADAPTIVE_DEFAULTS["adaptive_compare_bypass"]
    assert target.adaptive_monitor_enabled == ADAPTIVE_DEFAULTS["adaptive_monitor_enabled"]


def test_secondary_output_state_round_trip_keeps_device_and_filter_controls():
    state = make_state(
        secondary_output_enabled=True,
        secondary_output_device="Bass Shaker Out",
        secondary_output_gain_db=4.5,
        secondary_output_lowpass_hz=70.0,
        secondary_output_highpass_hz=18.0,
        secondary_output_delay_ms=12.5,
        secondary_output_source_mode="lfe_or_bass_mix",
        secondary_output_mono_sum=False,
    )

    secondary_state = build_secondary_output_state_from_object(state)

    assert secondary_state == {
        "secondary_output_enabled": True,
        "secondary_output_device": "Bass Shaker Out",
        "secondary_output_gain_db": 4.5,
        "secondary_output_lowpass_hz": 70.0,
        "secondary_output_highpass_hz": 18.0,
        "secondary_output_delay_ms": 12.5,
        "secondary_output_source_mode": "lfe_or_bass_mix",
        "secondary_output_mono_sum": False,
    }


def test_apply_secondary_output_state_restores_defaults_for_missing_fields():
    state = make_state(
        secondary_output_enabled=True,
        secondary_output_device="Old Device",
        secondary_output_gain_db=9.0,
        secondary_output_lowpass_hz=120.0,
    )

    apply_secondary_output_state_to_object(
        state,
        {
            "secondary_output_enabled": True,
            "secondary_output_device": "New Device",
            "secondary_output_gain_db": 2.25,
        },
    )

    assert state.secondary_output_enabled is True
    assert state.secondary_output_device == "New Device"
    assert state.secondary_output_gain_db == 2.25
    assert state.secondary_output_lowpass_hz == SECONDARY_OUTPUT_DEFAULTS["secondary_output_lowpass_hz"]
    assert state.secondary_output_source_mode == SECONDARY_OUTPUT_DEFAULTS["secondary_output_source_mode"]
