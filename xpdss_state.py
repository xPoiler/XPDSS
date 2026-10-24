from __future__ import annotations

from typing import Any, Dict

PROFILE_DEFAULTS: Dict[str, Any] = {
    "upmix_enabled": False,
    "manual_upmix": False,
    "lfe_muted": False,
    "center_extraction": False,
    "independent_panning": False,
    "stage_equalizer": False,
    "panning_x": 0.0,
    "panning_y": 0.0,
    "center_spread": 0.0,
    "side_gain_db": 0.0,
    "width_factor": 1.0,
    "dynamic_eq": False,
    "cinema_steering": False,
    "gaming_panning": False,
    "widening_mode": False,
    "vocal_lock": False,
    "dialogue_focus": False,
    "spectral_tilt": False,
    "sub_synth": False,
    "sub_gain": 0.35,
    "sub_drive_cutoff": 80.0,
    "sub_low_cutoff": 20.0,
    "transaural_widening": False,
    "transaural_delay": 7,
    "transaural_gain": 0.35,
    "transaural_feedback": 0.08,
    "head_tracking": False,
    "headphone_tracking": False,
    "head_shadow_eq": False,
    "head_tracking_preview": False,
    "speaker_width_cm": 60.0,
    "camera_fov": 70.0,
    "speaker_distance_cm": 70.0,
    "room_width_m": 4.0,
    "harmonic_exciter": False,
    "loudness_norm": False,
    "transient_restore": False,
    "transient_amount": 4.0,
    "declip": False,
    "freq_dep_width": False,
    "hp_externalization": False,
    "surround_decorrelation": False,
    "micro_dynamics": False,
    "micro_dyn_amount": 4.0,
    "clarity_dsp": False,
    "clarity_amount": 4.0,
    "virtual_room": False,
    "virtual_room_mix": 0.25,
    "air_exciter": False,
    "air_amount": 0.3,
}

ADAPTIVE_DEFAULTS: Dict[str, Any] = {
    "adaptive_monitor_enabled": True,
    "adaptive_observe_only": True,
    "adaptive_mode": "observe",
    "adaptive_profile": "neutral",
    "adaptive_compare_bypass": False,
}

SECONDARY_OUTPUT_DEFAULTS: Dict[str, Any] = {
    "secondary_output_enabled": False,
    "secondary_output_device": "",
    "secondary_output_gain_db": 0.0,
    "secondary_output_lowpass_hz": 80.0,
    "secondary_output_highpass_hz": 20.0,
    "secondary_output_delay_ms": 0.0,
    "secondary_output_source_mode": "lfe_or_bass_mix",
    "secondary_output_mono_sum": True,
}


def _merge_known(defaults: Dict[str, Any], values: Dict[str, Any] | None) -> Dict[str, Any]:
    """Defaults overlaid with `values`, ignoring keys that are not settings.

    Config files (especially old flat ones) can contain unrelated keys; they
    must never be written onto the audio engine as attributes.
    """
    data = dict(defaults)
    if values:
        data.update({key: value for key, value in values.items() if key in defaults})
    return data


def build_profile_from_object(obj: Any) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for key, default in PROFILE_DEFAULTS.items():
        attr_name = key
        if key == "panning_x":
            attr_name = "gui_panning_x"
        elif key == "panning_y":
            attr_name = "gui_panning_y"
        profile[key] = getattr(obj, attr_name, default)
    return profile


def apply_profile_to_object(obj: Any, profile: Dict[str, Any] | None) -> Dict[str, Any]:
    data = _merge_known(PROFILE_DEFAULTS, profile)

    for key, value in data.items():
        attr_name = key
        if key == "panning_x":
            attr_name = "gui_panning_x"
        elif key == "panning_y":
            attr_name = "gui_panning_y"
        setattr(obj, attr_name, value)
    return data


def build_adaptive_state_from_object(obj: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for key, default in ADAPTIVE_DEFAULTS.items():
        state[key] = getattr(obj, key, default)
    return state


def apply_adaptive_state_to_object(obj: Any, state: Dict[str, Any] | None) -> Dict[str, Any]:
    data = _merge_known(ADAPTIVE_DEFAULTS, state)

    for key, value in data.items():
        setattr(obj, key, value)
    return data


def build_secondary_output_state_from_object(obj: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for key, default in SECONDARY_OUTPUT_DEFAULTS.items():
        state[key] = getattr(obj, key, default)
    return state


def apply_secondary_output_state_to_object(obj: Any, state: Dict[str, Any] | None) -> Dict[str, Any]:
    data = _merge_known(SECONDARY_OUTPUT_DEFAULTS, state)

    for key, value in data.items():
        setattr(obj, key, value)
    return data
