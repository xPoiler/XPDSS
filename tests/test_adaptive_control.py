from adaptive_control import AdaptiveController


def test_dialogue_snapshot_pushes_clarity_and_dialogue_focus_higher_than_music_snapshot():
    controller = AdaptiveController(smoothing=0.0)

    dialogue_state = controller.update({
        "dialogue_score": 0.92,
        "immersion_score": 0.18,
        "compression_score": 0.25,
        "mud_score": 0.72,
        "air_score": 0.18,
        "surround_discreteness": 0.05,
    }, output_mode="speakers", profile="neutral")

    music_state = controller.update({
        "dialogue_score": 0.22,
        "immersion_score": 0.88,
        "compression_score": 0.20,
        "mud_score": 0.18,
        "air_score": 0.72,
        "surround_discreteness": 0.40,
    }, output_mode="speakers", profile="music")

    assert dialogue_state.targets.clarity_amount > music_state.targets.clarity_amount
    assert dialogue_state.targets.dialogue_focus is True
    assert music_state.targets.width_factor > dialogue_state.targets.width_factor


def test_compressed_snapshot_pushes_dynamics_recovery_higher():
    controller = AdaptiveController(smoothing=0.0)

    dynamic_state = controller.update({
        "dialogue_score": 0.35,
        "immersion_score": 0.40,
        "compression_score": 0.10,
        "mud_score": 0.25,
        "air_score": 0.45,
        "surround_discreteness": 0.05,
    }, output_mode="speakers", profile="neutral")

    compressed_state = controller.update({
        "dialogue_score": 0.35,
        "immersion_score": 0.40,
        "compression_score": 0.90,
        "mud_score": 0.25,
        "air_score": 0.45,
        "surround_discreteness": 0.05,
    }, output_mode="speakers", profile="neutral")

    assert compressed_state.targets.micro_dyn_amount > dynamic_state.targets.micro_dyn_amount
    assert compressed_state.targets.transient_amount > dynamic_state.targets.transient_amount


def test_night_profile_biases_toward_dialogue_and_less_room_than_cinema():
    snapshot = {
        "dialogue_score": 0.60,
        "immersion_score": 0.55,
        "compression_score": 0.25,
        "mud_score": 0.30,
        "air_score": 0.40,
        "surround_discreteness": 0.20,
    }
    controller = AdaptiveController(smoothing=0.0)

    cinema_state = controller.update(snapshot, output_mode="speakers", profile="cinema")
    night_state = controller.update(snapshot, output_mode="speakers", profile="night")

    assert night_state.targets.dialogue_focus is True
    assert night_state.targets.virtual_room_mix < cinema_state.targets.virtual_room_mix
    assert night_state.targets.width_factor < cinema_state.targets.width_factor


def test_headphone_targets_never_claim_speaker_only_adaptive_modules():
    controller = AdaptiveController(smoothing=0.0)

    state = controller.update({
        "dialogue_score": 0.92,
        "immersion_score": 0.88,
        "compression_score": 0.20,
        "mud_score": 0.25,
        "air_score": 0.25,
        "surround_discreteness": 0.10,
    }, output_mode="headphones", profile="cinema")

    assert state.targets.dialogue_focus is True
    assert state.targets.vocal_lock is False
    assert state.targets.hp_externalization is True


def test_adaptive_targets_do_not_expose_manual_dynamics_toggles():
    controller = AdaptiveController(smoothing=0.0)

    state = controller.update({
        "dialogue_score": 0.30,
        "immersion_score": 0.40,
        "compression_score": 0.12,
        "mud_score": 0.20,
        "air_score": 0.45,
        "surround_discreteness": 0.05,
    }, output_mode="speakers", profile="neutral")

    targets = state.targets.to_dict()
    assert 'micro_dynamics' not in targets
    assert 'transient_restore' not in targets
