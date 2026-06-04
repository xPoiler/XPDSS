from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@dataclass
class AdaptiveTargets:
    clarity_amount: float = 4.0
    width_factor: float = 1.0
    virtual_room_mix: float = 0.18
    air_amount: float = 0.18
    micro_dyn_amount: float = 4.0
    transient_amount: float = 4.0
    dialogue_focus: bool = False
    vocal_lock: bool = False
    hp_externalization: bool = False

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


@dataclass
class AdaptiveState:
    targets: AdaptiveTargets
    scores: Dict[str, float]
    output_mode: str
    profile: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "targets": self.targets.to_dict(),
            "scores": dict(self.scores),
            "output_mode": self.output_mode,
            "profile": self.profile,
        }


PROFILE_BIASES = {
    "neutral": {
        "dialogue_bias": 0.00,
        "room_bias": 0.00,
        "width_bias": 0.00,
        "air_bias": 0.00,
        "dynamics_bias": 0.00,
    },
    "cinema": {
        "dialogue_bias": 0.08,
        "room_bias": 0.08,
        "width_bias": 0.10,
        "air_bias": 0.03,
        "dynamics_bias": 0.02,
    },
    "music": {
        "dialogue_bias": -0.10,
        "room_bias": 0.04,
        "width_bias": 0.18,
        "air_bias": 0.08,
        "dynamics_bias": 0.04,
    },
    "night": {
        "dialogue_bias": 0.18,
        "room_bias": -0.10,
        "width_bias": -0.12,
        "air_bias": -0.02,
        "dynamics_bias": 0.08,
    },
}


class AdaptiveController:
    def __init__(self, smoothing: float = 0.82):
        self.smoothing = _clip01(smoothing)
        self._state: AdaptiveState | None = None

    def update(self, snapshot: Dict[str, float] | None, output_mode: str = "speakers", profile: str = "neutral") -> AdaptiveState:
        scores = self._normalize_scores(snapshot or {})
        biases = PROFILE_BIASES.get(profile, PROFILE_BIASES["neutral"])

        dialogue = _clip01(scores["dialogue_score"] + biases["dialogue_bias"])
        immersion = _clip01(scores["immersion_score"] + biases["width_bias"] * 0.5)
        compression = _clip01(scores["compression_score"] + biases["dynamics_bias"])
        mud = scores["mud_score"]
        air = scores["air_score"]
        surround = scores["surround_discreteness"]

        air_need = _clip01(1.0 - air + max(0.0, biases["air_bias"]))

        width = 0.88 + 0.65 * immersion - 0.30 * dialogue + biases["width_bias"]
        if profile == "night":
            width -= 0.05 * compression
        width_factor = float(min(1.75, max(0.75, width)))

        room_mix = 0.06 + 0.18 * immersion + 0.10 * surround - 0.10 * dialogue + biases["room_bias"]
        if output_mode == "headphones":
            room_mix *= 0.65
        virtual_room_mix = float(min(0.35, max(0.02, room_mix)))

        clarity_amount = float(min(6.0, max(1.2, 1.4 + 3.3 * mud + 1.2 * dialogue)))
        air_amount = float(min(0.45, max(0.04, 0.06 + 0.33 * air_need + biases["air_bias"] * 0.2)))
        micro_dyn_amount = float(min(8.0, max(1.5, 1.5 + 5.5 * compression)))
        transient_amount = float(min(8.0, max(1.5, 1.8 + 4.8 * compression - 0.7 * dialogue)))

        targets = AdaptiveTargets(
            clarity_amount=clarity_amount,
            width_factor=width_factor,
            virtual_room_mix=virtual_room_mix,
            air_amount=air_amount,
            micro_dyn_amount=micro_dyn_amount,
            transient_amount=transient_amount,
            dialogue_focus=dialogue >= 0.52,
            vocal_lock=dialogue >= 0.74 and output_mode == "speakers",
            hp_externalization=output_mode == "headphones" and immersion >= 0.45,
        )

        raw_state = AdaptiveState(targets=targets, scores=scores, output_mode=output_mode, profile=profile)
        self._state = self._smooth_state(raw_state)
        return self._state

    def latest_state(self) -> AdaptiveState | None:
        return self._state

    def latest_state_dict(self) -> Dict[str, object]:
        if self._state is None:
            return {}
        return self._state.to_dict()

    def _smooth_state(self, raw_state: AdaptiveState) -> AdaptiveState:
        if self._state is None or self.smoothing <= 0.0:
            return raw_state

        prev = self._state
        alpha = self.smoothing
        prev_targets = prev.targets.to_dict()
        next_targets = raw_state.targets.to_dict()
        blended_targets = {}
        for key, value in next_targets.items():
            if isinstance(value, bool):
                blended_targets[key] = value
            else:
                blended_targets[key] = alpha * float(prev_targets[key]) + (1.0 - alpha) * float(value)

        blended_scores = {
            key: alpha * float(prev.scores.get(key, 0.0)) + (1.0 - alpha) * float(value)
            for key, value in raw_state.scores.items()
        }
        return AdaptiveState(
            targets=AdaptiveTargets(**blended_targets),
            scores=blended_scores,
            output_mode=raw_state.output_mode,
            profile=raw_state.profile,
        )

    @staticmethod
    def _normalize_scores(snapshot: Dict[str, float]) -> Dict[str, float]:
        keys = (
            "dialogue_score",
            "immersion_score",
            "compression_score",
            "mud_score",
            "air_score",
            "surround_discreteness",
        )
        return {key: _clip01(float(snapshot.get(key, 0.0))) for key in keys}
