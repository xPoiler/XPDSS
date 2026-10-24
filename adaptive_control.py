"""Adaptive renderer: maps the long-term input analysis to effect amounts.

Only the *amount* sliders of the enhancement effects are driven. Nothing
that changes where sounds come from (upmix spread / side gain / width,
XTC, head-tracking geometry) and no on/off toggle is ever touched.

Mapping (all values stay inside the GUI slider ranges):

  clarity_amount    1.5 - 5.5 dB  more cut when the low-mids are congested
  transient_amount  1.5 - 6.0 dB  more punch for compressed masters,
                                  a bit less when the content is dialogue
  micro_dyn_amount  1.5 - 5.0 dB  more lift for dynamic content (quiet detail
                                  in film/classical), little for compressed
                                  masters that have no quiet detail to lift
  air_amount        10 - 40 %     more when the top octave is lacking, less
                                  for dialogue (sibilance)
  virtual_room_mix   8 - 33 %     more room for dry/narrow content, less when
                                  the mix is already spacious or is dialogue;
                                  x0.7 on headphones
"""
from __future__ import annotations

import math
from typing import Dict, Mapping

# effect amount slider -> the toggle that enables that effect
ADAPTIVE_PARAMS: Dict[str, str] = {
    "clarity_amount": "clarity_dsp",
    "transient_amount": "transient_restore",
    "micro_dyn_amount": "micro_dynamics",
    "air_amount": "air_exciter",
    "virtual_room_mix": "virtual_room",
}


def _clip01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return min(1.0, max(0.0, float(value)))


def compute_targets(scores: Mapping[str, float], output_mode: str = "speakers") -> Dict[str, float]:
    """Instantaneous target amounts for a snapshot (no smoothing)."""
    mud = _clip01(scores.get("mud_score", 0.0))
    air = _clip01(scores.get("air_score", 0.0))
    compression = _clip01(scores.get("compression_score", 0.5))
    dialogue = _clip01(scores.get("dialogue_score", 0.0))
    immersion = _clip01(scores.get("immersion_score", 0.0))

    room = 0.08 + 0.25 * (1.0 - immersion) - 0.10 * dialogue
    if output_mode == "headphones":
        room *= 0.7
    return {
        "clarity_amount": 1.5 + 4.0 * mud,
        "transient_amount": max(1.5, 1.5 + 4.5 * compression - 1.0 * dialogue),
        "micro_dyn_amount": 1.5 + 3.5 * (1.0 - compression),
        "air_amount": max(0.10, 0.10 + 0.30 * (1.0 - air) - 0.10 * dialogue),
        "virtual_room_mix": min(0.33, max(0.08 * (0.7 if output_mode == "headphones" else 1.0), room)),
    }


class AdaptiveController:
    """Glides the effect amounts toward the targets for the current content.

    `tau` is the glide time constant in seconds; together with the analysis
    integration this keeps changes slow enough to never be heard as pumping.
    """

    def __init__(self, tau: float = 2.0):
        self.tau = max(1e-3, float(tau))
        self._values: Dict[str, float] = {}

    def reset(self, start: Mapping[str, float]) -> None:
        """Start gliding from the given (manual) values."""
        self._values = {key: float(start[key]) for key in ADAPTIVE_PARAMS if key in start}

    def update(self, scores: Mapping[str, float] | None, dt: float,
               output_mode: str = "speakers") -> Dict[str, float]:
        """Advance by `dt` seconds. With `scores` None (silence) the values hold."""
        if scores:
            self._glide(compute_targets(scores, output_mode), dt, self.tau)
        return dict(self._values)

    def glide_to(self, targets: Mapping[str, float], dt: float, tau: float = 0.3) -> bool:
        """Glide toward explicit targets (used to return to manual values).
        Returns True once every value has arrived."""
        self._glide(targets, dt, tau)
        return all(abs(self._values.get(k, v) - v) <= 1e-3 * max(1.0, abs(v)) for k, v in targets.items())

    def values(self) -> Dict[str, float]:
        return dict(self._values)

    def _glide(self, targets: Mapping[str, float], dt: float, tau: float) -> None:
        alpha = 1.0 - math.exp(-max(0.0, dt) / max(1e-3, tau))
        for key, target in targets.items():
            current = self._values.get(key, float(target))
            self._values[key] = current + alpha * (float(target) - current)
