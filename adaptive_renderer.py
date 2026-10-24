from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


EPS = 1e-12

# Input blocks quieter than this are ignored: pauses and silence must not
# drag the measurements (and therefore the sliders) around.
SILENCE_DB = -55.0


@dataclass
class AdaptiveSnapshot:
    level_db: float
    stereo_correlation: float
    side_ratio: float
    dynamic_range_db: float
    compression_score: float
    mud_score: float
    presence_score: float
    air_score: float
    dialogue_score: float
    immersion_score: float
    surround_discreteness: float

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class AdaptiveFeatureEngine:
    """Long-term description of the input signal for the adaptive renderer.

    Every measurement is integrated over seconds (time constant `tau`), so
    the result describes the *content* rather than the current 10 ms block:

    - tonal balance (mud / presence / air) from a smoothed long-term spectrum
    - stereo image (correlation, side ratio) smoothed the same way
    - dynamics: the loudness range of ~0.4 s segments over the last
      `history_seconds`; heavily compressed masters sit around 2-4 dB,
      dynamic film / classical content well above 10 dB
    Silent blocks are skipped entirely.
    """

    BANDS = {
        "mud": (200.0, 500.0),
        "presence": (2000.0, 4000.0),
        "air": (8000.0, 12000.0),
        "ref_low": (80.0, 5000.0),
        "ref_presence": (250.0, 5000.0),
        "ref_air": (2000.0, 14000.0),
    }

    def __init__(self, sample_rate: int, tau: float = 3.0, history_seconds: float = 10.0,
                 segment_seconds: float = 0.4):
        self.sample_rate = int(sample_rate)
        self.tau = max(1e-3, float(tau))
        self.segment_seconds = float(segment_seconds)
        self._segments: deque = deque(maxlen=max(4, int(history_seconds / segment_seconds)))
        self._segment_energy = 0.0
        self._segment_count = 0
        self._segment_time = 0.0
        self._smooth: Dict[str, float] | None = None
        self._state: AdaptiveSnapshot | None = None
        # False when the last block was silent (measurements were held)
        self.last_block_active = False

    # ── public API ────────────────────────────────────────────────────
    def analyze_block(self, block: np.ndarray, dt: float | None = None) -> AdaptiveSnapshot | None:
        """Feed one input block. `dt` is the time this block represents
        (defaults to its duration). Returns the latest snapshot, or None until
        the first non-silent block has been seen."""
        data = np.asarray(block, dtype=np.float32)
        self.last_block_active = False
        if data.ndim != 2 or data.shape[0] < 32 or data.shape[1] < 2:
            return self._state
        if dt is None:
            dt = data.shape[0] / self.sample_rate

        fronts = data[:, :2].astype(np.float64)
        level_db = 10.0 * np.log10(np.mean(fronts ** 2) + EPS)
        if level_db < SILENCE_DB:
            return self._state
        self.last_block_active = True

        raw = self._block_features(data)
        alpha = 1.0 - np.exp(-dt / self.tau)
        if self._smooth is None:
            self._smooth = dict(raw)
        else:
            for key, value in raw.items():
                self._smooth[key] += alpha * (value - self._smooth[key])
        self._smooth["level_db"] = level_db if "level_db" not in self._smooth else (
            self._smooth["level_db"] + alpha * (level_db - self._smooth["level_db"]))

        self._update_dynamics(fronts, dt)
        self._state = self._snapshot()
        return self._state

    def latest_snapshot(self) -> AdaptiveSnapshot | None:
        return self._state

    def latest_snapshot_dict(self) -> Dict[str, float]:
        return {} if self._state is None else self._state.to_dict()

    # ── internals ─────────────────────────────────────────────────────
    def _block_features(self, data: np.ndarray) -> Dict[str, float]:
        left = data[:, 0].astype(np.float64)
        right = data[:, 1].astype(np.float64)
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        feats = {
            "stereo_correlation": self._stereo_correlation(left, right),
            "side_ratio": self._side_ratio(mid, side),
            "surround_discreteness": self._surround_discreteness(data),
        }
        # Band powers (smoothed as powers, ratios formed afterwards)
        spectrum = np.abs(np.fft.rfft(mid * np.hanning(mid.shape[0]))) ** 2
        freqs = np.fft.rfftfreq(mid.shape[0], d=1.0 / self.sample_rate)
        for name, (lo, hi) in self.BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            feats[f"p_{name}"] = float(np.sum(spectrum[mask])) if np.any(mask) else 0.0
        return feats

    def _update_dynamics(self, fronts: np.ndarray, dt: float) -> None:
        self._segment_energy += float(np.mean(fronts ** 2))
        self._segment_count += 1
        self._segment_time += dt
        if self._segment_time >= self.segment_seconds:
            self._segments.append(10.0 * np.log10(self._segment_energy / self._segment_count + EPS))
            self._segment_energy = 0.0
            self._segment_count = 0
            self._segment_time = 0.0

    def _dynamic_range_db(self) -> float | None:
        if len(self._segments) < 8:
            return None
        levels = np.asarray(self._segments)
        return float(np.percentile(levels, 95) - np.percentile(levels, 10))

    def _snapshot(self) -> AdaptiveSnapshot:
        s = self._smooth
        mud = _clip01(s["p_mud"] / (s["p_ref_low"] + EPS) * 3.2)
        presence = _clip01(s["p_presence"] / (s["p_ref_presence"] + EPS) * 4.2)
        air = _clip01(s["p_air"] / (s["p_ref_air"] + EPS) * 5.0)
        correlation = _clip01(s["stereo_correlation"])
        side_ratio = _clip01(s["side_ratio"])
        surround = _clip01(s["surround_discreteness"])

        dyn_range = self._dynamic_range_db()
        if dyn_range is None:
            compression = 0.5  # not enough history yet: neutral
            dyn_range = float("nan")
        else:
            # <= 3 dB loudness range -> fully compressed, >= 10 dB -> dynamic
            compression = _clip01((10.0 - dyn_range) / 7.0)

        dialogue = _clip01(0.42 * presence + 0.28 * (1.0 - side_ratio)
                           + 0.20 * correlation + 0.10 * (1.0 - _clip01(mud * 0.7)))
        immersion = _clip01(0.48 * side_ratio + 0.22 * (1.0 - correlation)
                            + 0.20 * air + 0.10 * surround)
        return AdaptiveSnapshot(
            level_db=float(s["level_db"]),
            stereo_correlation=correlation,
            side_ratio=side_ratio,
            dynamic_range_db=dyn_range,
            compression_score=compression,
            mud_score=mud,
            presence_score=presence,
            air_score=air,
            dialogue_score=dialogue,
            immersion_score=immersion,
            surround_discreteness=surround,
        )

    @staticmethod
    def _stereo_correlation(left: np.ndarray, right: np.ndarray) -> float:
        left_c = left - np.mean(left)
        right_c = right - np.mean(right)
        denom = np.sqrt(np.sum(left_c ** 2) * np.sum(right_c ** 2))
        if denom <= 1e-9:
            return 0.0
        return float(np.clip(abs(np.sum(left_c * right_c) / denom), 0.0, 1.0))

    @staticmethod
    def _side_ratio(mid: np.ndarray, side: np.ndarray) -> float:
        mid_rms = float(np.sqrt(np.mean(mid ** 2) + EPS))
        side_rms = float(np.sqrt(np.mean(side ** 2) + EPS))
        return float(side_rms / (mid_rms + side_rms + EPS))

    def _surround_discreteness(self, data: np.ndarray) -> float:
        if data.shape[1] <= 4:
            return 0.0
        front = np.mean(data[:, :2], axis=1).astype(np.float64)
        rear = np.mean(data[:, 4:], axis=1).astype(np.float64)
        rear_energy = float(np.sqrt(np.mean(rear ** 2) + EPS))
        front_energy = float(np.sqrt(np.mean(front ** 2) + EPS))
        energy_ratio = rear_energy / (front_energy + rear_energy + EPS)
        if rear_energy < 1e-6:
            return 0.0
        decorrelation = 1.0 - self._stereo_correlation(front, rear)
        return _clip01(0.6 * energy_ratio + 0.4 * decorrelation)
