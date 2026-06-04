from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


EPS = 1e-9


@dataclass
class AdaptiveSnapshot:
    stereo_correlation: float
    side_ratio: float
    crest_factor: float
    compression_score: float
    mud_score: float
    presence_score: float
    air_score: float
    dialogue_score: float
    immersion_score: float
    surround_discreteness: float

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


class AdaptiveFeatureEngine:
    """Lightweight perceptual feature extraction for XPDSS.

    Phase 1 is observe-only: this engine produces stable metrics but does not
    mutate DSP parameters directly.
    """

    def __init__(self, sample_rate: int, smoothing: float = 0.85):
        self.sample_rate = int(sample_rate)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self._state: AdaptiveSnapshot | None = None

    def analyze_block(self, block: np.ndarray) -> AdaptiveSnapshot:
        data = np.asarray(block, dtype=np.float32)
        if data.ndim != 2 or data.shape[0] < 32 or data.shape[1] < 2:
            snapshot = AdaptiveSnapshot(
                stereo_correlation=0.0,
                side_ratio=0.0,
                crest_factor=1.0,
                compression_score=0.0,
                mud_score=0.0,
                presence_score=0.0,
                air_score=0.0,
                dialogue_score=0.0,
                immersion_score=0.0,
                surround_discreteness=0.0,
            )
            self._state = snapshot
            return snapshot

        left = data[:, 0]
        right = data[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        stereo_correlation = self._stereo_correlation(left, right)
        side_ratio = self._side_ratio(mid, side)
        crest_factor = self._crest_factor(data[:, :2])
        compression_score = self._compression_score(crest_factor)

        front_mono = np.mean(data[:, :2], axis=1)
        front_ref = np.mean(np.abs(front_mono)) + EPS
        mud_score = self._band_ratio_score(front_mono, 200.0, 500.0, 80.0, 5000.0, weight=3.2)
        presence_score = self._band_ratio_score(front_mono, 2000.0, 4000.0, 250.0, 5000.0, weight=4.2)
        air_score = self._band_ratio_score(front_mono, 8000.0, 12000.0, 2000.0, 14000.0, weight=5.0)

        surround_discreteness = self._surround_discreteness(data)

        dialogue_score = self._clip01(
            0.42 * presence_score
            + 0.28 * (1.0 - side_ratio)
            + 0.20 * stereo_correlation
            + 0.10 * (1.0 - self._clip01(mud_score * 0.7))
        )
        immersion_score = self._clip01(
            0.48 * side_ratio
            + 0.22 * (1.0 - stereo_correlation)
            + 0.20 * air_score
            + 0.10 * surround_discreteness
        )

        raw_snapshot = AdaptiveSnapshot(
            stereo_correlation=stereo_correlation,
            side_ratio=side_ratio,
            crest_factor=crest_factor,
            compression_score=compression_score,
            mud_score=mud_score,
            presence_score=presence_score,
            air_score=air_score,
            dialogue_score=dialogue_score,
            immersion_score=immersion_score,
            surround_discreteness=surround_discreteness,
        )
        self._state = self._smooth_snapshot(raw_snapshot)
        return self._state

    def latest_snapshot(self) -> AdaptiveSnapshot | None:
        return self._state

    def latest_snapshot_dict(self) -> Dict[str, float]:
        if self._state is None:
            return {}
        return self._state.to_dict()

    def _smooth_snapshot(self, raw: AdaptiveSnapshot) -> AdaptiveSnapshot:
        if self._state is None or self.smoothing <= 0.0:
            return raw

        prev = self._state
        alpha = self.smoothing
        values = {
            field: alpha * getattr(prev, field) + (1.0 - alpha) * getattr(raw, field)
            for field in raw.__dataclass_fields__
        }
        return AdaptiveSnapshot(**values)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _stereo_correlation(left: np.ndarray, right: np.ndarray) -> float:
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        denom = np.sqrt(np.sum(left_centered ** 2) * np.sum(right_centered ** 2))
        if denom <= EPS:
            return 0.0
        corr = float(np.sum(left_centered * right_centered) / denom)
        return float(np.clip(abs(corr), 0.0, 1.0))

    @staticmethod
    def _side_ratio(mid: np.ndarray, side: np.ndarray) -> float:
        mid_rms = float(np.sqrt(np.mean(mid ** 2) + EPS))
        side_rms = float(np.sqrt(np.mean(side ** 2) + EPS))
        return float(side_rms / (mid_rms + side_rms + EPS))

    @staticmethod
    def _crest_factor(stereo: np.ndarray) -> float:
        peak = float(np.max(np.abs(stereo)))
        rms = float(np.sqrt(np.mean(stereo ** 2) + EPS))
        return peak / (rms + EPS)

    @classmethod
    def _compression_score(cls, crest_factor: float) -> float:
        # Crest near ~1.4 means flattened/clipped; >3 means comfortably dynamic.
        return cls._clip01((2.2 - crest_factor) / 1.0)

    def _surround_discreteness(self, data: np.ndarray) -> float:
        if data.shape[1] <= 4:
            return 0.0

        front = np.mean(data[:, :2], axis=1)
        rear = np.mean(data[:, 4:], axis=1)
        rear_energy = float(np.sqrt(np.mean(rear ** 2) + EPS))
        front_energy = float(np.sqrt(np.mean(front ** 2) + EPS))
        energy_ratio = rear_energy / (front_energy + rear_energy + EPS)
        decorrelation = 1.0 - self._stereo_correlation(front, rear)
        return self._clip01(0.6 * energy_ratio + 0.4 * decorrelation)

    def _band_ratio_score(
        self,
        signal: np.ndarray,
        band_low: float,
        band_high: float,
        ref_low: float,
        ref_high: float,
        weight: float,
    ) -> float:
        freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / self.sample_rate)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.shape[0])))

        band_mask = (freqs >= band_low) & (freqs <= band_high)
        ref_mask = (freqs >= ref_low) & (freqs <= ref_high)
        if not np.any(ref_mask):
            return 0.0

        band_energy = float(np.sum(spectrum[band_mask] ** 2))
        ref_energy = float(np.sum(spectrum[ref_mask] ** 2)) + EPS
        return self._clip01((band_energy / ref_energy) * weight)
