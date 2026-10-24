"""Frequency-domain (STFT) stereo upmixer.

Each STFT bin of the stereo input is split into a *direct* (primary) and an
*ambient* part with the classic primary-ambient model

    L = S + A_L,    R = a * S + A_R,    E|A_L|^2 = E|A_R|^2 = P_A,  A_L, A_R, S uncorrelated

whose statistics are tracked per bin (recursive averages of |L|^2, |R|^2 and
L R*). The ambient power is the smaller eigenvalue of the 2x2 covariance,
the direct signal is estimated by least squares (Wiener), and the
decomposition is complementary: direct + ambient == input, bin by bin, so
mis-estimates move energy between speakers but never lose it.

The direct part of every bin is then re-panned across the front L/C/R
speakers from its own panning position (amplitude ratio |a|), so several
sources at different frequencies each land at their own place in the front
stage. Only the ambient part is meant for the surrounds.

Front/back: the inter-channel phase of the direct part tells front from
rear. In-phase content is front; content in opposite phase between L and R
is how matrix-encoded stereo (Dolby Surround / Pro Logic) and many games
carry rear sounds. Steering modes:
  STEER_OFF     rear-phase content stays in L/R as a phantom (plain stereo)
  STEER_CINEMA  rear-phase content becomes a point source in the surrounds,
                left/right from its panning
  STEER_GAMING  as cinema, with sharper front/back decisions and faster
                tracking; on 7.1 partly-rear sources go to the sides and
                fully-rear ones to the backs

Streaming: frames of `fft_size` with 75 % overlap and sqrt-Hann
analysis/synthesis windows (perfect reconstruction). Latency is exactly
`fft_size` samples whatever the host block size.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12

# Output streams of SpectralUpmixer.process
DIRECT_L, DIRECT_R, DIRECT_C, AMBIENT_L, AMBIENT_R, SIDE_L, SIDE_R, BACK_L, BACK_R = range(9)
N_STREAMS = 9

STEER_OFF, STEER_CINEMA, STEER_GAMING = 0, 1, 2


class SpectralUpmixer:
    def __init__(self, sample_rate: int, fft_size: int | None = None,
                 stats_tau: float = 0.10, gaming_tau: float = 0.04, debias: float = 2.0,
                 bass_lo_hz: float = 150.0, bass_hi_hz: float = 250.0):
        self.sample_rate = int(sample_rate)
        if fft_size is None:
            # ~21 ms frames: fine enough in frequency (47 Hz bins at 48 kHz)
            # for panning analysis, short enough to keep transients tight.
            fft_size = 1024 if self.sample_rate <= 48000 else 2048
        self.n = int(fft_size)
        self.hop = self.n // 4
        self.bins = self.n // 2 + 1

        # sqrt-Hann (periodic) analysis and synthesis windows
        win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n) / self.n))
        self.win = win
        # overlap-add normalisation (constant for this window / hop)
        ola = np.zeros(self.hop)
        for k in range(self.n // self.hop):
            ola += (win * win)[k * self.hop:(k + 1) * self.hop]
        self.syn_win = win / ola[0]

        dt = self.hop / self.sample_rate
        self.alpha = 1.0 - math.exp(-dt / stats_tau)
        self.alpha_gaming = 1.0 - math.exp(-dt / gaming_tau)
        # Short-term averages make uncorrelated (diffuse) signals look partly
        # correlated: the expected spurious |L R*|^2 of an exponential average
        # is about alpha / (2 - alpha) * |L|^2 |R|^2. Subtracting it (x `debias`)
        # keeps reverb and room sound out of the center and fronts.
        self.debias_factor = float(debias)
        self.debias = self.debias_factor * self.alpha / (2.0 - self.alpha)

        freqs = np.fft.rfftfreq(self.n, 1.0 / self.sample_rate)
        # 0 below bass_lo (bass stays untouched in L/R), 1 above bass_hi.
        # Kept well above the bass: with ~47 Hz bins a low tone spreads over
        # several bins, and direction is not audible down there anyway.
        ramp = np.clip((freqs - bass_lo_hz) / (bass_hi_hz - bass_lo_hz), 0.0, 1.0)
        self.process_weight = 0.5 - 0.5 * np.cos(np.pi * ramp)

        self.reset()

    @property
    def latency(self) -> int:
        return self.n

    def reset(self) -> None:
        self._frame = np.zeros((self.n, 2))
        self._pending = np.zeros((0, 2))
        self._ola = np.zeros((self.n, N_STREAMS))
        # Output FIFO primed with one hop: latency is n whatever the block size
        self._out = np.zeros((self.hop, N_STREAMS))
        self._pll = np.zeros(self.bins)
        self._prr = np.zeros(self.bins)
        self._plr = np.zeros(self.bins, dtype=np.complex128)
        self._primed = False

    # ── streaming ─────────────────────────────────────────────────────
    def process(self, stereo: np.ndarray, center_spread: float = 0.0, width: float = 1.0,
                center: bool = True, steering: int = STEER_OFF, split_sides_backs: bool = False) -> np.ndarray:
        """Feed (frames, 2) samples, get (frames, N_STREAMS) streams delayed by `latency`.

        Streams: direct L/R/C (front stage), ambient L/R, and the steered
        rear point sources SIDE_L/R and BACK_L/R (zero unless steering is
        on; BACK_* only with split_sides_backs, i.e. a 7.1 layout).
        center_spread 0 = sources panned across L/C/R, 1 = no center (direct
        stays a phantom image in L/R). width scales the panning positions.
        """
        frames = stereo.shape[0]
        spread = 1.0 if not center else float(np.clip(center_spread, 0.0, 1.0))
        self._pending = np.concatenate((self._pending, np.asarray(stereo, dtype=np.float64)))
        produced = []
        while self._pending.shape[0] >= self.hop:
            hop_in = self._pending[:self.hop]
            self._pending = self._pending[self.hop:]
            self._frame = np.concatenate((self._frame[self.hop:], hop_in))
            self._ola += self._process_frame(self._frame, spread, float(width), int(steering), bool(split_sides_backs))
            produced.append(self._ola[:self.hop].copy())
            self._ola = np.concatenate((self._ola[self.hop:], np.zeros((self.hop, N_STREAMS))))
        if produced:
            self._out = np.concatenate([self._out] + produced)
        out = self._out[:frames]
        self._out = self._out[frames:]
        return out

    # ── one STFT frame ────────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray, spread: float, width: float,
                       steering: int = STEER_OFF, split: bool = False) -> np.ndarray:
        X = np.fft.rfft(frame * self.win[:, None], axis=0)
        L, R = X[:, 0], X[:, 1]

        # Recursive per-bin statistics (seeded on the first frame)
        pll_i, prr_i, plr_i = np.abs(L) ** 2, np.abs(R) ** 2, L * np.conj(R)
        if not self._primed:
            self._pll, self._prr, self._plr = pll_i, prr_i, plr_i
            self._primed = True
        else:
            a = self.alpha_gaming if steering == STEER_GAMING else self.alpha
            self._pll = self._pll + a * (pll_i - self._pll)
            self._prr = self._prr + a * (prr_i - self._prr)
            self._plr = self._plr + a * (plr_i - self._plr)
        pll = self._smooth_bins(self._pll)
        prr = self._smooth_bins(self._prr)
        plr = self._smooth_bins(self._plr)
        debias = self.debias
        if steering == STEER_GAMING:
            debias = self.debias_factor * self.alpha_gaming / (2.0 - self.alpha_gaming)
        if debias:
            # The spurious part vanishes for fully coherent bins, so the
            # correction scales with (1 - coherence^2)^2: mono stays mono.
            plr_mag2 = np.abs(plr) ** 2
            power = pll * prr
            coh2 = np.clip(plr_mag2 / np.maximum(power, EPS), 0.0, 1.0)
            corrected = np.maximum(plr_mag2 - debias * (1.0 - coh2) ** 2 * power, 0.0)
            plr = plr * np.sqrt(corrected / np.maximum(plr_mag2, EPS))

        # Primary-ambient decomposition, using the louder channel of each bin
        # as reference (X1 = S + A1, X2 = b*S + A2 with |b| <= 1). A fixed
        # reference channel is ill-conditioned for sources panned to the other
        # side and leaked them into the ambience.
        swap = prr > pll
        p11 = np.where(swap, prr, pll)
        p22 = np.where(swap, pll, prr)
        p12 = np.where(swap, np.conj(plr), plr)          # E[X1 X2*]
        X1 = np.where(swap, R, L)
        X2 = np.where(swap, L, R)

        pa = 0.5 * (p11 + p22 - np.sqrt((p11 - p22) ** 2 + 4.0 * np.abs(p12) ** 2))
        pa = np.maximum(pa, 0.0)
        ps = np.maximum(p11 - pa, EPS)
        b_c = np.conj(p12) / ps                         # X2 ~= b_c * S
        den = ps * (1.0 + np.abs(b_c) ** 2) + pa + EPS
        S = (ps * X1 + p12 * X2) / den                  # least-squares direct estimate
        D1 = S
        D2 = b_c * S
        DL = np.where(swap, D2, D1)
        DR = np.where(swap, D1, D2)
        AL = L - DL
        AR = R - DR

        ratio = np.abs(b_c)                             # quieter / louder channel
        # amplitude ratio R/L as an angle: 0 = hard left, pi/4 = center, pi/2 = hard right
        angle = np.where(swap, np.arctan2(1.0, ratio), np.arctan2(ratio, 1.0))

        # Re-pan the direct part across L / C / R (constant power)
        pan = (angle - math.pi / 4) / (math.pi / 4)   # -1 left .. +1 right
        pan = np.clip(pan * width, -1.0, 1.0)
        left_half = pan < 0
        t = np.where(left_half, pan + 1.0, pan) * (math.pi / 2)
        g_l = np.where(left_half, np.cos(t), 0.0)
        g_c = np.where(left_half, np.sin(t), np.cos(t))
        g_r = np.where(left_half, 0.0, np.sin(t))

        mag = np.abs(S) * np.sqrt(1.0 + ratio ** 2)     # total direct amplitude
        u_l = self._unit(DL)
        u_r = self._unit(DR, fallback=u_l)
        u_c = self._unit(DL + DR, fallback=u_l)

        # Front/back from the inter-channel phase of the direct part:
        # rear = 0 for phase differences up to 90 deg, 1 at 180 deg.
        cos_phase = np.real(plr) / np.maximum(np.abs(plr), EPS)
        rear = np.clip(-cos_phase, 0.0, 1.0)
        rear_raw = rear
        if steering == STEER_GAMING:
            rear = np.clip((rear - 0.25) / 0.5, 0.0, 1.0)  # decisive front/back
        front = 1.0 - rear

        if steering == STEER_OFF:
            # Linear split: the rear-phase part stays exactly as it was in L/R
            f_amp, r_amp = front, rear
            keep_l, keep_r = rear * DL, rear * DR
        else:
            # Steered: constant power between the front stage and the surrounds
            f_amp, r_amp = np.sqrt(front), np.sqrt(rear)
            keep_l = keep_r = 0.0

        lcr_l = g_l * mag * f_amp * u_l
        lcr_r = g_r * mag * f_amp * u_r
        lcr_c = g_c * mag * f_amp * u_c

        out_l = (1.0 - spread) * lcr_l + spread * DL * f_amp + keep_l
        out_r = (1.0 - spread) * lcr_r + spread * DR * f_amp + keep_r
        out_c = (1.0 - spread) * lcr_c

        side_l = side_r = back_l = back_r = 0.0
        if steering != STEER_OFF:
            # Rear point source, panned left/right by its position (constant power)
            th = (pan + 1.0) * (math.pi / 4)
            rear_mag = mag * r_amp
            src_l = np.cos(th) * rear_mag * u_l     # one common phase: a coherent
            src_r = np.sin(th) * rear_mag * u_l     # rear image, not an anti-phase one
            if split and steering == STEER_GAMING:
                # 7.1: beside (~110-120 deg phase) -> sides, behind (180 deg) -> backs
                behind = np.clip((rear_raw - 0.5) / 0.5, 0.0, 1.0) * (math.pi / 2)
                g_side = np.cos(behind)
                g_back = np.sin(behind)
            elif split:
                g_side, g_back = 0.8, 0.6
            else:
                g_side, g_back = 1.0, 0.0
            side_l, side_r = g_side * src_l, g_side * src_r
            back_l, back_r = g_back * src_l, g_back * src_r

        # Below ~150 Hz nothing is moved: bass stays in L/R as it is
        w = self.process_weight
        streams = np.zeros((self.bins, N_STREAMS), dtype=np.complex128)
        streams[:, DIRECT_L] = w * out_l + (1.0 - w) * L
        streams[:, DIRECT_R] = w * out_r + (1.0 - w) * R
        streams[:, DIRECT_C] = w * out_c
        streams[:, AMBIENT_L] = w * AL
        streams[:, AMBIENT_R] = w * AR
        streams[:, SIDE_L] = w * side_l
        streams[:, SIDE_R] = w * side_r
        streams[:, BACK_L] = w * back_l
        streams[:, BACK_R] = w * back_r

        # Only synthesise the streams in use (the rear ones are silent unless
        # steering is on; the backs only on 7.1)
        active = list(range(AMBIENT_R + 1))
        if steering != STEER_OFF:
            active += [SIDE_L, SIDE_R] + ([BACK_L, BACK_R] if split else [])
        y = np.zeros((self.n, N_STREAMS))
        y[:, active] = np.fft.irfft(streams[:, active], n=self.n, axis=0) * self.syn_win[:, None]
        return y

    @staticmethod
    def _smooth_bins(x: np.ndarray) -> np.ndarray:
        """Light smoothing across neighbouring bins (reduces 'musical noise')."""
        out = x.copy()
        out[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        return out

    @staticmethod
    def _unit(z: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        m = np.abs(z)
        u = np.where(m > 1e-12, z / np.maximum(m, 1e-12), 0.0)
        if fallback is not None:
            u = np.where(m > 1e-12, u, fallback)
        return u
