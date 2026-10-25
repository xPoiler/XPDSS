"""Frequency-domain (STFT) stereo upmixer.

Each STFT bin of the stereo input is split into a *direct* and an *ambient*
part by principal components: the principal eigenvector v of the smoothed
2x2 L/R covariance gives the direction of the dominant source, the smaller
eigenvalue the ambient power. The direct part is the bin's projection on v,
weighted by the direct-to-total power ratio; ambient = input - direct, so
the decomposition is complementary and nothing is lost. Every output is a
linear combination of the original L/R bins with weights from smoothed
statistics only.

The direct part of every bin is then re-panned across the front L/C/R
speakers from its own panning position, so several sources at different
frequencies each land at their own place in the front stage. Only the
ambient part is meant for the surrounds.

Musical noise ("low-bitrate" twinkle) is avoided by moving a bin only as
much as its direction can be trusted: a confidence from the L/R coherence
measured over perceptual bands (~1/8 octave) and smoothed over 300 ms.
Noise-like content (reverb, cymbals, crowds) therefore stays where it was
instead of being scattered bin by bin between L, C and R; directions
themselves come from narrow-band statistics, so close notes stay separate.

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
CONF_POWER = 3     # confidence = coherence ** CONF_POWER


class SpectralUpmixer:
    def __init__(self, sample_rate: int, fft_size: int | None = None,
                 stats_tau: float = 0.10, gaming_tau: float = 0.04, debias: float = 2.0,
                 bass_lo_hz: float = 150.0, bass_hi_hz: float = 250.0, band_fraction: float = 0.06):
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
        self.conf_alpha = 1.0 - math.exp(-dt / 0.3)
        self.band_move = 1.0     # how far low-confidence bins follow their band direction (0 = stay put)
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

        # Perceptual frequency bands for the confidence: a moving average whose
        # width grows with frequency (+/- ~band_fraction * f * 1.7, at least
        # one bin), computed with running sums (O(bins)). Per-bin decisions on
        # noise-like content (reverb, cymbals) otherwise fluctuate
        # independently and turn into "musical noise".
        bin_hz = self.sample_rate / self.n
        half = np.maximum(1, np.round(1.7 * band_fraction * freqs / bin_hz)).astype(int)
        k = np.arange(self.bins)
        self._band_lo = np.clip(k - half, 0, self.bins - 1)
        self._band_hi = np.clip(k + half, 0, self.bins - 1) + 1
        self._band_len = (self._band_hi - self._band_lo).astype(np.float64)

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
        self._prr = np.zeros(self.bins, dtype=np.float32)
        self._plr = np.zeros(self.bins, dtype=np.complex64)
        self._primed = False
        self._conf = None

    # ── streaming ─────────────────────────────────────────────────────
    def process(self, stereo: np.ndarray, center_spread: float = 0.0, width: float = 1.0,
                center: bool = True, steering: int = STEER_OFF, split_sides_backs: bool = False,
                envelopment: float = 0.0, wrap: float = 0.0, hard_pan_sides: bool = False,
                wide_sides: bool = False) -> np.ndarray:
        """Feed (frames, 2) samples, get (frames, N_STREAMS) streams delayed by `latency`.

        Streams: direct L/R/C (front stage), ambient L/R, and the steered
        rear point sources SIDE_L/R and BACK_L/R (zero unless steering is
        on; BACK_* only with split_sides_backs, i.e. a 7.1 layout).
        center_spread 0 = sources panned across L/C/R, 1 = no center (direct
        stays a phantom image in L/R). width scales the panning positions.
        envelopment (0..1): how much partly-coherent sound (stereo reverb,
        wide pads, room) counts as ambience instead of direct sound.
        wrap (0..1): how far sources panned wide in the mix extend toward the
        side surrounds (SIDE_L/R streams), like a wider front stage.
        hard_pan_sides: sources panned (nearly) hard left/right play from the
        side surrounds instead of the fronts (ramp from 60 % to 95 % pan).
        wide_sides: stereo-widened sources (L/R partly out of phase, 45-120
        degrees: chorused guitars, wide synths, stereo FX) play from the side
        surrounds as sources instead of becoming front / ambience wash.
        """
        frames = stereo.shape[0]
        spread = 1.0 if not center else float(np.clip(center_spread, 0.0, 1.0))
        self._pending = np.concatenate((self._pending, np.asarray(stereo, dtype=np.float64)))
        # Collect every complete hop of this block and process those frames
        # together (vectorised over frames: far less per-call overhead)
        frame_list = []
        while self._pending.shape[0] >= self.hop:
            hop_in = self._pending[:self.hop]
            self._pending = self._pending[self.hop:]
            self._frame = np.concatenate((self._frame[self.hop:], hop_in))
            frame_list.append(self._frame)
        produced = []
        if frame_list:
            ys = self._process_frame(np.stack(frame_list), spread, float(width), int(steering), bool(split_sides_backs),
                                     float(np.clip(envelopment, 0.0, 1.0)), float(np.clip(wrap, 0.0, 1.0)),
                                     bool(hard_pan_sides), bool(wide_sides))
            for y in ys:
                self._ola += y
                produced.append(self._ola[:self.hop].copy())
                self._ola = np.concatenate((self._ola[self.hop:], np.zeros((self.hop, N_STREAMS))))
        if produced:
            self._out = np.concatenate([self._out] + produced)
        out = self._out[:frames]
        self._out = self._out[frames:]
        return out

    # ── STFT frames ───────────────────────────────────────────────────
    def _process_frame(self, frames: np.ndarray, spread: float, width: float,
                       steering: int = STEER_OFF, split: bool = False,
                       envelopment: float = 0.0, wrap: float = 0.0, hard_sides: bool = False,
                       wide_sides: bool = False) -> np.ndarray:
        """frames: (F, n, 2) consecutive analysis frames -> (F, n, N_STREAMS)."""
        X = np.fft.rfft(frames * self.win[None, :, None], axis=1).astype(np.complex64)
        L, R = X[..., 0], X[..., 1]                       # (F, bins)

        # Recursive per-bin statistics (seeded on the first frame); the
        # recursion is sequential over frames, everything else is vectorised.
        pll_i, prr_i, plr_i = np.abs(L) ** 2, np.abs(R) ** 2, L * np.conj(R)
        n_frames = L.shape[0]
        raw = np.empty((n_frames, 4, self.bins), dtype=np.float32)
        a = self.alpha_gaming if steering == STEER_GAMING else self.alpha
        for f in range(n_frames):
            if not self._primed:
                self._pll, self._prr, self._plr = pll_i[f], prr_i[f], plr_i[f]
                self._primed = True
            else:
                self._pll = self._pll + a * (pll_i[f] - self._pll)
                self._prr = self._prr + a * (prr_i[f] - self._prr)
                self._plr = self._plr + a * (plr_i[f] - self._plr)
            raw[f, 0], raw[f, 1] = self._pll, self._prr
            raw[f, 2], raw[f, 3] = self._plr.real, self._plr.imag
        csum = np.concatenate((np.zeros((n_frames, 4, 1), dtype=np.float32), np.cumsum(raw, axis=-1)), axis=-1)
        banded = (csum[..., self._band_hi] - csum[..., self._band_lo]) / self._band_len
        narrow = raw.copy()
        narrow[..., 1:-1] = 0.25 * raw[..., :-2] + 0.5 * raw[..., 1:-1] + 0.25 * raw[..., 2:]
        pll, prr = narrow[:, 0], narrow[:, 1]
        plr = narrow[:, 2] + 1j * narrow[:, 3]
        b_pll, b_prr = banded[:, 0], banded[:, 1]
        b_plr = banded[:, 2] + 1j * banded[:, 3]
        debias = self.debias
        if steering == STEER_GAMING:
            debias = self.debias_factor * self.alpha_gaming / (2.0 - self.alpha_gaming)
        if debias:
            b_mag2 = np.abs(b_plr) ** 2
            b_pow = b_pll * b_prr
            b_coh2 = np.clip(b_mag2 / np.maximum(b_pow, EPS), 0.0, 1.0)
            b_corr = np.maximum(b_mag2 - debias * (1.0 - b_coh2) ** 2 * b_pow, 0.0)
            b_plr = b_plr * np.sqrt(b_corr / np.maximum(b_mag2, EPS))
            # The spurious part vanishes for fully coherent bins, so the
            # correction scales with (1 - coherence^2)^2: mono stays mono.
            plr_mag2 = np.abs(plr) ** 2
            power = pll * prr
            coh2 = np.clip(plr_mag2 / np.maximum(power, EPS), 0.0, 1.0)
            corrected = np.maximum(plr_mag2 - debias * (1.0 - coh2) ** 2 * power, 0.0)
            plr = plr * np.sqrt(corrected / np.maximum(plr_mag2, EPS))

        # Primary-ambient decomposition by principal components (symmetric,
        # no reference channel): v = principal eigenvector of the smoothed
        # 2x2 covariance, lambda2 = ambient power. The direct part of a bin is
        # its projection on v, weighted by the direct-to-total power ratio.
        # Every output is a *linear* combination of the original L/R bins with
        # weights from smoothed statistics only, so the per-bin gains change
        # slowly: no frame-by-frame magnitude rebuilding (musical noise).
        half_diff = 0.5 * (pll - prr)
        root = np.sqrt(half_diff ** 2 + np.abs(plr) ** 2)
        lam1 = 0.5 * (pll + prr) + root
        lam2 = np.maximum(0.5 * (pll + prr) - root, 0.0)
        va1, va2 = lam1 - prr, np.conj(plr)
        vb1, vb2 = plr, lam1 - pll
        use_a = (np.abs(va1) ** 2 + np.abs(va2) ** 2) >= (np.abs(vb1) ** 2 + np.abs(vb2) ** 2)
        v1 = np.where(use_a, va1, vb1)
        v2 = np.where(use_a, va2, vb2)
        norm = np.sqrt(np.abs(v1) ** 2 + np.abs(v2) ** 2)
        degenerate = norm < 1e-12
        v1 = np.where(degenerate, math.sqrt(0.5), v1 / np.maximum(norm, 1e-12))
        v2 = np.where(degenerate, math.sqrt(0.5), v2 / np.maximum(norm, 1e-12))

        w_direct = np.clip((lam1 - lam2) / np.maximum(lam1, EPS), 0.0, 1.0)
        # Confidence in the bin's direction: only coherent (source-like) bins
        # are moved; noise-like bins keep their direct part where it is.
        coherence = np.abs(b_plr) / np.sqrt(np.maximum(b_pll * b_prr, EPS))
        conf_now = np.clip(coherence, 0.0, 1.0) ** CONF_POWER
        # Slow (300 ms) smoothing: source-vs-noise is a slow property; a fast
        # confidence would follow beats between close notes and modulate them.
        confidence = np.empty_like(conf_now)
        for f in range(n_frames):
            if self._conf is None:
                self._conf = conf_now[f]
            else:
                self._conf = self._conf + self.conf_alpha * (conf_now[f] - self._conf)
            confidence[f] = self._conf
        if envelopment > 0.0:
            # Envelopment: partly coherent sound (stereo reverb, wide pads,
            # room) is mostly ambience; PCA alone counts only the uncorrelated
            # remainder as ambient. Scale the direct weight by the slow band
            # coherence (= cbrt(confidence)) ** (2 * envelopment): coherent
            # sources keep their weight, diffuse-ish sound becomes ambience.
            # Smoothed statistics only, so no musical noise.
            w_direct = w_direct * np.cbrt(confidence) ** (2.0 * envelopment)
        s_dir = w_direct * (np.conj(v1) * L + np.conj(v2) * R)   # direct source signal
        DL = v1 * s_dir
        DR = v2 * s_dir
        AL = L - DL
        AR = R - DR

        a1, a2 = np.abs(v1), np.abs(v2)
        # amplitude ratio R/L as an angle: 0 = hard left, pi/4 = center, pi/2 = hard right
        angle = np.arctan2(a2, a1)
        # Low-confidence bins (noise-like: snare and cymbal tops, sibilance,
        # breath) take the direction of their band instead of their own: it is
        # smooth across frequency and time, so moving them cannot twinkle, and a
        # source's upper spectrum lands where its body does instead of staying
        # behind as a phantom (a voice/snare split between C and L/R by frequency).
        b_half = 0.5 * (b_pll - b_prr)
        b_root = np.sqrt(b_half ** 2 + np.abs(b_plr) ** 2)
        b_lam1 = 0.5 * (b_pll + b_prr) + b_root
        bw1 = np.maximum(b_lam1 - b_prr, 0.0)       # |v1|^2 and |v2|^2 up to a common factor
        bw2 = np.maximum(b_lam1 - b_pll, 0.0)
        angle_band = np.arctan2(np.sqrt(bw2), np.sqrt(bw1))
        angle = confidence * angle + (1.0 - confidence) * angle_band

        # Re-pan the direct part across L / C / R (constant power)
        pan = (angle - math.pi / 4) / (math.pi / 4)   # -1 left .. +1 right
        pan = np.clip(pan * width, -1.0, 1.0)
        left_half = pan < 0
        t = np.where(left_half, pan + 1.0, pan) * (math.pi / 2)
        g_l = np.where(left_half, np.cos(t), 0.0)
        g_c = np.where(left_half, np.sin(t), np.cos(t))
        g_r = np.where(left_half, 0.0, np.sin(t))

        # Unit phasors (zero where undefined: those bins get zero gain anyway)
        u_l = v1 / (a1 + 1e-20)
        u_r = v2 / (a2 + 1e-20)
        vc = v1 + v2
        u_c = vc / (np.abs(vc) + 1e-20)
        mag = s_dir                                      # complex direct signal, |s_dir| = total direct amplitude

        # Front/back from the inter-channel phase of the direct part:
        # rear = 0 for phase differences up to 90 deg, 1 at 180 deg.
        cos_phase = np.real(np.conj(v1) * v2) / np.maximum(a1 * a2, EPS)
        rear = np.clip(-cos_phase, 0.0, 1.0)
        rear = np.where(np.minimum(a1, a2) < 1e-6, 0.0, rear)   # hard-panned: no phase info
        rear_raw = rear
        if steering == STEER_GAMING:
            rear = np.clip((rear - 0.25) / 0.5, 0.0, 1.0)  # decisive front/back
        # Below ~150 Hz nothing is moved (bass stays in L/R as it is); across
        # the 150-250 Hz handover every decision fades in with w. Folding w into
        # the decisions (instead of crossfading the processed and the original
        # signal) keeps the handover at constant power: no dip in the body of
        # voices and instruments there.
        w = self.process_weight
        rear = rear * w
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

        # (weighted by the band's slow-smoothed coherence: fully diffuse sound
        # stays where it is, and beating close notes do not modulate the move)
        move = (1.0 - spread) * (confidence + (1.0 - confidence) * self.band_move * np.cbrt(confidence)) * w
        # Partly moved bins blend two pannings of the same signal; an amplitude
        # blend of two different speaker sets loses power (up to 3 dB at 50/50:
        # duller transients and voices), so the blend is renormalised to the
        # direct power. Gains come from smoothed statistics only.
        c_l = move * g_l * u_l + (1.0 - move) * v1
        c_r = move * g_r * u_r + (1.0 - move) * v2
        c_c = move * g_c
        norm = 1.0 / np.sqrt(np.maximum(np.abs(c_l) ** 2 + np.abs(c_r) ** 2 + c_c ** 2, 0.25))
        out_l = (move * lcr_l + (1.0 - move) * DL * f_amp) * norm + keep_l
        out_r = (move * lcr_r + (1.0 - move) * DR * f_amp) * norm + keep_r
        out_c = move * lcr_c * norm

        side_l = side_r = back_l = back_r = 0.0
        if wrap > 0.0 or hard_sides:
            # Wrap: sources panned wide in the mix extend from the front
            # speaker toward the side surround (constant power, up to an equal
            # split: a hard-panned source then sits around 60-70 degrees
            # instead of 30). Positions depend on the source's pan only, so
            # nothing moves around; the stage gets wider.
            wr = wrap * np.clip((np.abs(pan) - 0.3) / 0.7, 0.0, 1.0) * (math.pi / 4)
            if hard_sides:
                # Hard pans to the sides: from 60 % pan the source moves on to
                # the side surround, entirely from 95 % (hard-panned sounds).
                wr = np.maximum(wr, np.clip((np.abs(pan) - 0.6) / 0.35, 0.0, 1.0) * (math.pi / 2))
            wr = wr * w
            cw, sw = np.cos(wr), np.sin(wr)
            side_l = sw * out_l
            side_r = sw * out_r
            out_l = cw * out_l
            out_r = cw * out_r
        if wide_sides:
            # Stereo-widened direct sources (inter-channel phase 45-120 deg)
            # move from the front stage to the side surrounds (constant power),
            # left/right by their panning; centred wide sources become a
            # phantom between the two side surrounds. Coherent sources only
            # (confidence), so reverb and room stay ambience.
            phase = np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))
            amt = np.clip((phase - 45.0) / 75.0, 0.0, 1.0) * confidence * w
            amt = np.where(np.minimum(a1, a2) < 1e-6, 0.0, amt)
            psi = amt * (math.pi / 2)
            cps, sps = np.cos(psi), np.sin(psi)
            c_share = out_c * (sps * math.sqrt(0.5))
            side_l = side_l + sps * out_l + c_share
            side_r = side_r + sps * out_r + c_share
            out_l, out_r, out_c = cps * out_l, cps * out_r, cps * out_c
        if steering != STEER_OFF:
            # Rear point source, panned left/right by its position (constant power)
            th = (pan + 1.0) * (math.pi / 4)
            rear_mag = mag * r_amp * confidence
            src_l = np.cos(th) * rear_mag * u_l     # one common phase: a coherent
            src_r = np.sin(th) * rear_mag * u_l     # rear image, not an anti-phase one
            # The rear part of low-confidence bins (noise-like content, and any
            # source while its statistics settle after a front->rear move) is
            # not re-panned bin by bin, but it still belongs behind: it goes to
            # the surrounds as its own L/R pair with the right channel's
            # polarity flipped (same-polarity pair, levels untouched), instead
            # of staying in the fronts as an anti-phase phantom. Linear and
            # constant power with the front part (f_amp^2 + r_amp^2 = 1).
            low = (1.0 - confidence) * r_amp
            src_l = src_l + low * DL
            src_r = src_r - low * DR
            if split and steering == STEER_GAMING:
                # 7.1: beside (~110-120 deg phase) -> sides, behind (180 deg) -> backs
                behind = np.clip((rear_raw - 0.5) / 0.5, 0.0, 1.0) * (math.pi / 2)
                g_side = np.cos(behind)
                g_back = np.sin(behind)
            elif split:
                g_side, g_back = 0.8, 0.6
            else:
                g_side, g_back = 1.0, 0.0
            side_l, side_r = side_l + g_side * src_l, side_r + g_side * src_r
            back_l, back_r = g_back * src_l, g_back * src_r

        streams = np.zeros((n_frames, self.bins, N_STREAMS), dtype=np.complex64)
        streams[..., DIRECT_L] = out_l + (1.0 - w) * AL   # unprocessed ambience stays in L/R
        streams[..., DIRECT_R] = out_r + (1.0 - w) * AR
        streams[..., DIRECT_C] = out_c
        streams[..., AMBIENT_L] = w * AL
        streams[..., AMBIENT_R] = w * AR
        streams[..., SIDE_L] = side_l
        streams[..., SIDE_R] = side_r
        streams[..., BACK_L] = back_l
        streams[..., BACK_R] = back_r

        # Only synthesise the streams in use (the rear ones are silent unless
        # steering is on; the backs only on 7.1)
        active = list(range(AMBIENT_R + 1))
        if steering != STEER_OFF or wrap > 0.0 or hard_sides or wide_sides:
            active += [SIDE_L, SIDE_R]
        if steering != STEER_OFF and split:
            active += [BACK_L, BACK_R]
        y = np.zeros((n_frames, self.n, N_STREAMS))
        y[:, :, active] = np.fft.irfft(streams[:, :, active], n=self.n, axis=1) * self.syn_win[None, :, None]
        return y
