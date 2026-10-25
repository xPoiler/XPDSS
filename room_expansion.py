"""Room expansion: a light late reverb for the surrounds (IMMERSIVE mode).

Commercial upmixers (Auro-Matic, the "room" modes of AVRs) add a diffuse,
decorrelated late field around the listener so that even dry mixes feel
like they are played in a larger space. This is a 4-line feedback delay
network (Hadamard feedback, per-line damping, ~1 s RT60) fed with the
band-limited program, with a pre-delay so it arrives after the direct sound
and early ambience. Each output is a different combination of the lines,
so the speakers get mutually decorrelated tails.

Every delay line is longer than any processing chunk, so the feedback for a
chunk only depends on samples already written: the network runs on whole
chunks with numpy (no per-sample Python loop).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

# Delay lengths at 48 kHz (mutually prime, 33-54 ms)
_DELAYS_48K = (1601, 1867, 2207, 2593)
# Output combinations of the 4 lines (orthogonal rows: decorrelated outputs)
_TAPS = np.array([[1, 1, 1, 1],
                  [1, -1, 1, -1],
                  [1, 1, -1, -1],
                  [1, -1, -1, 1]], dtype=np.float64) * 0.5


class RoomExpander:
    def __init__(self, sample_rate: int, rt60: float = 1.0, predelay_ms: float = 20.0,
                 damping_hz: float = 5000.0):
        self.sr = int(sample_rate)
        scale = self.sr / 48000.0
        self.delays = np.array([max(64, int(round(d * scale))) for d in _DELAYS_48K])
        self.max_chunk = int(self.delays.min())
        self.gains = 10.0 ** (-3.0 * self.delays / (rt60 * self.sr))
        self.H = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float64)
        a = np.exp(-2.0 * np.pi * damping_hz / self.sr)
        self.damp_b, self.damp_a = np.array([1.0 - a]), np.array([1.0, -a])
        # input band: no boomy low end, no fizzy top
        self.in_sos = butter(2, [200.0 / (self.sr / 2), min(0.95, 8000.0 / (self.sr / 2))], btype='band', output='sos')
        self.predelay = int(predelay_ms * self.sr / 1000.0)
        self.reset()

    def reset(self) -> None:
        self.buf_len = int(self.delays.max()) + self.max_chunk + 1
        self.bufs = np.zeros((4, self.buf_len))
        self.ptr = 0
        self.damp_zi = np.zeros((4, 1))
        self.in_zi = np.zeros((self.in_sos.shape[0], 2))
        self.pre = np.zeros(self.predelay)

    def process(self, mono: np.ndarray, n_out: int) -> np.ndarray:
        """mono (n,) -> (n, n_out) decorrelated late-reverb feeds (n_out <= 4)."""
        x, self.in_zi = sosfilt(self.in_sos, np.asarray(mono, dtype=np.float64), zi=self.in_zi)
        if self.predelay:
            ext = np.concatenate((self.pre, x))
            x, self.pre = ext[:len(x)], ext[len(x):]
        out = np.empty((len(x), n_out))
        pos = 0
        while pos < len(x):
            n = min(self.max_chunk, len(x) - pos)
            idx = (self.ptr + np.arange(n)) % self.buf_len
            ys = np.empty((4, n))
            for i in range(4):
                rd = (idx - self.delays[i]) % self.buf_len
                y, self.damp_zi[i] = lfilter(self.damp_b, self.damp_a, self.bufs[i, rd], zi=self.damp_zi[i])
                ys[i] = y * self.gains[i]
            fb = self.H @ ys
            for j in range(4):
                self.bufs[j, idx] = x[pos:pos + n] + fb[j]
            out[pos:pos + n] = (_TAPS[:n_out] @ ys).T
            self.ptr = (self.ptr + n) % self.buf_len
            pos += n
        return out
