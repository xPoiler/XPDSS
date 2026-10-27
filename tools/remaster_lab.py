"""Isolated remaster feasibility prototypes. Never imported by the application.

A: trained StemgenRT analysis guides bounded, causal filters. The control law
is hand designed, not a trained mastering model. B: pretrained RNNoise directly
enhances the waveform. Neither prototype is qualified for production playback.
"""
from pathlib import Path
import ctypes
import math
import time

import numpy as np
from scipy.signal import butter, sosfilt


class Guidance:
    """Called on the separator worker; publishes historical, smoothed controls."""
    def __init__(self):
        self.target = (0., 0., 0.)
        self.vocal_share = 0.
        self.updates = 0
        self._pending = []
        self.last_update = 0.

    def observe(self, stems):
        self._pending.append(np.asarray(stems).copy())  # [4,128,2]
        if len(self._pending) < 16:
            return
        s = np.concatenate(self._pending, axis=1)
        self._pending.clear()
        mix = s.sum(axis=0)
        energy = np.mean(s.astype(np.float64)**2, axis=(1,2))
        if np.mean(mix**2) < 1e-7:
            target = np.zeros(3)
        else:
            voice = float(np.clip(energy[3]/(energy.sum()+1e-12), 0, 1))
            self.vocal_share = voice
            power = abs(np.fft.rfft(mix*np.hanning(len(mix))[:,None], axis=0))**2
            hz = np.fft.rfftfreq(len(mix), 1/44100)
            def band(lo, hi):
                return float(power[(hz>=lo)&(hz<hi)].sum())
            total = band(80, 14000)+1e-12
            mud = band(180, 500)/total
            harsh = band(3000, 6500)/total
            # Broad tonal correction; neural voice evidence gates presence lift.
            target = np.array([-min(3., max(0., (mud-.26)*10)),
                               min(1.8, 2.2*voice)*max(0., 1-2*harsh),
                               -min(2.5, max(0., (harsh-.18)*8))])
        self.target = tuple(np.asarray(self.target)*.92+target*.08)
        self.updates += 1
        self.last_update = time.monotonic()


class ObservedModel:
    def __init__(self, model, guidance):
        self.model, self.guidance = model, guidance
        self.compute_s, self.p95_s = model.compute_s, model.p95_s

    def process(self, block):
        stems = self.model.process(block)
        self.guidance.observe(stems)
        return stems

    def reset(self):
        self.model.reset()

    def close(self):
        self.model.close()


class GuidedRemaster:
    """Zero queued audio; no future samples; linked processing across channels.

    Three fixed second-order band filters with slowly varying parallel gains.
    LFE is unchanged. On stale guidance (>0.5 s), correction glides to neutral.
    This does have frequency-dependent filter phase, not a flat delayed copy.
    """
    queued_delay_samples = 0

    def __init__(self, rate=48000, channels=2, amount=1.):
        self.rate, self.channels = rate, channels
        self.amount = float(np.clip(amount, 0, 1))
        self.filters = [butter(1, bounds, btype='bandpass', fs=rate, output='sos')
                        for bounds in [(180,500), (1500,3200), (3200,6500)]]
        self.states = [np.zeros((len(s),2,channels), np.float64) for s in self.filters]
        self.gain = np.zeros(3)

    def process(self, x, guidance=None, *, target=None, enabled=True):
        x = np.asarray(x, np.float32)
        if not enabled or self.amount == 0:
            return x.copy()
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError('Channel layout changed; create a fresh processor')
        if target is None:
            fresh = guidance is not None and time.monotonic()-guidance.last_update < .5
            target = guidance.target if fresh else (0.,0.,0.)
        target = np.clip(np.asarray(target, float), [-3,0,-2.5], [0,1.8,0])*self.amount
        finish = self.gain+(target-self.gain)*(1-math.exp(-len(x)/(self.rate*.15)))
        y = x.astype(np.float64)
        for k, sos in enumerate(self.filters):
            band, self.states[k] = sosfilt(sos, x, axis=0, zi=self.states[k])
            db = np.linspace(self.gain[k], finish[k], len(x), endpoint=False)
            y += band*(10**(db[:,None]/20)-1)
        self.gain = finish
        if self.channels >= 6:
            y[:,3] = x[:,3]
        return y.astype(np.float32)


class RNNoiseWaveform:
    """Direct C API, pretrained default model; no Python package installation."""
    def __init__(self, dll, channels=2):
        self.lib = ctypes.CDLL(str(Path(dll).resolve()))
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.rnnoise_process_frame.restype = ctypes.c_float
        self.lib.rnnoise_get_frame_size.restype = ctypes.c_int
        self.frame_size = self.lib.rnnoise_get_frame_size()
        self.states = [self.lib.rnnoise_create(None) for _ in range(channels)]
        if not all(self.states):
            self.close()
            raise RuntimeError('RNNoise allocation failed')

    def process(self, block):
        if block.shape != (self.frame_size, len(self.states)):
            raise ValueError('RNNoise requires one complete 480-frame block per channel')
        out = np.empty_like(block, dtype=np.float32)
        for ch, state in enumerate(self.states):
            x = np.ascontiguousarray(block[:,ch]*32768, dtype=np.float32)
            y = np.empty_like(x)
            self.lib.rnnoise_process_frame(state,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            out[:,ch] = y/32768
        return out

    def close(self):
        for s in self.states:
            if s:
                self.lib.rnnoise_destroy(s)
        self.states = []
