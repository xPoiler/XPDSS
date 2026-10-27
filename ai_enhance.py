"""Optional AI-guided tonal balance; no lookahead or audio routing.

The existing separator supplies content evidence (how much of the signal is
vocals, drums, bass, other). The mix's long-term octave balance is compared
with a reference curve and partially corrected with five EQ bands, but only in
ways the stems justify: lows follow a real bass part, presence follows a voice,
cymbals are spared from harshness cuts and hiss is never boosted. This is a
heuristic tonal-balance corrector, not a trained mastering or restoration model.
"""
import math
import time

import numpy as np
from scipy.signal import butter, sosfilt

# (kind, centre Hz) of the five correction bands, in snapshot order.
FILTERS = (('low', 90.), ('peak', 300.), ('peak', 3000.), ('peak', 5800.), ('high', 11000.))
LIMITS_LO = np.array([-3., -4., -2., -4., -2.])
LIMITS_HI = np.array([3., 1., 3., 0., 2.5])
# Analysis bands (Hz) and the reference per-octave level of each relative to
# the 450–1400 Hz body band. Reference is a typical modern mix balance.
BANDS = ((40, 120), (200, 450), (450, 1400), (2000, 4000), (4500, 7500), (9000, 16000))
REFERENCE_DB = np.array([6., 2., 0., -5., -10., -17.])
NEUTRAL = (0.,) * len(FILTERS)


def analysis_fold(block):
    """Analysis only: preserve fronts, include center/surrounds, exclude LFE."""
    if block.shape[1] == 1:
        return np.repeat(block, 2, axis=1)
    if block.shape[1] <= 2:
        return block
    x = block[:, :2].copy()
    x += block[:, 2:3] * .707
    for left in (4, 6):
        if block.shape[1] > left:
            x[:, 0] += block[:, left] * .5
        if block.shape[1] > left + 1:
            x[:, 1] += block[:, left + 1] * .5
    return x * .4


def _biquad(kind, f0, gain_db, rate, q=.9):
    """RBJ cookbook section; exactly identity at 0 dB."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / rate
    cw, sw = math.cos(w0), math.sin(w0)
    if kind == 'peak':
        alpha = sw / (2 * q)
        b = (1 + alpha*A, -2*cw, 1 - alpha*A)
        a = (1 + alpha/A, -2*cw, 1 - alpha/A)
    else:
        sa = 2 * math.sqrt(A) * sw / 2 * math.sqrt(2)   # shelf slope S = 1
        if kind == 'low':
            b = (A*((A+1) - (A-1)*cw + sa), 2*A*((A-1) - (A+1)*cw), A*((A+1) - (A-1)*cw - sa))
            a = ((A+1) + (A-1)*cw + sa, -2*((A-1) + (A+1)*cw), (A+1) + (A-1)*cw - sa)
        else:
            b = (A*((A+1) + (A-1)*cw + sa), -2*A*((A-1) + (A+1)*cw), A*((A+1) + (A-1)*cw - sa))
            a = ((A+1) - (A-1)*cw + sa, 2*((A-1) - (A+1)*cw), (A+1) - (A-1)*cw - sa)
    return [b[0]/a[0], b[1]/a[0], b[2]/a[0], 1., a[1]/a[0], a[2]/a[0]]


class EnhanceGuidance:
    """Worker-owned state; an immutable snapshot is published to rendering."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._pending = []
        self._bands = None      # smoothed per-octave band levels, dB
        self._shares = None     # smoothed stem energy shares (drums, bass, other, vocals)
        self._air_flatness = 0.
        self.snapshot = (0., NEUTRAL)
        self.updates = 0
        self.error = ''

    def observe(self, stems):
        self._pending.append(stems.copy())
        if len(self._pending) < 16:
            return
        s = np.concatenate(self._pending, axis=1).astype(np.float64)
        self._pending.clear()
        mix = s.sum(axis=0)
        level = float(np.sqrt(np.mean(mix*mix)))
        if level > .003:
            energy = np.mean(s*s, axis=(1, 2))
            shares = energy / (energy.sum() + 1e-12)
            power = np.mean(abs(np.fft.rfft(mix * np.hanning(len(mix))[:, None], axis=0))**2, axis=1)
            hz = np.fft.rfftfreq(len(mix), 1/44100)
            bands = np.array([power[(hz >= a) & (hz < b)].sum() / math.log2(b/a) for a, b in BANDS])
            bands = 10*np.log10(bands + 1e-20)
            air = power[(hz >= 9000) & (hz < 16000)] + 1e-20
            flat = math.exp(float(np.mean(np.log(air)))) / float(np.mean(air))
            # Long-term balance (~1.5 s); stem shares a little faster (~0.5 s).
            if self._bands is None:
                self._bands, self._shares, self._air_flatness = bands, shares, flat
            else:
                self._bands += .03 * (bands - self._bands)
                self._shares += .09 * (shares - self._shares)
                self._air_flatness += .05 * (flat - self._air_flatness)
            targets = self._targets()
        else:
            # Silence: hold the last decision rather than re-learning from noise.
            targets = self.snapshot[1] if self.updates else NEUTRAL
        self.snapshot = (time.monotonic(), tuple(float(t) for t in targets))
        self.updates += 1

    def _targets(self):
        drums, bass, other, voice = self._shares
        rel = self._bands - self._bands[2]
        # Indexed like BANDS: 0 low, 1 mud, 2 body (reference), 3 presence, 4 harsh, 5 air.
        err = rel - REFERENCE_DB          # > 0: band too loud, < 0: too quiet
        vocal = float(np.clip((voice - .15) / .35, 0, 1))
        t = np.zeros(len(FILTERS))
        # Lows: fill only when a bass part exists; trim boom only when the
        # low end does not come from bass/drums (speech, rumble, room boom).
        if err[0] < -2 and bass > .08:
            t[0] = .5 * (-err[0] - 2) * min(1., bass / .2)
        elif err[0] > 3 and bass + drums < .25:
            t[0] = -.5 * (err[0] - 3)
        # Low-mid mud masks voices and instruments; cut harder when a voice is present.
        t[1] = -.6 * max(0., err[1] - 2) * (.7 + .3 * vocal) - 1.2 * vocal
        # Presence: voice clarity lift, or a cut if the band is already forward.
        if err[3] > 3:
            t[2] = -.5 * (err[3] - 3)
        else:
            t[2] = vocal * (1.5 + .5 * max(0., -err[3] - 1))
        # Harshness: cut sustained excess, sparing cymbals/percussion.
        t[3] = -.6 * max(0., err[4] - 2) * (1 - min(.7, 2 * drums))
        # Air: never lift an empty band (band-limited source) or hiss; trim
        # noise-like top end that has no percussion behind it.
        hiss = self._air_flatness > .55 and drums < .1
        if hiss:
            t[4] = -1.5
        elif -20 < err[5] < -2:
            t[4] = .4 * (-err[5] - 2)
        return np.clip(t, LIMITS_LO, LIMITS_HI)


class EnhanceProcessor:
    """Linked full-range channels, unchanged LFE, 40 ms toggle fade, no queue."""
    queued_delay_samples = 0

    def __init__(self, rate, channels):
        self.rate, self.channels = rate, channels
        self.state = np.zeros((len(FILTERS), 2, channels))
        self.gain = np.zeros(len(FILTERS))
        self.wet = 0.

    def process(self, block, guidance, enabled=True):
        if not enabled and self.wet == 0:
            return block
        stamp, target = guidance.snapshot if guidance is not None else (0., NEUTRAL)
        if not enabled or time.monotonic()-stamp > .5 or getattr(guidance, 'error', ''):
            target = NEUTRAL
        target = np.clip(np.resize(np.asarray(target, float), len(FILTERS)), LIMITS_LO, LIMITS_HI)
        n = len(block)
        # Slow controls keep historical analysis from modulating transients.
        self.gain = self.gain + (target-self.gain)*(1-math.exp(-n/(self.rate*.20)))
        sos = np.array([_biquad(kind, f0, g, self.rate) for (kind, f0), g in zip(FILTERS, self.gain)])
        y, self.state = sosfilt(sos, block, axis=0, zi=self.state)
        step = (1 if enabled else -1) / max(1, round(self.rate*.040))
        wet = np.clip(self.wet + step*np.arange(1, n+1), 0, 1)
        self.wet = float(wet[-1])
        y = (block + (y-block)*wet[:, None]).astype(np.float32)
        if self.channels >= 6:
            y[:, 3] = block[:, 3]
        return y


def _one_pole(seconds, rate):
    c = math.exp(-1 / max(1., seconds * rate))
    return np.array([1 - c]), np.array([1., -c])


class StemEnhancer:
    """AI Live only: remaster the separated stems before they are placed.

    The stems arrive already time-aligned with the playing audio, so this adds
    no latency. Every stage is causal:
      vocals  low-mid cleanup, presence and air, levelling, +level (up front)
      drums   transient punch, kick weight
      bass    harmonics an octave up (audible on small speakers), low shelf
      other   air and stereo width
    The processed stems are slowly matched to the original loudness, so on/off
    compares clarity rather than volume. The residual is never touched.
    """
    queued_delay_samples = 0

    def __init__(self, rate):
        self.rate = rate
        self.amount = None
        self.wet = 0.
        self.match_gain = 1.
        self._power = None
        lp = butter(2, 160, btype='lowpass', fs=rate, output='sos')
        hp = butter(2, 200, btype='highpass', fs=rate, output='sos')
        self._bass_split = (lp, hp)
        self._fast = _one_pole(.0015, rate)
        self._slow = _one_pole(.060, rate)
        self._lev_fast = _one_pole(.030, rate)
        self._lev_slow = _one_pole(.800, rate)
        self._bass_env = _one_pole(.030, rate)
        self._zi = {}

    def _eq(self, a):
        r = self.rate
        return {
            'vocals': np.array([_biquad('peak', 250., -3*a, r, .8), _biquad('peak', 3200., 5*a, r, .8),
                                _biquad('high', 9000., 3*a, r)]),
            'drums': np.array([_biquad('low', 70., 2.5*a, r), _biquad('high', 8000., 1.5*a, r)]),
            'bass': np.array([_biquad('low', 60., 2*a, r)]),
            'other': np.array([_biquad('high', 8000., 3*a, r), _biquad('peak', 300., -1.5*a, r, .8)]),
        }

    def _filter(self, key, sos, x, axis=-1):
        zi = self._zi.get(key)
        if zi is None:
            shape = list(x.shape)
            shape[axis] = 2
            zi = np.zeros((len(sos),) + tuple(shape))
        y, self._zi[key] = sosfilt(sos, x, axis=axis, zi=zi)
        return y

    def _lfilter(self, key, ba, x):
        from scipy.signal import lfilter
        zi = self._zi.get(key)
        if zi is None:
            zi = np.zeros(x.shape[:-1] + (1,))
        y, self._zi[key] = lfilter(ba[0], ba[1], x, axis=-1, zi=zi)
        return y

    def process(self, stems, amount, enabled=True):
        """stems: (4, 2, frames) drums, bass, other, vocals. Returns new stems."""
        if not enabled and self.wet == 0:
            return stems
        a = float(np.clip(amount, 0, 1.5))
        if a != self.amount:
            self.amount, self._sos = a, self._eq(a)
        x = stems.astype(np.float64)
        drums, bass, other, vocals = x
        n = x.shape[-1]
        db = math.log(10) / 20

        # Vocals: EQ, then level the phrase dynamics (2:1 around the slow level).
        v = self._filter('v_eq', self._sos['vocals'], vocals)
        p = np.mean(v*v, axis=0)
        fast = self._lfilter('v_fast', self._lev_fast, p[None])[0]
        slow = self._lfilter('v_slow', self._lev_slow, p[None])[0]
        lev = np.clip(.25*a*10*np.log10((slow+1e-10)/(fast+1e-10)), -6, 6)
        v = v * np.exp((lev + 4*a)*db)[None]

        # Drums: fast/slow envelope ratio drives attack emphasis.
        d = self._filter('d_eq', self._sos['drums'], drums)
        m = np.abs(d).mean(axis=0)
        ef = self._lfilter('d_fast', self._fast, m[None])[0]
        es = self._lfilter('d_slow', self._slow, m[None])[0]
        punch = np.clip(10*a*np.log10((ef+1e-7)/(es+1e-7)), -3, 8)
        d = d * np.exp((punch + a)*db)[None]

        # Bass: level-independent soft saturation of the low band; keep only
        # the generated harmonics (>200 Hz) and add them to the bass stem.
        lp, hp = self._bass_split
        low = self._filter('b_lp', lp, bass)
        env = self._lfilter('b_env', self._bass_env, np.abs(low).mean(axis=0)[None])[0] + 1e-5
        sat = np.tanh(low / env * .8) * env / .8
        harm = self._filter('b_hp', hp, sat)
        b = self._filter('b_eq', self._sos['bass'], bass) + 1.6*a*harm
        b = b * np.exp(a*db)

        # Other: air and width.
        o = self._filter('o_eq', self._sos['other'], other)
        mid, side = (o[0]+o[1])*.5, (o[0]-o[1])*.5*(1+.6*a)
        o = np.stack((mid+side, mid-side))

        y = np.stack((d, b, o, v))
        # Slow loudness match (~1.5 s) to the original stems, then the toggle fade.
        powers = np.array([np.mean(x.sum(axis=0)**2), np.mean(y.sum(axis=0)**2)])
        k = 1 - math.exp(-n / (self.rate*1.5))
        self._power = powers if self._power is None else self._power + k*(powers-self._power)
        target = float(np.clip(math.sqrt((self._power[0]+1e-12)/(self._power[1]+1e-12)), .5, 1.12))
        gain = np.linspace(self.match_gain, target, n, endpoint=False)
        self.match_gain = target
        y *= gain
        step = (1 if enabled else -1) / max(1, round(self.rate*.040))
        wet = np.clip(self.wet + step*np.arange(1, n+1), 0, 1)
        self.wet = float(wet[-1])
        return (x + (y-x)*wet).astype(np.float32)
