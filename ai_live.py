"""Opt-in, bounded-latency four-stem upmix backend (StemgenRT, MIT).

No model imports, sessions or workers are created until AI Live is requested.
The three published partitions preserve the original model weights and states.
App stem order is drums/bass/other/vocals; model hops emit the previous hop.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import math
import os
import threading
import time
import urllib.request

import numpy as np
from scipy.signal import firwin

MODEL_SR = 44100
HOP = 128
LIVE_RESERVE_MS = 17  # inference overlaps this input reserve; no second delay
SUPPORTED_RATES = (44100, 48000, 88200, 96000, 192000)
COMMIT = '7f42f68a31947d51da090ddc18e16726fe20a980'
MODEL_FILES = {
    'drums': (12267886, 'f2a3fc5dd90f45690536370f02d9b9377e6625ab08cb32b3e921371ad134d874'),
    'bass': (8660860, '1187b414f22344028035caba0a763d2211d13fe8bfce366c5f6b7ee4af77df40'),
    'vocals': (11019332, 'f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148'),
}


def default_model_dir():
    return Path(__file__).resolve().parent / 'models'


def model_path(directory, stem):
    return Path(directory) / f'stemgenrt-live-{stem}.onnx'


def models_present(directory=None):
    directory = directory or default_model_dir()
    return all(model_path(directory, stem).is_file() and
               model_path(directory, stem).stat().st_size == size
               for stem, (size, _) in MODEL_FILES.items())


def download_models(directory=None, progress=None):
    directory = Path(directory or default_model_dir())
    directory.mkdir(parents=True, exist_ok=True)
    for stem, (size, digest) in MODEL_FILES.items():
        path = model_path(directory, stem)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        tmp = path.with_suffix('.part')
        try:
            url = f'https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/{COMMIT}/model/parallel/{stem}.onnx'
            h, count = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > size:
                        raise ValueError('AI Live model exceeds published size')
                    h.update(chunk)
                    out.write(chunk)
                    if progress:
                        progress(f'{stem}: {count / 1e6:.1f}/{size / 1e6:.1f} MB')
            if count != size or h.hexdigest() != digest:
                raise ValueError('AI Live model checksum mismatch')
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


class StreamingResampler:
    """Causal rational FIR converter with continuous phase across callbacks."""
    def __init__(self, input_rate, output_rate):
        gcd = math.gcd(int(input_rate), int(output_rate))
        self.up, self.down = int(output_rate)//gcd, int(input_rate)//gcd
        half = 10 * max(self.up, self.down) if self.up != self.down else 0
        h = (firwin(2*half+1, 1/max(self.up, self.down), window=('kaiser', 5.0))*self.up
             if half else np.ones(1))
        self.taps = np.zeros((self.up, math.ceil(len(h)/self.up)), np.float32)
        for phase in range(self.up):
            part = h[phase::self.up]
            self.taps[phase, :len(part)] = part
        self.delay_output = half / self.down
        self.total = self.emitted = 0
        self.history = None

    def process(self, block):
        x = np.asarray(block, np.float32)
        if not len(x):
            return x.copy()
        if self.up == self.down:
            return x.copy()
        hist = self.history if self.history is not None else np.zeros((0, x.shape[1]), np.float32)
        base = self.total - len(hist)
        data = np.concatenate((hist, x))
        self.total += len(x)
        end = (self.total*self.up + self.down-1)//self.down
        m = np.arange(self.emitted, end, dtype=np.int64)*self.down
        indices = m[:, None]//self.up - np.arange(self.taps.shape[1])[None, :] - base
        valid = (indices >= 0) & (indices < len(data))
        values = data[np.clip(indices, 0, len(data)-1)]
        weights = self.taps[m % self.up] * valid
        out = np.einsum('nkc,nk->nc', values, weights, optimize=False)
        self.history = data[-(self.taps.shape[1]+2):].copy()
        self.emitted = end
        return out.astype(np.float32)


class ParallelModel:
    def __init__(self, directory):
        import onnxruntime as ort
        self.branches = []
        self.pool = None
        for stem, (_, digest) in MODEL_FILES.items():
            path = model_path(directory, stem)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'{stem} model checksum mismatch; download again')
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = opt.inter_op_num_threads = 1
            opt.add_session_config_entry('session.intra_op.allow_spinning', '0')
            opt.add_session_config_entry('session.inter_op.allow_spinning', '0')
            sess = ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])
            self.branches.append((sess, {}, [o.name for o in sess.get_outputs()]))
        self.pool = ThreadPoolExecutor(2, thread_name_prefix='ai-live-stem')
        self.reset()

    def reset(self):
        for sess, state, _ in self.branches:
            state.clear()
            state.update({i.name: np.zeros(i.shape, np.float32) for i in sess.get_inputs()})
        self.previous = np.zeros((HOP, 2), np.float32)

    @staticmethod
    def _run(branch, chunk):
        sess, state, names = branch
        state['audio_chunk'] = chunk
        outputs = sess.run(None, state)
        for name, value in zip(names, outputs):
            if name.startswith('next_'):
                state[name[5:]] = value
        return outputs[0][0, 0].T

    def process(self, block):
        chunk = np.ascontiguousarray(block.T[None], np.float32)
        drum_job = self.pool.submit(self._run, self.branches[0], chunk)
        vocal_job = self.pool.submit(self._run, self.branches[2], chunk)
        # Always gather every branch, including on failure, before reset/close.
        try:
            bass = self._run(self.branches[1], chunk)
            drums, vocals = drum_job.result(), vocal_job.result()
        finally:
            from concurrent.futures import wait
            wait((drum_job, vocal_job))
        other = self.previous - ((drums + bass) + vocals)
        self.previous = block.copy()
        return np.stack((drums, bass, other, vocals))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


def load_model(directory):
    model = ParallelModel(directory)
    try:
        x = np.random.default_rng(7).normal(0, .03, (HOP, 2)).astype(np.float32)
        timings = []
        for i in range(350):
            t = time.perf_counter()
            y = model.process(x)
            if not np.isfinite(y).all():
                raise ValueError('Non-finite AI Live output')
            if i >= 50:
                timings.append(time.perf_counter()-t)
        model.compute_s = float(np.mean(timings))
        model.p95_s = float(np.percentile(timings, 95))
        if model.compute_s >= .8*HOP/MODEL_SR or model.p95_s >= HOP/MODEL_SR:
            raise RuntimeError(f'CPU cannot sustain AI Live: {model.compute_s*1000:.1f} ms average, '
                               f'{model.p95_s*1000:.1f} ms p95 per 2.9 ms hop')
        model.reset()
        return model
    except Exception:
        model.close()
        raise


class LiveDezipper:
    """Remove the boundary step when a missing AI block changes speaker levels.

    Carries one endpoint, not old waveform blocks. A three-millisecond cosine
    correction settles to the current fallback without buffering/delaying audio.
    """
    def __init__(self, sample_rate):
        self.length = max(1, round(sample_rate*.003))
        self.last = None
        self.correction = None
        self.position = self.length
        self.was_ai = False

    def process(self, block, is_ai, ends_ai=None, discontinuity=False):
        if self.last is not None and self.last.shape != block[0].shape:
            self.last, self.position = None, self.length
        if self.last is not None and (discontinuity or (self.was_ai and not is_ai)):
            self.correction = self.last-block[0]
            self.position = 0
        if self.position < self.length:
            n = min(len(block), self.length-self.position)
            phase = (self.position+np.arange(n))/self.length
            weight = (.5+.5*np.cos(np.pi*phase)).astype(np.float32)
            block[:n] += weight[:, None]*self.correction
            self.position += n
        self.last = block[-1].copy()
        self.was_ai = bool(is_ai if ends_ai is None else ends_ai)


def live_mix_envelope(valid, initial, fade_samples):
    """Fade toward Classic before a known missing interval, without delay."""
    pos = np.arange(len(valid))
    previous_gap = np.maximum.accumulate(np.where(valid, -initial*fade_samples, pos))
    next_gap = np.minimum.accumulate(np.where(valid, len(valid)+fade_samples, pos)[::-1])[::-1]
    rise = np.clip((pos-previous_gap)/fade_samples, 0, 1)
    fall = np.clip((next_gap-pos)/fade_samples, 0, 1)
    return np.minimum(rise, fall).astype(np.float32)


class LiveSeparator:
    def __init__(self, sample_rate, model):
        if int(sample_rate) not in SUPPORTED_RATES:
            raise ValueError('AI Live requires 44.1, 48, 88.2, 96 or 192 kHz')
        self.sr, self.model = int(sample_rate), model
        self.latency = int(math.ceil(self.sr*LIVE_RESERVE_MS/1000))
        self.max_pending = int(self.sr*.040)
        self.pending_samples = self.fed = 0
        self.misses = self.resets = 0
        self.recent_results = deque(maxlen=200)
        self.error = ''
        self._queue = deque()
        self._cv = threading.Condition()
        self._epoch = 0
        self._worker_epoch = -1
        self._stopped = False
        self._thread = None
        self._capacity = int(self.sr*.25)
        self._audio = np.zeros((self._capacity, 4, 2), np.float32)
        self._tags = np.full(self._capacity, -1, np.int64)

    def feed(self, block, discontinuity=False):
        x = np.asarray(block[:, :2], np.float32).copy()
        with self._cv:
            start = self.fed
            self.fed += len(x)
            if self._stopped or self.error:
                return
            if discontinuity or self.pending_samples + len(x) > self.max_pending:
                self._queue.clear()
                self.pending_samples = 0
                self._epoch += 1
                self.resets += 1
                self._tags.fill(-1)
            if len(x) > self.max_pending:
                start += len(x)-self.max_pending
                x = x[-self.max_pending:]
            self._queue.append((self._epoch, start, x))
            self.pending_samples += len(x)
            self._cv.notify_all()

    def _reset(self, epoch, start):
        self.model.reset()
        guidance = getattr(self, 'enhance_guidance', None)
        if guidance is not None:
            guidance.reset()
        self._to_model = StreamingResampler(self.sr, MODEL_SR)
        self._to_host = StreamingResampler(MODEL_SR, self.sr)
        self._natural_delay = round((self._to_model.delay_output+HOP)*self.sr/MODEL_SR
                                    + self._to_host.delay_output)
        self._partial = np.zeros((0, 2), np.float32)
        self._output_index = start
        self._epoch_start = start
        self._worker_epoch = epoch

    def step(self):
        with self._cv:
            if not self._queue or self._stopped:
                return False
            epoch, start, x = self._queue.popleft()
            self.pending_samples -= len(x)
        if epoch != self._worker_epoch:
            self._reset(epoch, start)
        data = np.concatenate((self._partial, self._to_model.process(x)))
        used = len(data)//HOP*HOP
        for at in range(0, used, HOP):
            stems = self.model.process(data[at:at+HOP])  # [4, 128, 2]
            if not np.isfinite(stems).all():
                raise ValueError('Non-finite AI Live stems')
            guidance = getattr(self, 'enhance_guidance', None)
            if guidance is not None and not guidance.error:
                try:
                    guidance.observe(stems)
                except Exception as exc:
                    # Enhancement failure must not stop separation or playback.
                    guidance.error = str(exc)
            host = self._to_host.process(stems.transpose(1, 0, 2).reshape(HOP, 8)).reshape(-1, 4, 2)
            tags = np.arange(self._output_index, self._output_index+len(host)) - self._natural_delay
            self._output_index += len(host)
            valid = tags >= self._epoch_start
            # A queue reset can arrive while inference is in flight: discard it.
            with self._cv:
                if epoch != self._epoch or self._stopped:
                    return True
                slots = tags[valid] % self._capacity
                self._audio[slots] = host[valid]
                self._tags[slots] = tags[valid]
                self._cv.notify_all()
        self._partial = data[used:].copy()
        return True

    def pull(self, start, n, wait_s=0.0):
        return self._pull(start, n, wait_s, partial=False)

    def pull_available(self, start, n, wait_s=0.0):
        return self._pull(start, n, wait_s, partial=True)

    def _pull(self, start, n, wait_s, partial):
        if start < 0:
            return None
        tags = np.arange(start, start+n)
        slots = tags % self._capacity
        with self._cv:
            epoch = self._epoch
            def ready():
                return np.array_equal(self._tags[slots], tags)
            # Only use the caller's existing render-time allowance, never grow
            # an audio queue to wait for AI. Condition.wait releases the GIL.
            if wait_s > 0 and not ready() and self.fed >= start+n:
                self._cv.wait_for(lambda: self._stopped or self.error or
                                  epoch != self._epoch or ready(), timeout=wait_s)
            valid = self._tags[slots] == tags
            if self._stopped or self.error or epoch != self._epoch:
                valid.fill(False)
            complete = bool(valid.all())
            self.recent_results.append(float(valid.mean()))
            if not complete:
                self.misses += 1
            if not valid.any() or (not partial and not complete):
                return None
            result = self._audio[slots].transpose(1, 2, 0).copy()
            if partial:
                result[:, :, ~valid] = 0
                return result, valid
            return result

    def recent_coverage(self):
        with self._cv:
            values = tuple(self.recent_results)
        return 100*sum(values)/len(values) if values else None

    def start(self):
        self._thread = threading.Thread(target=self._loop, name='ai-live', daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._stopped or self._queue)
                    if self._stopped:
                        return
                self.step()
        except Exception as exc:
            with self._cv:
                self.error = f'{type(exc).__name__}: {exc}'
                self._cv.notify_all()

    def stop(self):
        with self._cv:
            self._stopped = True
            self._queue.clear()
            self.pending_samples = 0
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()
        self.model.close()


class LiveController:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or default_model_dir())
        self.separator = None
        self.state, self.detail = 'off', ''
        self._desired = (False, 48000)
        self._lock = threading.RLock()
        self._job = None
        self._version = 0
        self._restart = False

    @property
    def status_text(self):
        if self.state == 'ready' and self.separator:
            s = self.separator
            coverage = s.recent_coverage()
            availability = f'{coverage:.1f}%' if coverage is not None else 'warming up'
            return (f'AI Live: CPU · {getattr(s.model, "compute_s", 0)*1000:.1f} ms/hop · '
                    f'recent AI availability {availability} · total late blocks {s.misses} · resets {s.resets}')
        return {'off': 'AI Live: off', 'loading': 'AI Live: loading and checking CPU speed…',
                'missing': 'AI Live: model missing (32 MB) — conventional upmix',
                'error': f'AI Live: fallback — {self.detail}',
                'downloading': f'AI Live: downloading {self.detail}'}.get(self.state, self.state)

    def ensure(self, enabled, sample_rate):
        with self._lock:
            desired = (bool(enabled), int(sample_rate))
            changed = desired != self._desired
            if changed:
                self._version += 1
            self._desired = desired
            if self.state == 'ready' and self.separator and self.separator.error:
                self.state, self.detail = 'error', self.separator.error
            busy = self._job is not None and self._job.is_alive()
            if busy:
                return
            wrong_rate = self.separator is not None and self.separator.sr != sample_rate
            if (not enabled and (self.separator is not None or self.state != 'off')) or \
                    (enabled and (self.state == 'off' or self._restart or wrong_rate or (changed and self.state == 'error'))):
                self._job = threading.Thread(target=self._reconcile, name='ai-live-control', daemon=True)
                self._job.start()

    def _reconcile(self):
        with self._lock:
            enabled, rate = self._desired
            version = self._version
            self._restart = False
            old, self.separator = self.separator, None
            self.state = 'loading' if enabled else 'off'
        if old:
            old.stop()
        if not enabled:
            return
        if rate not in SUPPORTED_RATES:
            self.state, self.detail = 'error', f'Unsupported sample rate: {rate} Hz'
            return
        if not models_present(self.model_dir):
            self.state = 'missing'
            return
        model = None
        try:
            model = load_model(self.model_dir)
            sep = LiveSeparator(rate, model)
            with self._lock:
                if self._desired != (True, rate) or self._version != version:
                    self.state = 'off'
                    model.close()
                    return
                sep.start()
                self.separator, self.state = sep, 'ready'
        except Exception as exc:
            if model is not None:
                model.close()
            self.state, self.detail = 'error', str(exc)

    def invalidate(self):
        """Device/format changes invalidate pending loads and recurrent history."""
        with self._lock:
            self._version += 1
            self._restart = True

    def download(self):
        with self._lock:
            if self._job is not None and self._job.is_alive():
                return
            def job():
                try:
                    self.state = 'downloading'
                    download_models(self.model_dir, lambda s: setattr(self, 'detail', s))
                    self.state = 'off'
                except Exception as exc:
                    self.state, self.detail = 'error', str(exc)
            self._job = threading.Thread(target=job, name='ai-live-download', daemon=True)
            self._job.start()

    def wait_idle(self, timeout=None):
        job = self._job
        if job:
            job.join(timeout)
