"""AI music mode: streaming stem separation with HTDemucs (ONNX).

The model (Demucs v4 / HTDemucs, MIT, ONNX conversion by smank) splits a
stereo mix into drums, bass, other and vocals. It works on a fixed 7.8 s
window at 44.1 kHz, so it is run on a sliding window over the input:

    window end   = t + hop + xfade + future
    kept region  = [t, t + hop + xfade)   (the newest `future` seconds are context)

Measured on a real track, the stems produced with 0.5 s of future context
differ from a full-context separation by about -21 dB (and the model's own
separation error is around -8 dB), so the quality cost of streaming is
negligible from ~0.5 s. Total latency = future + hop + compute margin.

A worker thread does the inference (ONNX Runtime releases the GIL); the
audio thread only feeds input and pulls ready stems. When stems are not
ready in time, pull() returns None and the caller plays a fallback with the
same delay, so nothing ever jumps in time.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.request

import numpy as np
from scipy.signal import resample_poly

MODEL_URL = "https://huggingface.co/smank/htdemucs-onnx/resolve/main/htdemucs.onnx"
MODEL_SHA256 = "d2b401f322558cd57d67a752ed7be3fa55178a0626011eda8ac7bb74e17280c0"
MODEL_FILE = "htdemucs.onnx"
STEMS = ("drums", "bass", "other", "vocals")
MODEL_SR = 44100
SEG = 343980                      # model window (7.8 s at 44.1 kHz)


def default_model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_FILE)


def model_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == 304321552


def download_model(path: str, progress=None) -> None:
    """Download the model next to the app and verify its SHA-256."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    h = hashlib.sha256()
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if h.hexdigest() != MODEL_SHA256:
        os.remove(tmp)
        raise IOError("model download corrupted (SHA-256 mismatch)")
    os.replace(tmp, path)


# Providers tried in order; each must pass a real test run on the model.
# DirectML's vendor "metacommands" can fail on some ConvTranspose layers of
# the decoder (seen on an RTX 3060 Ti), hence the second DirectML attempt.
PROVIDER_CHAIN = (
    ("CUDA", "CUDAExecutionProvider", {}),
    ("DirectML", "DmlExecutionProvider", {}),
    ("DirectML (no metacommands)", "DmlExecutionProvider", {"disable_metacommands": "true"}),
    ("CPU", "CPUExecutionProvider", {}),
)


def _preload_gpu_libraries() -> None:
    """Make CUDA/cuDNN installed as pip packages (onnxruntime-gpu[cuda,cudnn])
    loadable; harmless when they are absent."""
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass


def make_session(path: str, provider: str = "CPUExecutionProvider", options: dict | None = None):
    """ONNX Runtime session on one provider (CPU kept as fallback for single
    unsupported nodes). Raises if the requested provider did not load."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # leave cores for the audio thread and the rest of the system
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 2)
    if provider == "DmlExecutionProvider":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [(provider, dict(options or {}))]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(path, so, providers=providers)
    if sess.get_providers()[0] != provider:
        raise RuntimeError(f"{provider} did not load (got {sess.get_providers()[0]})")
    return sess


def available_chain():
    import onnxruntime as ort
    avail = set(ort.get_available_providers())
    return [c for c in PROVIDER_CHAIN if c[1] in avail]


class StreamingSeparator:
    """Sliding-window HTDemucs separation for a live stream at `sample_rate`."""

    def __init__(self, sample_rate: int, session=None, future_s: float = 0.5, hop_s: float = 0.25,
                 xfade_s: float = 0.05, run_fn=None):
        self.sr = int(sample_rate)
        self.session = session
        self._run_fn = run_fn               # injectable for tests
        self.future = int(future_s * self.sr)
        self.hop = int(hop_s * self.sr)
        self.xfade = max(1, int(xfade_s * self.sr))
        # window length at the host rate (7.8 s), resampled to/from 44.1 kHz
        self.win = int(np.ceil(SEG * self.sr / MODEL_SR))
        g = np.gcd(self.sr, MODEL_SR)
        self._up, self._down = MODEL_SR // g, self.sr // g
        self.provider = None
        self.compute_s = None               # measured inference time (s)
        self.latency = None                 # samples, fixed after warm-up
        self._hist = np.zeros((2, self.win + self.sr * 4), dtype=np.float32)
        self._hist_end = 0                  # absolute input index of the sample after the newest
        self._out = {}                      # region start -> (4, 2, hop + xfade) stems
        self._next_t = None                 # next region start to compute
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._thread = None
        self._stop = False
        self.misses = 0
        self.error = None

    # ── model ────────────────────────────────────────────────────────
    def _run_model(self, mix44: np.ndarray) -> np.ndarray:
        if self._run_fn is not None:
            return self._run_fn(mix44)
        return self.session.run(None, {"mix": mix44[None].astype(np.float32)})[0][0]

    def separate_window(self, window: np.ndarray) -> np.ndarray:
        """(2, win) at the host rate -> (4, 2, win) stems at the host rate."""
        x44 = resample_poly(window, self._up, self._down, axis=1)
        seg = np.zeros((2, SEG), dtype=np.float32)
        n = min(SEG, x44.shape[1])
        seg[:, SEG - n:] = x44[:, x44.shape[1] - n:]
        y = self._run_model(seg)                                      # (4, 2, SEG)
        back = resample_poly(y, self._down, self._up, axis=2)
        out = np.zeros((4, 2, self.win), dtype=np.float32)
        m = min(self.win, back.shape[2])
        out[:, :, self.win - m:] = back[:, :, back.shape[2] - m:]
        return out

    def warm_up(self, runs: int = 3) -> float:
        """Time the model and fix the latency: future + hop + xfade + 1.5 x compute."""
        x = (np.random.default_rng(0).standard_normal((2, self.win)) * 0.05).astype(np.float32)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            self.separate_window(x)
            times.append(time.perf_counter() - t)
        self.compute_s = float(np.median(times[1:] if len(times) > 1 else times))
        self.latency = self.latency_for(self.compute_s)
        return self.compute_s

    def latency_for(self, compute_s: float) -> int:
        """A region [t, t+hop) is ready once the input reaches t + hop + xfade +
        future and the model has run; the output must also cover one audio
        block. Latency = future + hop + xfade + 1.5 x compute + 50 ms."""
        return self.future + self.hop + self.xfade + int((1.5 * compute_s + 0.05) * self.sr)

    @property
    def realtime_ok(self) -> bool:
        return self.compute_s is not None and self.compute_s < 0.8 * self.hop / self.sr

    # ── streaming ────────────────────────────────────────────────────
    def feed(self, stereo: np.ndarray) -> None:
        """Audio thread: append (n, 2) input."""
        x = np.asarray(stereo[:, :2], dtype=np.float32).T
        n = x.shape[1]
        with self._cv:
            cap = self._hist.shape[1]
            if n >= cap:
                self._hist[:] = x[:, -cap:]
            else:
                self._hist[:, :-n] = self._hist[:, n:]
                self._hist[:, -n:] = x
            self._hist_end += n
            if self._next_t is None:
                self._next_t = max(0, self._hist_end - n)
            self._cv.notify()

    def step(self) -> bool:
        """Compute the next region if its input is available (worker / tests)."""
        with self._lock:
            t = self._next_t
            if t is None:
                return False
            end = t + self.hop + self.xfade + self.future
            if self._hist_end < end:
                return False
            cap = self._hist.shape[1]
            off = cap - (self._hist_end - end)
            window = self._hist[:, off - self.win:off].copy() if off - self.win >= 0 else None
            if window is None:                                 # not enough history yet: pad
                window = np.zeros((2, self.win), dtype=np.float32)
                avail = self._hist[:, max(0, off - self.win):off]
                window[:, self.win - avail.shape[1]:] = avail
        stems = self.separate_window(window)
        s0 = self.win - self.future - self.hop - self.xfade
        region = stems[:, :, s0:s0 + self.hop + self.xfade].copy()
        with self._lock:
            self._out[t] = region
            self._next_t = t + self.hop
            # forget regions far behind
            for k in [k for k in self._out if k < t - 8 * self.hop]:
                del self._out[k]
        return True

    def pull(self, start: int, n: int):
        """Stems for input samples [start, start + n) as (4, 2, n), or None
        if any part is not ready. Consecutive regions are crossfaded."""
        out = np.zeros((4, 2, n), dtype=np.float32)
        with self._lock:
            pos = start
            while pos < start + n:
                k = pos - ((pos - self._region0()) % self.hop) if self._region0() is not None else None
                if k is None or k not in self._out:
                    self.misses += 1
                    return None
                reg = self._out[k]
                i0 = pos - k
                take = min(self.hop - i0, start + n - pos)
                seg = reg[:, :, i0:i0 + take].copy()
                # crossfade the first `xfade` samples of a region with the tail of the previous one
                prev = self._out.get(k - self.hop)
                if prev is not None and i0 < self.xfade:
                    m = min(take, self.xfade - i0)
                    w = (np.arange(i0, i0 + m) + 0.5) / self.xfade
                    seg[:, :, :m] = prev[:, :, self.hop + i0:self.hop + i0 + m] * (1 - w) + seg[:, :, :m] * w
                out[:, :, pos - start:pos - start + take] = seg
                pos += take
        return out

    def _region0(self):
        return min(self._out) if self._out else None

    # ── worker thread ────────────────────────────────────────────────
    def start(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="ai-separation", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        try:
            while True:
                with self._cv:
                    while not self._stop and (self._next_t is None or
                                              self._hist_end < self._next_t + self.hop + self.xfade + self.future):
                        self._cv.wait(timeout=0.5)
                    if self._stop:
                        return
                self.step()
        except Exception as e:          # never take the audio down with it
            self.error = f"{type(e).__name__}: {e}"


class AIController:
    """Lifecycle of the AI music mode, off the audio thread.

    States: off, missing (model not downloaded), downloading, loading,
    ready, error. Loading creates the ONNX session, times the model and
    picks the hop: 0.25 s when the hardware keeps up, otherwise the
    smallest hop it can sustain (more latency, but never dropouts).
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or default_model_path()
        self.state = "off"
        self.detail = ""
        self.separator: StreamingSeparator | None = None
        self.provider = None
        self.failures = []
        self._lock = threading.Lock()
        self._busy = False

    @property
    def status_text(self) -> str:
        if self.state == "ready" and self.separator is not None:
            s = self.separator
            return (f"AI: running on {self.provider}, "
                    f"{s.compute_s * 1000:.0f} ms per window, delay {s.latency / s.sr:.2f} s"
                    + (f", fallbacks {s.misses}" if s.misses else "")
                    + (" (skipped: " + ", ".join(f.split(":")[0] for f in self.failures) + ")" if self.failures else ""))
        return {"off": "AI: off", "missing": "AI: model not downloaded (304 MB)",
                "downloading": f"AI: downloading model {self.detail}", "loading": "AI: loading model...",
                "error": f"AI: error - {self.detail}"}.get(self.state, self.state)

    def ensure(self, enabled: bool, sample_rate: int) -> None:
        """Called from the audio thread every block: cheap, never blocks."""
        if enabled:
            if self.state in ("off",) and not self._busy:
                if not model_ok(self.model_path):
                    self.state = "missing"
                    return
                self._spawn(self._load, sample_rate)
            elif self.state == "ready" and self.separator is not None and self.separator.error:
                self.state, self.detail = "error", self.separator.error
        else:
            if self.state in ("ready", "error", "missing") and not self._busy:
                self._spawn(self._unload)

    def download(self) -> None:
        if self._busy:
            return

        def job():
            try:
                self.state = "downloading"
                download_model(self.model_path, lambda d, t: setattr(self, "detail", f"{d / 1e6:.0f}/{t / 1e6:.0f} MB"))
                self.state = "off"
            except Exception as e:
                self.state, self.detail = "error", f"download failed: {e}"
        self._spawn(job)

    def _spawn(self, fn, *args) -> None:
        self._busy = True

        def run():
            try:
                fn(*args)
            finally:
                self._busy = False
        threading.Thread(target=run, name="ai-controller", daemon=True).start()

    def _load(self, sample_rate: int) -> None:
        """Try each available provider in order with a real test run; the first
        that works is used (a failure only moves on to the next one)."""
        self.state = "loading"
        failures = []
        if any(c[1] == "CUDAExecutionProvider" for c in PROVIDER_CHAIN):
            _preload_gpu_libraries()
        try:
            chain = available_chain()
        except Exception as e:
            self.state, self.detail = "error", f"ONNX Runtime not installed ({e})"
            return
        for name, provider, options in chain:
            try:
                sess = make_session(self.model_path, provider, options)
                probe = StreamingSeparator(sample_rate, session=sess)
                compute = probe.warm_up(3)
            except Exception as e:
                failures.append(f"{name}: {str(e).splitlines()[0][:120]}")
                continue
            hop_s = max(0.25, 1.3 * compute)          # keep up with the hardware
            sep = StreamingSeparator(sample_rate, session=sess, hop_s=hop_s, xfade_s=0.05)
            sep.compute_s = compute
            sep.latency = sep.latency_for(compute)
            sep.start()
            self.provider, self.failures = name, failures
            self.separator = sep
            self.state = "ready"
            if failures:
                print("[AI] fell back past: " + " | ".join(failures))
            return
        self.state, self.detail = "error", "no provider worked: " + " | ".join(failures)

    def _unload(self) -> None:
        sep, self.separator = self.separator, None
        if sep is not None:
            sep.stop()
        self.state = "off"
