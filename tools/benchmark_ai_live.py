"""Paced model-worker benchmark; never opens an audio device or plays sound.

Run from the project root: python tools/benchmark_ai_live.py --seconds 30
This measures sustained readiness, not subjective separation or acoustic delay.
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_live import LiveSeparator, load_model, default_model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--rate', type=int, default=48000)
    parser.add_argument('--block', type=int, default=480)
    parser.add_argument('--capture-block', type=int, help='Input delivery burst size (defaults to output block size)')
    parser.add_argument('--output-phase-ms', type=float, default=0)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--pipeline', action='store_true', help='Include the real input buffer and upmixer')
    parser.add_argument('--reserve-ms', type=float, help='Benchmark-only reserve override')
    args = parser.parse_args()
    if args.reserve_ms is not None:
        import ai_live
        ai_live.LIVE_RESERVE_MS = args.reserve_ms
    model = load_model(default_model_dir())
    sep = LiveSeparator(args.rate, model)
    sep.start()
    app = None
    scratch = tempfile.TemporaryDirectory(prefix='xpdss-live-bench-')
    if args.pipeline:
        from audio_analyzer import AudioBypass
        AudioBypass._setup_tray = lambda self: None
        app = AudioBypass('', '', simulation=True, config_file=str(Path(scratch.name)/'config.json'))
        if args.rate != app.sample_rate:
            raise ValueError('Integrated benchmark currently requires 48000 Hz')
        app.upmix_enabled = app.center_extraction = app.ai_live_mode = True
        app.current_channels = 6
        ctl = app._live_controller()
        ctl.separator, ctl.state = sep, 'ready'
        ctl.ensure = lambda *a: None
        # Match the real app's callback scheduling policy.
        sys.setswitchinterval(.001)
    samples = int(args.rate*args.seconds)
    t = np.arange(samples)/args.rate
    audio = np.stack((.06*np.sin(2*np.pi*300*t)+.03*np.sin(2*np.pi*3200*t),
                      .06*np.sin(2*np.pi*700*t)+.03*np.sin(2*np.pi*6000*t)), 1).astype(np.float32)
    started = time.perf_counter()
    misses = checked = fully_missing = available_samples = checked_samples = 0
    max_reconstruction_error = 0.0
    feed_ms, pull_ms, lateness_ms = [], [], []
    capture_block = args.capture_block or args.block
    events = [(at/args.rate, 0, at) for at in range(0, samples-capture_block+1, capture_block)]
    events += [(at/args.rate+args.output_phase_ms/1000, 1, at) for at in range(0, samples-args.block+1, args.block)]
    events.sort()
    try:
        for when, kind, at in events:
            deadline = started+when
            wait = deadline-time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            lateness_ms.append(max(0, time.perf_counter()-deadline)*1000)
            begin = time.perf_counter()
            if kind == 0:
                if app:
                    block = np.zeros((capture_block, 8), np.float32)
                    block[:, :2] = audio[at:at+capture_block]
                    app.input_callback(block, capture_block, None, None)
                else:
                    sep.feed(audio[at:at+capture_block])
                feed_ms.append((time.perf_counter()-begin)*1000)
                continue
            begin = time.perf_counter()
            if app:
                source = app._read_input_block(args.block)
                stems = app._live_stems
                valid = app._live_valid
                if source is not None:
                    ref = source[:, :2].T.copy()
                    rendered = app._apply_upmix(source, 6)
                    if not np.isfinite(rendered).all():
                        raise ValueError('Non-finite upmix output')
            else:
                stems = sep.pull(at-sep.latency, args.block)
                valid = np.ones(args.block, bool) if stems is not None else None
                ref = audio[at-sep.latency:at-sep.latency+args.block].T
            pull_ms.append((time.perf_counter()-begin)*1000)
            if at > args.rate:
                checked += 1
                checked_samples += args.block
                if stems is None:
                    misses += 1
                    fully_missing += 1
                else:
                    available_samples += int(valid.sum())
                    misses += int(not valid.all())
                    max_reconstruction_error = max(max_reconstruction_error, float(np.max(np.abs(stems.sum(0)[:, valid]-ref[:, valid]))))
    finally:
        sep.stop()
        scratch.cleanup()
    result = dict(seconds=args.seconds, rate=args.rate, block=args.block,
                  capture_block=capture_block, output_phase_ms=args.output_phase_ms,
                  model_mean_ms=model.compute_s*1000, model_p95_ms=model.p95_s*1000,
                  pipeline=args.pipeline,
                  input_reserve_ms=(1000*app.target_latency_samples/args.rate if app else 1000*sep.latency/args.rate),
                  checked_blocks=checked, missed_blocks=misses, queue_resets=sep.resets,
                  fully_missing_blocks=fully_missing, available_ai_samples_percent=100*available_samples/max(1,checked_samples),
                  max_reconstruction_error=max_reconstruction_error, error=sep.error,
                  callback_feed_p99_ms=float(np.percentile(feed_ms,99)),
                  callback_pull_p99_ms=float(np.percentile(pull_ms,99)),
                  host_lateness_p99_ms=float(np.percentile(lateness_ms,99)))
    print(json.dumps(result, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2))
    return int(bool(sep.error or misses or sep.resets))


if __name__ == '__main__':
    raise SystemExit(main())
