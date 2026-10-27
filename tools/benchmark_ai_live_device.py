"""Exercise AI Live against real WASAPI callback timing; output is always silent.

Uses the configured input/output device names, opens shared streams, injects
synthetic input, and discards rendered output. Does not save settings. Reported
ADC-to-DAC timestamps exclude virtual-cable source and receiver latency.
"""
import argparse
import hashlib
from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ai_live
from audio_analyzer import AudioBypass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--reserve-ms', type=float, default=ai_live.LIVE_RESERVE_MS)
    parser.add_argument('--app-load', action='store_true', help='Run the analysis worker and offscreen Qt mixer too')
    parser.add_argument('--source-input', action='store_true', help='Process captured audio, still outputting only silence')
    parser.add_argument('--report', type=Path, help='Save timing measurements as JSON (no audio)')
    parser.add_argument('--enhance', action='store_true', help='Exercise the production AI Enhance toggle')
    parser.add_argument('--native', action='store_true', help='Preserve native channels instead of AI Live upmix')
    parser.add_argument('--probe', type=Path, help='Inject an existing public/synthetic fixture; output remains silent')
    parser.add_argument('--ui-snapshot', type=Path, help='Save the offscreen mixer window during --app-load')
    args = parser.parse_args()
    probe = np.load(args.probe) if args.probe else None
    ai_live.LIVE_RESERVE_MS = args.reserve_ms
    config_bytes = (ROOT/'config.json').read_bytes()
    config = json.loads(config_bytes)
    api = next(i for i, h in enumerate(sd.query_hostapis()) if 'WASAPI' in h['name'])
    devices = sd.query_devices()
    def device(name, direction):
        return next(i for i, d in enumerate(devices) if d['hostapi'] == api and
                    d['name'] == name and d[f'max_{direction}_channels'])
    inp, out = device(config['input_device'], 'input'), device(config['output_device'], 'output')
    sr = 48000
    AudioBypass._setup_tray = lambda self: None
    with tempfile.TemporaryDirectory(prefix='xpdss-live-device-') as scratch:
        app = AudioBypass('', '', simulation=True, config_file=str(Path(scratch)/'config.json'))
        app._apply_profile_dict(config.get(config.get('current_mode', 'speakers'), {}))
        app.ai_music_mode = False  # do not start the unrelated music model
        app.ai_live_mode = app.upmix_enabled = True
        app.ai_enhance = args.enhance
        app.is_multichannel_detected = args.native
        app.manual_upmix = False
        app.current_channels = min(8, int(devices[out]['max_output_channels']))
        sep = ai_live.LiveSeparator(sr, ai_live.load_model(ai_live.default_model_dir()))
        ctl = app._live_controller()
        ctl.separator, ctl.state = sep, 'ready'
        ctl.ensure = lambda *a: None
        stamps, delays, costs = deque(maxlen=200), [], []
        misses_detail, capture_times, render_times = [], [], []
        wait_budgets, dac_slack = [], []
        capture_age, buffer_fill, buffer_after_read, deadline_remaining = [], [], [], []
        original_pull = sep.pull_available
        def diagnostic_pull(at, n, **kwargs):
            result = original_pull(at, n, **kwargs)
            if (result is None or not result[1].all()) and at >= 0:
                produced = getattr(sep, '_output_index', 0)-getattr(sep, '_natural_delay', 0)
                misses_detail.append((sep.fed-at-n, at+n-produced, sep.pending_samples))
            return result
        sep.pull_available = diagnostic_pull
        checked = missed = empty = fully_missing = available_samples = checked_samples = 0
        errors = []
        sys.setswitchinterval(.001)
        start = time.perf_counter()
        def capture(data, frames, timing, status):
            try:
                capture_times.append((time.perf_counter(), frames))
                capture_age.append(1000*(timing.currentTime-timing.inputBufferAdcTime))
                at = app._input_total_samples
                stamps.append((at, timing.inputBufferAdcTime))
                t = np.arange(at, at+frames)/sr
                x = np.zeros((frames, 8), np.float32)
                x[:, 0] = .06*np.sin(2*np.pi*300*t)
                x[:, 1] = .06*np.sin(2*np.pi*700*t)
                if args.source_input:
                    x[:, :min(8, data.shape[1])] = data[:, :8]
                if probe is not None:
                    x.fill(0)
                    ch = min(x.shape[1], probe.shape[1])
                    x[:, :ch] = probe[(at+np.arange(frames)) % len(probe), :ch]
                app.input_callback(x, frames, timing, status)
            except Exception as exc:
                errors.append(str(exc))
        def render(data, frames, timing, status):
            nonlocal checked, missed, empty, fully_missing, available_samples, checked_samples
            data.fill(0)
            begin = time.perf_counter()
            render_times.append((begin, frames))
            try:
                if getattr(status, 'output_underflow', False):
                    app.io_stats['out_underflow'] += 1
                allowance = app._live_wait_budget(frames, timing)
                wait_budgets.append(allowance*1000)
                dac_slack.append((timing.outputBufferDacTime-timing.currentTime)*1000)
                with app.buffer_lock:
                    buffer_fill.append(1000*app.available_samples/sr)
                source = app._read_input_block(frames, live_wait_s=allowance)
                with app.buffer_lock:
                    buffer_after_read.append(1000*app.available_samples/sr)
                if begin-start > 2:
                    checked += 1
                    checked_samples += frames
                    missing = app._live_stems is None
                    fully_missing += int(missing)
                    missed += int(missing or not app._live_valid.all())
                    available_samples += 0 if missing else int(app._live_valid.sum())
                    empty += int(source is None)
                if source is not None:
                    # Read pointer identifies the first source sample rendered.
                    at = app._input_total_samples-app.available_samples-frames
                    stamp = next(((n, ts) for n, ts in reversed(stamps) if n <= at), None)
                    if stamp and begin-start > 2:
                        delays.append(1000*(timing.outputBufferDacTime-stamp[1]-(at-stamp[0])/sr))
                    y = app._apply_upmix(source, app.current_channels)
                    if not np.isfinite(y).all():
                        errors.append('Non-finite output')
                elapsed_ms = (time.perf_counter()-begin)*1000
                costs.append(elapsed_ms)
                deadline_remaining.append(dac_slack[-1]-elapsed_ms)
            except Exception as exc:
                errors.append(str(exc))
        sep.start()
        qt = win = analysis = None
        if args.app_load:
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QTimer
            from PySide6.QtGui import QFontDatabase, QFont
            from gui_components import UpmixerWindow
            qt = QApplication([])
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
            qt.setFont(QFont('Segoe UI', 9))
            win = UpmixerWindow(app)
            win.resize(1300, 900)
            win.show()
            analysis = threading.Thread(target=app._analysis_worker, daemon=True)
            analysis.start()
        try:
            with sd.InputStream(device=inp, samplerate=sr, channels=int(devices[inp]['max_input_channels']),
                                latency='low', blocksize=0, callback=capture) as ins:
                with sd.OutputStream(device=out, samplerate=sr, channels=app.current_channels,
                                     latency='low', blocksize=0, callback=render) as outs:
                    if qt:
                        if args.ui_snapshot:
                            QTimer.singleShot(1500, lambda: win.grab().save(str(args.ui_snapshot)))
                        QTimer.singleShot(int(args.seconds*1000), lambda: qt.exit(0))
                        qt.exec()
                    else:
                        time.sleep(args.seconds)
                    driver_ms = 1000*(ins.latency+outs.latency)
                    streams = {
                        'mode': 'WASAPI shared', 'sample_rate': sr,
                        'blocksize_request': 0, 'latency_request': 'low',
                        'input': {'name': devices[inp]['name'], 'channels': ins.channels,
                                  'default_low_latency_ms': 1000*devices[inp]['default_low_input_latency'],
                                  'reported_latency_ms': 1000*ins.latency},
                        'output': {'name': devices[out]['name'], 'channels': outs.channels,
                                   'default_low_latency_ms': 1000*devices[out]['default_low_output_latency'],
                                   'reported_latency_ms': 1000*outs.latency},
                    }
        finally:
            if win:
                win.timer.stop()
                win._save_timer.stop()
                win.hide()
            app.stop_event.set()
            if analysis:
                analysis.join(2)
            sep.stop()
        effect_ms = 1.5+(127000/sr if app.spectral_tilt else 0)
        result = dict(seconds=args.seconds, reserve_ms=args.reserve_ms, driver_reported_ms=driver_ms,
                      checked_blocks=checked, missing_stem_blocks=missed, empty_input_blocks=empty,
                      fully_missing_blocks=fully_missing,
                      available_ai_samples_percent=100*available_samples/max(1,checked_samples),
                      queue_resets=sep.resets, errors=errors, model_error=sep.error,
                      render_p99_ms=float(np.percentile(costs,99)),
                      adc_to_dac_plus_effect_ms_percentiles=(np.percentile(delays,[0,50,95,99,100])+effect_ms).tolist() if delays else [],
                      timing_note='Driver timestamps, not an acoustic measurement; excludes source and receiver')
        result['app_load'] = args.app_load
        result['enhance'] = args.enhance
        result['native_passthrough'] = args.native
        result['final_route'] = getattr(app, '_route_mode', None)
        result['final_reserve_ms'] = 1000*app.target_latency_samples/sr
        guidance = getattr(app, '_enhance_guidance', None)
        result['enhance_updates'] = guidance.updates if guidance else 0
        result['enhance_final_controls_db'] = guidance.snapshot[1] if guidance else []
        result['enhance_error'] = guidance.error if guidance else ''
        result['enhance_added_queue_samples'] = 0
        result['source_input'] = args.source_input
        result['streams'] = streams
        result['effect_estimate_ms'] = effect_ms
        route_delay_ms = 0.
        if getattr(app, '_route_mode', None) in ('spectral', 'pass_d'):
            route_delay_ms = 1024/sr*1000  # existing spectral analysis delay
        result['routing_delay_estimate_ms'] = route_delay_ms
        result['conservative_path_estimate_ms'] = driver_ms+1000*app.target_latency_samples/sr+effect_ms+route_delay_ms
        if args.native:
            result['available_ai_samples_percent'] = None  # no stem routing or result waits
            result['fully_missing_blocks'] = result['missing_stem_blocks'] = 0
        result['config_sha256'] = hashlib.sha256(config_bytes).hexdigest()
        result['config_unchanged'] = (ROOT/'config.json').read_bytes() == config_bytes
        result['percentile_order'] = ['min', 'p5', 'p50', 'p95', 'p99', 'max']
        for label, values in [('capture_age_from_driver_timestamps_ms', capture_age),
                              ('buffer_fill_before_read_ms', buffer_fill),
                              ('buffer_fill_after_read_ms', buffer_after_read),
                              ('render_cost_ms', costs),
                              ('output_deadline_remaining_ms', deadline_remaining)]:
            result[label] = np.percentile(values, [0,5,50,95,99,100]).tolist() if values else []
        result['negative_output_deadline_count'] = sum(v < 0 for v in deadline_remaining)
        result['capture_timestamps_plausible'] = bool(capture_age) and min(capture_age) >= -0.01
        if not result['capture_timestamps_plausible']:
            result['timing_note'] += '; capture timestamps are ahead of callback currentTime: ADC-to-DAC percentiles are not a valid latency measurement'
        result['buffer_note'] = 'Instantaneous occupancy, includes startup; capture can arrive between snapshots. Not an additional delay to add to the reserve.'
        result['deadline_note'] = 'Driver timestamp allowance minus callback wall time; scheduling before Python callback entry is not separately measured.'
        result['wait_budget_ms_p5_p50_p95'] = np.percentile(wait_budgets,[5,50,95]).tolist()
        result['dac_slack_ms_p5_p50_p95'] = np.percentile(dac_slack,[5,50,95]).tolist()
        result['io_stats'] = app.io_stats
        for label, calls in [('capture', capture_times), ('render', render_times)]:
            result[label+'_frames'] = sorted(set(n for _, n in calls))
            result[label+'_gap_ms_p50_p95_max'] = np.percentile(np.diff([t for t, _ in calls])*1000,[50,95,100]).tolist()
        if misses_detail:
            result['miss_buffer_ms_p5_p50_p95'] = (np.percentile(np.array(misses_detail)[:,0],[5,50,95])/sr*1000).tolist()
            result['miss_not_ready_ms_p5_p50_p95'] = (np.percentile(np.array(misses_detail)[:,1],[5,50,95])/sr*1000).tolist()
            result['miss_pending_ms_p5_p50_p95'] = (np.percentile(np.array(misses_detail)[:,2],[5,50,95])/sr*1000).tolist()
        print(json.dumps(result, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        return int(bool(errors or sep.error or empty or (missed and not args.native) or sep.resets))


if __name__ == '__main__':
    raise SystemExit(main())
