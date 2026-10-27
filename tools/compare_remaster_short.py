"""Eight short comparisons using existing licensed evaluation fixtures.

No device access or application changes. RNNoise delay is compensated only for
offline comparison; this does not remove that delay from live playback.
"""
from pathlib import Path
import sys
import json
import time
import html

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_remaster_quality import LAB, sf, rms, aligned_error_db
from tools.remaster_lab import RNNoiseWaveform


def main():
    assets = LAB / 'listening'
    dest = LAB / 'comparison-short'
    dest.mkdir(exist_ok=True)
    dll = next((LAB / 'rnnoise').rglob('rnnoise.dll'))
    rows = []
    timings = []
    for name in ('choice', 'brahms', 'libri1', 'mixed_proxy'):
        ref, rate = sf.read(assets / f'{name}-clean-original.wav', dtype='float32', always_2d=True)
        for condition in ('clean', 'hiss'):
            base = f'{name}-{condition}'
            x, sr = sf.read(assets / f'{base}-original.wav', dtype='float32', always_2d=True)
            guided, gs = sf.read(assets / f'{base}-guided.wav', dtype='float32', always_2d=True)
            assert rate == sr == gs == 48000 and x.shape == guided.shape == ref.shape
            p = RNNoiseWaveform(dll, x.shape[1])
            # Flush measured 960-sample latency before trimming to input length.
            padded = np.pad(x, ((0, 960), (0, 0)))
            output = []
            try:
                for at in range(0, len(padded), 480):
                    start = time.perf_counter()
                    output.append(p.process(padded[at:at + 480]))
                    if at >= 48000:
                        timings.append(1000 * (time.perf_counter() - start))
            finally:
                p.close()
            y = np.concatenate(output)[960:960 + len(x)]
            assert y.shape == x.shape and np.isfinite(y).all()
            # Exclude first second for RNNoise startup; same interval for all.
            reference, source, a, b = [z[48000:] for z in (ref, x, guided, y)]
            input_score = aligned_error_db(reference, source)
            row = {'clip': name, 'condition': condition,
                   'input_si_sdr_db': input_score,
                   'guided_si_sdr_db': aligned_error_db(reference, a),
                   'waveform_si_sdr_db': aligned_error_db(reference, b),
                   'waveform_raw_gain_db': 20 * np.log10(rms(b) / rms(source)),
                   'waveform_peak': float(abs(y).max()),
                   'waveform_finite': True, 'delay_compensated_samples': 960}
            if condition == 'hiss':
                row['guided_improvement_db'] = row['guided_si_sdr_db'] - input_score
                row['waveform_improvement_db'] = row['waveform_si_sdr_db'] - input_score
            matched = [source, a * rms(source) / rms(a), b * rms(source) / rms(b)]
            common_gain = min(1., .8 / max(float(abs(z).max()) for z in matched))
            errors = []
            for label, z in zip(('original', 'guided', 'waveform'), matched):
                path = dest / f'{base}-{label}.wav'
                sf.write(path, z * common_gain, sr, subtype='PCM_24')
                decoded, _ = sf.read(path, always_2d=True)
                errors.append(20 * np.log10(rms(decoded) / rms(source * common_gain)))
            row['saved_rms_matching_errors_db'] = errors
            assert max(abs(e) for e in errors) < .01
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {'cases': rows, 'sample_rate': 48000, 'evaluation_duration_s': 4,
              'stereo_compute_ms': dict(zip(('median', 'p99', 'max'), np.percentile(timings, [50, 99, 100]).tolist())),
              'latency': 'Existing guided estimate about 65 ms; waveform optimistic estimate about 68 ms. No new end-to-end device measurement.',
              'limitations': ['Eight excerpts, including constructed mixed proxy; no actual film/game corpus.',
                              'RMS matched, not perceptually loudness matched; raw attenuation reported separately.',
                              'Objective preservation/restoration scores do not establish listening preference.',
                              'Guided samples reuse settled prior render; RNNoise gets one second startup exclusion.']}
    (LAB / 'comparison-short.json').write_text(json.dumps(result, indent=2) + '\n')
    page = ['<!doctype html><meta charset="utf-8"><title>Remaster comparison</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#16191f;color:#eef}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #555}audio{width:250px}pre{white-space:pre-wrap}</style>',
            '<h1>Remaster comparison</h1><p>Four-second examples at matching RMS levels. Original input, AI-guided tone processing, and direct neural processing. No autoplay. The direct model\'s 20 ms delay is aligned away here for comparison only. Hiss is deliberately added. These are experiments, not active app settings.</p>',
            '<table><tr><th>Example</th><th>Original input</th><th>AI-guided</th><th>Direct neural</th></tr>']
    for row in rows:
        base = row['clip'] + '-' + row['condition']
        page.append('<tr><td>' + html.escape(base) + '</td>' + ''.join(
            f'<td><audio controls preload="none" src="{base}-{label}.wav"></audio></td>'
            for label in ('original', 'guided', 'waveform')) + '</tr>')
    page.append('</table><h2>Attribution</h2>')
    for path in sorted((LAB / 'sources').glob('*.txt')):
        page.append('<pre>' + html.escape(path.read_text(encoding='utf-8')) + '</pre>')
    (dest / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(json.dumps(result['stereo_compute_ms']), flush=True)


if __name__ == '__main__':
    main()
