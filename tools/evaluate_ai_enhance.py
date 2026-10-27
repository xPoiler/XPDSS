"""Six short production-enhancer comparisons, using existing licensed sources."""
import html
import json
import math
import time

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from evaluate_remaster_quality import LAB, sf, rms, aligned_error_db
from ai_live import load_model, default_model_dir, StreamingResampler
from ai_enhance import EnhanceGuidance, EnhanceProcessor


def main():
    samples = 6*48000
    clips = {}
    for name in ('choice', 'brahms', 'libri1', 'robin'):
        x, sr = sf.read(LAB/'sources'/f'{name}.hq.ogg', dtype='float32', always_2d=True)
        div = math.gcd(sr, 48000)
        x = resample_poly(x, 48000//div, sr//div)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        # Later source material than the first feasibility probe where possible.
        start = min(10*48000, max(0, len(x)-samples))
        x = x[start:start+samples, :2]
        x = np.tile(x, (math.ceil(samples/len(x)), 1))[:samples]
        clips[name] = x*(.35/max(float(abs(x).max()), 1e-6))
    clips['mixed'] = clips['choice']*.45 + clips['libri1']*.7 + clips['robin']*.25
    specs = [('brahms', 'clean'), ('libri1', 'clean'), ('mixed', 'clean'),
             ('libri1', 'mud'), ('mixed', 'harsh'), ('mixed', 'hiss')]
    dest = LAB/'enhance-preview'
    dest.mkdir(exist_ok=True)
    rng = np.random.default_rng(712)
    model = load_model(default_model_dir())
    cases = []
    try:
        for name, condition in specs:
            clean = clips[name]
            x = clean.copy()
            if condition in ('mud', 'harsh'):
                band = (180, 500) if condition == 'mud' else (3200, 6500)
                x += (10**(8/20)-1)*sosfilt(butter(1, band, btype='bandpass', fs=48000, output='sos'), clean, axis=0)
            elif condition == 'hiss':
                x += rng.normal(0, rms(clean)*.1, clean.shape)
            model.reset()
            guidance = EnhanceGuidance()
            processor = EnhanceProcessor(48000, 2)
            converter = StreamingResampler(48000, 44100)
            partial = np.zeros((0, 2), np.float32)
            out, timings, controls = [], [], []
            for at in range(0, len(x), 480):
                block = x[at:at+480]
                before = time.perf_counter()
                out.append(processor.process(block, guidance))
                timings.append(1000*(time.perf_counter()-before))
                pending = np.concatenate((partial, converter.process(block)))
                used = len(pending)//128*128
                for k in range(0, used, 128):
                    guidance.observe(model.process(pending[k:k+128]))
                partial = pending[used:]
                controls.append(guidance.snapshot[1])
            y = np.concatenate(out)
            assert y.shape == x.shape and np.isfinite(y).all()
            ref, source, result = [z[2*48000:] for z in (clean, x, y)]
            matched = result*rms(source)/rms(result)
            gain = min(1., .8/max(float(abs(source).max()), float(abs(matched).max())))
            base = f'{name}-{condition}'
            sf.write(dest/f'{base}-original.wav', source*gain, 48000, subtype='PCM_24')
            sf.write(dest/f'{base}-enhanced.wav', matched*gain, 48000, subtype='PCM_24')
            row = {'clip': name, 'condition': condition,
                   'input_si_sdr_db': aligned_error_db(ref, source),
                   'enhanced_si_sdr_db': aligned_error_db(ref, result),
                   'raw_gain_change_db': 20*np.log10(rms(result)/rms(source)),
                   'max_controls_db': np.max(np.abs(controls), axis=0).tolist(),
                   'dsp_p99_ms': float(np.percentile(timings, 99)),
                   'neural_updates': guidance.updates}
            cases.append(row)
            print(json.dumps(row), flush=True)
    finally:
        model.close()
    (LAB/'enhance-preview.json').write_text(json.dumps({'cases': cases, 'queued_samples': 0,
        'note': 'Small level-matched comparison; restoration scores are not listening preferences.'}, indent=2)+'\n')
    page = ['<!doctype html><meta charset="utf-8"><title>AI Enhance preview</title>',
            '<style>body{font:16px system-ui;max-width:950px;margin:32px auto;background:#16191f;color:#eef}td,th{padding:12px}pre{white-space:pre-wrap}</style>',
            '<h1>AI Enhance preview</h1><p>Original and enhanced audio at matching RMS levels. Four-second excerpts; effects are intentionally subtle. No autoplay. Mud, harshness, and hiss were deliberately added in those examples. These samples demonstrate the new processor, without other app effects or room acoustics.</p>',
            '<table><tr><th>Example</th><th>Original</th><th>AI Enhance</th></tr>']
    for row in cases:
        base = row['clip']+'-'+row['condition']
        page.append(f'<tr><td>{base}</td><td><audio controls preload="none" src="{base}-original.wav"></audio></td><td><audio controls preload="none" src="{base}-enhanced.wav"></audio></td></tr>')
    page.append('</table><h2>Attribution</h2>')
    for path in sorted((LAB/'sources').glob('*.txt')):
        page.append('<pre>'+html.escape(path.read_text(encoding='utf-8'))+'</pre>')
    (dest/'index.html').write_text('\n'.join(page), encoding='utf-8')


if __name__ == '__main__':
    main()
