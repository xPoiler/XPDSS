"""Run isolated candidate delay and CPU gates; does not open audio devices."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import correlate, correlation_lags

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.remaster_lab import GuidedRemaster, RNNoiseWaveform


def main():
    root = Path(__file__).resolve().parents[1]/'.work/ai-remaster'
    dll = next((root/'rnnoise').rglob('rnnoise.dll'))
    report = {'latency_target':'Nominal 65 ms; roughly 68 ms acceptable if quality earns it.', 'driver_reported_ms':44.,
              'existing_effect_ms':1.5+127000/48000, 'reserve_ms':17.}
    delays = []
    for offset in (0,1,137,239,478,479):
        p = RNNoiseWaveform(dll, 1)
        x = np.zeros((48000,1),np.float32)
        at = 24000+offset; x[at,0] = .5
        y = np.concatenate([p.process(x[i:i+480]) for i in range(0,len(x),480)])
        delays.append(int(np.argmax(abs(y[:,0])))-at)
        p.close()
    timings = {}
    rng = np.random.default_rng(87)
    for ch in (2,6,8):
        p = RNNoiseWaveform(dll, ch)
        x = rng.normal(0,.03,(480,ch)).astype(np.float32)
        ts=[]
        for i in range(350):
            start=time.perf_counter(); y=p.process(x); elapsed=time.perf_counter()-start
            assert np.isfinite(y).all()
            if i>=50: ts.append(elapsed*1000)
        p.close()
        timings[str(ch)] = dict(zip(('p50','p95','p99','max'),np.percentile(ts,[50,95,99,100]).tolist()))
    p = RNNoiseWaveform(dll, 1)
    x = rng.normal(0,.1,(48000,1)).astype(np.float32)
    y = np.concatenate([p.process(x[i:i+480]) for i in range(0,len(x),480)])
    p.close()
    corr=correlate(y[:,0],x[:,0],method='fft')
    lag=int(correlation_lags(len(y),len(x))[np.argmax(abs(corr))])
    model_ms=max(delays)/48
    lower=44+max(17,model_ms)+report['existing_effect_ms']
    report['waveform_rnnoise']={'impulse_offsets':[0,1,137,239,478,479],
        'delay_samples':delays,'broadband_correlation_lag_samples':lag,
        'algorithmic_delay_ms':model_ms,'compute_ms_by_channels':timings,
        'optimistic_total_lower_bound_ms':lower,
        'timing_screen':'Eligible around 68 ms; integrated timing not established.',
        'quality_test':'See comparison-short.json; run tools/compare_remaster_short.py separately.'}
    timings={}
    for ch in (2,6,8):
        p=GuidedRemaster(channels=ch)
        x=rng.normal(0,.03,(480,ch)).astype(np.float32); ts=[]
        for i in range(1050):
            start=time.perf_counter(); y=p.process(x,target=(-2,1.2,-1)); elapsed=time.perf_counter()-start
            assert np.isfinite(y).all()
            if i>=50: ts.append(elapsed*1000)
        timings[str(ch)]=dict(zip(('p50','p95','p99','max'),np.percentile(ts,[50,95,99,100]).tolist()))
    report['guided']={'queued_delay_samples':0, 'dsp_compute_ms_by_channels':timings,
        'neural_analysis_compute':'Existing StemgenRT on worker; tested in device trial.',
        'current_total_estimate_ms':44+17+report['existing_effect_ms'],
        'timing_screen':'Preserves accepted nominal 65 ms estimate; actual total latency not independently measured.'}
    report['music_restoration_pure']={'source':'https://arxiv.org/html/2607.12872v1',
        'algorithmic_delay_ms':1024/44.1,'optimistic_total_lower_bound_ms':44+1024/44.1+report['existing_effect_ms'],
        'timing_screen':'Around 71 ms optimistic, outside current target.', 'neural_inference_executed':False,
        'checkpoint_status':'Documented manoskary/pure-crn and pure-dfn HF references returned HTTP 401.'}
    (root/'candidate-gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
