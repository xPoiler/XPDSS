"""Small, reproducible quality probe. Metrics are not a listening verdict.

Uses licensed clips downloaded into .work/ai-remaster/sources and isolated
soundfile in .work/ai-remaster/deps. No live audio is recorded or played.
"""
from pathlib import Path
import json
import sys
import time
import math
import html

import numpy as np
from scipy.signal import resample_poly, butter, sosfilt

ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/'.work/ai-remaster'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(LAB/'deps'))
import soundfile as sf
from ai_live import load_model, default_model_dir, StreamingResampler
from tools.remaster_lab import Guidance, GuidedRemaster, ObservedModel


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)+1e-15))


def aligned_error_db(reference, signal):
    # Scale-invariant distortion at the known zero sample offset.
    r=reference.astype(np.float64).ravel(); s=signal.astype(np.float64).ravel()
    a=float(np.dot(r,s)/(np.dot(r,r)+1e-15))
    return float(10*np.log10((np.sum((a*r)**2)+1e-15)/(np.sum((s-a*r)**2)+1e-15)))


def main():
    samples=8*48000; clips={}; report=[]
    for name in ('choice','brahms','libri1','robin'):
        x,sr=sf.read(LAB/'sources'/f'{name}.hq.ogg',dtype='float32',always_2d=True)
        gcd=math.gcd(sr,48000); x=resample_poly(x,48000//gcd,sr//gcd)
        if x.shape[1]==1: x=np.repeat(x,2,axis=1)
        if len(x)>samples+48000: x=x[48000:]
        x=np.tile(x,(math.ceil(samples/len(x)),1))[:samples,:2]
        clips[name]=x*(.35/max(float(abs(x).max()),1e-6))
    clips['mixed_proxy']=clips['choice']*.45+clips['libri1']*.7+clips['robin']*.25
    model=load_model(default_model_dir())
    assets=LAB/'listening'; assets.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261027)
    try:
        for name,clean in clips.items():
            mud=sosfilt(butter(1,[180,500],btype='bandpass',fs=48000,output='sos'),clean,axis=0)
            harsh=sosfilt(butter(1,[3200,6500],btype='bandpass',fs=48000,output='sos'),clean,axis=0)
            variants={'clean':clean,'mud':clean+(10**(8/20)-1)*mud,
                      'harsh':clean+(10**(8/20)-1)*harsh,
                      'hiss':clean+rng.normal(0,rms(clean)*.1,clean.shape)}
            for defect,x in variants.items():
                x=x.astype(np.float32)
                model.reset(); g=Guidance(); observed=ObservedModel(model,g)
                rs=StreamingResampler(48000,44100); partial=np.zeros((0,2),np.float32)
                p=GuidedRemaster(); outputs=[]; neural_ms=[]; dsp_ms=[]
                for at in range(0,len(x),480):
                    block=x[at:at+480]
                    # Render using only guidance from PREVIOUS audio blocks.
                    begin=time.perf_counter(); outputs.append(p.process(block,g))
                    dsp_ms.append(1000*(time.perf_counter()-begin))
                    pending=np.concatenate((partial,rs.process(block)))
                    used=len(pending)//128*128
                    for k in range(0,used,128):
                        begin=time.perf_counter(); observed.process(pending[k:k+128])
                        neural_ms.append(1000*(time.perf_counter()-begin))
                    partial=pending[used:]
                y=np.concatenate(outputs)
                assert np.isfinite(y).all() and y.shape==x.shape
                ref=clean[3*48000:]; source=x[3*48000:]; out=y[3*48000:]
                inp_score=aligned_error_db(ref,source); out_score=aligned_error_db(ref,out)
                matched=out*rms(source)/rms(out)
                # Same gain for both files AFTER matching, preserving relative level.
                gain=min(1.,.8/max(float(abs(source).max()),float(abs(matched).max()),1e-6))
                base=f'{name}-{defect}'
                sf.write(assets/f'{base}-original.wav',source*gain,48000,subtype='PCM_24')
                sf.write(assets/f'{base}-guided.wav',matched*gain,48000,subtype='PCM_24')
                report.append({'clip':name,'condition':defect,
                    'input_si_sdr_db':inp_score,'output_si_sdr_db':out_score,
                    'delta_si_sdr_db':None if defect=='clean' else out_score-inp_score,
                    'raw_gain_change_db':20*np.log10(rms(out)/rms(source)),
                    'peak':float(abs(y).max()),'final_control_db':g.target,
                    'final_neural_vocal_share':g.vocal_share,
                    'model_p99_ms':float(np.percentile(neural_ms,99)),
                    'dsp_p99_ms':float(np.percentile(dsp_ms,99)),
                    'rms_matched_ab_error_db':20*np.log10(rms(matched)/rms(source)),
                    'file_base':base})
                print(f'{name}/{defect}: complete',flush=True)
    finally:
        model.close()
    result={'sample_rate':48000,'clip_duration_s':8,'evaluation_after_s':3,
            'model':'Existing checksum-verified StemgenRT partitions',
            'limitations':['Small public corpus; mixed_proxy is constructed, not an actual film/game.',
                          'Objective restoration metrics do not measure listener preference.',
                          'A/B is RMS matched, not BS.1770 integrated-loudness matched.',
                          'Isolated enhancer outputs; existing DSP and room/speakers not modeled here.'],
            'cases':report}
    (LAB/'quality-results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['<!doctype html><meta charset="utf-8"><title>AI Remaster laboratory comparisons</title>',
           '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#16191f;color:#eef}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #444}audio{width:280px}a{color:#9cf}</style>',
           '<h1>AI Remaster laboratory comparisons</h1><p>Experimental, not enabled in XPDSS. Same RMS level in each pair; no autoplay. Clean clips check unwanted changes. Mud/harsh/hiss are deliberately added defects. These clips do not establish that the processing sounds better.</p>',
           '<table><tr><th>Clip / condition</th><th>Original input</th><th>AI-guided prototype</th></tr>']
    for r in report:
        b=r['file_base']
        lines.append(f'<tr><td>{html.escape(r["clip"]+" / "+r["condition"])}</td><td><audio controls src="{b}-original.wav"></audio></td><td><audio controls src="{b}-guided.wav"></audio></td></tr>')
    lines.append('</table><h2>Attribution</h2>')
    for f in sorted((LAB/'sources').glob('*.txt')):
        lines.append('<pre>'+html.escape(f.read_text(encoding='utf-8'))+'</pre>')
    (assets/'index.html').write_text('\n'.join(lines),encoding='utf-8')
    print('Quality probe complete; listening samples and metrics saved.',flush=True)


if __name__=='__main__': main()
