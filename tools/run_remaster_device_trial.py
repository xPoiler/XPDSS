"""Test-only adapters around the existing silent device benchmark.

No product modules are edited. Each invocation is a fresh process. The native
trial uses the same 17 ms reserve solely for comparison with the agreed ceiling;
it preserves channel positions and takes guidance from a separate stereo fold.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ai_live
from audio_analyzer import AudioBypass
from tools.remaster_lab import Guidance, GuidedRemaster, ObservedModel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['baseline','guided-upmix','guided-native'],required=True)
    p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--probe',type=Path,help='Preloaded synthetic/public fixture; never played')
    args=p.parse_args()
    probe=np.load(args.probe) if args.probe else None
    def source_block(self,data):
        if probe is None:
            return data
        result=np.zeros_like(data)
        n=min(result.shape[1],probe.shape[1])
        result[:,:n]=probe[(self._input_total_samples+np.arange(len(data)))%len(probe),:n]
        return result
    guidance=Guidance(); costs=[]; control_updates=[]
    if args.mode != 'baseline':
        original_load=ai_live.load_model
        ai_live.load_model=lambda directory: ObservedModel(original_load(directory),guidance)
        original_post=AudioBypass._apply_post_effects
        def post(self, out, out_ch, **kwargs):
            original_post(self,out,out_ch,**kwargs)
            if not hasattr(self,'_lab_remaster') or self._lab_remaster.channels != out_ch:
                self._lab_remaster=GuidedRemaster(self.sample_rate,out_ch)
            start=time.perf_counter()
            out[:]=self._lab_remaster.process(out,guidance)
            costs.append(1000*(time.perf_counter()-start))
            control_updates.append(guidance.updates)
        AudioBypass._apply_post_effects=post
    if args.mode == 'guided-native':
        # Keep all six captured channels in place; neural analysis gets a fold.
        original_capture=AudioBypass.input_callback
        original_feed=ai_live.LiveSeparator.feed
        current_fold=[None]
        def capture(self,data,frames,time_info,status):
            data=source_block(self,data)
            fold=data[:,:2].copy()
            if data.shape[1]>=6:
                fold += data[:,2:3]*.707
                fold[:,0] += data[:,4]*.5
                fold[:,1] += data[:,5]*.5
            if data.shape[1]>=8:
                fold[:,0] += data[:,6]*.5
                fold[:,1] += data[:,7]*.5
            current_fold[0]=fold*.4
            return original_capture(self,data,frames,time_info,status)
        def feed(self,data,*a,**kw):
            return original_feed(self,current_fold[0] if current_fold[0] is not None else data,*a,**kw)
        AudioBypass.input_callback=capture
        ai_live.LiveSeparator.feed=feed
        # Analysis and reserve active; render pass-through on that timeline.
        AudioBypass._live_requested=lambda self,out_ch=None: True
        original_apply=AudioBypass._apply_upmix
        def apply(self,source,out_ch=None):
            self.is_multichannel_detected=True
            self.manual_upmix=False
            return original_apply(self,source,out_ch)
        AudioBypass._apply_upmix=apply
        def route(self,source,out,out_ch,classic=None):
            self._live_delayed_src=source
            self._route('pass',source,out,out_ch)
        AudioBypass._route_live=route
    elif probe is not None:
        original_capture=AudioBypass.input_callback
        def capture(self,data,frames,time_info,status):
            return original_capture(self,source_block(self,data),frames,time_info,status)
        AudioBypass.input_callback=capture
    sys.argv=['benchmark_ai_live_device.py','--seconds',str(args.seconds),
              '--app-load','--source-input','--report',str(args.report)]
    try:
        runpy.run_path(str(ROOT/'tools/benchmark_ai_live_device.py'),run_name='__main__')
    except SystemExit as e:
        benchmark_exit=e.code
    report=json.loads(args.report.read_text())
    report['trial_mode']=args.mode
    report['prototype_dsp_ms_p50_p95_p99_max']=np.percentile(costs,[50,95,99,100]).tolist() if costs else []
    report['neural_guidance_updates']=guidance.updates
    report['final_guidance_db']=guidance.target
    report['baseline_benchmark_exit']=benchmark_exit
    report['added_audio_queue_samples']=0
    report['normal_app_changed']=False
    report['controlled_probe']=str(args.probe) if args.probe else None
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['trial_mode','prototype_dsp_ms_p50_p95_p99_max',
        'neural_guidance_updates','final_guidance_db','available_ai_samples_percent','negative_output_deadline_count']},indent=2))


if __name__=='__main__': main()
