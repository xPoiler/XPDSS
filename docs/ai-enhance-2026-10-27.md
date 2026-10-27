# AI Enhance — first experimental implementation

AI ENHANCE is a separate default-off switch in AUDIO ENHANCE. It uses the
existing 32 MB Live model, shared with AI Live when that route is active.
There is no new download or Python dependency. Settings persist per profile.

When off, the existing DSP path is used. Disabling releases the enhancement
over 40 ms, then removes the processor. Selected routing and buffering stay
unchanged whether enhancement is on or off. Master Bypass suppresses enhancement entirely.
Shared mode and the existing Equalizer APO path are retained.

Correction after user feedback: the initial implementation paused Music/Spectral
and forced a 17 ms reserve. This made Spectral switch to Classic, moving sounds
when the user wanted only enhancement. Those overrides have been removed.
AI Enhance now preserves Classic, Spectral, Music, AI Live and passthrough route
selection and their buffering policies. The enhancer adds no lookahead or queued
samples; it does not make a higher-latency selected engine fit the 65 ms target.
The existing approximately 65 ms estimate applies to AI Live, not every route.

The user also reported no audible improvement. The small comparison metrics
below did not demonstrate an improvement either. Treat this as an unsuccessful
quality experiment so far, not a completed general-purpose remaster feature.
No stronger effect or volume boost was added in response to that feedback.

## Processing

The separation worker supplies vocal/percussion evidence and spectral measures
about every 46 ms. Persistent evidence, noise-like spectra, and percussion
activity gate small corrections. This is a heuristic controller informed by a
trained separator, not a trained remastering network or a speech classifier.

- Low-mid attenuation up to 1.2 dB, gated by vocal evidence and persistent excess.
- Presence lift up to 0.6 dB, reduced on bright/noise-like content.
- Sustained harshness attenuation up to 1.5 dB, reduced around percussion.
- Slow dynamics adjustment between -0.8 and +0.5 dB; existing peak limiter retained.

Controls smooth over 200 ms. No fresh analysis for 500 ms, or an analysis error,
returns the controls toward neutral. Enhancement analysis failures cannot stop
the separator. A new stream/model resets analysis history.

Processing is linked across full-range channels; no content moves between them.
LFE is unchanged by enhancement. Native surround supplies a separate analysis
fold including center/surround channels and excluding LFE. Its audio is never
replaced by the fold. There is no wait for enhancement guidance.

## Limited verification

Device figures below describe the earlier implementation. After removing the
route override, only targeted route-preservation/off-path checks were run; no
new device or audio quality campaign was performed.

14 focused checks passed initially; after the latency-policy correction, six
affected cases passed (five repeated, one new; 15 distinct checks overall).
They include existing golden-output comparisons for Classic, Spectral and
Bypass, restoration after enabling/disabling with AI Live and native surround,
buffer/route restoration, profile persistence, LFE/channel preservation,
causality, stale analysis and isolated analysis failures. No full suite was run.

Six short audio examples used later sections of the existing licensed public
clips where possible: clean orchestral music, speech and mixed content, plus
added muddiness, harshness and hiss. The three clean controls changed by only
0.23–0.33 dB overall; their scale-invariant preservation scores were 40–44 dB.
The three degraded-reference score changes were approximately -0.002, -0.031,
and -0.030 dB. These results show small changes, **not a demonstrated restoration
improvement or listening preference**. The feature is deliberately labeled
experimental. Twelve RMS-matched files are provided for listening comparisons.
Isolated stereo enhancement p99 cost was 0.29–0.39 ms per 10 ms block.

| Device check | Duration | Estimated path | Result |
|---|---:|---:|---|
| AI Live + Enhance, saved effects and offscreen GUI/analysis | 25 s | ~65 ms | 99.94% AI availability; no wholly missing blocks, input-buffer empties, hardware underruns, queue resets or model errors |
| Corrected native passthrough + Enhance | 10 s | ~65 ms | Actual route `pass`, 17 ms reserve; no input-buffer empties, hardware underruns, queue resets or model errors |

The first 12-second native trial exposed inherited Spectral delay and one
input-buffer-empty event. Its old total estimate omitted that routing delay;
do not use it as the final latency result. The on-only policy was corrected and
the final 10-second check above verified it. Total device-check time was 47 s.

The upmix check had ten partly late stem blocks and three nominal callback
deadline exceedances; the final native check had one nominal exceedance of
0.020 ms. Render p99 was 7.14 ms and 2.10 ms respectively. These short runs do
not establish a hard acoustic maximum; driver timestamp anomalies remain as
documented previously. The ordinary app remained open during diagnostics.
All diagnostic output was silent and the saved configuration hash was unchanged.

The GUI was exercised offscreen and its snapshot inspected. The running user
process was not restarted; reopen XPDSS to load the new toggle.

## Files

- Implementation: `ai_enhance.py`, integration in `audio_analyzer.py`,
  `ai_live.py`, `gui_components.py`, `xpdss_state.py`.
- Focused tests: `tests/test_ai_enhance.py` and existing AI Live regression cases.
- Listening page: `.work/ai-remaster/enhance-preview/index.html`.
- Audio metrics: `.work/ai-remaster/enhance-preview.json`.
- Device results: `.work/ai-remaster/enhance-device-upmix.json`,
  `.work/ai-remaster/enhance-device-native-final.json`.
