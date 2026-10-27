# XPDSS implementation handoff for Opus 5.5

Prepared 2026-10-27. Workspace: `D:\XPDSS` (Windows, Python 3.11, PowerShell).
This is a chronological record of code actually implemented in this conversation.
It excludes unimplemented proposals and research candidates. Experimental tools
are explicitly separated from application features. Historical test results are
reported as historical; no tests were run to write this handoff.

## 1. Startup silence / needing Apply Devices

**Reported problem:** opening XPDSS produced silence until the user opened Device
Setup and clicked Apply Devices. It occurred both at Windows startup and when
reopening the application later.

**Implemented in `audio_analyzer.py`:**

- Initialized Windows COM on the actual audio worker thread, not just the GUI/main
  thread. `AudioBypass.run()` calls `CoInitializeEx(None, 0)` before opening audio
  streams through `_run_audio_loop()`.
- Balanced successful COM initialization (`S_OK` or `S_FALSE`) with
  `CoUninitialize()` after shutdown. Handles `RPC_E_CHANGED_MODE` without claiming
  ownership of an initialization it did not perform.
- Added `_stop_lock` and separated `_stop_resources()` from `stop()` so worker COM
  teardown waits for stream cleanup, including when another thread initiates exit.
- Added/reworked `_check_stream_health()`: every two seconds, retry missing or
  inactive input/output streams. Acquire the stream lock non-blockingly; do not
  reopen during shutdown or simulation.
- Retry failed output opens without permanently treating a startup/driver failure
  as an unsupported channel layout. Clear failed channel counts when the output
  is unavailable, but preserve a working stereo fallback.
- Stream-open failure cleanup closes failed streams before subsequent retries.
  An existing secondary stream that stops is also refreshed by the health check.

**Relevant tests:** `tests/test_stream_recovery.py`. Covers startup/open/start
failures, cleanup before retry, working fallback preservation, shutdown behavior,
COM lifetime and concurrent external shutdown. Do not infer a new machine reboot
verification from this handoff; this turn only inspected the existing code/tests.

## 2. Separate low-latency AI Live upmix

The user wanted a new mode alongside existing high-latency AI Music, for films,
TV, gaming and music. AI Music was retained. The initial 50–60 ms target became
approximately 65 ms with explicit user acceptance.

**Implemented files:**

- `ai_live.py`: model loading, resampling, streaming separator and lifecycle.
- `audio_analyzer.py`: capture integration, aligned stem rendering, conditional
  buffer policy and fallback route.
- `xpdss_state.py`: profile setting `ai_live_mode`, default `False`.
- `gui_components.py`: AI LIVE toggle, separate model download and status.
- `tests/test_ai_live.py`, `tests/ai_live_off_reference.npz`.
- `tools/benchmark_ai_live.py`, `tools/benchmark_ai_live_device.py`.

### Backend and model actually used

- StemgenRT, pinned upstream commit
  `7f42f68a31947d51da090ddc18e16726fe20a980`.
- Three ONNX CPU partitions: `models/stemgenrt-live-drums.onnx`,
  `models/stemgenrt-live-bass.onnx`, `models/stemgenrt-live-vocals.onnx`.
  Approximately 32 MB total. Expected lengths/hashes are in `ai_live.MODEL_FILES`;
  downloads are checksum verified. License: `models/STEMGENRT-LICENSE.txt`.
- `ParallelModel` runs the partitions using two helper threads plus the caller,
  with one inference thread per ONNX session. Existing model weights retained.
- Model rate 44.1 kHz, hop 128 samples. Outputs correspond to the preceding hop.
  App stem order is drums, bass, other, vocals; the residual supplies `other`.
- `StreamingResampler` maintains state across blocks. Supported host rates:
  44100, 48000, 88200, 96000 and 192000 Hz.
- Loading verifies/warm-ups the model and checks whether CPU inference can keep
  up. `LiveController` handles asynchronous loading, stopping, invalidation,
  missing models, errors, and cancellation/rate changes during loading.

### Routing/buffering implementation

- `_live_requested()` limits the Live route to enabled upmix, surround output,
  no master bypass, and stereo input unless manual upmix is forced. Native
  multichannel normally retains passthrough.
- Capture feeds inference **before** the input reserve, so inference overlaps
  buffering. The Live reserve is 17 ms (`LIVE_RESERVE_MS`), not a second queue
  added after inference.
- The worker backlog is bounded to 40 ms. Discontinuities/overflow reset history
  and invalidate stale output tags. Results carry absolute source-sample tags.
- `_read_input_block()` associates stems with exactly the source samples read,
  including input-ring clock-drift compensation. It saves/restores the prior
  ordinary buffer target when entering/leaving Live and trims excess on entry.
- `_route_live()` puts drums/bass in front, vocal mid in center when enabled,
  and spreads the other stem according to the existing AI spread control.
  Residual content remains represented; center-disabled behavior is supported.
- Missing AI falls back to current, time-aligned Classic audio. It does not replay
  old waveform blocks. Existing effects/limiter remain downstream.
- AI Live temporarily takes precedence over AI Music without overwriting its
  saved setting. Classic/Live crossfades render shared Classic state only once.
- With Live off, original routing/buffering remains available. Existing Classic,
  Spectral and Bypass outputs have stored bit-for-bit regression references.

## 3. Corrected growing fallback counts and audible clicks

The first AI Live implementation was not usable for the user: fallback was
frequent and transitions clicked. This was a real implementation problem.

**Corrections actually implemented:**

- Allow a bounded wait for in-flight inference within the existing render-time
  allowance: at most 5 ms, at most half the callback period, constrained by DAC
  slack and elapsed input-read time. `_live_wait_budget()` implements the budget.
- Release the capture lock before waiting so new input can continue arriving.
  Notify waiters on result publication, reset, stop and failure.
- `pull_available()` retains correctly tagged partial results rather than
  rejecting an entire block because a small fragment is missing.
- Carry a conservative validity mask through clock-drift resampling.
- `live_mix_envelope()` fades toward Classic before known gaps. `LiveDezipper`
  smooths unexpected boundary jumps over approximately 3 ms, including the case
  where a valid prefix is shorter than the fade. No added reserve/lookahead.
- GUI distinguishes recent AI sample availability from cumulative late blocks;
  the cumulative count naturally increases and is not itself a current failure
  percentage.

**Historical verification:** the recorded full suite after this correction was
216 passing tests. A final 90-second native-device run then recorded 100% AI
availability over 8,860 checked blocks, no input/output underruns or model errors,
and renderer p99 3.86 ms. Other runs had partial misses; this is not a guarantee
of uninterrupted inference under every Windows scheduling condition.

The user subsequently reported satisfaction with this AI Live version. Preserve
this working feature independently of the unsuccessful enhancement work below.

Detailed historical record:
`docs/superpowers/plans/2026-10-27-ai-live.md` and matching spec. Their old skill
instructions are historical planning material, not instructions for the next
agent to install or invoke that workflow.

## 4. Shared-mode latency instrumentation and baseline

The user changed VB-CABLE settings themselves and asked for measurements, then
authorized only the baseline/measurement step of latency investigation.

**Implemented diagnostic changes in `tools/benchmark_ai_live_device.py`:**

- JSON report output; requested/default/opened-stream latency metadata.
- Actual callback sizes and timing, buffer fill before/after reads, render costs,
  bounded wait budgets, nominal callback deadlines, partial/whole AI misses,
  underruns, resets, errors, and unchanged-config verification.
- Flags for GUI/analysis load and captured source input. Hardware output is
  always silence; captured user audio is not saved.
- Explicitly mark implausible capture timestamps so the raw ADC-to-DAC field
  is not mistaken for an acoustic latency measurement.

**Recorded baseline:** `docs/latency-baseline-2026-10-27.md` and
`.work/ai-live/latency-step1.json`.

- WASAPI **shared** input: CABLE Output (VB-Audio Virtual Cable).
- Shared output: S/PDIF Out (AAF Family DCH Audio), six channels, 48 kHz.
- Both opened streams reported 22 ms despite a `low` request and 3 ms advertised
  default. Callback blocks were 480 samples / 10 ms, with uneven capture delivery.
- Estimated path: 44 ms drivers + 17 ms Live reserve + 1.5 ms limiter +
  127/48000 seconds Tilt Correct = approximately **65 ms**.
- That is an estimate excluding unmeasured source/receiver contributions.
  Capture timestamps were ahead of callback time and unsuitable for absolute
  latency measurement.
- The 90-second baseline had 99.9976% AI sample availability, three partially
  late blocks, no wholly missing blocks and no reported hardware underruns.

No driver optimization, exclusive mode, new latency slider, or buffer reduction
was shipped in this phase. The user depends on Equalizer APO: **no exclusive
mode**. The user accepts nominal 65 ms and roughly 68 ms if quality warrants it;
do not repeat the earlier mistake of treating 65.15 versus 65.00 as meaningful.

## 5. Remaster experiment tools that were actually written

These were isolated development tools, **not app processing modes**. They remain
on disk and may help interpret the subsequent implementation.

- `tools/remaster_lab.py`: first heuristic AI-guided tonal processor plus an
  RNNoise native-DLL waveform processor. Existing StemgenRT supplies guidance.
- `tools/test_remaster_candidates.py`: model-delay and processing-time probes.
- `tools/run_remaster_device_trial.py`: test-only adapters around silent device
  trials, including a native-channel analysis fold.
- `tools/evaluate_remaster_quality.py`: initial 20-case licensed-audio probe and
  RMS-matched listening files.
- `tools/compare_remaster_short.py`: eight short comparisons of original,
  first guided prototype, and direct RNNoise processing; compensates measured
  RNNoise delay only for offline comparison.
- `tests/test_remaster_lab.py`: isolated prototype integrity checks.

Assets/results live under `.work/ai-remaster/`: `sources/` contains public source
clips, hashes and licenses; `rnnoise/` contains the isolated runtime; `deps/`
contains isolated SoundFile. These were not installed as app features or global
dependency upgrades. Local listening pages are `listening/index.html` and
`comparison-short/index.html`.

The RNNoise runtime measured 960 samples / 20 ms algorithmic delay at 48 kHz.
The short comparison showed clean music attenuation of approximately 18–21 dB
before volume matching, plus substantial distortion beyond overall gain. It was
not integrated into playback. The guided prototype had mixed results and no
established universal improvement. Neither experiment constitutes a trained
general-purpose remastering model.

Reports: `docs/ai-remaster-comparison-2026-10-27.md`,
`.work/ai-remaster/quality-results.json`, `comparison-short.json`, and device JSONs.
Some raw early gate reports retain the superseded strict-65-ms rejection; read
the updated comparison document for interpretation.

## 6. AI Enhance feature actually integrated — quality goal not achieved

Following user approval, an experimental default-off **AI ENHANCE** toggle was
added to AUDIO ENHANCE. The user explicitly required existing processing when
disabled. This feature exists in the current source, but the user reports no
audible improvement and the comparisons did not demonstrate one.

**Files:** new `ai_enhance.py`; integration in `audio_analyzer.py`, `ai_live.py`,
`gui_components.py`, `xpdss_state.py`; tests in `tests/test_ai_enhance.py`.

### Current DSP implementation

- `EnhanceGuidance.observe()` runs on the existing Live separator worker. It
  aggregates 16 model hops, approximately 46 ms, and uses stem-energy shares,
  spectral flatness and band-energy ratios to publish an immutable snapshot.
- This is **handwritten heuristic control informed by a music separator**. It is
  not trained mastering, not a validated speech classifier, and not restoration
  of missing detail. No denoising/reconstruction model is in the app path.
- Persistent vocal/percussion/spectral evidence gates bounded corrections:
  low-mid cut up to 1.2 dB, presence lift up to 0.6 dB, harshness cut up to 1.5 dB,
  and slow gain riding between -0.8 and +0.5 dB.
- `EnhanceProcessor` uses three parallel causal band filters and linked gain.
  Controls smooth over 200 ms. Toggles fade over 40 ms; after disabling, the
  processor is removed once dry. There is no enhancer audio queue/lookahead.
- Full-range channels share controls; filter states are separate. LFE index 3
  in surround layouts is copied unchanged by this processor.
- `analysis_fold()` includes center/surrounds, excludes LFE, and is used only
  for analysis when AI Live routing is not active. It never replaces playback
  channels. AI Live and Enhance share the existing separator/model when both on.
- Guidance stale for 500 ms or in error returns controls toward neutral.
  Observer exceptions do not stop Live separation. Stream/model changes reset
  guidance; unsupported model rates leave enhancement DSP inactive.
- Enhancement runs after existing post-effects/center match and before output
  headroom/limiter. Master Bypass suppresses it.
- New profile key: `ai_enhance=False`. UI includes an experimental tooltip and
  status. Existing Live model download is reused.

### Implemented verification tools/results

`tools/evaluate_ai_enhance.py` generated six short comparisons and twelve
RMS-matched WAVs: `.work/ai-remaster/enhance-preview/index.html` and
`enhance-preview.json`. Clean examples changed overall level by about 0.23–0.33
dB. Known-defect restoration scores changed approximately -0.002, -0.031 and
-0.030 dB. **These results did not demonstrate improvement.**

The device benchmark gained `--enhance`, `--native`, `--probe`, `--ui-snapshot`,
route/reserve reporting and a spectral routing-delay estimate. A 25-second
AI Live + Enhance trial had 99.94% AI availability and no hardware underruns;
render p99 was 7.14 ms. Its path estimate remained approximately 65 ms.

Fourteen focused checks passed initially. Subsequent route/buffer changes had
small targeted checks. Do not claim the earlier 216-test full suite was rerun
against the enhancement implementation; it was not.

### Incorrect policy briefly implemented, then removed

An on-only policy was added to pause Music/Spectral and impose the Live 17 ms
reserve during Enhance. It attempted to keep every enhancement route near 65 ms.
For Spectral users, this switched spatial rendering to Classic and moved sounds.
The user noticed this with AI Enhance on and AI Live off. This was an unwanted
behavior change, not evidence that the enhancement model was placing sources.

A 12-second native diagnostic preceded that policy correction; a 10-second run
verified the forced low-latency native path. Those native numbers describe an
intermediate implementation and are **not current route-wide latency evidence**.

## 7. Latest implemented fix: AI Enhance no longer changes routing

After the user's report, the Music/Spectral overrides and Enhance-specific
buffer policy were removed from `audio_analyzer.py`.

**Current behavior:**

- `_spectral_selected()` is no longer gated by AI Enhance.
- AI Music activation/selection is no longer gated by AI Enhance.
- `_read_input_block()` applies the 17 ms reserve override only for AI Live,
  as before. AI Enhance by itself does not select or resize the input reserve.
- Enhancement still starts/shares neural analysis as needed, but analysis
  availability does not select the playback route.
- UI no longer claims Enhance pauses Music/Spectral or guarantees ~65 ms for
  every selected engine. It says selected routing/buffering remains unchanged.
- The enhancer adds no queued delay, but the chosen engine retains its existing
  latency. In particular, Spectral/Music do not become ~65 ms merely by enabling
  Enhance. There is no current all-routes 65 ms guarantee.

**Latest checks actually run: six passed in 1.91 seconds.**

- Two parameterized cases preserve Spectral upmix / delayed native passthrough
  and buffering through Enhance toggles. With enhancement guidance neutral,
  output matches a separate unchanged engine, with no routing crossfade.
- One case confirms Enhance does not pause/select away from AI Music.
- Three existing golden-output cases verify original Classic/Spectral/Bypass
  output with the feature off.

No new hardware or quality campaign followed this last correction. No stronger
effect or compensating volume boost was added after the user's negative feedback.
The running app was not forcibly restarted; source changes require reopening it.

## Current handoff boundaries

- **Keep:** startup recovery and the user-accepted AI Live upmix/fallback fixes.
- **Current but unsuccessful experiment:** AI Enhance. It remains implemented,
  default off for configs without the key, and has no demonstrated audible gain.
  The user explicitly said they hear no improvement. Do not market it as solved.
- **Required invariants:** existing behavior with new toggles off; no unwanted
  placement changes from Enhance; shared mode/Equalizer APO compatibility;
  bounded latency for the accepted AI Live route.
- The user is concerned about wasted usage and excessive testing. Do not repeat
  full suites, research campaigns or the existing audio comparisons without a
  concrete reason. This handoff request authorizes documentation, not more DSP
  development or app changes.
- Changes were made in place; no Git commits/worktree were created for this work.
  `.work/ai-live/` contains earlier baseline source copies, not a complete current
  rollback. Do not overwrite current production files from those copies blindly.
- Historical diagnostics kept normal playback running alongside a second test
  process, emitted silence, and checked that `config.json` stayed unchanged.
  Do not treat their timings as isolated-process or acoustic guarantees.
- Existing high-latency AI Music, Classic/Spectral engines and unrelated effects
  predate this work. They are not new implementations from this conversation.

For navigation, start with this file, then inspect the named functions in current
source. `docs/ai-enhance-2026-10-27.md` contains additional measured details, but
its device results explicitly precede the final route-preservation fix.

## 8. AI Enhance rework (Claude, 2026-10-27)

Diagnosis: the integration/plumbing was correct, but the old heuristic was
effectively inert. Caps were 0.6–1.5 dB, and the gates almost never opened. Its
own preview showed a 0.0 dB mud cut on audio with +8 dB of mud deliberately
added. `ai_enhance.py` is now a stem-aware tonal-balance corrector:

- It smooths octave-band levels over about 1.5 s and compares them with a
  reference mix curve.
- Five RBJ bands, identity at 0 dB: low shelf 90 Hz ±3, mud 300 Hz -4/+1,
  presence 3 kHz -2/+3, harsh 5.8 kHz -4/0, air shelf 11 kHz -2/+2.5.
- The stems gate each correction. Lows are filled only when there is a bass
  part. Presence is lifted only with a voice. Drums soften the harshness cut.
  The air band is never lifted when it is empty or hiss-like; hiss is trimmed.
- The gain-riding control was removed. Routing, latency and the toggle fade
  are unchanged.

Re-run of `tools/evaluate_ai_enhance.py`: restoration score for added mud went
from 6.46 to 8.41 dB, and for added harshness from 11.74 to 16.98 dB. Hiss is
1.4 dB worse (not a de-noiser). Full suite: 235 passed.

## 9. AI Enhance stem remaster on AI Live (Claude, 2026-10-27)

The user found the tonal-balance EQ (section 8) barely audible. When the AI Live
route is active, `ai_enhance.StemEnhancer` now processes the time-aligned Live
stems inside `_route_live`, before placement. This adds 0 ms of latency, costs
about 0.3 ms per 10 ms block, and leaves the residual untouched.

- Vocals: low-mid cut, presence and air lift, 2:1 levelling, then a level lift.
- Drums: transient punch from a fast/slow envelope ratio, plus kick shelf.
- Bass: harmonics generated above 200 Hz, plus a low shelf.
- Other: air and stereo width.
- A ~1.5 s loudness match keeps on and off at the same level.
- The new profile key `ai_enhance_amount` (default 0.7, slider 0–150%) sets
  the strength.
- The mix-level tonal EQ steps aside while on the Live route, so the two never
  stack. Other routes keep the tonal EQ.
- Stereo-fold previews: `.work/ai-remaster/stem-enhance-preview/`.
- Full suite: 237 passed.
