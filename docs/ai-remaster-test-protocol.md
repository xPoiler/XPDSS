# AI Remaster comparison protocol

Scope: isolated feasibility tests, not a shipped feature. Shared mode only,
Equalizer APO compatibility retained, no edits to app behavior or saved settings.

## Candidates

A: existing pretrained StemgenRT supplies neural stem evidence to an experimental
bounded tonal controller and causal channel-linked DSP. Its mastering decisions
are heuristic; this is AI-guided DSP, not a separately trained mastering model.
Historical guidance may lag, but audio never waits for guidance. Stereo and native
surround retain their channel assignment; LFE is preserved.

B: direct neural waveform enhancement. Evaluate a compact available RNNoise
runtime at 48 kHz as a low-delay baseline; do not mistake its speech training for
a universal remaster model. Screen music-trained PURE models as well: the cited
protocol has 23.22 ms algorithmic delay and its public checkpoint references
returned HTTP 401 during this investigation. Do not run random weights and call
that a quality evaluation.

## Gates, in order

1. Fixed delay: measure candidate delay with impulses at multiple hop offsets and
   broadband correlation where applicable. Record inference time separately.
   For direct enhancement, use the optimistic overlapping bound
   `44 + max(17, model_delay_ms) + 4.145833 ms`, not double-counted buffering.
   Treat the existing approximately 65 ms mode as accepted and roughly 68 ms
   as eligible if quality earns the extra delay, per the user's clarification.
   An optimistic bound is a screening estimate, not an integrated measurement.
2. Timing/reliability: test real trained guidance and causal DSP under actual
   shared-mode callback timing with saved effects, mixer, and analysis active.
   Compare to the unchanged baseline, record partial fallback and underruns.
   Count callback deadline exceedances separately from driver underrun reports.
3. Signal integrity: disabled identity, no queued delay, causality, silence,
   channel isolation, exact LFE preservation, finite output, and stale-guidance
   neutralization. A finite/correct output is not a quality endorsement.
4. Quality for survivors: licensed public speech, music, environmental effects,
   and a constructed mixed-content proxy. Use clean material plus deterministic
   known EQ/noise defects, preserve clean controls, and volume-match A/B samples.
   Measure gain, spectrum, and channel relationships. Do not claim preference
   from these objective metrics; provide examples for listening.

## Practical latency interpretation

The existing estimate of 65.15 ms is the accepted nominal 65 ms mode. Roughly
68 ms is allowed if the quality improvement is worthwhile. Do not reject on
fractions of a millisecond. Use small representative comparisons and short
timing/stability checks; reuse valid previous measurements rather than running
an exhaustive campaign. Capture timestamps are known to be implausible, so these
remain path estimates rather than acoustic measurements. No normal playback
buffers are changed as part of this experiment.

## Sources

- https://github.com/xiph/rnnoise
- https://github.com/pengzhendong/pyrnnoise (isolated wheel 0.4.5, SHA recorded)
- https://github.com/manoskary/audio-enhancement
- https://arxiv.org/html/2607.12872v1

Production promotion requires both timing qualification and useful audible
benefit. If neither approach qualifies, select neither for release rather than
manufacture a passing result.
