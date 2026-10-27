# AI Remaster comparison — 2026-10-27

Historical comparison of the first prototypes. The subsequently authorized,
separate experimental AI Enhance implementation and its limited checks are
documented in `ai-enhance-2026-10-27.md`; the measurements below remain historical.

## Decision

**Prefer AI-guided processing as the development direction. Reject the tested
direct waveform enhancer on content preservation. Neither prototype yet earns
a general-purpose remaster toggle.**

The guided prototype adds no queued audio and can operate alongside AI Live or
on native surround without moving channels. Its quality is not consistently
better across content, however. The direct enhancer has a measured 20 ms signal
delay and an optimistic total around 68 ms, now explicitly acceptable to the
user if the quality earns it. Its completed quality probe instead shows severe
suppression of music. The earlier rejection before quality testing was premature
and is superseded by the results below.

No production toggle was added. No app routing, saved settings, models, driver
settings, or dependencies were changed. All device tests were shared mode with
silent output; the user's existing XPDSS process remained running. Equalizer APO
was not bypassed. No user audio was saved.

## Latency qualification

The existing mode's estimate is **about 65 ms**: 44 ms reported driver latency +
17 ms reserve + about 4.15 ms limiter/Tilt Correct delay. The guided candidate
preserves that estimate. The user's clarified acceptance is the existing 65 ms
mode, or roughly 68 ms if the quality improvement is worthwhile. Fractional-ms
differences are not a rejection criterion.

Capture timestamps remain ahead of the callback clock and cannot establish
acoustic delay. Device tests also showed occasional callback deadline overruns,
including in the baseline. None of these trials proves a hard maximum through
Windows, VB-CABLE, Equalizer APO, and the receiver.

| Candidate | Fixed delay / buffering | Optimistic path estimate | Decision |
|---|---|---:|---|
| A: AI-guided causal DSP | No new queued samples; existing 17 ms reserve | About 65 ms | Preferred development direction; quality needs work |
| B: RNNoise waveform enhancement, PyRnNoise 0.4.5 native runtime | Measured 960 samples at 48 kHz = 20 ms | About 68 ms optimistic | Timing eligible for evaluation; reject tested model on music preservation |
| Music-trained PURE published streaming configuration | Published 1024-sample analysis at 44.1 kHz = 23.220 ms | About 71 ms optimistic | Not pursued: outside current target and checkpoints unavailable |

For direct enhancement the optimistic calculation is
`44 + max(17, model_delay) + 4.145833` ms. This gives the candidate every benefit
of overlapping its work with existing buffering; it does not incorrectly add
both buffers in full. Scheduling, sample-rate conversion, and further inference
requirements can make it worse, not rescue this bound.
In particular, this is not a measured integrated RNNoise-plus-upmix path; serial
enhancement and separation could need more buffering. Since the quality test
rejects this candidate, no integration or further device trials are warranted.

### Direct enhancer measurements

RNNoise's pretrained native runtime was downloaded into the experiment directory,
with its wheel SHA-256 verified against PyPI metadata and recorded locally. The
normal Python environment was not modified. Direct C calls avoided wrapper
resampling or extra queues.

- Six impulse positions within a 480-sample hop all measured exactly **960 samples**
  of displacement. Broadband cross-correlation independently measured 960 samples.
- Stereo inference: median 0.829 ms, p99 0.884 ms per 10 ms block.
- Six-channel inference: median 2.489 ms, p99 2.561 ms per block.
- Eight-channel inference: median 3.322 ms, p99 3.490 ms per block.
- A short follow-up on real clips measured stereo median 0.829 ms, p99 0.931 ms,
  maximum 1.256 ms per block: consistent with the previous timing measurements.
- Quality testing is now complete for eight cases, summarized below. RNNoise
  is speech trained; these results do not eliminate all waveform enhancers.

PURE was screened from the authors' published protocol and source. Its advertised
checkpoint references `manoskary/pure-crn` and `manoskary/pure-dfn` returned HTTP
401 during this investigation. No random-weight model was substituted and no
claim is made that PURE inference was executed locally. Its documented delay
already exceeds the available budget. These findings apply to the tested and
screened candidates, not every possible future waveform model.

## Guided prototype design and signal tests

The existing checksum-verified StemgenRT partitions provide neural stem evidence.
A hand-designed controller uses the vocal energy share and spectral measurements
to guide bounded low-mid cuts, presence lift, and harshness reduction. This is
**AI-guided DSP, not a trained mastering model**. It does not claim neural
declipping, bandwidth reconstruction, or denoising.

Analysis runs on the separator worker. Rendering uses prior guidance and never
waits for it. Guidance older than 0.5 s returns smoothly toward neutral. Three
causal, linked filters process all full-range channels; LFE is unchanged.
There is no additional audio queue. The filters do introduce frequency-dependent
phase, as ordinary EQ does; zero queued delay is not a claim of zero phase change.

Six prototype integrity tests passed: disabled/zero-amount identity after active
processing; no future dependence and unchanged impulse location; eight-channel
isolation and exact LFE preservation; proportional stereo preservation; finite
silence and stale-guidance recovery; neural stem evidence affecting controls.
Four existing AI Live default/off-path regression cases also passed: **10 total**.

Isolated DSP timing for six channels: median 0.148 ms, p99 0.177 ms, maximum
0.226 ms per 10 ms block. In the full app environment Python scheduling raised
observed remaster work to p99 0.665–1.387 ms. Use the full-app results for
deployment decisions, not the favorable isolated number alone.

## Shared-mode device trials

All trials used 48 kHz, six-channel shared output, automatic 480-frame callbacks,
saved speaker effects, mixer and analysis load, and the 17 ms reserve. The normal
app ran concurrently, so these were duplicate-workload experiments. Trials were
sequential and their timing variation is not evidence that enhancement improves
reliability. Each report verified that the saved configuration was unchanged.

| Trial | Duration | Checked blocks | AI sample availability | Partly late blocks | Wholly missing blocks | Empty input after warm-up | Nominal deadline exceedances |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged baseline, captured input | 30 s | 2876 | 99.89424% | 5 | 2 | 2 | 2 |
| Guided upmix, captured input | 60 s | 5858 | 99.98261% | 11 | 0 | 0 | 1 |
| Guided upmix, controlled mixed fixture | 30 s | 2859 | 100% | 0 | 0 | 0 | 2 |
| Guided native surround, distinct-channel fixture | 60 s | 5856 | 99.99634% | 3 | 0 | 0 | 2 |

The controlled upmix and native trials each also recorded one input-buffer-empty
event during warm-up. All four trials reported zero hardware input overflows and
output underruns, zero model errors, and zero model queue resets. Deadline
exceedances and input buffering events are not hidden by those hardware counters.

The native-surround trial keeps source channels in place while a separate stereo
fold supplies historical neural guidance. It uses the same reserve as the AI Live
comparison, not the potentially shorter current unenhanced passthrough route.
The fixture contains music in the fronts, speech in the center, a 50 Hz LFE signal,
and environmental audio in the surrounds. No fixture signal was played aloud.

## Quality probe

Twenty cases: four public source excerpts (drum/bass music, orchestral music,
speech, bird/environmental sound) plus a constructed mixed-content proxy, each
tested clean and with added muddiness, harshness, or hiss. Neural guidance was
actually executed and only past-block guidance was used for rendering. Eight
seconds were processed per case; metrics use the final five seconds after settling.

Known EQ defects were +8 dB parallel band boosts; hiss used a seeded random source
at 20 dB RMS signal-to-noise ratio. Scale-invariant SDR measures recovery toward
the known clean reference at zero sample offset. It does **not** predict listener
preference or prove that a polished sound is better. A 0.1 dB threshold separates
improvement from worsening in this descriptive table.

| Added defect | Improved | Worsened | Mean SI-SDR change |
|---|---:|---:|---:|
| Muddiness | 4 / 5 | 1 / 5 | +1.23 dB |
| Harshness | 3 / 5 | 2 / 5 | +0.26 dB |
| Hiss | 0 / 5 | 5 / 5 | -0.99 dB |

Clean controls changed in level by -0.10 to -0.62 dB before matching. This alone
does not prove acceptable transparency. Hiss regression and changes to clean
environmental content show that the current controller is insufficient for an
always-on universal remaster. No post-hoc retuning was performed to make this
small corpus pass. Film/game coverage is a constructed proxy, not a representative
collection of actual titles. A later revision needs held-out material.

Forty listening files were generated as twenty pairs. Each pair is RMS matched
before one common peak-safe scaling; matching errors are recorded. This is not
BS.1770 loudness matching and no human preference result is claimed. Attribution
is included in the player page and source directory. Samples are for evaluation;
they are not bundled into the application.

## What the result supports

AI-guided processing is the technically stronger route under the existing nominal
65 ms budget, and it works experimentally without rearranging native channels.
The current prototype is not ready to be labeled general-purpose AI Remaster:
its quality is inconsistent. The tested direct waveform enhancer is eliminated
on music preservation, not the approximately 3 ms additional delay.

Normal playback remains unchanged. No latency setting was reduced to force a pass,
and no new toggle was added before qualification.

## Short follow-up under the clarified latency target

Eight cases only: clean and added-hiss versions of drum/bass music, orchestral
music, speech, and the constructed mixed-content proxy. Existing guided outputs
were reused. RNNoise processed the same inputs, with its measured 960-sample
delay compensated offline and its tail flushed. The first second was excluded
from all comparisons, leaving four-second examples. This alignment is solely
for comparison and does not remove live delay. All outputs were finite.

| Material | Direct model gain on clean input, before matching | Clean-input SI-SDR, guided / direct |
|---|---:|---:|
| Drum/bass music | -17.89 dB | 28.70 / -10.79 dB |
| Orchestral music | -20.53 dB | 23.18 / -6.09 dB |
| Speech | -0.29 dB | 33.85 / 10.74 dB |
| Constructed mixed content | -2.44 dB | 34.21 / 4.48 dB |

Scale-invariant scores compensate for overall gain; the very poor music scores
therefore also show changes beyond a volume reduction. This is unsuitable for
the requested broad entertainment use. Speech is much less attenuated, but the
test does not establish a perceptual improvement there either.

On the four hiss cases, guided restoration-score changes ranged from -0.25 to
-1.81 dB, versus -9.03 to -25.58 dB for the direct model. These scores measure
fidelity to the known clean reference, not pleasantness or noise removal alone.
They support rejecting this direct model for the intended use, not declaring
the guided prototype a universal quality improvement.

The 24 exported clips compare input, guided, and direct output at matching RMS
levels (verified after saving within 0.01 dB), with shared peak-safe scaling and
source attribution. Matching makes suppressed content audible for comparison;
the table above preserves the otherwise-hidden raw attenuation. No subjective
listening verdict is claimed. No further device run or full test suite was
needed: production code was unchanged and this candidate fails on content.

## Reproduction and artifacts

- Protocol: `docs/ai-remaster-test-protocol.md`
- Prototypes: `tools/remaster_lab.py`
- Integrity tests: `tests/test_remaster_lab.py`
- Delay/CPU gates: `tools/test_remaster_candidates.py`
- Shared-mode trial runner: `tools/run_remaster_device_trial.py`
- Quality probe: `tools/evaluate_remaster_quality.py`
- Raw results: `.work/ai-remaster/candidate-gates.json`, `device-*.json`, `quality-results.json`
- Listening page: `.work/ai-remaster/listening/index.html`
- Short three-way comparison: `.work/ai-remaster/comparison-short/index.html`
- Follow-up results/script: `.work/ai-remaster/comparison-short.json`, `tools/compare_remaster_short.py`
- Source hashes/licenses: `.work/ai-remaster/sources/`, `rnnoise-source.json`

## Primary references

- RNNoise: https://github.com/xiph/rnnoise
- Runtime packaging: https://github.com/pengzhendong/pyrnnoise
- Music-restoration runtime protocol: https://arxiv.org/html/2607.12872v1
- PURE source, inspected commit `ac0cbd66e02e1ff6b42c17cc15511df25124d319`:
  https://github.com/manoskary/audio-enhancement
- Public audio registry and attribution: https://librosa.org/doc/latest/recordings.html
