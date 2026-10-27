# Shared-mode latency baseline — step 1 only

Measured for 90 seconds on 2026-10-27 after the user's VB-CABLE settings change.
No buffer tuning, exclusive mode, dependency changes, or production-code changes.
The saved configuration was byte-for-byte unchanged across the run.

## Conditions

- Input: CABLE Output (VB-Audio Virtual Cable), WASAPI shared, six channels.
- Output: S/PDIF Out (AAF Family DCH Audio), WASAPI shared, six channels.
- 48 kHz, automatic callback block size, latency request `low` on both streams.
- Same AI model and 17 ms reserve; saved speaker effects loaded, including Tilt Correct.
- Captured input processed in memory; output always silence; no audio recorded.
- Mixer and analysis worker active in the diagnostic. The existing XPDSS instance
  (PID 22412, launched without extra arguments) remained running alongside it.
  This is a duplicate workload, not instrumentation inside that existing process.
- Shared mode preserves eligibility for Equalizer APO; silent output does not
  independently verify the receiver's audible output or APO's processing.

## Requested versus reported buffering

| Stage | Request / advertised default | Opened-stream report |
|---|---:|---:|
| Capture | low / 3 ms | 22 ms |
| Playback | low / 3 ms | 22 ms |
| AI input reserve | 17 ms | median post-read occupancy 17 ms |
| Limiter plus Tilt Correct | unchanged | estimated 4.146 ms |

The existing estimate is therefore 22 + 22 + 17 + 4.146 = **65.146 ms**.
This is neither an acoustic measurement nor a guaranteed maximum. The 3 ms
default is a requested value, not proof of achievable 3 ms stream latency.
The stream reports do not decompose the 22 ms into Windows, PortAudio, driver,
and hardware contributions. The 19 ms difference per stream is not established
as removable overhead.

## Delivery, occupancy, and processing

Both callbacks delivered 480 frames, representing 10 ms at 48 kHz.

| Measurement | Median | 95th percentile | Maximum |
|---|---:|---:|---:|
| Time between capture callbacks | 10.06 ms | 20.23 ms | 28.97 ms |
| Time between output callbacks | 10.08 ms | 11.42 ms | 27.23 ms |
| Input-buffer occupancy after read | 17.00 ms | 29.05 ms | 43.27 ms |
| Render callback work, including allowed AI wait | 2.92 ms | 4.72 ms | 27.18 ms |

Render work's 99th percentile was 6.03 ms. The output timestamp allowance was
10 ms, and the permitted AI wait was 5 ms. Two render calls exceeded that nominal
10 ms allowance; the driver reported zero output underruns. These are distinct
observations, not proof that all callbacks met their deadlines. Timings include
startup, and scheduling before Python callback entry is not separately measured.

Occupancy varies with delivery bursts and clock-drift correction. It is not an
extra delay to add again on top of the 17 ms reserve. Snapshot reads include
startup and can straddle an arriving capture callback.

## AI availability and health

- 8,856 checked output blocks after warm-up.
- **99.9976005%** of checked samples had AI results available.
- Three blocks contained a late fragment; zero blocks were wholly missing AI.
- Zero input overflows, output underruns, or empty-input events reported.
- Zero model errors or queue resets; one buffer resynchronization during the run.
- Diagnostic exits with code 1 for any late AI fragment; here that reflects the
  three partial misses, not a crash. No processing errors were reported.

## Timestamp limitation

The capture timestamp was 10–20 ms ahead of its callback's `currentTime`, giving
a negative capture age. Consequently the raw ADC-to-DAC percentile field in the
JSON must not be interpreted as real latency or used to claim an improvement.
The report explicitly flags these timestamps as implausible.

VB-CABLE's internal transfer time, the source application's buffering, and the
receiver's delay remain unmeasured. Determining these needs an independent
signal/loopback or acoustic measurement; no test tones were played in this run.

## Conclusion

The baseline identifies 10 ms callback blocks with uneven capture delivery,
22 ms reported latency on each opened stream, and almost complete AI availability
with the 17 ms reserve. It does not establish why each stream reports 22 ms or
prove that changing the request will lower it. Step 2 has not been attempted.

Diagnostic: `tools/benchmark_ai_live_device.py`

Raw measurements: `.work/ai-live/latency-step1.json`

Reproduce with:

```text
python tools/benchmark_ai_live_device.py --seconds 90 --app-load --source-input --report .work/ai-live/latency-step1.json
```
