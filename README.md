This is a personal project, updated as-needed and without any roadmap.

it consists of an audio processing engine (that works best with vb-audio's vb cable) to intelligently route windows audio to get instant switching between multichannel and stereo on my VERY-SPECIFIC setup. YMMV a lot.

it also contains a lot of advanced audio enhancements . ranging from upmixing to very advanced head tracking based processing.  it's basically a full vst suite for both speakers  and headphones for free and all contained in a single python based app. 

it also makes running speakers and bass shakers at the same time possible even on separate outputs in windows without using voicemeeter.

some of the features are not currently supported in development. 

AI LIVE is a separate, default-off streaming AI upmix option in Mixer & Steering.
It uses the 32 MB StemgenRT model on the CPU to put vocals in the centre,
drums/bass in front and other content toward the sides. CENTER FOCUS controls
vocal placement; AI INSTRUMENT SPREAD controls the side spread. Native surround
continues through the existing passthrough route unless manual upmix is selected.

Enable UPMIX and AI LIVE with a surround output. If needed, use DOWNLOAD LIVE MODEL
once. The three model files are checksum-verified before loading. AI MUSIC MODE
remains available: Live temporarily takes priority, and turning it off restores
the selected route and previous adaptive buffering. No device settings or other
saved effect settings are changed. Loading, missing models and slow CPUs use
conventional upmix on the same audio timeline. Partially late results keep their
valid samples and fade around the missing fragments; old stem blocks are never replayed.

Live runs inference during a fixed 17 ms input reserve, without adding the Music
or spectral upmixer's delay. On the tested 48 kHz VB-Cable / AAF S/PDIF path, the
driver-buffer estimate plus reserve, limiter and Tilt Correct is approximately
65 ms. This is an estimate, not measured source-to-speaker latency: the player,
virtual cable's source side and receiver can add delay. Lip sync needs a playback
check on the complete setup. Other sound devices or heavy CPU load can differ.
The model separates music stems; film/game dialogue and effects are not separate
trained categories, and separation quality has not been listening-tested here.

The initial version discarded a whole block when any part was late and switched
speaker levels abruptly. The playback fix uses a bounded allowance inside the
existing output deadline, after releasing the capture lock. It keeps partial
results, fades before missing regions and smooths unexpected level changes over
3 ms. This does not add another audio buffer. The status shows recent AI sample
availability; its cumulative late-block count also includes small late fragments.

Validation after the playback fix: 216 regression tests pass. A 30-second test
with 20 ms input bursts retained 99.17% of AI samples, with no wholly missing AI
blocks. A final 90-second native-device test with captured input, the Qt mixer
and analysis worker retained 100% of AI samples across 8860 checked blocks, with
zero input/output underruns and a 3.86 ms render-time p99. Availability can still
vary with Windows delivery timing and other load;
automated checks do not establish subjective sound quality or perceived lip sync.

The optional model's MIT license is in STEMGENRT-LICENSE.txt. Its source and weights
come from https://github.com/sweetspotsoundsystem/stemgen-rt at the immutable commit
recorded in ai_live.py. Existing dependencies are used without upgrades.

Developer timing checks (no audible output):

```text
python tools/benchmark_ai_live.py --pipeline --seconds 30
python tools/benchmark_ai_live_device.py --seconds 30
```

The first paces synthetic capture through the actual upmixer. The second opens
the configured WASAPI devices, injects synthetic data and sends only silence.
Driver timestamp results are diagnostic, not an acoustic lip-sync measurement.



LICENSE: 

the app is free to use but the code must not be used to create paid software of any kind. Only free distribution is allowed.
