# VoiceType

Global hotkey voice-to-text for Linux. Press `Ctrl+Super+H` to record, press again to transcribe and type into the focused window.

## Install

```bash
git clone https://github.com/lee101/voicetype && cd voicetype && ./install.sh
```

## Setup

Set your [text-generator.io](https://text-generator.io) API key:

```bash
echo 'TEXT_GENERATOR_API_KEY=your_key_here' >> ~/.config/voicetype/env
```

If you have Groq credentials, add it for primary transcription:

```bash
echo 'GROQ_API_KEY=your_groq_key_here' >> ~/.config/voicetype/env
```

### Fallback ASR (optional)

Transcription order is: Groq → text-generator → [fal.ai Whisper](https://fal.ai/models/fal-ai/whisper) → local ASR. Set your fal key:

```bash
echo 'FAL_KEY=your_fal_key' >> ~/.config/voicetype/env
```

## Local model behavior

When recording starts, VoiceType warms the local model server in the background so the same model instance is reused for the current session. The model runs as a single socket server, so repeated chunks/sessions avoid repeated cold loads.

## Usage

- `Ctrl+Super+H` - Start/stop recording
- `Enter` - When actively recording, pressing Enter in any app now stops recording and transcribes
- Shows visualizer while recording
- Transcribed text is typed into the previously focused window

### Global enter-to-stop behavior

Pressing Enter only controls recording state if a recording is in progress. That keeps normal Enter behavior unchanged when not recording.

### Encoder behavior

`vad_chunker.py` defaults to WER-safe chunk encoding (no silence trimming and no `atempo` speedup). If you need the old aggressive shortening behavior, run the chunker with:

```bash
VOICETYPE_AGGRESSIVE_ENCODER=1 /usr/bin/python3 vad_chunker.py
```

You can also toggle capture/transcription safety for debugging misses:

```bash
VOICETYPE_DISABLE_VAD=1 /usr/bin/python3 vad_chunker.py              # disable speech detection
VOICETYPE_DISABLE_PREPROCESS=1 /usr/bin/python3 vad_chunker.py         # skip AudioPreprocessor
VOICETYPE_SAVE_RAW_CHUNK=1 /usr/bin/python3 vad_chunker.py             # additionally write chunk-*.raw.ogg for debugging
```

## WER Testing & Optimization

Place reference audio in `test_audio/` and run:

```bash
python3 wer_test.py --generate-ground-truth  # transcribe reference files once
python3 wer_test.py                           # measure WER with current params
python3 wer_test.py --baseline                # measure WER without preprocessing
python3 optimize.py --budget 100              # auto-optimize preprocessing params
```

For optimization on real captured samples, use:

```bash
python3 optimize.py --use-real-samples --top-samples 10 --refresh-samples --aggressive-encode
```

### Automatic online learning

After each recording, the daemon can auto-run optimization against the best recent samples (`samples/`) and update `optimized_params.json` automatically.

Enable with:

```bash
export VOICETYPE_AUTO_LEARN=1
```

Useful tuning knobs:

- `VOICETYPE_AUTO_LEARN_MIN_SAMPLES` (default `10`)
- `VOICETYPE_AUTO_LEARN_TOP_SAMPLES` (default `10`)
- `VOICETYPE_AUTO_LEARN_BUDGET` (default `90`)
- `VOICETYPE_AUTO_LEARN_COOLDOWN_MIN` (default `45`)
- `VOICETYPE_AUTO_LEARN_REFRESH_SAMPLES` (`0`/`1`, default `0`)
- `VOICETYPE_AUTO_LEARN_AGGRESSIVE` (`0`/`1`, default `0`, enables `--aggressive-encode`)

Keep `~/.config/voicetype/env` populated with:

```bash
TEXT_GENERATOR_API_KEY=...
FAL_KEY=...
```

When sample capture is running, `vad_chunker.py` now keeps up to the best 10 snippets in `samples/` (plus full utterances in `samples/utterances/`). If recordings are long, full chunks are still written to `/tmp/voicetype-chunks` as before.

Optimized params are saved to `optimized_params.json` and loaded automatically.

If you are running the installed user service (`~/.local/bin/voicetype`), make sure to redeploy after edits:

```bash
go build -o ~/.local/bin/voicetype .
cp *.py ~/.local/share/voicetype/
systemctl --user restart voicetype
```

This repo keeps working scripts at `~/code/voicetype/...`; after rebuild, the service will also pick them up automatically.

## Debug Pipeline vs Raw Transcription

Use the debug script to compare:

- raw text-generator transcription,
- raw fal whisper transcription,
- pipeline-processed transcription (current default: preprocessing only, no silence trim, no atempo),
- and a legacy aggressive pipeline mode (`--aggressive`) with silence trim + speedup.

```bash
python3 debug_pipeline_compare.py --record 8
python3 debug_pipeline_compare.py --record 8 --aggressive
python3 debug_pipeline_compare.py /path/to/audio.wav
```

Artifacts are written under `/tmp/voicetype-debug` for direct waveform inspection.

## Uninstall

```bash
./uninstall.sh
```
