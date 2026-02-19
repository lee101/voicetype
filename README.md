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

### Fallback ASR (optional)

VoiceType prefers local ASR when available. If local inference is unavailable or fails, it falls back to text-generator API, then [fal.ai Whisper](https://fal.ai/models/fal-ai/whisper). Set your fal key:

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
VOICETYPE_SAVE_RAW_CHUNK=1 /usr/bin/python3 vad_chunker.py             # always write chunk-*.raw.ogg
```

## WER Testing & Optimization

Place reference audio in `test_audio/` and run:

```bash
python3 wer_test.py --generate-ground-truth  # transcribe reference files once
python3 wer_test.py                           # measure WER with current params
python3 wer_test.py --baseline                # measure WER without preprocessing
python3 optimize.py --budget 100              # auto-optimize preprocessing params
```

Optimized params are saved to `optimized_params.json` and loaded automatically.

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
