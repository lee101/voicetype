#!/usr/bin/env python3
"""Record audio, check levels, and test transcription pipeline."""
import subprocess
import struct
import sys
import os
import time
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5
CHUNK_DIR = "/tmp/voicetype-chunks"
OUT_WAV = "/tmp/voicetype-benchmark.wav"
OUT_OGG = "/tmp/voicetype-benchmark.ogg"

def write_wav(path, samples, sr):
    import struct
    n = len(samples)
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + n * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', n * 2))
        f.write(samples.tobytes())

def record(duration):
    print(f"Recording {duration}s... speak now")
    proc = subprocess.Popen(
        ['parec', '--raw', '--format=s16le', f'--rate={SAMPLE_RATE}', '--channels=1'],
        stdout=subprocess.PIPE
    )
    frames = []
    total = SAMPLE_RATE * duration * 2
    read = 0
    while read < total:
        chunk = proc.stdout.read(min(4096, total - read))
        if not chunk:
            break
        frames.append(chunk)
        read += len(chunk)
    proc.terminate()
    proc.wait()
    raw = b''.join(frames)
    return np.frombuffer(raw, dtype=np.int16)

def analyze(samples):
    f32 = samples.astype(np.float32) / 32768.0
    peak = np.max(np.abs(f32))
    rms = np.sqrt(np.mean(f32 ** 2))
    silent_frames = np.sum(np.abs(f32) < 0.01) / len(f32) * 100
    print(f"  peak: {peak:.4f}")
    print(f"  rms:  {rms:.4f}")
    print(f"  silent: {silent_frames:.1f}%")
    if peak < 0.01:
        print("  WARNING: no audio detected - mic may not be working")
    elif peak < 0.05:
        print("  WARNING: very low audio levels")
    return peak > 0.01

def get_api_key():
    key = os.environ.get('TEXT_GENERATOR_API_KEY', '')
    if not key:
        env_file = os.path.expanduser('~/.config/voicetype/env')
        if os.path.exists(env_file):
            for line in open(env_file):
                line = line.strip()
                if line.startswith('TEXT_GENERATOR_API_KEY='):
                    key = line.split('=', 1)[1].strip().strip('"').strip("'")
    return key

def transcribe_api(ogg_path):
    import json
    key = get_api_key()
    if not key:
        print("  no API key found")
        return ""

    result = subprocess.run([
        'curl', '-s',
        '-X', 'POST',
        'https://api.text-generator.io/api/v1/audio-file-extraction',
        '-H', f'secret: {key}',
        '-F', f'audio_file=@{ogg_path}',
        '-F', 'translate_to_english=false',
    ], capture_output=True, text=True, timeout=60)

    try:
        data = json.loads(result.stdout)
        return data.get('text', '')
    except Exception as e:
        print(f"  API error: {e} | {result.stdout[:200]}")
        return ""

def main():
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else DURATION

    # list audio sources
    print("=== PulseAudio sources ===")
    subprocess.run(['pactl', 'list', 'short', 'sources'])
    print()
    print(f"=== Default source ===")
    subprocess.run(['pactl', 'get-default-source'])
    print()

    # record
    print(f"=== Recording ({dur}s) ===")
    samples = record(dur)
    print(f"  captured {len(samples)} samples ({len(samples)/SAMPLE_RATE:.1f}s)")

    # analyze
    print("=== Audio levels ===")
    has_audio = analyze(samples)

    # save
    write_wav(OUT_WAV, samples, SAMPLE_RATE)
    print(f"  saved: {OUT_WAV}")

    # compress
    subprocess.run([
        'ffmpeg', '-y', '-i', OUT_WAV,
        '-ac', '1', '-ar', '16000',
        '-c:a', 'libopus', '-b:a', '24k', '-application', 'voip',
        OUT_OGG
    ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    ogg_size = os.path.getsize(OUT_OGG)
    print(f"  compressed: {OUT_OGG} ({ogg_size} bytes)")

    # normalize
    print("\n=== Normalized ===")
    f32 = samples.astype(np.float32) / 32768.0
    peak = np.max(np.abs(f32))
    if peak > 0.005:
        gain = 0.9 / peak
        normalized = np.clip(f32 * gain, -1.0, 1.0)
        norm_int16 = (normalized * 32767).astype(np.int16)
        norm_wav = "/tmp/voicetype-benchmark-norm.wav"
        write_wav(norm_wav, norm_int16, SAMPLE_RATE)
        norm_peak = np.max(np.abs(normalized))
        norm_rms = np.sqrt(np.mean(normalized ** 2))
        print(f"  gain: {gain:.1f}x")
        print(f"  peak: {norm_peak:.4f}")
        print(f"  rms:  {norm_rms:.4f}")

        # compress normalized with silence removal
        norm_ogg = "/tmp/voicetype-benchmark-norm.ogg"
        subprocess.run([
            'ffmpeg', '-y', '-i', norm_wav,
            '-af', 'silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,'
                   'areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,'
                   'areverse,atempo=1.3',
            '-ac', '1', '-ar', '16000',
            '-c:a', 'libopus', '-b:a', '24k', '-application', 'voip',
            norm_ogg
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        norm_size = os.path.getsize(norm_ogg)
        print(f"  compressed: {norm_ogg} ({norm_size} bytes vs {ogg_size} raw)")
    else:
        norm_ogg = OUT_OGG

    if not has_audio:
        print("\nNo audio detected, skipping transcription test")
        print("Check: pactl get-default-source")
        return

    # transcribe both
    print("\n=== Transcription (raw) ===")
    t0 = time.time()
    text = transcribe_api(OUT_OGG)
    elapsed = time.time() - t0
    print(f"  result: {text}")
    print(f"  time: {elapsed:.2f}s")

    print("\n=== Transcription (normalized + trimmed) ===")
    t0 = time.time()
    text2 = transcribe_api(norm_ogg)
    elapsed2 = time.time() - t0
    print(f"  result: {text2}")
    print(f"  time: {elapsed2:.2f}s")

    # playback
    print("\n=== Playback ===")
    subprocess.run(['paplay', '--raw', f'--rate={SAMPLE_RATE}', '--format=s16le', '--channels=1', OUT_WAV.replace('.wav', '_raw')], timeout=dur+2) if False else None
    print("done")

if __name__ == '__main__':
    main()
