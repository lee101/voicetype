#!/usr/bin/env python3
import os
import sys
import json
import time
import struct
import subprocess
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_AUDIO_DIR = os.path.join(SCRIPT_DIR, 'test_audio')
GROUND_TRUTH_PATH = os.path.join(TEST_AUDIO_DIR, 'ground_truth.json')
WER_RESULTS_PATH = os.path.join(SCRIPT_DIR, 'wer_results.json')
WER_HISTORY_PATH = os.path.join(SCRIPT_DIR, 'wer_history.json')
SAMPLE_RATE = 16000


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
    key = get_api_key()
    if key:
        try:
            result = subprocess.run([
                'curl', '-s', '-X', 'POST',
                'https://api.text-generator.io/api/v1/audio-file-extraction',
                '-H', f'secret: {key}',
                '-F', f'audio_file=@{ogg_path}',
                '-F', 'translate_to_english=false',
            ], capture_output=True, text=True, timeout=60)
            text = json.loads(result.stdout).get('text', '')
            if text:
                return text
        except Exception as e:
            print(f"primary API error: {e}", file=sys.stderr)

    # fallback to fal whisper
    try:
        from fal_whisper import transcribe as fal_transcribe
        text = fal_transcribe(ogg_path)
        if text:
            print(f"  (fal fallback)", file=sys.stderr)
            return text
    except Exception as e:
        print(f"fal fallback error: {e}", file=sys.stderr)

    return ""


def load_audio(path):
    try:
        proc = subprocess.run(
            ['ffmpeg', '-i', path, '-f', 's16le', '-ac', '1', '-ar', str(SAMPLE_RATE), '-'],
            capture_output=True, timeout=10
        )
        if proc.returncode != 0:
            return None
        return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None


def write_wav(path, audio_f32):
    samples = (audio_f32 * 32767).astype(np.int16)
    n = len(samples)
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + n * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', n * 2))
        f.write(samples.tobytes())


def compress_to_ogg(audio_f32, path=None, aggressive=False):
    import tempfile
    wav_path = path or tempfile.mktemp(suffix='.wav')
    ogg_path = wav_path.replace('.wav', '.ogg')
    write_wav(wav_path, audio_f32)

    cmd = ['ffmpeg', '-y', '-i', wav_path]
    if aggressive:
        cmd += ['-af',
            'silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,'
            'areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,'
            'areverse,atempo=1.3']
    cmd += ['-ac', '1', '-ar', '16000',
            '-c:a', 'libopus', '-b:a', '16k', '-application', 'voip',
            '-vbr', 'on', '-compression_level', '10', ogg_path]
    subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    if aggressive and (not os.path.exists(ogg_path) or os.path.getsize(ogg_path) < 100):
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-af', 'atempo=1.3',
            '-ac', '1', '-ar', '16000',
            '-c:a', 'libopus', '-b:a', '16k', '-application', 'voip',
            ogg_path
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    os.remove(wav_path)
    return ogg_path


def normalize_text(text):
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def wer(ref, hyp):
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            sub = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    return d[len(r)][len(h)] / len(r)


def generate_ground_truth():
    print("generating ground truth from test_audio/...")
    gt = {}
    for fname in sorted(os.listdir(TEST_AUDIO_DIR)):
        if not fname.endswith(('.ogg', '.mp3', '.wav')):
            continue
        fpath = os.path.join(TEST_AUDIO_DIR, fname)
        print(f"  transcribing {fname}...", end=' ', flush=True)
        text = transcribe_api(fpath)
        print(f"'{text[:80]}...'") if len(text) > 80 else print(f"'{text}'")
        gt[fname] = text
    with open(GROUND_TRUTH_PATH, 'w') as f:
        json.dump(gt, f, indent=2)
    print(f"saved ground truth for {len(gt)} files")
    return gt


def load_ground_truth():
    if not os.path.exists(GROUND_TRUTH_PATH):
        return generate_ground_truth()
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def test_params(params=None, ground_truth=None, aggressive=False):
    from audio_preprocess import AudioPreprocessor

    if ground_truth is None:
        ground_truth = load_ground_truth()

    preprocessor = AudioPreprocessor()
    if params:
        preprocessor.apply_params(params)

    results = {}
    for fname, ref_text in ground_truth.items():
        if not ref_text:
            continue
        fpath = os.path.join(TEST_AUDIO_DIR, fname)
        audio = load_audio(fpath)
        if audio is None:
            continue

        processed = preprocessor.process(audio.copy())
        ogg_path = compress_to_ogg(processed, aggressive=aggressive)
        hyp_text = transcribe_api(ogg_path)
        w = wer(ref_text, hyp_text)
        results[fname] = {"wer": w, "ref": ref_text, "hyp": hyp_text}
        print(f"  {fname}: WER={w:.3f}")
        try:
            os.remove(ogg_path)
        except Exception:
            pass

    if results:
        avg = np.mean([r["wer"] for r in results.values()])
        results["avg_wer"] = float(avg)
        print(f"  avg WER: {avg:.3f}")

    return results


def test_baseline(ground_truth=None, aggressive=False):
    if ground_truth is None:
        ground_truth = load_ground_truth()

    results = {}
    for fname, ref_text in ground_truth.items():
        if not ref_text:
            continue
        fpath = os.path.join(TEST_AUDIO_DIR, fname)
        audio = load_audio(fpath)
        if audio is None:
            continue
        peak = np.max(np.abs(audio))
        if peak > 0.005:
            audio = audio * (0.9 / peak)
        ogg_path = compress_to_ogg(audio, aggressive=aggressive)
        hyp_text = transcribe_api(ogg_path)
        w = wer(ref_text, hyp_text)
        results[fname] = {"wer": w, "ref": ref_text, "hyp": hyp_text}
        print(f"  {fname}: WER={w:.3f} (baseline)")
        try:
            os.remove(ogg_path)
        except Exception:
            pass

    if results:
        avg = np.mean([r["wer"] for r in results.values()])
        results["avg_wer"] = float(avg)
        print(f"  avg WER: {avg:.3f} (baseline)")
    return results


def save_results(results, params=None):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params or "default",
        "results": results,
    }
    with open(WER_RESULTS_PATH, 'w') as f:
        json.dump(entry, f, indent=2)

    history = []
    if os.path.exists(WER_HISTORY_PATH):
        try:
            with open(WER_HISTORY_PATH) as f:
                history = json.load(f)
        except Exception:
            pass
    history.append(entry)
    with open(WER_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-ground-truth', action='store_true')
    parser.add_argument('--params', help='path to params JSON')
    parser.add_argument('--baseline', action='store_true', help='test with no preprocessing')
    parser.add_argument('--aggressive', action='store_true',
                        help='apply silenceremove+atempo=1.3 encoding (off by default to match live pipeline)')
    args = parser.parse_args()

    if args.generate_ground_truth:
        generate_ground_truth()
        return

    gt = load_ground_truth()

    if args.baseline:
        print("=== baseline (peak normalize only) ===")
        results = test_baseline(gt, aggressive=args.aggressive)
        save_results(results, "baseline")
        return

    params = None
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        print(f"=== testing params from {args.params} ===")
    else:
        print("=== testing with current/default params ===")

    results = test_params(params, gt, aggressive=args.aggressive)
    save_results(results, params)


if __name__ == '__main__':
    main()
