#!/usr/bin/env python3
"""End-to-end latency + WER benchmark for the live transcription path."""

import argparse
import json
import os
import struct
import subprocess
import time
import tempfile
import numpy as np

from wer_test import normalize_text, wer, compress_to_ogg as encode_audio_to_ogg, transcribe_with_provider

SAMPLE_RATE = 16000
OUT_DIR = tempfile.gettempdir()
OUT_WAV = os.path.join(OUT_DIR, "voicetype-e2e-bench.wav")


def write_wav(path: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    audio_int16 = (samples * 32767).astype(np.int16)
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_int16) * 2))
        f.write(audio_int16.tobytes())


def record(seconds: float) -> np.ndarray:
    print(f"Recording {seconds:.1f}s ...")
    proc = subprocess.Popen(
        ['parec', '--raw', '--format=s16le', f'--rate={SAMPLE_RATE}', '--channels=1'],
        stdout=subprocess.PIPE,
    )
    frames = []
    target = int(SAMPLE_RATE * seconds * 2)
    got = 0
    while got < target:
        chunk = proc.stdout.read(min(4096, target - got))
        if not chunk:
            break
        frames.append(chunk)
        got += len(chunk)

    proc.terminate()
    proc.wait()
    raw = b''.join(frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def analyze(samples: np.ndarray) -> str:
    if len(samples) == 0:
        return "silent"
    f32 = samples.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(f32)))
    rms = float(np.sqrt(np.mean(f32 ** 2)))
    ratio = (np.abs(f32) < 0.01).sum() / float(len(f32)) * 100.0
    return f"duration={len(samples)/SAMPLE_RATE:.2f}s peak={peak:.4f} rms={rms:.4f} silence={ratio:.1f}%"


def basic_normalize(audio_f32: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio_f32)))
    if peak < 0.005:
        return audio_f32
    return np.clip(audio_f32 * (0.9 / peak), -1.0, 1.0)


def preprocess_audio(audio_f32: np.ndarray) -> np.ndarray:
    try:
        from audio_preprocess import AudioPreprocessor
        pre = AudioPreprocessor()
        if pre.enabled:
            return pre.process(audio_f32.copy())
    except Exception:
        pass
    return basic_normalize(audio_f32)


def run_variant(audio_f32: np.ndarray, label: str, do_preprocess=True, aggressive=False, provider="auto", encode_profile=None):
    timings = {}

    t0 = time.time()
    if do_preprocess:
        processed = preprocess_audio(audio_f32)
    else:
        processed = basic_normalize(audio_f32)
    timings['preprocess_s'] = time.time() - t0

    t0 = time.time()
    ogg = encode_audio_to_ogg(
        processed,
        aggressive=aggressive,
        provider=provider,
        profile=encode_profile,
    )
    timings['encode_s'] = time.time() - t0

    t0 = time.time()
    text = transcribe_with_provider(ogg, provider=provider)
    timings['transcribe_s'] = time.time() - t0

    size = os.path.getsize(ogg) if os.path.exists(ogg) else 0
    try:
        os.remove(ogg)
    except Exception:
        pass
    return text, size, timings


def run_optimize(top_samples, budget, refresh, aggressive_encode, min_samples, provider="auto", encode_profile=None):
    optimize_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimize.py')
    if not os.path.exists(optimize_script):
        return False, 'optimize.py not found'

    cmd = [
        'python3', optimize_script,
        '--use-real-samples',
        '--top-samples', str(top_samples),
        '--budget', str(budget),
    ]
    if provider:
        cmd.extend(['--provider', provider])
    if encode_profile:
        cmd.extend(['--encode-profile', encode_profile])
    if refresh:
        cmd.append('--refresh-samples')
    if aggressive_encode:
        cmd.append('--aggressive-encode')

    if min_samples is not None:
        env = os.environ.copy()
        env['VOICETYPE_AUTO_LEARN_MIN_SAMPLES'] = str(min_samples)
    else:
        env = None

    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (p.stdout or '') + '\n' + (p.stderr or '')
    if p.returncode != 0:
        return False, out.strip()

    return True, out.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--aggressive', action='store_true', help='use silence trim + atempo in encode path')
    parser.add_argument('--reference', default='', help='ground truth text for WER check')
    parser.add_argument('--optimize', action='store_true', help='run optimize.py after benchmark using real samples')
    parser.add_argument('--optimize-iterations', type=int, default=1, help='repeat optimization passes')
    parser.add_argument('--optimize-budget', type=int, default=90)
    parser.add_argument('--optimize-top', type=int, default=10)
    parser.add_argument('--optimize-refresh', action='store_true', help='refresh sample GT each optimization run')
    parser.add_argument('--optimize-aggressive', action='store_true', help='optimize with aggressive encoding')
    parser.add_argument('--optimize-min-samples', type=int, default=10)
    parser.add_argument('--provider', default='auto', help='provider for benchmark and optimize run: auto|groq|fal')
    parser.add_argument('--encode-profile', default=None, help='encoder profile to test')
    args = parser.parse_args()

    print('=== PulseAudio sources ===')
    subprocess.run(['pactl', 'list', 'short', 'sources'])
    print()

    ref = args.reference.strip()
    if ref:
        print(f'=== Reference WER enabled ===')
        print(f"  reference: {ref}")
    else:
        print('=== Reference WER disabled (no --reference provided) ===')

    rows = []

    for i in range(max(1, args.iterations)):
        print(f"\n=== Iteration {i + 1}/{args.iterations} ===")
        t0 = time.time()
        samples = record(args.seconds)
        capture_s = time.time() - t0
        print(f"capture: {capture_s:.3f}s | {analyze(samples)}")

        if len(samples) == 0:
            print('No audio captured. aborting this iteration')
            continue

        write_wav(OUT_WAV, samples, SAMPLE_RATE)

        baseline_text, baseline_bytes, baseline_t = run_variant(
            samples,
            f'base_{i}',
            do_preprocess=False,
            aggressive=False,
            provider=args.provider,
            encode_profile=args.encode_profile,
        )
        pipe_text, pipe_bytes, pipe_t = run_variant(
            samples,
            f'pipe_{i}',
            do_preprocess=True,
            aggressive=args.aggressive,
            provider=args.provider,
            encode_profile=args.encode_profile,
        )

        baseline_wer = wer(normalize_text(ref), normalize_text(baseline_text)) if ref else None
        pipe_wer = wer(normalize_text(ref), normalize_text(pipe_text)) if ref else None

        print(f"  baseline(raw): {baseline_text}")
        print(f"  pipeline : {pipe_text}")

        t_tot = {
            **{f'base_{k}': v for k, v in baseline_t.items()},
            **{f'pipe_{k}': v for k, v in pipe_t.items()},
            'capture_s': capture_s,
        }

        print(f"  baseline preprocess={t_tot['base_preprocess_s']:.3f}s encode={t_tot['base_encode_s']:.3f}s transcribe={t_tot['base_transcribe_s']:.3f}s size={baseline_bytes} bytes")
        print(f"  pipeline preprocess={t_tot['pipe_preprocess_s']:.3f}s encode={t_tot['pipe_encode_s']:.3f}s transcribe={t_tot['pipe_transcribe_s']:.3f}s size={pipe_bytes} bytes")

        if ref:
            print(f"  WER baseline:  {baseline_wer:.4f}")
            print(f"  WER pipeline: {pipe_wer:.4f}")

        rows.append({
            'capture_s': capture_s,
            'baseline': baseline_t,
            'pipeline': pipe_t,
            'bytes': {'baseline': baseline_bytes, 'pipeline': pipe_bytes},
            'wer': {'baseline': baseline_wer, 'pipeline': pipe_wer},
        })

    print('\n=== summary ===')
    if rows:
        cap = np.mean([r['capture_s'] for r in rows])
        bprep = np.mean([r['baseline']['preprocess_s'] for r in rows])
        ben = np.mean([r['baseline']['encode_s'] for r in rows])
        bt = np.mean([r['baseline']['transcribe_s'] for r in rows])
        pprep = np.mean([r['pipeline']['preprocess_s'] for r in rows])
        pen = np.mean([r['pipeline']['encode_s'] for r in rows])
        pt = np.mean([r['pipeline']['transcribe_s'] for r in rows])
        print(f'capture:  {cap:.3f}s')
        print(f'baseline: preprocess={bprep:.3f}s encode={ben:.3f}s transcribe={bt:.3f}s')
        print(f'pipeline: preprocess={pprep:.3f}s encode={pen:.3f}s transcribe={pt:.3f}s')

        if args.reference:
            base_wer = [r['wer']['baseline'] for r in rows if r['wer']['baseline'] is not None]
            pipe_wer = [r['wer']['pipeline'] for r in rows if r['wer']['pipeline'] is not None]
            if base_wer:
                print(f'WER baseline avg: {np.mean(base_wer):.4f}')
            if pipe_wer:
                print(f'WER pipeline avg: {np.mean(pipe_wer):.4f}')

    if args.optimize:
        print('\n=== auto optimize on captured samples ===')
        rounds = max(1, args.optimize_iterations)
        for i in range(rounds):
            if rounds > 1:
                print(f"\n  optimize round {i + 1}/{rounds}")

            ok, out = run_optimize(
                top_samples=args.optimize_top,
                budget=args.optimize_budget,
                refresh=args.optimize_refresh,
                aggressive_encode=args.optimize_aggressive,
                min_samples=args.optimize_min_samples,
                provider=args.provider,
                encode_profile=args.encode_profile,
            )
            if not ok:
                print(f'optimize failed: {out[:400]}')
                break

            params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimized_params.json')
            if os.path.exists(params_path):
                try:
                    with open(params_path) as f:
                        data = json.load(f)
                    print('  optimize completed')
                    print(f"  optimized WER (last run): {data.get('wer', '?')}")
                    print(f"  optimized baseline WER: {data.get('baseline_wer', '?')}")
                except Exception as e:
                    print(f'  could not parse optimized params: {e}')


if __name__ == '__main__':
    main()
