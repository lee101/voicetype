#!/usr/bin/env python3
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, 'optimization_cache.json')
OPTIMIZED_PATH = os.path.join(SCRIPT_DIR, 'optimized_params.json')
SAMPLES_DIR = os.path.join(SCRIPT_DIR, 'samples')
SAMPLE_GROUND_TRUTH_PATH = os.path.join(SAMPLES_DIR, 'ground_truth.json')
SAMPLE_EXTS = ('.wav', '.mp3', '.ogg', '.m4a', '.webm', '.flac')
DEFAULT_REAL_SAMPLE_COUNT = 10
SAMPLE_RATE = 16000

from audio_preprocess import DEFAULT_PARAMS
from wer_test import load_ground_truth, load_audio, compress_to_ogg, transcribe_with_provider, list_encode_profiles, _normalize_provider, wer, normalize_text

TEST_AUDIO_DIR = os.path.join(SCRIPT_DIR, 'test_audio')

BASE_WEIGHTS = np.array([0.5, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 0.5])

SWEEP_SPACE = {
    "weights_scale": [0.0, 0.5, 1.0, 1.5, 2.0],
    "energy_boost_factor": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
    "alpha": [0.05, 0.1, 0.2, 0.5, 1.0],
    "gain_clip_max": [5.0, 10.0, 20.0, 50.0],
    "gain_clip_min": [0.1, 0.3, 0.5, 0.8],
}

PRE_EQ_PRESETS = {
    "flat": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "low_cut": [0.3, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "speech_boost": [0.5, 0.8, 1.2, 1.5, 1.5, 1.2, 0.8, 0.5],
    "high_boost": [0.5, 0.7, 1.0, 1.0, 1.0, 1.3, 1.5, 1.5],
}


def cache_key(params, fname, aggressive=False, baseline=False, provider="auto", encode_profile=None):
    payload = {
        "params": params,
        "fname": fname,
        "aggressive": bool(aggressive),
        "baseline": bool(baseline),
        "provider": _normalize_provider(provider),
        "encode_profile": encode_profile or "default",
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def resolve_audio_path(key, audio_dir):
    if audio_dir is None or os.path.isabs(key):
        return key
    return os.path.join(audio_dir, key)


def transcribe_reference(audio_path):
    try:
        from fal_whisper import transcribe as transcribe_fal

        text = transcribe_fal(audio_path)
        if text:
            return text
    except Exception as e:
        print(f"FAL reference transcription failed for {os.path.basename(audio_path)}: {e}", file=sys.stderr)

    return transcribe_with_provider(audio_path, provider="auto")


def list_sample_candidates(samples_dir):
    roots = [samples_dir, os.path.join(samples_dir, 'utterances'), os.path.join(samples_dir, 'rolling')]
    paths = []

    for root in roots:
        if not os.path.isdir(root):
            continue
        for fname in sorted(os.listdir(root)):
            if fname.lower().endswith(SAMPLE_EXTS):
                paths.append(os.path.join(root, fname))

    # keep deterministic order and dedupe
    seen = set()
    deduped = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def score_sample(audio_f32):
    if audio_f32 is None or len(audio_f32) == 0:
        return 0.0

    dur = len(audio_f32) / float(SAMPLE_RATE)
    if dur < 0.3:
        return 0.0

    peak = float(np.max(np.abs(audio_f32)))
    if peak < 0.002:
        return 0.0

    rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
    return 0.7 * rms + 0.3 * min(1.0, dur / 8.0)


def load_real_sample_ground_truth(samples_dir=SAMPLES_DIR, top_n=DEFAULT_REAL_SAMPLE_COUNT, refresh=False):
    if top_n <= 0:
        return {}

    os.makedirs(samples_dir, exist_ok=True)

    existing = {}
    if os.path.exists(SAMPLE_GROUND_TRUTH_PATH) and not refresh:
        try:
            with open(SAMPLE_GROUND_TRUTH_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    # prune stale items from cache
    existing = {
        path: text
        for path, text in existing.items()
        if os.path.exists(path)
    }

    ranked = []
    for path in list_sample_candidates(samples_dir):
        if not os.path.exists(path) or os.path.basename(path) == 'ground_truth.json':
            continue

        if path in existing and not refresh:
            # still included below for rank; transcription already available
            audio = load_audio(path)
            if audio is None:
                continue
            s = score_sample(audio)
            if s > 0:
                ranked.append((s, path))
            continue

        audio = load_audio(path)
        if audio is None:
            continue

        s = score_sample(audio)
        if s <= 0:
            continue
        ranked.append((s, path))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = [path for _, path in ranked[:top_n]]

    payload = {}
    for path in selected:
        text = existing.get(path, '')
        if not text:
            text = transcribe_reference(path)
            text = normalize_text(text)
        if text:
            payload[path] = text

    if payload:
        with open(SAMPLE_GROUND_TRUTH_PATH, 'w') as f:
            json.dump(payload, f, indent=2)

    return payload


def make_params(weights_scale=1.0, energy_boost=0.3, alpha=0.1,
                gain_clip_min=0.5, gain_clip_max=50.0, pre_eq=None):
    return {
        "weights": (BASE_WEIGHTS * weights_scale).tolist(),
        "energy_boost_factor": energy_boost,
        "alpha": alpha,
        "gain_clip_min": gain_clip_min,
        "gain_clip_max": gain_clip_max,
        "pre_eq": pre_eq or [1.0] * 8,
    }


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache):
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)


def evaluate(params, ground_truth, cache, api_calls, audio_dir=TEST_AUDIO_DIR, aggressive_encode=False, provider="auto", encode_profile=None):
    from audio_preprocess import AudioPreprocessor

    results = {}
    all_cached = True

    for fname, ref_text in ground_truth.items():
        if not ref_text:
            continue

        ck = cache_key(
            params,
            fname,
            aggressive=aggressive_encode,
            provider=provider,
            encode_profile=encode_profile,
        )
        if ck in cache:
            hyp = cache[ck]
        else:
            all_cached = False
            fpath = resolve_audio_path(fname, audio_dir)
            audio = load_audio(fpath)
            if audio is None:
                continue

            preprocessor = AudioPreprocessor()
            preprocessor.apply_params(params)
            processed = preprocessor.process(audio.copy())
            ogg_path = compress_to_ogg(
                processed,
                aggressive=aggressive_encode,
                provider=provider,
                profile=encode_profile,
            )
            hyp = transcribe_with_provider(ogg_path, provider=provider)
            api_calls[0] += 1
            try:
                os.remove(ogg_path)
            except Exception:
                pass
            cache[ck] = hyp
            save_cache(cache)

        w = wer(ref_text, hyp)
        results[fname] = w

    if not results:
        return 1.0
    if all_cached:
        return float(np.mean(list(results.values())))
    return float(np.mean(list(results.values())))


def evaluate_baseline(ground_truth, cache, api_calls, audio_dir=TEST_AUDIO_DIR, aggressive_encode=False, provider="auto", encode_profile=None):
    results = {}
    for fname, ref_text in ground_truth.items():
        if not ref_text:
            continue

        ck = cache_key(
            {"_baseline": True},
            fname,
            aggressive=aggressive_encode,
            baseline=True,
            provider=provider,
            encode_profile=encode_profile,
        )
        if ck in cache:
            hyp = cache[ck]
        else:
            fpath = resolve_audio_path(fname, audio_dir)
            audio = load_audio(fpath)
            if audio is None:
                continue

            peak = np.max(np.abs(audio))
            if peak > 0.005:
                audio = audio * (0.9 / peak)
            ogg_path = compress_to_ogg(
                audio,
                aggressive=aggressive_encode,
                provider=provider,
                profile=encode_profile,
            )
            hyp = transcribe_with_provider(ogg_path, provider=provider)
            api_calls[0] += 1
            try:
                os.remove(ogg_path)
            except Exception:
                pass
            cache[ck] = hyp
            save_cache(cache)

        w = wer(ref_text, hyp)
        results[fname] = w

    if not results:
        return 1.0
    return float(np.mean(list(results.values())))


def optimize(
    budget=150,
    use_real_samples=False,
    samples_dir=SAMPLES_DIR,
    top_samples=DEFAULT_REAL_SAMPLE_COUNT,
    refresh_samples=False,
    aggressive_encode=False,
    provider="auto",
    encode_profile=None,
):
    provider = _normalize_provider(provider)
    available_profiles = list_encode_profiles(provider)
    if encode_profile and encode_profile not in available_profiles:
        print(f'WARN: encode profile "{encode_profile}" not valid for provider "{provider}", using default')
        encode_profile = None
    if not encode_profile:
        encode_profile = available_profiles[0] if available_profiles else "default"

    if use_real_samples:
        gt = load_real_sample_ground_truth(samples_dir=samples_dir, top_n=top_samples, refresh=refresh_samples)
        audio_dir = None
        print('=== real samples mode (FAL ground truth) ===')
        print(f'  samples_dir: {samples_dir}')
        print(f'  top samples: {top_samples}')
        print(f'  ground-truth entries: {len(gt)}')
    else:
        gt = load_ground_truth()
        audio_dir = TEST_AUDIO_DIR

    if not gt:
        print('No ground-truth audio found. Add files to test_audio/ or samples/.')
        return

    if use_real_samples:
        gt = {path: text for path, text in gt.items() if os.path.exists(path)}
        if not gt:
            print('No usable samples after filtering missing files.')
            return

    cache = load_cache()
    api_calls = [0]

    print('=== baseline (no preprocessing) ===')
    print(f'  provider: {provider} profile: {encode_profile}')
    baseline_wer = evaluate_baseline(
        gt,
        cache,
        api_calls,
        audio_dir=audio_dir,
        aggressive_encode=aggressive_encode,
        provider=provider,
        encode_profile=encode_profile,
    )
    print(f'  WER: {baseline_wer:.4f} ({api_calls[0]} API calls)')

    default_params = make_params()
    print('\n=== default params ===')
    default_wer = evaluate(
        default_params,
        gt,
        cache,
        api_calls,
        audio_dir=audio_dir,
        aggressive_encode=aggressive_encode,
        provider=provider,
        encode_profile=encode_profile,
    )
    print(f'  WER: {default_wer:.4f} ({api_calls[0]} API calls)')

    best = {"params": default_params, "wer": default_wer}
    sensitivity = {}

    print('\n=== phase 1: parameter sweeps ===')
    for param_name, values in SWEEP_SPACE.items():
        print(f'\n--- {param_name} ---')
        wers = []
        for val in values:
            if api_calls[0] >= budget:
                print(f'budget exhausted at {api_calls[0]} calls')
                break

            if param_name == "weights_scale":
                p = make_params(weights_scale=val)
            elif param_name == "energy_boost_factor":
                p = make_params(energy_boost=val)
            elif param_name == "alpha":
                p = make_params(alpha=val)
            elif param_name == "gain_clip_max":
                p = make_params(gain_clip_max=val)
            elif param_name == "gain_clip_min":
                p = make_params(gain_clip_min=val)
            else:
                continue

            w = evaluate(
                p,
                gt,
                cache,
                api_calls,
                audio_dir=audio_dir,
                aggressive_encode=aggressive_encode,
                provider=provider,
                encode_profile=encode_profile,
            )
            wers.append((val, w))
            marker = ' ***' if w < best['wer'] else ''
            print(f'  {param_name}={val}: WER={w:.4f}{marker}')
            if w < best['wer']:
                best = {"params": p, "wer": w}

        if wers:
            wer_values = [w for _, w in wers]
            sensitivity[param_name] = max(wer_values) - min(wer_values)

    print('\n--- pre_eq presets ---')
    for name, eq in PRE_EQ_PRESETS.items():
        if api_calls[0] >= budget:
            break
        p = make_params(pre_eq=eq)
        w = evaluate(
            p,
            gt,
            cache,
            api_calls,
            audio_dir=audio_dir,
            aggressive_encode=aggressive_encode,
            provider=provider,
            encode_profile=encode_profile,
        )
        marker = ' ***' if w < best['wer'] else ''
        print(f'  {name}: WER={w:.4f}{marker}')
        if w < best['wer']:
            best = {"params": p, "wer": w}

    if sensitivity and api_calls[0] < budget:
        top2 = sorted(sensitivity, key=sensitivity.get, reverse=True)[:2]
        print(f'\n=== phase 2: focused grid on {top2} ===')
        print(f'  sensitivity: {json.dumps({k: f"{v:.4f}" for k, v in sensitivity.items()})}')

        best_vals = {}
        gt_keys = list(gt.keys())
        first_key = gt_keys[0] if gt_keys else ''

        for pname in top2:
            sweep_results = []
            for val in SWEEP_SPACE[pname]:
                if pname == "weights_scale":
                    p = make_params(weights_scale=val)
                elif pname == "energy_boost_factor":
                    p = make_params(energy_boost=val)
                elif pname == "alpha":
                    p = make_params(alpha=val)
                elif pname == "gain_clip_max":
                    p = make_params(gain_clip_max=val)
                elif pname == "gain_clip_min":
                    p = make_params(gain_clip_min=val)
                else:
                    continue

                ck = cache_key(
                    p,
                    first_key,
                    aggressive=aggressive_encode,
                    provider=provider,
                    encode_profile=encode_profile,
                )
                if ck in cache:
                    w = evaluate(
                        p,
                        gt,
                        cache,
                        api_calls,
                        audio_dir=audio_dir,
                        aggressive_encode=aggressive_encode,
                        provider=provider,
                        encode_profile=encode_profile,
                    )
                    sweep_results.append((val, w))
                elif api_calls[0] >= budget:
                    break
                else:
                    w = evaluate(
                        p,
                        gt,
                        cache,
                        api_calls,
                        audio_dir=audio_dir,
                        aggressive_encode=aggressive_encode,
                        provider=provider,
                        encode_profile=encode_profile,
                    )
                    sweep_results.append((val, w))

            if sweep_results:
                best_vals[pname] = min(sweep_results, key=lambda x: x[1])[0]

        if len(best_vals) >= 2:
            p1, p2 = top2[0], top2[1]
            v1, v2 = best_vals[p1], best_vals[p2]
            offsets = [0.7, 1.0, 1.3]

            for m1 in offsets:
                for m2 in offsets:
                    if api_calls[0] >= budget:
                        break
                    kwargs = {}
                    for pname, base_v, mult in ((p1, v1, m1), (p2, v2, m2)):
                        adj = base_v * mult
                        if pname == "weights_scale":
                            kwargs["weights_scale"] = adj
                        elif pname == "energy_boost_factor":
                            kwargs["energy_boost"] = adj
                        elif pname == "alpha":
                            kwargs["alpha"] = adj
                        elif pname == "gain_clip_max":
                            kwargs["gain_clip_max"] = adj
                        elif pname == "gain_clip_min":
                            kwargs["gain_clip_min"] = adj
                    p = make_params(**kwargs)
                    w = evaluate(
                        p,
                        gt,
                        cache,
                        api_calls,
                        audio_dir=audio_dir,
                        aggressive_encode=aggressive_encode,
                        provider=provider,
                        encode_profile=encode_profile,
                    )
                    marker = ' ***' if w < best['wer'] else ''
                    print(f'  {p1}={v1*m1:.3f}, {p2}={v2*m2:.3f}: WER={w:.4f}{marker}')
                    if w < best['wer']:
                        best = {"params": p, "wer": w}

    print(f'\n=== results ===')
    print(f'  baseline WER: {baseline_wer:.4f}')
    print(f'  default WER:  {default_wer:.4f}')
    print(f'  best WER:     {best["wer"]:.4f}')
    print(f'  API calls:    {api_calls[0]}')

    improvement = (default_wer - best["wer"]) / default_wer * 100 if default_wer > 0 else 0
    print(f'  improvement:  {improvement:.1f}%')

    result = dict(best["params"])
    result["samples_mode"] = bool(use_real_samples)
    result["sample_dir"] = os.path.abspath(samples_dir)
    result["top_samples"] = top_samples
    result["aggressive_encoding"] = bool(aggressive_encode)
    result["provider"] = provider
    result["encode_profile"] = encode_profile
    result["wer"] = best["wer"]
    result["baseline_wer"] = baseline_wer
    result["optimized_at"] = time.strftime('%Y-%m-%dT%H:%M:%S')
    result["api_calls"] = api_calls[0]

    with open(OPTIMIZED_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  saved to {OPTIMIZED_PATH}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=150, help='max API calls')
    parser.add_argument('--use-real-samples', action='store_true', help='optimize on recent voice samples in samples/')
    parser.add_argument('--samples-dir', default=SAMPLES_DIR, help='directory that contains real samples')
    parser.add_argument('--top-samples', type=int, default=DEFAULT_REAL_SAMPLE_COUNT, help='max real samples to use')
    parser.add_argument('--refresh-samples', action='store_true', help='re-transcribe real samples and refresh sample GT cache')
    parser.add_argument('--aggressive-encode', action='store_true', help='evaluate with silence trim + atempo=1.3')
    parser.add_argument('--provider', default='auto', help='auto|groq|fal')
    parser.add_argument('--encode-profile', default=None, help='encoder profile override')
    args = parser.parse_args()

    optimize(
        budget=args.budget,
        use_real_samples=args.use_real_samples,
        samples_dir=args.samples_dir,
        top_samples=args.top_samples,
        refresh_samples=args.refresh_samples,
        aggressive_encode=args.aggressive_encode,
        provider=args.provider,
        encode_profile=args.encode_profile,
    )


if __name__ == '__main__':
    main()
