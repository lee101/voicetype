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

from audio_preprocess import DEFAULT_PARAMS
from wer_test import load_ground_truth, load_audio, compress_to_ogg, transcribe_api, wer, normalize_text

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


def cache_key(params, fname):
    s = json.dumps(params, sort_keys=True) + fname
    return hashlib.md5(s.encode()).hexdigest()


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


def evaluate(params, ground_truth, cache, api_calls):
    from audio_preprocess import AudioPreprocessor

    results = {}
    all_cached = True

    for fname, ref_text in ground_truth.items():
        if not ref_text:
            continue
        ck = cache_key(params, fname)
        if ck in cache:
            hyp = cache[ck]
        else:
            all_cached = False
            fpath = os.path.join(TEST_AUDIO_DIR, fname)
            audio = load_audio(fpath)
            if audio is None:
                continue
            preprocessor = AudioPreprocessor()
            preprocessor.apply_params(params)
            processed = preprocessor.process(audio.copy())
            ogg_path = compress_to_ogg(processed)
            hyp = transcribe_api(ogg_path)
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


def evaluate_baseline(ground_truth, cache, api_calls):
    results = {}
    for fname, ref_text in ground_truth.items():
        if not ref_text:
            continue
        ck = cache_key({"_baseline": True}, fname)
        if ck in cache:
            hyp = cache[ck]
        else:
            fpath = os.path.join(TEST_AUDIO_DIR, fname)
            audio = load_audio(fpath)
            if audio is None:
                continue
            peak = np.max(np.abs(audio))
            if peak > 0.005:
                audio = audio * (0.9 / peak)
            ogg_path = compress_to_ogg(audio)
            hyp = transcribe_api(ogg_path)
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


def optimize(budget=150):
    gt = load_ground_truth()
    cache = load_cache()
    api_calls = [0]

    print("=== baseline (no preprocessing) ===")
    baseline_wer = evaluate_baseline(gt, cache, api_calls)
    print(f"  WER: {baseline_wer:.4f} ({api_calls[0]} API calls)")

    default_params = make_params()
    print("\n=== default params ===")
    default_wer = evaluate(default_params, gt, cache, api_calls)
    print(f"  WER: {default_wer:.4f} ({api_calls[0]} API calls)")

    best = {"params": default_params, "wer": default_wer}
    sensitivity = {}

    # phase 1: one-at-a-time sweeps
    print("\n=== phase 1: parameter sweeps ===")
    for param_name, values in SWEEP_SPACE.items():
        print(f"\n--- {param_name} ---")
        wers = []
        for val in values:
            if api_calls[0] >= budget:
                print(f"budget exhausted at {api_calls[0]} calls")
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

            w = evaluate(p, gt, cache, api_calls)
            wers.append((val, w))
            marker = " ***" if w < best["wer"] else ""
            print(f"  {param_name}={val}: WER={w:.4f}{marker}")
            if w < best["wer"]:
                best = {"params": p, "wer": w}

        if wers:
            wer_values = [w for _, w in wers]
            sensitivity[param_name] = max(wer_values) - min(wer_values)

    # pre-EQ presets
    print("\n--- pre_eq presets ---")
    for name, eq in PRE_EQ_PRESETS.items():
        if api_calls[0] >= budget:
            break
        p = make_params(pre_eq=eq)
        w = evaluate(p, gt, cache, api_calls)
        marker = " ***" if w < best["wer"] else ""
        print(f"  {name}: WER={w:.4f}{marker}")
        if w < best["wer"]:
            best = {"params": p, "wer": w}

    # phase 2: focused grid on top-2 sensitive params
    if sensitivity and api_calls[0] < budget:
        top2 = sorted(sensitivity, key=sensitivity.get, reverse=True)[:2]
        print(f"\n=== phase 2: focused grid on {top2} ===")
        print(f"  sensitivity: {json.dumps({k: f'{v:.4f}' for k, v in sensitivity.items()})}")

        # find best value for each of top2 from phase 1
        best_vals = {}
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
                ck_test = cache_key(p, list(gt.keys())[0])
                if ck_test in cache:
                    w = evaluate(p, gt, cache, api_calls)
                    sweep_results.append((val, w))
            if sweep_results:
                best_vals[pname] = min(sweep_results, key=lambda x: x[1])[0]

        # small grid around best values
        if len(best_vals) >= 2:
            p1, p2 = top2[0], top2[1]
            v1, v2 = best_vals[p1], best_vals[p2]
            offsets = [0.7, 1.0, 1.3]
            for m1 in offsets:
                for m2 in offsets:
                    if api_calls[0] >= budget:
                        break
                    kwargs = {}
                    for pname, base_v, mult in [(p1, v1, m1), (p2, v2, m2)]:
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
                    w = evaluate(p, gt, cache, api_calls)
                    marker = " ***" if w < best["wer"] else ""
                    print(f"  {p1}={v1*m1:.3f}, {p2}={v2*m2:.3f}: WER={w:.4f}{marker}")
                    if w < best["wer"]:
                        best = {"params": p, "wer": w}

    # save best
    print(f"\n=== results ===")
    print(f"  baseline WER: {baseline_wer:.4f}")
    print(f"  default WER:  {default_wer:.4f}")
    print(f"  best WER:     {best['wer']:.4f}")
    print(f"  API calls:    {api_calls[0]}")
    improvement = (default_wer - best["wer"]) / default_wer * 100 if default_wer > 0 else 0
    print(f"  improvement:  {improvement:.1f}%")

    result = dict(best["params"])
    result["wer"] = best["wer"]
    result["baseline_wer"] = baseline_wer
    result["optimized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    result["api_calls"] = api_calls[0]

    with open(OPTIMIZED_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  saved to {OPTIMIZED_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=150, help='max API calls')
    args = parser.parse_args()
    optimize(budget=args.budget)


if __name__ == '__main__':
    main()
