#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import subprocess

STATS_PATH = os.path.join(os.path.dirname(__file__), 'ref_stats.json')
OPTIMIZED_PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'optimized_params.json')
TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'test_audio')
SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP = 256
BAND_EDGES = [0, 200, 400, 800, 1200, 2000, 3500, 5500, 8000]

DEFAULT_PARAMS = {
    "weights": [0.5, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 0.5],
    "energy_boost_factor": 0.3,
    "alpha": 0.1,
    "gain_clip_min": 0.5,
    "gain_clip_max": 50.0,
    "pre_eq": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def compute_spectral_stats(audio_f32, sr=SAMPLE_RATE):
    """Compute mean/std of log mel-like spectral energy in bands."""
    n_frames = (len(audio_f32) - FFT_SIZE) // HOP
    if n_frames < 1:
        return None, None

    bands = np.zeros((n_frames, 8))
    freqs_per_bin = sr / FFT_SIZE

    for i in range(n_frames):
        frame = audio_f32[i * HOP:i * HOP + FFT_SIZE]
        frame = frame * np.hanning(FFT_SIZE)
        spec = np.abs(np.fft.rfft(frame)) ** 2

        for b in range(8):
            lo = int(BAND_EDGES[b] / freqs_per_bin)
            hi = int(BAND_EDGES[b + 1] / freqs_per_bin)
            hi = min(hi, len(spec))
            if lo < hi:
                bands[i, b] = np.mean(spec[lo:hi])

    log_bands = np.log1p(bands * 1e4)
    return log_bands.mean(axis=0), log_bands.std(axis=0)


def load_reference_audio(path):
    """Load audio file to float32 array at 16kHz mono."""
    try:
        proc = subprocess.run(
            ['ffmpeg', '-i', path, '-f', 's16le', '-ac', '1', '-ar', str(SAMPLE_RATE), '-'],
            capture_output=True, timeout=10
        )
        if proc.returncode != 0:
            return None
        samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return samples
    except Exception:
        return None


def learn_reference_stats():
    """Compute spectral stats from test audio files."""
    all_means = []
    all_stds = []

    for fname in sorted(os.listdir(TEST_AUDIO_DIR)):
        fpath = os.path.join(TEST_AUDIO_DIR, fname)
        audio = load_reference_audio(fpath)
        if audio is None or len(audio) < SAMPLE_RATE:
            continue

        m, s = compute_spectral_stats(audio)
        if m is not None:
            all_means.append(m)
            all_stds.append(s)
            print(f"  {fname}: rms={np.sqrt(np.mean(audio**2)):.4f} peak={np.max(np.abs(audio)):.4f}", file=sys.stderr)

    if not all_means:
        print("no reference audio found", file=sys.stderr)
        return None

    ref_mean = np.mean(all_means, axis=0)
    ref_std = np.mean(all_stds, axis=0)
    ref_std = np.maximum(ref_std, 0.01)

    stats = {
        'mean': ref_mean.tolist(),
        'std': ref_std.tolist(),
        'n_files': len(all_means),
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f)
    print(f"saved ref stats from {len(all_means)} files", file=sys.stderr)
    return stats


def load_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            return json.load(f)
    return learn_reference_stats()


class AudioPreprocessor:
    def __init__(self):
        stats = load_stats()
        if stats:
            self.ref_mean = np.array(stats['mean'])
            self.ref_std = np.array(stats['std'])
            self.enabled = True
        else:
            self.enabled = False
        self.running_mean = None
        self.running_std = None
        self.params = self._load_params()
        self.alpha = self.params["alpha"]

    def _load_params(self):
        params = dict(DEFAULT_PARAMS)
        if os.path.exists(OPTIMIZED_PARAMS_PATH):
            try:
                with open(OPTIMIZED_PARAMS_PATH) as f:
                    saved = json.load(f)
                params.update({k: v for k, v in saved.items() if k in params})
                print(f"loaded optimized params (WER={saved.get('wer', '?')})", file=sys.stderr)
            except Exception:
                pass
        return params

    def apply_params(self, params):
        self.params.update(params)
        self.alpha = self.params["alpha"]
        self.running_mean = None
        self.running_std = None

    def _apply_eq(self, audio_f32, band_gains):
        n = len(audio_f32)
        spec = np.fft.rfft(audio_f32)
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        gain_curve = np.ones(len(spec))
        for b in range(8):
            mask = (freqs >= BAND_EDGES[b]) & (freqs < BAND_EDGES[b + 1])
            gain_curve[mask] = band_gains[b]
        return np.fft.irfft(spec * gain_curve, n=n).astype(np.float32)

    def process(self, audio_f32):
        if not self.enabled or len(audio_f32) < FFT_SIZE * 2:
            return self._basic_normalize(audio_f32)

        pre_eq = np.array(self.params["pre_eq"])
        if not np.allclose(pre_eq, 1.0):
            audio_f32 = self._apply_eq(audio_f32, pre_eq)

        cur_mean, cur_std = compute_spectral_stats(audio_f32)
        if cur_mean is None:
            return self._basic_normalize(audio_f32)

        if self.running_mean is None:
            self.running_mean = cur_mean
            self.running_std = cur_std
        else:
            self.running_mean = self.alpha * cur_mean + (1 - self.alpha) * self.running_mean
            self.running_std = self.alpha * cur_std + (1 - self.alpha) * self.running_std

        safe_std = np.maximum(self.running_std, 0.01)
        gain_per_band = self.ref_std / safe_std

        weights = np.array(self.params["weights"])
        wsum = weights.sum()
        if wsum > 0:
            weights = weights / wsum
        else:
            weights = np.ones(8) / 8
        broadband_gain = float(np.sum(gain_per_band * weights))

        mean_diff = float(np.sum((self.ref_mean - self.running_mean) * weights))
        energy_boost = np.exp(mean_diff * self.params["energy_boost_factor"])

        total_gain = broadband_gain * energy_boost
        total_gain = np.clip(total_gain, self.params["gain_clip_min"], self.params["gain_clip_max"])

        audio_out = audio_f32 * total_gain
        audio_out = np.tanh(audio_out)
        return audio_out

    def _basic_normalize(self, audio_f32):
        peak = np.max(np.abs(audio_f32))
        if peak > 0.005:
            return audio_f32 * (0.9 / peak)
        return audio_f32


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'learn':
        learn_reference_stats()
    else:
        stats = load_stats()
        if stats:
            print(f"ref stats: {stats['n_files']} files")
            print(f"  mean: {[f'{x:.2f}' for x in stats['mean']]}")
            print(f"  std:  {[f'{x:.2f}' for x in stats['std']]}")
        else:
            print("no stats available, run: python3 audio_preprocess.py learn")
