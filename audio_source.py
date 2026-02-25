#!/usr/bin/env python3
import subprocess
import sys
from typing import Optional, Tuple

import numpy as np

AUDIO_SOURCE_CACHE = '/tmp/voicetype-audio-source'
MIN_WORKING_PEAK = 0.0008


def list_input_sources():
    try:
        result = subprocess.run(
            ['pactl', 'list', 'short', 'sources'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    sources = []
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if not name or '.monitor' in name:
            continue
        sources.append(name)
    return sources


def probe_source_signal(src: str, sample_rate: int = 16000, probe_seconds: float = 0.5) -> Optional[Tuple[float, float]]:
    probe_bytes = int(sample_rate * 2 * probe_seconds)
    proc = None
    try:
        proc = subprocess.Popen(
            [
                'parec',
                '--raw',
                '--format=s16le',
                f'--rate={sample_rate}',
                '--channels=1',
                f'--device={src}',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(probe_bytes)
        if not raw:
            return None
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        return peak, rms
    except Exception:
        return None
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def detect_best_source(log_file=sys.stderr):
    sources = list_input_sources()
    if not sources:
        return None

    best_source = None
    best_peak = -1.0

    for src in sources:
        signal = probe_source_signal(src)
        if signal is None:
            print(f'  source {src}: probe failed', file=log_file)
            continue
        peak, rms = signal
        print(f'  source {src}: peak={peak:.4f} rms={rms:.4f}', file=log_file)
        if peak > best_peak:
            best_peak = peak
            best_source = src

    if best_source:
        if best_peak >= MIN_WORKING_PEAK:
            print(f'  selected: {best_source} (peak={best_peak:.4f})', file=log_file)
        else:
            print(f'  selected quiet source: {best_source} (peak={best_peak:.4f})', file=log_file)
    return best_source


def get_cached_source():
    try:
        with open(AUDIO_SOURCE_CACHE) as f:
            src = f.read().strip()
        if src:
            return src
    except Exception:
        pass
    return None


def cache_source(source: str):
    if not source:
        return
    try:
        with open(AUDIO_SOURCE_CACHE, 'w') as f:
            f.write(source)
    except Exception:
        pass


def ensure_working_source(preferred: Optional[str] = None, log_file=sys.stderr):
    checked = set()
    for candidate in [preferred, get_cached_source()]:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        signal = probe_source_signal(candidate)
        if signal is None:
            print(f'  cached source unavailable: {candidate}', file=log_file)
            continue
        peak, rms = signal
        print(f'  cached source {candidate}: peak={peak:.4f} rms={rms:.4f}', file=log_file)
        if peak >= MIN_WORKING_PEAK:
            cache_source(candidate)
            return candidate

    best = detect_best_source(log_file=log_file)
    if best:
        cache_source(best)
    return best
