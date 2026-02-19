#!/usr/bin/env python3
"""Compare raw vs pipeline-processed transcription paths for debugging clip loss."""

import argparse
import json
import os
import subprocess
import struct
import sys
import tempfile
import time
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
WORK_DIR = os.path.join(tempfile.gettempdir(), "voicetype-debug")
os.makedirs(WORK_DIR, exist_ok=True)


def _read_key(env_name: str, file_key: str) -> str:
    value = os.getenv(env_name, "").strip()
    if value:
        return value

    env_path = os.path.expanduser("~/.config/voicetype/env")
    if os.path.exists(env_path):
        try:
            for line in open(env_path):
                line = line.strip()
                if line.startswith(f"{file_key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def write_wav(path: str, audio_f32: np.ndarray) -> None:
    audio_int16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with open(path, "wb") as f:
        n = len(audio_int16)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", SAMPLE_RATE))
        byte_rate = SAMPLE_RATE * 2
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        f.write(audio_int16.tobytes())


def load_audio(path: str) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-i",
            path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore')[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def record(seconds: float) -> np.ndarray:
    print(f"Recording for {seconds:.1f}s; speak now...")
    proc = subprocess.Popen(
        [
            "parec",
            "--raw",
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
        ],
        stdout=subprocess.PIPE,
    )
    frames = []
    target = int(SAMPLE_RATE * seconds * 2)
    read = 0
    while read < target:
        chunk = proc.stdout.read(min(4096, target - read))
        if not chunk:
            break
        frames.append(chunk)
        read += len(chunk)

    proc.terminate()
    proc.wait()

    raw = b"".join(frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def analyze(audio: np.ndarray) -> str:
    if len(audio) == 0:
        return "empty"
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return f"duration={len(audio)/SAMPLE_RATE:.2f}s peak={peak:.4f} rms={rms:.4f}"

def basic_normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.005:
        return audio
    return np.clip(audio * (0.9 / peak), -1.0, 1.0)


def pipeline_preprocess(audio: np.ndarray, skip_preprocess: bool):
    if skip_preprocess:
        return basic_normalize(audio)

    try:
        from audio_preprocess import AudioPreprocessor

        pre = AudioPreprocessor()
        if pre.enabled:
            return pre.process(audio.copy())
    except Exception:
        pass

    return basic_normalize(audio)


def save_raw_ogg(wav_path: str, ogg_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
            "-application",
            "voip",
            ogg_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=20,
    )


def pipeline_encode(
    wav_path: str,
    ogg_path: str,
    keep_silence: bool = True,
    keep_atempo: bool = True,
) -> None:
    filters = []
    if keep_silence:
        filters.extend(
            [
                "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB",
                "areverse",
                "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB",
                "areverse",
            ]
        )
    if keep_atempo:
        filters.append("atempo=1.3")

    base_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        wav_path,
    ]
    if filters:
        base_cmd += ["-af", ",".join(filters)]
    base_cmd += [
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        "16k",
        "-application",
        "voip",
        "-vbr",
        "on",
        "-compression_level",
        "10",
        ogg_path,
    ]

    subprocess.run(
        base_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
        check=True,
    )

    if not os.path.exists(ogg_path) or os.path.getsize(ogg_path) < 100:
        print(f"chunk {ogg_path}: pipeline filter removed audio, retrying plain")
        retry_filters = []
        if keep_atempo:
            retry_filters.append("atempo=1.3")

        retry_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
        ]
        if retry_filters:
            retry_cmd += ["-af", ",".join(retry_filters)]
        retry_cmd += [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            "16k",
            "-application",
            "voip",
            "-vbr",
            "on",
            "-compression_level",
            "10",
            ogg_path,
        ]

        subprocess.run(
            retry_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=True,
        )


def _safe_json_load(payload: str):
    try:
        return json.loads(payload)
    except Exception:
        return None


def transcribe_text_generator(audio_path: str) -> tuple[str, str]:
    key = _read_key("TEXT_GENERATOR_API_KEY", "TEXT_GENERATOR_API_KEY")
    if not key:
        return "", "missing TEXT_GENERATOR_API_KEY"

    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "https://api.text-generator.io/api/v1/audio-file-extraction",
            "-H",
            f"secret: {key}",
            "-F",
            f"audio_file=@{audio_path}",
            "-F",
            "translate_to_english=false",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )

    if result.returncode != 0:
        return "", f"HTTP request error: {result.stderr.strip() or result.stdout.strip()}"

    data = _safe_json_load(result.stdout or "")
    if not data:
        return "", f"invalid JSON: {result.stdout[:200]}"

    text = str(data.get("text", "") or "").strip()
    err = str(data.get("error", "") or "").strip()
    if not text and err:
        return "", err
    return text, ""


def transcribe_fal(audio_path: str) -> tuple[str, str]:
    key = _read_key("FAL_KEY", "FAL_KEY")
    if not key:
        return "", "missing FAL_KEY"
    try:
        from fal_whisper import transcribe as transcribe_fal_api

        text = transcribe_fal_api(audio_path)
        if text:
            return text.strip(), ""
        return "", "empty result"
    except Exception as e:
        return "", str(e)


def print_result(label: str, path: str, text: str, err: str, elapsed: float):
    status = "ok" if text else f"warn({err})"
    print(f"[{status}] {label} ({elapsed:.2f}s) -> {path}")
    if text:
        print(f"  text: {text}")


def make_path(name: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=name + "-", suffix=suffix, dir=WORK_DIR)
    os.close(fd)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", help="Path to audio file (wav/mp3/ogg/m4a). If omitted, records from mic.")
    parser.add_argument("--record", type=float, default=6.0, help="Recording seconds if no file provided")
    parser.add_argument("--no-silence", action="store_true", help="Disable silenceremove in pipeline path")
    parser.add_argument("--no-atempo", action="store_true", help="Disable atempo in pipeline path")
    parser.add_argument("--aggressive", action="store_true", help="Enable legacy silence removal + atempo in pipeline path")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip AudioPreprocessor / basic normalization")
    args = parser.parse_args()

    if args.audio:
        audio = load_audio(args.audio)
        print(f"Loaded: {args.audio}")
    else:
        if args.record <= 0:
            raise SystemExit("--record must be > 0")
        audio = record(args.record)

    if len(audio) == 0:
        raise SystemExit("No audio captured")

    raw_wav = make_path("raw", ".wav")
    write_wav(raw_wav, audio)

    print(f"Input: {analyze(audio)}")
    print(f"Saved source wav: {raw_wav}")

    raw_ogg = make_path("raw", ".ogg")
    save_raw_ogg(raw_wav, raw_ogg)
    print(f"Saved raw API wav->ogg: {raw_ogg}")

    t0 = time.time()
    tg_raw, tg_raw_err = transcribe_text_generator(raw_ogg)
    print_result("text-generator raw", raw_ogg, tg_raw, tg_raw_err, time.time() - t0)

    t1 = time.time()
    fal_raw, fal_raw_err = transcribe_fal(raw_ogg)
    print_result("fal raw", raw_ogg, fal_raw, fal_raw_err, time.time() - t1)

    pipeline_wav = make_path("pipeline", ".wav")
    processed = pipeline_preprocess(audio, args.skip_preprocess)
    write_wav(pipeline_wav, processed)
    print(f"Applied pipeline preprocessing, saved: {pipeline_wav}")

    pipeline_ogg = make_path("pipeline", ".ogg")
    pipeline_encode(
        pipeline_wav,
        pipeline_ogg,
        keep_silence=args.aggressive and not args.no_silence,
        keep_atempo=args.aggressive and not args.no_atempo,
    )
    print(f"Saved pipeline encoded: {pipeline_ogg}")

    t2 = time.time()
    tg_pipe, tg_pipe_err = transcribe_text_generator(pipeline_ogg)
    print_result("text-generator pipeline", pipeline_ogg, tg_pipe, tg_pipe_err, time.time() - t2)

    t3 = time.time()
    fal_pipe, fal_pipe_err = transcribe_fal(pipeline_ogg)
    print_result("fal pipeline", pipeline_ogg, fal_pipe, fal_pipe_err, time.time() - t3)

    print("\nComparison:")
    print(f"text-generator: raw='{tg_raw}'")
    print(f"text-generator: pipeline='{tg_pipe}'")
    print(f"fal: raw='{fal_raw}'")
    print(f"fal: pipeline='{fal_pipe}'")

    print(f"Artifacts in: {WORK_DIR}")


if __name__ == "__main__":
    main()
