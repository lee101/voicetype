#!/usr/bin/python3
"""Local Parakeet ASR fallback - only loaded when API fails"""
import sys
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "2"

def transcribe(audio_path):
    try:
        import torch
        torch.set_num_threads(2)
        from nemo.collections.asr.models import ASRModel
        import soundfile as sf

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy import signal
            audio = signal.resample_poly(audio, 16000, sr)
        audio = audio.astype(np.float32)
        audio /= (np.abs(audio).max() + 1e-9)

        model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2", map_location="cpu").eval()
        with torch.inference_mode():
            result = model.transcribe([audio])
        print(result[0].text)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fallback_asr.py <audio_file>", file=sys.stderr)
        sys.exit(1)
    transcribe(sys.argv[1])
