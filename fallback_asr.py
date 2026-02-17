#!/usr/bin/python3
"""Local ASR - uses model server if available, else loads directly"""
import sys
import os
import json
import socket

SOCKET_PATH = "/tmp/voicetype-model.sock"

def call_server(action, path=None):
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect(SOCKET_PATH)

        cmd = {"action": action}
        if path:
            cmd["path"] = path

        sock.send(json.dumps(cmd).encode())
        resp = json.loads(sock.recv(65536).decode())
        sock.close()
        return resp
    except:
        return None

def transcribe_direct(audio_path):
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2", map_location=device).eval()
    with torch.inference_mode():
        result = model.transcribe([audio])
    return result[0].text

def main():
    if len(sys.argv) < 2:
        print("Usage: fallback_asr.py <audio_file|preload>", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "preload":
        resp = call_server("preload")
        if resp and resp.get("status") == "ok":
            print("preloaded")
        else:
            print("server not running")
        return

    # Try server first
    resp = call_server("transcribe", arg)
    if resp and "text" in resp:
        print(resp["text"])
        return

    # Direct fallback
    try:
        text = transcribe_direct(arg)
        print(text)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
