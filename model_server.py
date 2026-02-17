#!/usr/bin/python3
"""Persistent ASR model server - keeps model warm on GPU"""
import os
import sys
import json
import socket
import threading
import time
import numpy as np

SOCKET_PATH = "/tmp/voicetype-model.sock"
DEVICE = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"

model = None
model_lock = threading.Lock()
last_used = 0
UNLOAD_AFTER = 300  # unload from GPU after 5min idle

def load_model():
    global model
    if model is not None:
        return

    import torch
    if DEVICE == "cuda":
        torch.cuda.set_device(0)

    from nemo.collections.asr.models import ASRModel

    print(f"Loading model on {DEVICE}...", file=sys.stderr)
    model = ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v2",
        map_location=DEVICE
    ).eval()

    # Warmup
    with torch.inference_mode():
        dummy = np.zeros(16000, dtype=np.float32)
        model.transcribe([dummy])

    print("Model ready", file=sys.stderr)

def unload_model():
    global model
    if model is None:
        return

    import torch
    del model
    model = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Model unloaded", file=sys.stderr)

def transcribe(audio_path):
    global last_used
    import torch
    import soundfile as sf

    with model_lock:
        load_model()
        last_used = time.time()

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy import signal
            audio = signal.resample_poly(audio, 16000, sr)
        audio = audio.astype(np.float32)
        audio /= (np.abs(audio).max() + 1e-9)

        with torch.inference_mode():
            result = model.transcribe([audio])

        return result[0].text

def handle_client(conn):
    try:
        data = conn.recv(4096).decode()
        cmd = json.loads(data)

        if cmd.get("action") == "preload":
            with model_lock:
                load_model()
            conn.send(json.dumps({"status": "ok"}).encode())

        elif cmd.get("action") == "transcribe":
            text = transcribe(cmd["path"])
            conn.send(json.dumps({"text": text}).encode())

        elif cmd.get("action") == "unload":
            with model_lock:
                unload_model()
            conn.send(json.dumps({"status": "ok"}).encode())

        else:
            conn.send(json.dumps({"error": "unknown action"}).encode())

    except Exception as e:
        conn.send(json.dumps({"error": str(e)}).encode())
    finally:
        conn.close()

def idle_unloader():
    global last_used
    while True:
        time.sleep(60)
        if model is not None and time.time() - last_used > UNLOAD_AFTER:
            with model_lock:
                if time.time() - last_used > UNLOAD_AFTER:
                    unload_model()

def main():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    os.chmod(SOCKET_PATH, 0o600)

    threading.Thread(target=idle_unloader, daemon=True).start()

    print(f"Model server listening on {SOCKET_PATH}", file=sys.stderr)

    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

if __name__ == "__main__":
    main()
