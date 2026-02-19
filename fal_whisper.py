#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import tempfile


def get_fal_key():
    key = os.environ.get('FAL_KEY', '')
    if not key:
        env_file = os.path.expanduser('~/.config/voicetype/env')
        if os.path.exists(env_file):
            for line in open(env_file):
                line = line.strip()
                if line.startswith('FAL_KEY='):
                    key = line.split('=', 1)[1].strip().strip('"').strip("'")
    return key


def upload_file(audio_path, key):
    """Upload audio to fal storage, return public URL."""
    ext = os.path.splitext(audio_path)[1].lstrip('.')
    mime_map = {'ogg': 'audio/ogg', 'mp3': 'audio/mpeg', 'wav': 'audio/wav',
                'mp4': 'video/mp4', 'm4a': 'audio/mp4', 'webm': 'audio/webm'}
    content_type = mime_map.get(ext, 'application/octet-stream')
    fname = os.path.basename(audio_path)

    # convert ogg to wav for better compatibility
    if ext == 'ogg':
        tmp_wav = tempfile.mktemp(suffix='.wav')
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', '16000', tmp_wav],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=10
        )
        audio_path = tmp_wav
        content_type = 'audio/wav'
        fname = fname.replace('.ogg', '.wav')

    # initiate upload
    r = subprocess.run([
        'curl', '-s', '-X', 'POST',
        'https://rest.alpha.fal.ai/storage/upload/initiate',
        '-H', f'Authorization: Key {key}',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({'file_name': fname, 'content_type': content_type}),
    ], capture_output=True, text=True, timeout=10)

    try:
        resp = json.loads(r.stdout)
    except Exception:
        print(f"fal upload initiate failed: {r.stdout[:200]}", file=sys.stderr)
        return ""

    upload_url = resp.get('upload_url', '')
    file_url = resp.get('file_url', '')
    if not upload_url or not file_url:
        print(f"fal upload no urls: {resp}", file=sys.stderr)
        return ""

    # PUT file data
    r2 = subprocess.run([
        'curl', '-s', '-X', 'PUT',
        upload_url,
        '-H', f'Content-Type: {content_type}',
        '--data-binary', f'@{audio_path}',
    ], capture_output=True, text=True, timeout=30)

    # cleanup temp wav
    if ext == 'ogg':
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return file_url


def transcribe(audio_path, timeout=120):
    key = get_fal_key()
    if not key:
        return ""

    file_url = upload_file(audio_path, key)
    if not file_url:
        return ""

    payload = json.dumps({
        "audio_url": file_url,
        "task": "transcribe",
        "chunk_level": "segment",
        "batch_size": 64,
    })

    # submit to queue
    payload_file = tempfile.mktemp(suffix='.json')
    with open(payload_file, 'w') as f:
        f.write(payload)

    try:
        result = subprocess.run([
            'curl', '-s', '-X', 'POST',
            'https://queue.fal.run/fal-ai/whisper',
            '-H', f'Authorization: Key {key}',
            '-H', 'Content-Type: application/json',
            '-d', f'@{payload_file}',
        ], capture_output=True, text=True, timeout=30)
    finally:
        try:
            os.remove(payload_file)
        except Exception:
            pass

    try:
        resp = json.loads(result.stdout)
    except Exception:
        print(f"fal submit failed: {result.stdout[:200]}", file=sys.stderr)
        return ""

    request_id = resp.get('request_id', '')
    if not request_id:
        if 'text' in resp:
            return resp['text']
        print(f"fal no request_id: {resp}", file=sys.stderr)
        return ""

    # poll for result
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(1)
        status_result = subprocess.run([
            'curl', '-s', '-X', 'GET',
            f'https://queue.fal.run/fal-ai/whisper/requests/{request_id}/status',
            '-H', f'Authorization: Key {key}',
        ], capture_output=True, text=True, timeout=10)

        try:
            status = json.loads(status_result.stdout)
        except Exception:
            continue

        if status.get('status') == 'COMPLETED':
            fetch_result = subprocess.run([
                'curl', '-s', '-X', 'GET',
                f'https://queue.fal.run/fal-ai/whisper/requests/{request_id}',
                '-H', f'Authorization: Key {key}',
            ], capture_output=True, text=True, timeout=30)
            try:
                data = json.loads(fetch_result.stdout)
                return data.get('text', '')
            except Exception:
                return ""
        elif status.get('status') in ('FAILED', 'CANCELLED'):
            print(f"fal failed: {status}", file=sys.stderr)
            return ""

    print("fal timeout", file=sys.stderr)
    return ""


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: fal_whisper.py <audio_file>")
        sys.exit(1)
    text = transcribe(sys.argv[1])
    print(text)
