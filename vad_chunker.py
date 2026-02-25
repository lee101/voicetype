#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import struct
import tempfile
import time
import queue
import threading
import numpy as np

from audio_source import detect_best_source, ensure_working_source

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # 16-bit

TARGET_CHUNK_S = 60.0
MIN_CHUNK_S = 50.0
MAX_CHUNK_S = 70.0
SILENCE_THRESHOLD_S = 0.3  # min silence to split
TARGET_PEAK = 0.9
NOISE_GATE = 0.005

CHUNK_DIR = "/tmp/voicetype-chunks"
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
ROLLING_DIR = os.path.join(SAMPLES_DIR, 'rolling')
UTTERANCE_DIR = os.path.join(SAMPLES_DIR, 'utterances')
MAX_SAVED_SAMPLES = 10
ROLLING_DURATION_S = 8.0
ROLLING_MAX = 10
UTTERANCE_MAX = 20
MIN_UTTERANCE_S = 0.5
SAMPLE_MIN_DURATION = 0.4
SAMPLE_MIN_PEAK = 0.001
SAMPLE_INDEX_PATH = os.path.join(SAMPLES_DIR, 'sample_index.json')


def env_true(name, default=False):
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on", "y"}


AGGRESSIVE_ENCODER = env_true("VOICETYPE_AGGRESSIVE_ENCODER")
FORCE_DISABLE_VAD = env_true("VOICETYPE_DISABLE_VAD")
FORCE_NO_PREPROCESS = env_true("VOICETYPE_DISABLE_PREPROCESS")
SAVE_RAW_CHUNKS = env_true("VOICETYPE_SAVE_RAW_CHUNK", False)
SAMPLE_QUEUE_SIZE = 16


class VADChunker:
    def __init__(self):
        self.buffer = []
        self.speech_duration = 0.0
        self.silence_duration = 0.0
        self.chunk_num = 0
        self.model = None
        self.use_vad = True
        self.preprocessor = None
        self.force_preprocess = not FORCE_NO_PREPROCESS
        self.sample_counter = 0
        self.sample_records = []
        # rolling capture
        self.rolling_buf = []
        self.rolling_duration = 0.0
        self.rolling_counter = 0
        # per-utterance capture
        self.in_utterance = False
        self.utterance_buf = []
        self.utterance_duration = 0.0
        self.utt_silence = 0.0
        self.utterance_counter = 0
        self.apply_silence_filter = AGGRESSIVE_ENCODER
        self.apply_atempo = AGGRESSIVE_ENCODER
        self.sample_queue = queue.Queue(maxsize=SAMPLE_QUEUE_SIZE)
        self._sample_worker = threading.Thread(target=self._sample_worker_loop, daemon=True)
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        os.makedirs(ROLLING_DIR, exist_ok=True)
        os.makedirs(UTTERANCE_DIR, exist_ok=True)
        self._load_sample_records()
        self._sample_worker.start()

    def load_preprocessor(self):
        if not self.force_preprocess:
            print("preprocessor disabled via VOICETYPE_DISABLE_PREPROCESS", file=sys.stderr)
            return
        try:
            from audio_preprocess import AudioPreprocessor
            self.preprocessor = AudioPreprocessor()
            if self.preprocessor.enabled:
                print("audio preprocessor loaded", file=sys.stderr)
            else:
                print("preprocessor: no ref stats, using basic normalize", file=sys.stderr)
        except Exception as e:
            print(f"preprocessor load failed: {e}", file=sys.stderr)

    def load_model(self):
        if FORCE_DISABLE_VAD:
            print("VAD disabled via VOICETYPE_DISABLE_VAD, using time-based chunking", file=sys.stderr)
            self.use_vad = False
            return
        try:
            import torch
            torch.set_num_threads(1)
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model.eval()
        except Exception as e:
            print(f"VAD load failed: {e}, using time-based chunking", file=sys.stderr)
            self.use_vad = False

    def is_speech(self, samples):
        if not self.use_vad or self.model is None:
            return True  # fallback: assume all speech
        import torch
        audio = torch.from_numpy(samples).float()
        prob = self.model(audio, SAMPLE_RATE).item()
        return prob > 0.5

    def _save_audio_bg(self, audio_f32, path):
        def _save():
            wav_path = path.replace('.ogg', '.wav')
            try:
                audio_int16 = (audio_f32 * 32767).astype(np.int16)
                with open(wav_path, 'wb') as f:
                    write_wav_header(f, len(audio_int16), SAMPLE_RATE)
                    f.write(audio_int16.tobytes())
                subprocess.run([
                    'ffmpeg', '-y', '-i', wav_path,
                    '-ac', '1', '-ar', '16000',
                    '-c:a', 'libopus', '-b:a', '24k', '-application', 'voip',
                    path
                ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                os.remove(wav_path)
            except Exception as e:
                print(f"save failed {path}: {e}", file=sys.stderr)
        threading.Thread(target=_save, daemon=True).start()

    def _save_audio_sync(self, audio_f32, path):
        audio_int16 = (audio_f32 * 32767).astype(np.int16)
        tmp_wav = path + ".tmp.wav"
        try:
            with open(tmp_wav, 'wb') as f:
                write_wav_header(f, len(audio_int16), SAMPLE_RATE)
                f.write(audio_int16.tobytes())
            subprocess.run([
                'ffmpeg', '-y', '-i', tmp_wav,
                '-ac', '1', '-ar', '16000',
                '-c:a', 'libopus', '-b:a', '24k', '-application', 'voip',
                path
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
            os.remove(tmp_wav)
            return True
        except Exception as e:
            print(f"sample sync save failed {path}: {e}", file=sys.stderr)
            try:
                if os.path.exists(tmp_wav):
                    os.replace(tmp_wav, path)
                    return True
            except Exception:
                pass
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
            try:
                os.remove(path)
            except Exception:
                pass
            return False

    def _save_rolling(self, audio_f32):
        idx = self.rolling_counter % ROLLING_MAX
        self.rolling_counter += 1
        path = os.path.join(ROLLING_DIR, f"rolling_{idx}.ogg")
        peak = float(np.max(np.abs(audio_f32)))
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        print(f"rolling_{idx} peak={peak:.3f} rms={rms:.3f}", file=sys.stderr)
        self._save_audio_bg(audio_f32, path)

    def _save_raw_chunk(self, audio_f32):
        if not SAVE_RAW_CHUNKS:
            return None

        raw_wav = os.path.join(CHUNK_DIR, f"chunk-{self.chunk_num}.raw.wav")
        raw_ogg = os.path.join(CHUNK_DIR, f"chunk-{self.chunk_num}.raw.ogg")
        try:
            audio_int16 = (audio_f32 * 32767).astype(np.int16)
            with open(raw_wav, 'wb') as f:
                write_wav_header(f, len(audio_int16), SAMPLE_RATE)
                f.write(audio_int16.tobytes())

            subprocess.run([
                'ffmpeg', '-y', '-i', raw_wav,
                '-ac', '1', '-ar', '16000',
                '-c:a', 'libopus', '-b:a', '16k', '-application', 'voip',
                '-vbr', 'on', '-compression_level', '10',
                raw_ogg
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            os.remove(raw_wav)
            return raw_ogg
        except Exception as e:
            print(f"save raw chunk failed for {raw_ogg}: {e}", file=sys.stderr)
            try:
                os.remove(raw_wav)
            except Exception:
                pass
            return None

    def _save_utterance(self, audio_f32):
        idx = self.utterance_counter % UTTERANCE_MAX
        self.utterance_counter += 1
        path = os.path.join(UTTERANCE_DIR, f"utterance_{idx}.ogg")
        dur = len(audio_f32) / SAMPLE_RATE
        peak = float(np.max(np.abs(audio_f32)))
        print(f"utterance_{idx} {dur:.1f}s peak={peak:.3f}", file=sys.stderr)
        self._save_audio_bg(audio_f32, path)
        self._save_sample_candidate(audio_f32)

    def _load_sample_records(self):
        if not os.path.exists(SAMPLE_INDEX_PATH):
            return
        try:
            with open(SAMPLE_INDEX_PATH) as f:
                data = json.load(f)
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                path = rec.get('path')
                if path and os.path.exists(path):
                    self.sample_records.append(rec)
            self.sample_records.sort(key=lambda r: r.get('quality', 0), reverse=True)
            self.sample_records = self.sample_records[:MAX_SAVED_SAMPLES]
        except Exception:
            self.sample_records = []

    def _save_sample_records(self):
        try:
            with open(SAMPLE_INDEX_PATH, 'w') as f:
                json.dump(self.sample_records, f, indent=2)
        except Exception:
            pass

    def _sample_worker_loop(self):
        while True:
            item = self.sample_queue.get()
            if item is None:
                return
            path, quality, duration, audio_f32 = item
            if self._save_audio_sync(audio_f32, path):
                self._register_top_sample(path, quality, duration)

    def _sample_quality(self, audio_f32):
        if audio_f32 is None or len(audio_f32) < int(SAMPLE_RATE * SAMPLE_MIN_DURATION):
            return 0.0
        dur = len(audio_f32) / float(SAMPLE_RATE)
        peak = float(np.max(np.abs(audio_f32)))
        if peak < SAMPLE_MIN_PEAK:
            return 0.0
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        return 0.7 * rms + 0.3 * min(1.0, dur / 8.0)

    def _register_top_sample(self, path, quality, duration):
        self.sample_records.append({
            "path": path,
            "quality": quality,
            "duration": duration,
            "created_at": int(time.time()),
        })
        self.sample_records.sort(key=lambda r: r.get('quality', 0), reverse=True)
        if len(self.sample_records) > MAX_SAVED_SAMPLES:
            for rec in self.sample_records[MAX_SAVED_SAMPLES:]:
                old_path = rec.get('path')
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass
            self.sample_records = self.sample_records[:MAX_SAVED_SAMPLES]
        self._save_sample_records()

    def _save_sample_candidate(self, audio_f32):
        quality = self._sample_quality(audio_f32)
        if quality <= 0:
            return
        worst_quality = self.sample_records[-1]["quality"] if self.sample_records else 0.0
        if len(self.sample_records) >= MAX_SAVED_SAMPLES and quality <= worst_quality:
            return
        duration = len(audio_f32) / float(SAMPLE_RATE)
        fname = f"sample_{int(time.time() * 1000)}_{self.sample_counter}.ogg"
        self.sample_counter += 1
        path = os.path.join(SAMPLES_DIR, fname)
        audio_copy = np.array(audio_f32, dtype=np.float32)
        try:
            self.sample_queue.put_nowait((path, quality, duration, audio_copy))
        except queue.Full:
            pass

    def process_frame(self, raw_bytes):
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        is_speech = self.is_speech(samples)

        # rolling capture (unconditional)
        self.rolling_buf.extend(samples)
        self.rolling_duration += FRAME_MS / 1000.0
        if self.rolling_duration >= ROLLING_DURATION_S:
            self._save_rolling(np.array(self.rolling_buf, dtype=np.float32))
            self.rolling_buf = []
            self.rolling_duration = 0.0

        # per-utterance capture
        if is_speech:
            if not self.in_utterance:
                self.in_utterance = True
                self.utterance_buf = []
                self.utterance_duration = 0.0
            self.utterance_buf.extend(samples)
            self.utterance_duration += FRAME_MS / 1000.0
            self.utt_silence = 0.0
        else:
            if self.in_utterance:
                self.utt_silence += FRAME_MS / 1000.0
                self.utterance_buf.extend(samples)
                if self.utt_silence >= SILENCE_THRESHOLD_S:
                    if self.utterance_duration >= MIN_UTTERANCE_S:
                        self._save_utterance(np.array(self.utterance_buf, dtype=np.float32))
                    self.in_utterance = False

        # existing chunk logic
        if is_speech:
            self.buffer.extend(samples)
            self.speech_duration += FRAME_MS / 1000.0
            self.silence_duration = 0.0
        else:
            self.silence_duration += FRAME_MS / 1000.0
            if self.speech_duration > 0:
                self.buffer.extend(samples)

        if self.should_emit():
            self.emit_chunk()

    def should_emit(self):
        if self.speech_duration < MIN_CHUNK_S:
            return False
        if self.speech_duration >= MAX_CHUNK_S:
            return True
        if self.speech_duration >= TARGET_CHUNK_S and self.silence_duration >= SILENCE_THRESHOLD_S:
            return True
        return False

    def save_sample(self, audio_f32):
        """Save rolling window of last N samples to disk."""
        self._save_sample_candidate(audio_f32)

    def emit_chunk(self):
        if len(self.buffer) < SAMPLE_RATE:
            return

        audio = np.array(self.buffer, dtype=np.float32)

        # save raw sample before processing
        self.save_sample(audio)
        self._save_raw_chunk(audio)

        # apply learned preprocessor or basic normalization
        if self.preprocessor and self.force_preprocess:
            audio = self.preprocessor.process(audio)
        else:
            peak = np.max(np.abs(audio))
            if peak > NOISE_GATE:
                audio = audio * (TARGET_PEAK / peak)

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        wav_path = os.path.join(CHUNK_DIR, f"chunk-{self.chunk_num}.wav")
        ogg_path = os.path.join(CHUNK_DIR, f"chunk-{self.chunk_num}.ogg")

        with open(wav_path, 'wb') as f:
            write_wav_header(f, len(audio_int16), SAMPLE_RATE)
            f.write(audio_int16.tobytes())

        # compress to opus (optional silence filter + speedup)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', wav_path
        ]
        if self.apply_silence_filter or self.apply_atempo:
            filters = []
            if self.apply_silence_filter:
                filters.extend([
                    'silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB',
                    'areverse',
                    'silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB',
                    'areverse',
                ])
            if self.apply_atempo:
                filters.append('atempo=1.3')
            ffmpeg_cmd += ['-af', ','.join(filters)]

        ffmpeg_cmd += [
            '-ac', '1', '-ar', '16000',
            '-c:a', 'libopus', '-b:a', '16k', '-application', 'voip',
            '-vbr', 'on', '-compression_level', '10',
            ogg_path
        ]
        subprocess.run(ffmpeg_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        # if silence removal ate everything, retry without it
        if not os.path.exists(ogg_path) or os.path.getsize(ogg_path) < 100:
            print(f"chunk {self.chunk_num}: encoded audio too small, retrying plain", file=sys.stderr)
            retry_cmd = [
                'ffmpeg', '-y', '-i', wav_path
            ]
            if self.apply_atempo:
                retry_cmd += ['-af', 'atempo=1.3']
            retry_cmd += [
                '-ac', '1', '-ar', '16000',
                '-c:a', 'libopus', '-b:a', '16k', '-application', 'voip',
                '-vbr', 'on', '-compression_level', '10',
                ogg_path
            ]
            subprocess.run(retry_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        os.remove(wav_path)

        # signal ready
        with open(os.path.join(CHUNK_DIR, "ready"), 'a') as f:
            f.write(f"{self.chunk_num}\n")
            f.flush()

        print(f"chunk {self.chunk_num}: {self.speech_duration:.1f}s speech", file=sys.stderr)

        self.chunk_num += 1
        self.buffer = []
        self.speech_duration = 0.0
        self.silence_duration = 0.0

    def flush(self):
        if self.in_utterance and self.utterance_buf:
            self._save_sample_candidate(np.array(self.utterance_buf, dtype=np.float32))
            self.in_utterance = False
            self.utterance_buf = []
            self.utterance_duration = 0.0
            self.utt_silence = 0.0

        if self.rolling_buf:
            self._save_sample_candidate(np.array(self.rolling_buf, dtype=np.float32))
            self.rolling_buf = []
            self.rolling_duration = 0.0

        if len(self.buffer) > SAMPLE_RATE:
            self.emit_chunk()
        try:
            self.sample_queue.put_nowait(None)
        except queue.Full:
            try:
                self.sample_queue.get_nowait()
            except Exception:
                pass
            try:
                self.sample_queue.put_nowait(None)
            except Exception:
                pass
        self._sample_worker.join(timeout=2.0)
        with open(os.path.join(CHUNK_DIR, "done"), 'w') as f:
            f.write("1")

def write_wav_header(f, num_samples, sample_rate):
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    f.write(b'RIFF')
    f.write(struct.pack('<I', 36 + data_size))
    f.write(b'WAVE')
    f.write(b'fmt ')
    f.write(struct.pack('<I', 16))
    f.write(struct.pack('<H', 1))  # PCM
    f.write(struct.pack('<H', num_channels))
    f.write(struct.pack('<I', sample_rate))
    f.write(struct.pack('<I', byte_rate))
    f.write(struct.pack('<H', block_align))
    f.write(struct.pack('<H', bits_per_sample))
    f.write(b'data')
    f.write(struct.pack('<I', data_size))

def update_source_cache():
    source = detect_best_source()
    if source:
        # ensure helper updates the cache file consistently
        ensure_working_source(preferred=source)


def main():
    import threading
    os.makedirs(CHUNK_DIR, exist_ok=True)
    for f in os.listdir(CHUNK_DIR):
        os.remove(os.path.join(CHUNK_DIR, f))

    source = ensure_working_source()
    if source:
        print(f"using source: {source}", file=sys.stderr)
    else:
        print("no source selected; using PulseAudio default", file=sys.stderr)

    # start parec IMMEDIATELY
    cmd = ['parec', '--raw', '--format=s16le', '--rate=16000', '--channels=1']
    if source:
        cmd.append(f'--device={source}')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    print("parec started", file=sys.stderr)

    # load model + preprocessor in background
    chunker = VADChunker()
    def bg_init():
        chunker.load_model()
        chunker.load_preprocessor()
    threading.Thread(target=bg_init, daemon=True).start()

    # update source cache in background for next run
    threading.Thread(target=update_source_cache, daemon=True).start()

    try:
        while True:
            data = proc.stdout.read(BYTES_PER_FRAME)
            if not data or len(data) < BYTES_PER_FRAME:
                break
            chunker.process_frame(data)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        chunker.flush()

if __name__ == "__main__":
    main()
