#!/usr/bin/env python3
import os
import sys
import subprocess
import struct
import tempfile
import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # 16-bit

TARGET_CHUNK_S = 60.0
MIN_CHUNK_S = 50.0
MAX_CHUNK_S = 70.0
SILENCE_THRESHOLD_S = 0.3  # min silence to split

CHUNK_DIR = "/tmp/voicetype-chunks"

class VADChunker:
    def __init__(self):
        self.buffer = []
        self.speech_duration = 0.0
        self.silence_duration = 0.0
        self.chunk_num = 0
        self.model = None
        self.use_vad = True

    def load_model(self):
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

    def process_frame(self, raw_bytes):
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        is_speech = self.is_speech(samples)

        if is_speech:
            self.buffer.extend(samples)
            self.speech_duration += FRAME_MS / 1000.0
            self.silence_duration = 0.0
        else:
            self.silence_duration += FRAME_MS / 1000.0
            if self.speech_duration > 0:
                self.buffer.extend(samples)  # keep some trailing silence

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

    def emit_chunk(self):
        if len(self.buffer) < SAMPLE_RATE:  # skip tiny chunks
            return

        audio = np.array(self.buffer, dtype=np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

        wav_path = os.path.join(CHUNK_DIR, f"chunk-{self.chunk_num}.wav")
        ogg_path = os.path.join(CHUNK_DIR, f"chunk-{self.chunk_num}.ogg")

        # write wav
        with open(wav_path, 'wb') as f:
            write_wav_header(f, len(audio_int16), SAMPLE_RATE)
            f.write(audio_int16.tobytes())

        # compress to opus with 1.3x speedup for faster transfer
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-af', 'atempo=1.3',
            '-ac', '1', '-ar', '16000',
            '-c:a', 'libopus', '-b:a', '16k', '-application', 'voip',
            '-vbr', 'on', '-compression_level', '10',
            ogg_path
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
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
        if len(self.buffer) > SAMPLE_RATE:
            self.emit_chunk()
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

def main():
    os.makedirs(CHUNK_DIR, exist_ok=True)
    for f in os.listdir(CHUNK_DIR):
        os.remove(os.path.join(CHUNK_DIR, f))

    chunker = VADChunker()
    chunker.load_model()

    proc = subprocess.Popen(
        ['parec', '--raw', '--format=s16le', '--rate=16000', '--channels=1'],
        stdout=subprocess.PIPE
    )

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
