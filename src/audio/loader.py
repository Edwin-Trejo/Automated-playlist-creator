import io

import librosa
import numpy as np
from pydub import AudioSegment

from src.config import CLIP_SAMPLES, SAMPLE_RATE
from src.http_utils import get_with_retry


def decode_mp3(data):
    """Decode MP3 bytes (needs ffmpeg on PATH) to mono float32 samples + sample rate."""
    audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    return samples, audio.frame_rate


def prepare(y, sr):
    """Peak-normalize, resample to 22,050 Hz, then pad/trim to exactly 30 s."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)


def download_audio(url, *, session=None):
    """Download a preview MP3 and return a prepared 30 s clip, or None if unusable."""
    resp = get_with_retry(url, session=session)
    if resp is None:
        return None
    try:
        samples, sr = decode_mp3(resp.content)
    except Exception:
        return None
    if samples.size == 0 or not np.any(samples):
        return None
    return prepare(samples, sr)
