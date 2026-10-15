import os
import tempfile

import librosa
import numpy as np

from src.config import CLIP_SECONDS, SAMPLE_RATE
from src.http_utils import get_with_retry


def decode_audio(data, suffix=".mp3"):
    """Decode audio bytes to mono float32 at 16 kHz, trimmed to 30 s.

    Mirrors the training notebook exactly (librosa.load, no normalization, no padding):
    the models were trained on audio prepared this way.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
    try:
        y, _ = librosa.load(f.name, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
    finally:
        os.remove(f.name)
    return y.astype(np.float32)


def download_audio(url, *, session=None):
    """Download a preview and return a decoded clip, or None if unusable."""
    resp = get_with_retry(url, session=session)
    if resp is None:
        return None
    try:
        y = decode_audio(resp.content)
    except Exception:
        return None
    if y.size == 0 or not np.any(y):
        return None
    return y
