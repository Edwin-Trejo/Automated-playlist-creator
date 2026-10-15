import io

import numpy as np
import soundfile as sf

from src.audio import loader
from src.config import CLIP_SAMPLES, SAMPLE_RATE
from tests.fakes import FakeResp


def wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def test_decode_resamples_to_16k_and_keeps_amplitude():
    sr = 44100
    t = np.arange(sr * 2) / sr
    y = loader.decode_audio(wav_bytes(0.25 * np.sin(2 * np.pi * 440 * t), sr), suffix=".wav")
    assert y.dtype == np.float32
    assert abs(len(y) - 2 * SAMPLE_RATE) <= 1
    assert 0.2 < np.abs(y).max() < 0.3  # no peak normalization: loudness is a model feature


def test_decode_trims_to_30_seconds_without_padding():
    long = loader.decode_audio(wav_bytes(np.full(SAMPLE_RATE * 40, 0.1), SAMPLE_RATE), suffix=".wav")
    short = loader.decode_audio(wav_bytes(np.full(SAMPLE_RATE * 5, 0.1), SAMPLE_RATE), suffix=".wav")
    assert len(long) == CLIP_SAMPLES
    assert len(short) == SAMPLE_RATE * 5


def test_download_audio_rejects_silent_preview(monkeypatch):
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"mp3"))
    monkeypatch.setattr(loader, "decode_audio", lambda data: np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_handles_decode_error(monkeypatch):
    def boom(data):
        raise ValueError("bad mp3")
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"x"))
    monkeypatch.setattr(loader, "decode_audio", boom)
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_returns_none_when_request_fails(monkeypatch):
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: None)
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_returns_clip(monkeypatch):
    clip = np.full(SAMPLE_RATE * 30, 0.1, dtype=np.float32)
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"mp3"))
    monkeypatch.setattr(loader, "decode_audio", lambda data: clip)
    assert loader.download_audio("http://p/1.mp3") is clip
