import numpy as np

from src.audio import loader
from src.config import CLIP_SAMPLES
from tests.fakes import FakeResp


def test_prepare_pads_short_audio():
    y = loader.prepare(np.ones(1000, dtype=np.float32), 22050)
    assert y.shape == (CLIP_SAMPLES,)
    assert y.dtype == np.float32


def test_prepare_resamples_before_trimming():
    # 40 s at 44.1 kHz must become exactly 30 s at 22.05 kHz (old bug kept only ~15 s of music)
    sr = 44100
    t = np.arange(sr * 40) / sr
    y = loader.prepare(np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    assert y.shape == (CLIP_SAMPLES,)
    assert np.abs(y[-1000:]).max() > 0.1  # end of clip is still music, not padding


def test_prepare_peak_normalizes():
    y = loader.prepare(np.full(CLIP_SAMPLES, 5000.0, dtype=np.float32), 22050)
    assert np.isclose(np.abs(y).max(), 1.0)


def test_prepare_silence_stays_finite():
    y = loader.prepare(np.zeros(CLIP_SAMPLES, dtype=np.float32), 22050)
    assert np.all(np.isfinite(y))


def test_prepare_empty_input():
    assert loader.prepare(np.zeros(0, dtype=np.float32), 44100).shape == (CLIP_SAMPLES,)


def test_download_audio_rejects_silent_preview(monkeypatch):
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"mp3"))
    monkeypatch.setattr(loader, "decode_mp3", lambda data: (np.zeros(44100, dtype=np.float32), 44100))
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_handles_decode_error(monkeypatch):
    def boom(data):
        raise ValueError("bad mp3")
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"x"))
    monkeypatch.setattr(loader, "decode_mp3", boom)
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_returns_prepared_clip(monkeypatch):
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"mp3"))
    monkeypatch.setattr(loader, "decode_mp3", lambda data: (np.random.rand(44100 * 30).astype(np.float32), 44100))
    assert loader.download_audio("http://p/1.mp3").shape == (CLIP_SAMPLES,)
