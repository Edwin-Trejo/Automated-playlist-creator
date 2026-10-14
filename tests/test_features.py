import numpy as np
import pytest

from src.audio.features import SUMMARY_FEATURE_NAMES, mel_spectrogram, summary_vector
from src.config import CLIP_SAMPLES, SAMPLE_RATE


@pytest.fixture(scope="module")
def clicks():
    # 120 BPM click track over a quiet tone
    y = 0.05 * np.sin(2 * np.pi * 220 * np.arange(CLIP_SAMPLES) / SAMPLE_RATE)
    for start in range(0, CLIP_SAMPLES, SAMPLE_RATE // 2):
        y[start:start + 500] += 0.9
    return y.astype(np.float32)


def test_mel_shape_matches_cnn_input(clicks):
    assert mel_spectrogram(clicks).shape == (128, 640)


def test_summary_vector_shape_and_names(clicks):
    vec = summary_vector(clicks)
    assert vec.shape == (len(SUMMARY_FEATURE_NAMES),) == (72,)
    assert np.all(np.isfinite(vec))
    assert len(set(SUMMARY_FEATURE_NAMES)) == 72


def test_tempo_feature_is_detected(clicks):
    tempo = summary_vector(clicks)[SUMMARY_FEATURE_NAMES.index("tempo")]
    assert 110 <= tempo <= 130 or 55 <= tempo <= 65 or 230 <= tempo <= 250


def test_silence_is_finite():
    silent = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    assert np.all(np.isfinite(summary_vector(silent)))
    assert np.all(np.isfinite(mel_spectrogram(silent)))
