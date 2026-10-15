import json

import numpy as np
import pytest

from src.audio.embeddings import EXTRA_NAMES, FEATURE_NAMES, FeatureExtractor
from src.config import MODELS_DIR, SAMPLE_RATE

YAMNET = MODELS_DIR / "yamnet.onnx"
needs_yamnet = pytest.mark.skipif(not YAMNET.exists(), reason="models/yamnet.onnx not present")


def test_feature_names_match_trained_models():
    # The trained LightGBM models expect exactly the notebook's 2,053 features, in this order.
    for meta in ("genre_meta.json", "mood_meta.json"):
        path = MODELS_DIR / meta
        if path.exists():
            assert json.loads(path.read_text())["feature_names"] == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 2053
    assert FEATURE_NAMES[-5:] == EXTRA_NAMES


@pytest.fixture(scope="module")
def extract():
    return FeatureExtractor(YAMNET)


@needs_yamnet
def test_click_track_features(extract):
    # 120 BPM clicks over a quiet tone, 10 s
    y = 0.05 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE * 10) / SAMPLE_RATE)
    for start in range(0, len(y), SAMPLE_RATE // 2):
        y[start:start + 200] += 0.9
    vec = extract(y.astype(np.float32))
    assert vec.shape == (2053,) and vec.dtype == np.float32
    assert np.all(np.isfinite(vec))
    tempo = vec[FEATURE_NAMES.index("tempo")]
    assert 110 <= tempo <= 130 or 55 <= tempo <= 65 or 230 <= tempo <= 250


@needs_yamnet
def test_silence_and_long_input_are_finite(extract):
    assert np.all(np.isfinite(extract(np.zeros(SAMPLE_RATE * 3, dtype=np.float32))))
    # Longer than 30 s is trimmed, so it matches the first 30 s exactly
    y = np.random.default_rng(0).uniform(-0.5, 0.5, SAMPLE_RATE * 40).astype(np.float32)
    np.testing.assert_allclose(extract(y), extract(y[: SAMPLE_RATE * 30]), rtol=1e-5)
