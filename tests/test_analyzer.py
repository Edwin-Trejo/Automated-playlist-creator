import numpy as np
import pytest

from src.config import CLIP_SAMPLES
from src.ml import analyzer as an


class FakeCNN:
    def predict(self, x, verbose=0):
        assert x.shape == (1, 128, 640, 1)
        return np.array([[0.2, 0.8]])  # classes: jazz, rock


class FakeLGBM:
    classes_ = np.array(["rock", "jazz"])  # different order on purpose

    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])  # rock 0.4, jazz 0.6


class FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value])


MOOD = {
    "models": {"danceability": FakeRegressor(0.7), "energy": FakeRegressor(1.4),
               "valence": FakeRegressor(0.3), "acousticness": FakeRegressor(-0.2)},
    "feature_names": list(an.SUMMARY_FEATURE_NAMES),
}


@pytest.fixture
def clip():
    return (np.random.default_rng(0).random(CLIP_SAMPLES).astype(np.float32) - 0.5)


def test_ensemble_averages_aligned_probabilities(clip):
    a = an.Analyzer(FakeCNN(), ["jazz", "rock"], {"model": FakeLGBM(), "classes": ["rock", "jazz"],
                                                  "feature_names": list(an.SUMMARY_FEATURE_NAMES)}, MOOD)
    out = a.analyze_audio(clip)
    # rock = (0.8 + 0.4) / 2 = 0.6 ; jazz = (0.2 + 0.6) / 2 = 0.4
    assert out["genre"] == "rock"
    assert out["genre_conf"] == pytest.approx(0.6)


def test_cnn_only_when_lgbm_missing(clip):
    out = an.Analyzer(FakeCNN(), ["jazz", "rock"], None, MOOD).analyze_audio(clip)
    assert out["genre"] == "rock" and out["genre_conf"] == pytest.approx(0.8)


def test_mood_values_clipped_and_tempo_present(clip):
    out = an.Analyzer(FakeCNN(), ["jazz", "rock"], None, MOOD).analyze_audio(clip)
    assert out["energy"] == 1.0 and out["acousticness"] == 0.0 and out["danceability"] == pytest.approx(0.7)
    assert out["tempo"] >= 0


def test_load_reports_missing_models(tmp_path):
    with pytest.raises(an.ModelsMissingError, match="train_mood_models"):
        an.Analyzer.load(tmp_path)


def test_ensemble_uses_union_of_classes(clip):
    class JazzLGBM:
        classes_ = np.array(["jazz", "rock", "blues"])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1, 0.0]])

    class RockBluesCNN:
        def predict(self, x, verbose=0):
            return np.array([[0.6, 0.4]])  # rock, blues

    lgbm = {"model": JazzLGBM(), "classes": ["jazz", "rock", "blues"], "feature_names": list(an.SUMMARY_FEATURE_NAMES)}
    out = an.Analyzer(RockBluesCNN(), ["rock", "blues"], lgbm, MOOD).analyze_audio(clip)
    # jazz = (0 + 0.9) / 2 = 0.45 ; rock = (0.6 + 0.1) / 2 = 0.35
    assert out["genre"] == "jazz"
    assert out["genre_conf"] == pytest.approx(0.45)
