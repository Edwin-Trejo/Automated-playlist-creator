import json

import numpy as np
import pytest

from src.audio.embeddings import FEATURE_NAMES
from src.ml import analyzer as an

TEMPO = FEATURE_NAMES.index("tempo")


def fake_features(y):
    vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    vec[TEMPO] = 123.44
    return vec


class FakeBooster:
    def __init__(self, out):
        self.out = np.asarray(out)
        self.seen = None

    def predict(self, X):
        self.seen = X
        return self.out


GENRES = ["jazz", "rock", "pop"]


def make_analyzer(genre_out=((0.1, 0.7, 0.2),), mood=None):
    mood = mood or {"danceability": 0.7, "energy": 1.4, "valence": 0.3, "acousticness": -0.2}
    genre = FakeBooster(genre_out)
    return an.Analyzer(fake_features, genre, GENRES, {t: FakeBooster([v]) for t, v in mood.items()}), genre


def test_genre_is_most_likely_class():
    a, genre = make_analyzer()
    out = a.analyze_audio(np.zeros(10, dtype=np.float32))
    assert out["genre"] == "rock"
    assert out["genre_conf"] == pytest.approx(0.7)
    assert genre.seen.shape == (1, len(FEATURE_NAMES))


def test_mood_values_are_clipped_and_tempo_comes_from_features():
    a, _ = make_analyzer()
    out = a.analyze_audio(np.zeros(10, dtype=np.float32))
    assert out["energy"] == 1.0 and out["acousticness"] == 0.0
    assert out["danceability"] == pytest.approx(0.7) and out["valence"] == pytest.approx(0.3)
    assert out["tempo"] == pytest.approx(123.4)


def test_load_reports_missing_models(tmp_path):
    with pytest.raises(an.ModelsMissingError, match="train_models_colab"):
        an.Analyzer.load(tmp_path)


def test_load_rejects_models_trained_on_different_features(tmp_path):
    for name in an.REQUIRED_FILES:
        (tmp_path / name).write_text("x")
    for meta in ("genre_meta.json", "mood_meta.json"):
        (tmp_path / meta).write_text(json.dumps({"feature_names": ["something_else"], "genres": GENRES,
                                                 "targets": ["energy"]}))
    with pytest.raises(an.ModelsMissingError, match="features"):
        an.Analyzer.load(tmp_path)


@pytest.mark.skipif(not (an.MODELS_DIR / "genre_lgbm.txt").exists(), reason="trained models not present")
def test_load_real_models():
    a = an.Analyzer.load()
    assert a.genres == json.loads((an.MODELS_DIR / "genre_meta.json").read_text())["genres"]
    assert set(a.mood_models) == {"danceability", "energy", "valence", "acousticness"}
