import json

import numpy as np
import pytest

from src.config import LYRICS_MODEL_JSON
from src.ml.lyrics_mood import LYRIC_MOODS, LyricsMood


def test_score_is_linear_and_clipped():
    embedder = lambda text: np.array([1.0, 0.0], dtype=np.float32)
    coef = [[0.5, 9.0], [2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.25, 0.0], [0.0, 0.0]]
    model = LyricsMood(embedder, coef, [0.1, 0.0, 0.0, 0.3, 0.0, 0.0], "l1")
    scores = model.score("some lyrics")
    assert list(scores) == list(LYRIC_MOODS)
    assert scores["sad"] == pytest.approx(0.6)
    assert scores["happy"] == 1.0 and scores["romantic"] == 0.0  # clipped to 0..1
    assert scores["angry"] == pytest.approx(0.3)


def test_load_rejects_a_model_for_other_moods(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"version": "x", "moods": ["sad"], "coef": [[0]], "intercept": [0]}))
    with pytest.raises(ValueError):
        LyricsMood.load(path, tmp_path)


def test_committed_model_matches_the_embedder():
    meta = json.loads(LYRICS_MODEL_JSON.read_text(encoding="utf-8"))
    assert tuple(meta["moods"]) == LYRIC_MOODS
    assert np.asarray(meta["coef"]).shape == (len(LYRIC_MOODS), 384)  # multilingual-e5-small vectors
    assert len(meta["intercept"]) == len(LYRIC_MOODS) and meta["version"]
