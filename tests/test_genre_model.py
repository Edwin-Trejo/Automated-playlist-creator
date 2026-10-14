import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.ml.genre_model import train_genre_classifier


def test_genre_classifier_learns_separable_classes():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, len(SUMMARY_FEATURE_NAMES))), columns=SUMMARY_FEATURE_NAMES)
    y = pd.Series(np.where(X.iloc[:, 0] > 0, "rock", "jazz"))
    bundle = train_genre_classifier(X, y)
    assert bundle["classes"] == ["jazz", "rock"]
    assert (bundle["model"].predict(X) == y).mean() > 0.9
