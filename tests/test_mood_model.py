import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.ml.mood_model import TARGETS, evaluate_regressors, predict_mood, train_regressors


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(SUMMARY_FEATURE_NAMES))), columns=SUMMARY_FEATURE_NAMES)
    y = pd.DataFrame({t: 1 / (1 + np.exp(-X.iloc[:, i])) for i, t in enumerate(TARGETS)})
    return X, y


def test_regressors_learn_signal_and_beat_baseline():
    X, y = make_data()
    bundle = train_regressors(X.iloc[:300], y.iloc[:300])
    metrics = evaluate_regressors(bundle, X.iloc[300:], y.iloc[300:])
    for t in TARGETS:
        assert metrics[t]["mae"] < metrics[t]["baseline_mae"]


def test_predict_mood_returns_clipped_values():
    X, y = make_data()
    bundle = train_regressors(X, y)
    out = predict_mood(bundle, X.iloc[0].to_numpy() * 100)
    assert set(out) == set(TARGETS)
    assert all(0.0 <= v <= 1.0 for v in out.values())
