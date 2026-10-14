import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.audio.features import SUMMARY_FEATURE_NAMES

TARGETS = ("danceability", "energy", "valence", "acousticness")
VERSION = "mood-v1"


def train_regressors(X, y, seed=42):
    models = {}
    for target in TARGETS:
        model = LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        )
        model.fit(X[SUMMARY_FEATURE_NAMES], y[target])
        models[target] = model
    return {
        "models": models,
        "feature_names": list(SUMMARY_FEATURE_NAMES),
        "train_means": {t: float(y[t].mean()) for t in TARGETS},
        "version": VERSION,
    }


def evaluate_regressors(bundle, X, y):
    metrics = {}
    for target, model in bundle["models"].items():
        pred = model.predict(X[bundle["feature_names"]])
        baseline = np.full(len(y), bundle["train_means"][target])
        metrics[target] = {
            "mae": float(mean_absolute_error(y[target], pred)),
            "r2": float(r2_score(y[target], pred)),
            "baseline_mae": float(mean_absolute_error(y[target], baseline)),
        }
    return metrics


def predict_mood(bundle, vec):
    X = pd.DataFrame([np.asarray(vec, dtype=np.float32)], columns=bundle["feature_names"])
    return {t: float(np.clip(m.predict(X)[0], 0.0, 1.0)) for t, m in bundle["models"].items()}
