from lightgbm import LGBMClassifier

from src.audio.features import SUMMARY_FEATURE_NAMES


def train_genre_classifier(X, y, seed=42):
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )
    model.fit(X[SUMMARY_FEATURE_NAMES], y)
    return {"model": model, "classes": [str(c) for c in model.classes_], "feature_names": list(SUMMARY_FEATURE_NAMES)}
