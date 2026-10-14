"""Train LightGBM genre classifier on GTZAN audio. Usage: python -m src.training.train_genre_lgbm"""
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import librosa
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.audio.features import SUMMARY_FEATURE_NAMES, summary_vector
from src.audio.loader import prepare
from src.config import CLIP_SECONDS, FEATURES_DIR, GTZAN_DIR, MODELS_DIR
from src.ml.genre_model import train_genre_classifier

CACHE = FEATURES_DIR / "gtzan_summary.parquet"


def _extract(path_and_genre):
    path, genre = path_and_genre
    try:
        y, sr = librosa.load(path, sr=None, duration=CLIP_SECONDS)
    except Exception as exc:  # GTZAN has a known corrupt file (jazz.00054.wav)
        print(f"skip {path}: {exc}")
        return None
    return {"genre": genre, **dict(zip(SUMMARY_FEATURE_NAMES, summary_vector(prepare(y, sr)).tolist()))}


def build_features():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    jobs = [(str(p), p.parent.name) for p in sorted(GTZAN_DIR.glob("*/*.wav"))]
    print(f"Extracting features from {len(jobs)} GTZAN clips...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = [r for r in pool.map(_extract, jobs, chunksize=8) if r]
    df = pd.DataFrame(rows)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def main():
    df = build_features()
    X, y = df[SUMMARY_FEATURE_NAMES], df["genre"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    bundle = train_genre_classifier(X_train, y_train)
    print(classification_report(y_test, bundle["model"].predict(X_test)))
    joblib.dump(bundle, MODELS_DIR / "genre_lgbm_audio.pkl")
    print(f"Saved {MODELS_DIR / 'genre_lgbm_audio.pkl'}")


if __name__ == "__main__":
    main()
