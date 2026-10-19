"""Train the free lyrics mood model (models/lyrics_ridge.json) from data/lyrics_labels.csv.

The labels are Claude Haiku 5.5's mood scores for ~6,600 dataset.csv songs, made once (about $0.72) by
reading what each song's lyrics mean. No lyrics text is stored, and retraining needs no API key: this
script fetches lyrics from LRCLIB (cached in data/features/), embeds them with the same Embedder the app
uses, and fits one ridge regression per mood.

    pip install scikit-learn scipy        # training only
    python scripts/train_lyrics_model.py [--version l2]

Bump --version after changing the labels or the embedder, so the next sync re-scores every song.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# onnxruntime (via lyrics_mood) must load before scipy/scikit-learn: the other order crashes on Windows.
from src.config import FEATURES_DIR, LYRICS_MODEL_JSON  # noqa: E402
from src.lyrics import LyricsUnavailable, find_lyrics  # noqa: E402
from src.ml.lyrics_mood import EMBEDDER_REPO, LYRIC_MOODS, Embedder  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

LABELS_CSV = ROOT / "data" / "lyrics_labels.csv"
LYRICS_CACHE = FEATURES_DIR / "lyrics_cache.json"
EMBED_CACHE = FEATURES_DIR / "lyrics_embeddings.npz"
ALPHAS = np.logspace(-2, 3, 12)


def fetch_lyrics(labels):
    cache = json.loads(LYRICS_CACHE.read_text(encoding="utf-8")) if LYRICS_CACHE.exists() else {}
    todo = [r for r in labels.itertuples() if r.track_id not in cache]
    print(f"lyrics: {len(labels) - len(todo)} cached, {len(todo)} to fetch from LRCLIB", flush=True)
    for i, r in enumerate(todo, 1):
        try:
            cache[r.track_id] = find_lyrics(r.track_name, r.artists.split(";")[0], r.album_name, r.duration_ms)
        except LyricsUnavailable:
            print("  LRCLIB unavailable; run again later to fetch the rest", flush=True)
            break
        if i % 250 == 0:
            LYRICS_CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  {i}/{len(todo)}", flush=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    LYRICS_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def embed(labels, cache):
    stored = dict(np.load(EMBED_CACHE, allow_pickle=False)) if EMBED_CACHE.exists() else {}
    known = dict(zip(stored.get("ids", []), stored.get("X", [])))
    rows = [r for r in labels.itertuples() if (cache.get(r.track_id) or {}).get("lyrics")]
    todo = [r for r in rows if r.track_id not in known]
    if todo:
        embedder, start = Embedder(), time.time()
        for i, r in enumerate(todo, 1):
            known[r.track_id] = embedder(cache[r.track_id]["lyrics"])
            if i % 500 == 0:
                print(f"  embedded {i}/{len(todo)} ({time.time() - start:.0f}s)", flush=True)
        ids = list(known)
        np.savez(EMBED_CACHE, ids=np.array(ids), X=np.array([known[k] for k in ids], dtype=np.float32))
    ids = [r.track_id for r in rows]
    return labels.set_index("track_id").loc[ids].reset_index(), np.array([known[k] for k in ids])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="l1")
    args = parser.parse_args()

    labels = pd.read_csv(LABELS_CSV)
    train, X = embed(labels, fetch_lyrics(labels))
    Y = train[list(LYRIC_MOODS)].clip(0, 1).to_numpy()
    groups = train["artists"].str.split(";").str[0]
    print(f"training on {len(train)} songs with lyrics ({groups.nunique()} artists)", flush=True)

    cv = {}
    for j, mood in enumerate(LYRIC_MOODS):
        scores = [spearmanr(RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr, j]).predict(X[va]), Y[va, j])[0]
                  for tr, va in GroupKFold(5).split(X, groups=groups)]
        cv[mood] = round(float(np.mean(scores)), 3)
    print("artist-grouped CV Spearman: " + ", ".join(f"{m} {s:.2f}" for m, s in cv.items()), flush=True)

    models = [RidgeCV(alphas=ALPHAS).fit(X, Y[:, j]) for j in range(len(LYRIC_MOODS))]
    LYRICS_MODEL_JSON.write_text(json.dumps({
        "version": args.version, "embedder": EMBEDDER_REPO, "moods": list(LYRIC_MOODS),
        "coef": [[float(f"{c:.6g}") for c in m.coef_] for m in models],
        "intercept": [float(f"{m.intercept_:.6g}") for m in models],
        "alpha": [float(m.alpha_) for m in models],
        "n_train": len(train), "cv_spearman": cv, "trained_at": date.today().isoformat(),
    }), encoding="utf-8")
    print(f"wrote {LYRICS_MODEL_JSON}", flush=True)


if __name__ == "__main__":
    main()
