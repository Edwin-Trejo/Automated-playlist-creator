"""Build mood training data: Deezer previews for dataset.csv tracks -> Librosa features + Spotify labels.

Usage: python -m src.training.collect_mood_dataset --per-genre 45 [--workers 8]
Resumable: re-running skips tracks already saved or already failed.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES, summary_vector
from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.config import DATASET_CSV, FEATURES_DIR
from src.ml.mood_model import TARGETS

OUT_PATH = FEATURES_DIR / "mood_train.parquet"
FAILED_PATH = FEATURES_DIR / "mood_failed.csv"
FLUSH_EVERY = 100


def sample_tracks(df, per_genre, seed=42):
    df = df.dropna(subset=["track_name", "artists"]).drop_duplicates("track_id")
    return df.sample(frac=1, random_state=seed).groupby("track_genre").head(per_genre).reset_index(drop=True)


def process_row(row):
    artist = str(row["artists"]).split(";")[0]
    url = find_preview(str(row["track_name"]), artist)
    if not url:
        return {"track_id": row["track_id"], "ok": False}
    y = download_audio(url)
    if y is None:
        return {"track_id": row["track_id"], "ok": False}
    out = {"track_id": row["track_id"], "ok": True, "track_genre": row["track_genre"]}
    out.update(dict(zip(SUMMARY_FEATURE_NAMES, summary_vector(y).tolist())))
    out.update({t: float(row[t]) for t in TARGETS})
    return out


def _flush(rows, failed):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        new = pd.DataFrame(rows)
        if OUT_PATH.exists():
            new = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True)
        new.to_parquet(OUT_PATH, index=False)
    if failed:
        pd.DataFrame({"track_id": failed}).to_csv(FAILED_PATH, mode="a", header=not FAILED_PATH.exists(), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-genre", type=int, default=45)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    sampled = sample_tracks(pd.read_csv(DATASET_CSV), args.per_genre)
    done = set(pd.read_parquet(OUT_PATH)["track_id"]) if OUT_PATH.exists() else set()
    if FAILED_PATH.exists():
        done |= set(pd.read_csv(FAILED_PATH)["track_id"])
    pending = [r for r in sampled.to_dict("records") if r["track_id"] not in done]
    print(f"{len(sampled)} sampled, {len(done)} already processed, {len(pending)} to go", flush=True)

    rows, failed, finished = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_row, r) for r in pending]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as exc:  # one bad track never stops the run
                print(f"worker error: {exc}", flush=True)
                continue
            if res["ok"]:
                rows.append(res)
            else:
                failed.append(res["track_id"])
            finished += 1
            if finished % FLUSH_EVERY == 0:
                _flush(rows, failed)
                rows, failed = [], []
                print(f"{finished}/{len(pending)} processed", flush=True)
    _flush(rows, failed)
    total = len(pd.read_parquet(OUT_PATH)) if OUT_PATH.exists() else 0
    print(f"Done. {total} usable tracks in {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
