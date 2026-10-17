"""Last-resort analysis for songs without any audio preview.

1. DatasetLookup: Spotify's real scores from data/dataset.csv, when the song is in it.
2. artist_estimate: the genre and average moods of the same artist's analyzed songs (marked as an estimate).
"""
import csv
import json
import re
import threading
from collections import Counter

from src.audio.preview import clean_title
from src.config import DATASET_CSV, GENRE_GROUPS_JSON

MOODS = ("danceability", "energy", "valence", "acousticness", "tempo")


def _norm(text):
    return re.sub(r"[^\w]+", " ", (text or "").lower()).strip()


def _key(name, artist):
    return _norm(clean_title(name)), _norm(artist)


class DatasetLookup:
    """Loads dataset.csv on first use (~114k rows) and finds songs by Spotify id, else by title + first artist."""

    def __init__(self, csv_path=DATASET_CSV, groups_path=GENRE_GROUPS_JSON):
        self.csv_path, self.groups_path = csv_path, groups_path
        self._by_id = self._by_name = None
        self._lock = threading.Lock()

    def _load(self):
        groups = json.loads(open(self.groups_path, encoding="utf-8").read())
        sub_to_genre = {sub: g for g, subs in groups.items() for sub in subs}
        by_id, by_name = {}, {}
        with open(self.csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("track_id") or not row.get("track_name"):
                    continue
                entry = by_id.get(row["track_id"])
                if entry is None:
                    entry = {m: float(row[m]) for m in MOODS}
                    entry["genres"] = Counter()
                    by_id[row["track_id"]] = entry
                    by_name.setdefault(_key(row["track_name"], row["artists"].split(";")[0]), entry)
                genre = sub_to_genre.get(row["track_genre"])
                if genre:
                    entry["genres"][genre] += 1
        self._by_id, self._by_name = by_id, by_name

    def find(self, track):
        """Analysis dict for a track ({spotify_id, name, artists}), or None if it isn't in the dataset."""
        with self._lock:
            if self._by_id is None:
                self._load()
        entry = self._by_id.get(track["spotify_id"]) or self._by_name.get(
            _key(track["name"], track["artists"].split(", ")[0]))
        if entry is None:
            return None
        genre = entry["genres"].most_common(1)[0][0] if entry["genres"] else None
        return {**{m: entry[m] for m in MOODS}, "genre": genre, "genre_conf": 1.0 if genre else None}


def artist_estimate(track, analyzed_tracks):
    """Genre and average moods of the same main artist's analyzed songs, or None if there are none."""
    artist = track["artists"].split(", ")[0].lower()
    same = [t for t in analyzed_tracks
            if t["artists"].split(", ")[0].lower() == artist and t.get("analysis_source") != "artist"]
    if not same:
        return None
    genres = Counter(t["genre"] for t in same if t.get("genre"))
    genre, count = genres.most_common(1)[0] if genres else (None, 0)
    out = {m: sum(t[m] for t in same) / len(same) for m in MOODS if all(t.get(m) is not None for t in same)}
    # Confidence: how consistent the artist is, halved because the song itself was never heard.
    return {**out, "genre": genre, "genre_conf": round(count / len(same) / 2, 4) if genre else None}
