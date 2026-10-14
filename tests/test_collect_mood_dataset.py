import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.training import collect_mood_dataset as cmd


def test_sample_tracks_dedupes_and_caps_per_genre():
    df = pd.DataFrame({
        "track_id": ["a", "a", "b", "c", "d", "e"],
        "track_genre": ["pop", "rock", "pop", "pop", "rock", "rock"],
        "track_name": list("123456"), "artists": list("xyzuvw"),
    })
    out = cmd.sample_tracks(df, per_genre=2)
    assert out["track_id"].is_unique
    assert out.groupby("track_genre").size().max() <= 2


ROW = {"track_id": "t1", "track_name": "Song", "artists": "Main Artist;Feat", "track_genre": "pop",
       "danceability": 0.5, "energy": 0.6, "valence": 0.7, "acousticness": 0.1}


def test_process_row_uses_first_artist_and_returns_features(monkeypatch):
    seen = {}
    monkeypatch.setattr(cmd, "find_preview", lambda name, artist: seen.setdefault("artist", artist) and "http://p")
    monkeypatch.setattr(cmd, "download_audio", lambda url: np.zeros(10, dtype=np.float32))
    monkeypatch.setattr(cmd, "summary_vector", lambda y: np.arange(72, dtype=np.float32))
    out = cmd.process_row(ROW)
    assert seen["artist"] == "Main Artist"
    assert out["ok"] is True
    assert out[SUMMARY_FEATURE_NAMES[5]] == 5.0
    assert out["energy"] == 0.6


def test_process_row_marks_missing_preview(monkeypatch):
    monkeypatch.setattr(cmd, "find_preview", lambda name, artist: None)
    assert cmd.process_row(ROW) == {"track_id": "t1", "ok": False}
