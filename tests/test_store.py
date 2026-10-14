import pytest

from src.store import TrackStore


@pytest.fixture
def store(tmp_path):
    return TrackStore(tmp_path / "t.db")


def row(sid, **kw):
    base = {"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": f"Song {sid}", "artists": "A",
            "preview_found": 1, "genre": "pop", "genre_conf": 0.9, "danceability": 0.5, "energy": 0.5,
            "valence": 0.5, "acousticness": 0.5, "tempo": 120.0, "model_version": "v1", "analyzed_at": "now"}
    base.update(kw)
    return base


def test_upsert_and_get(store):
    store.upsert(row("a"))
    assert store.get("a")["genre"] == "pop"
    store.upsert(row("a", genre="rock"))
    assert store.get("a")["genre"] == "rock"
    assert store.get("missing") is None


def test_partial_row_fills_nulls(store):
    store.upsert({"spotify_id": "x", "uri": "u", "name": "n", "artists": "a", "preview_found": 0, "model_version": "v1"})
    assert store.get("x")["energy"] is None


def test_versions_analyzed_and_counts(store):
    store.upsert(row("a"))
    store.upsert(row("b", preview_found=0, genre=None))
    store.upsert(row("c", model_version="v0"))
    assert store.ids_with_version("v1") == {"a", "b"}
    assert [t["spotify_id"] for t in store.analyzed_tracks()] == ["a", "c"]
    assert store.counts() == {"total": 3, "analyzed": 2, "no_preview": 1}


def test_get_many_preserves_order(store):
    for sid in "abc":
        store.upsert(row(sid))
    assert [t["spotify_id"] for t in store.get_many(["c", "zz", "a"])] == ["c", "a"]
