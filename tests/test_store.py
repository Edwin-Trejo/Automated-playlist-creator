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


def test_playlist_mapping(store):
    assert store.get_playlist("happy") is None
    store.set_playlist("happy", "p1", "Happy", "u1")
    store.set_playlist("happy", "p2", "Happy (2)", "u2")
    assert {k: store.get_playlist("happy")[k] for k in ("playlist_id", "name", "url")} == {
        "playlist_id": "p2", "name": "Happy (2)", "url": "u2"}


def test_sources_filter_tracks_and_counts(store):
    for sid in "abc":
        store.upsert(row(sid))
    store.upsert(row("d", preview_found=0))
    store.set_source_tracks("liked", ["a", "b", "d"])
    store.set_source_tracks("p1", ["b", "c"])
    store.set_source_tracks("p1", ["c"])  # replaces, does not add
    assert [t["spotify_id"] for t in store.analyzed_tracks(["liked"])] == ["a", "b"]
    assert [t["spotify_id"] for t in store.analyzed_tracks(["liked", "p1"])] == ["a", "b", "c"]
    assert store.analyzed_tracks([]) == []
    assert store.counts(["liked"]) == {"total": 3, "analyzed": 2, "no_preview": 1}
    assert store.source_counts() == {"liked": {"total": 3, "analyzed": 2}, "p1": {"total": 1, "analyzed": 1}}


def test_existing_library_becomes_liked_songs(tmp_path):
    old = TrackStore(tmp_path / "old.db")
    old.upsert(row("a"))
    with old._connect() as conn:  # simulate a database from before sources existed
        conn.execute("DELETE FROM track_sources")
    reopened = TrackStore(tmp_path / "old.db")
    assert [t["spotify_id"] for t in reopened.analyzed_tracks(["liked"])] == ["a"]


def test_get_many_preserves_order(store):
    for sid in "abc":
        store.upsert(row(sid))
    assert [t["spotify_id"] for t in store.get_many(["c", "zz", "a"])] == ["c", "a"]
