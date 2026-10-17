import numpy as np

from src.library_sync import LibrarySync, SyncProgress, iter_liked_tracks, iter_source_tracks, start_background_sync
from src.ml.analyzer import MODEL_VERSION
from src.store import TrackStore

ANALYSIS = {"genre": "pop", "genre_conf": 0.9, "danceability": 0.6, "energy": 0.7,
            "valence": 0.8, "acousticness": 0.1, "tempo": 120.0}


class FakeAnalyzer:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)

    def analyze_audio(self, y):
        if y[0] in self.fail_for:
            raise RuntimeError("model blew up")
        return dict(ANALYSIS)


def track(sid, name=None):
    return {"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": name or sid, "artists": "Main, Feat"}


class FakeSpotify:
    def __init__(self, pages):
        self.pages = pages
        self.offsets = []

    def current_user_saved_tracks(self, limit, offset):
        self.offsets.append(offset)
        return self.pages[len(self.offsets) - 1]


def item(sid, local=False):
    return {"track": {"id": sid, "uri": f"spotify:track:{sid}", "name": sid, "is_local": local,
                      "artists": [{"name": "A"}, {"name": "B"}]}}


def test_iter_liked_tracks_paginates_and_skips_local_and_null():
    sp = FakeSpotify([
        {"items": [item("a"), {"track": None}, item(None, local=True)], "next": "more"},
        {"items": [item("b")], "next": None},
    ])
    tracks = list(iter_liked_tracks(sp, page_size=3))
    assert [t["spotify_id"] for t in tracks] == ["a", "b"]
    assert tracks[0]["artists"] == "A, B"
    assert sp.offsets == [0, 3]


def test_iter_playlist_tracks_reads_item_field_and_skips_episodes():
    class PlaylistSp:
        def _get(self, url, **params):
            return {"items": [{"item": {"id": "a", "uri": "spotify:track:a", "name": "a", "artists": [{"name": "X"}]}},
                              {"item": {"id": "e", "uri": "spotify:episode:e", "name": "pod", "type": "episode"}},
                              {"track": {"id": "b", "uri": "spotify:track:b", "name": "b", "artists": []}}],
                    "next": None}

    tracks = list(iter_source_tracks(PlaylistSp(), "p1"))
    assert [t["spotify_id"] for t in tracks] == ["a", "b"]
    assert tracks[0]["artists"] == "X"


def make_sync(tmp_path, previews, audio, analyzer=None):
    store = TrackStore(tmp_path / "t.db")
    seen_artists = []

    def fake_find(name, artist):
        seen_artists.append(artist)
        return previews.get(name)

    sync = LibrarySync(store, analyzer or FakeAnalyzer(), find_preview=fake_find, download=lambda url: audio.get(url))
    return sync, store, seen_artists


def test_run_records_outcomes(tmp_path):
    previews = {"ok": "u-ok", "bad-audio": "u-bad"}
    audio = {"u-ok": np.array([1.0], dtype=np.float32)}  # u-bad -> None
    sync, store, artists = make_sync(tmp_path, previews, audio)
    progress = SyncProgress()
    sync.run([track("ok"), track("nopreview"), track("bad-audio")], progress)

    assert progress.to_dict()["analyzed"] == 1
    assert (progress.done, progress.analyzed, progress.no_preview, progress.failed) == (3, 1, 1, 1)
    assert progress.running is False
    assert store.get("ok")["energy"] == 0.7 and store.get("ok")["model_version"] == MODEL_VERSION
    assert store.get("nopreview")["preview_found"] == 0
    assert store.get("bad-audio") is None  # failed downloads are retried next sync
    assert artists[0] == "Main"


def test_run_skips_already_analyzed(tmp_path):
    sync, store, _ = make_sync(tmp_path, {"ok": "u"}, {"u": np.array([1.0], dtype=np.float32)})
    sync.run([track("ok")], SyncProgress())
    progress = SyncProgress()
    sync.run([track("ok")], progress)
    assert progress.total == 0


def test_analyzer_exception_does_not_abort(tmp_path):
    previews = {"boom": "u1", "ok": "u2"}
    audio = {"u1": np.array([9.0], dtype=np.float32), "u2": np.array([1.0], dtype=np.float32)}
    sync, store, _ = make_sync(tmp_path, previews, audio, FakeAnalyzer(fail_for=[9.0]))
    progress = SyncProgress()
    sync.run([track("boom"), track("ok")], progress)
    assert progress.failed == 1 and progress.analyzed == 1


def test_background_sync_runs_and_finishes(tmp_path):
    sync, store, _ = make_sync(tmp_path, {"ok": "u"}, {"u": np.array([1.0], dtype=np.float32)})
    progress = SyncProgress()
    thread = start_background_sync([track("ok")], sync, progress)
    assert progress.running is True or progress.done == 1
    thread.join(timeout=5)
    assert progress.running is False and progress.analyzed == 1
