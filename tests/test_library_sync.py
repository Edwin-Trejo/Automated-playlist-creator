import numpy as np

from src.library_sync import LibrarySync, SyncProgress, iter_liked_tracks, iter_source_tracks, start_background_sync
from src.lyrics import LOOKUP_VERSION
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


def track(sid, name=None, artists="Main, Feat"):
    return {"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": name or sid, "artists": artists}


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


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def find(self, track):
        return self.rows.get(track["spotify_id"])


def make_sync(tmp_path, previews, audio, analyzer=None, dataset=None):
    store = TrackStore(tmp_path / "t.db")
    seen_artists = []

    def fake_find(name, artist):
        seen_artists.append(artist)
        return previews.get(name)

    sync = LibrarySync(store, analyzer or FakeAnalyzer(), find_preview=fake_find, download=lambda url: audio.get(url),
                       dataset=dataset)
    return sync, store, seen_artists


def test_run_records_outcomes(tmp_path):
    previews = {"ok": "u-ok", "bad-audio": "u-bad"}
    audio = {"u-ok": np.array([1.0], dtype=np.float32)}  # u-bad -> None
    sync, store, artists = make_sync(tmp_path, previews, audio)
    progress = SyncProgress()
    sync.run([track("ok"), track("nopreview", artists="Other"), track("bad-audio", artists="Third")], progress)

    assert (progress.done, progress.analyzed, progress.no_preview, progress.failed) == (3, 1, 2, 0)
    assert progress.running is False
    ok = store.get("ok")
    assert ok["energy"] == 0.7 and ok["model_version"] == MODEL_VERSION and ok["analysis_source"] == "audio"
    assert store.get("nopreview")["preview_found"] == 0
    assert store.get("bad-audio")["preview_found"] == 0
    assert artists[0] == "Main"


def test_no_audio_falls_back_to_dataset(tmp_path):
    dataset = FakeDataset({"x": {"genre": "Latin", "genre_conf": 1.0, "danceability": 0.9, "energy": 0.8,
                                 "valence": 0.7, "acousticness": 0.1, "tempo": 100.0}})
    sync, store, _ = make_sync(tmp_path, {}, {}, dataset=dataset)
    progress = SyncProgress()
    sync.run([track("x")], progress)
    assert progress.from_dataset == 1
    assert store.get("x")["genre"] == "Latin" and store.get("x")["analysis_source"] == "dataset"


def test_no_audio_is_estimated_from_artist_after_their_other_songs(tmp_path):
    # "lost" comes first but is estimated at the end, once "ok" by the same artist is analyzed.
    sync, store, _ = make_sync(tmp_path, {"ok": "u"}, {"u": np.array([1.0], dtype=np.float32)})
    progress = SyncProgress()
    sync.run([track("lost"), track("ok")], progress)
    assert (progress.analyzed, progress.estimated, progress.done) == (1, 1, 2)
    lost = store.get("lost")
    assert lost["analysis_source"] == "artist" and lost["genre"] == "pop" and lost["energy"] == 0.7
    assert lost["genre_conf"] == 0.5


def test_no_preview_and_estimates_are_retried_next_sync(tmp_path):
    sync, store, _ = make_sync(tmp_path, {"ok": "u"}, {"u": np.array([1.0], dtype=np.float32)})
    sync.run([track("ok"), track("lost"), track("gone", artists="Nobody")], SyncProgress())
    progress = SyncProgress()
    sync.run([track("ok"), track("lost"), track("gone", artists="Nobody")], progress)
    assert progress.total == 2  # "ok" is done; the estimate and the no-preview song are tried again


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


class FakeScorer:
    version = "l1"

    def score(self, lyrics):
        return {"sad": 0.9 if "cry" in lyrics else 0.1, "happy": 0.5, "calm": 0.2}


def lyrics_sync(tmp_path, lyrics, scorer=FakeScorer):
    store = TrackStore(tmp_path / "t.db")
    calls = []

    def fake_lyrics(name, artist, album=None, duration_ms=None):
        calls.append((name, artist, album, duration_ms))
        found = lyrics[name]
        if isinstance(found, Exception):
            raise found
        return found

    sync = LibrarySync(store, FakeAnalyzer(), find_preview=lambda n, a: "u",
                       download=lambda url: np.array([1.0], dtype=np.float32), lyrics=scorer, find_lyrics=fake_lyrics)
    return sync, store, calls


def test_lyrics_are_scored_after_the_audio(tmp_path):
    from src.lyrics import LyricsUnavailable
    lyrics = {"a": {"lyrics": "I cry every night", "instrumental": False},
              "b": {"lyrics": None, "instrumental": True},
              "c": {"lyrics": None, "instrumental": False}}
    sync, store, calls = lyrics_sync(tmp_path, lyrics)
    progress = SyncProgress()
    sync.run([{**track("a"), "album": "Al", "duration_ms": 1000}, track("b"), track("c")], progress)
    assert (progress.lyrics_total, progress.lyrics_done, progress.lyrics_found) == (3, 3, 1)
    assert calls[0] == ("a", "Main", "Al", 1000)
    a, b, c = store.get("a"), store.get("b"), store.get("c")
    assert a["lyric_sad"] == 0.9 and a["lyrics_status"] == "found" and a["lyrics_version"] == "l1"
    assert b["lyrics_status"] == "instrumental" and b["lyric_sad"] is None
    assert c["lyrics_status"] == "none" and c["lyrics_version"] == f"l1+{LOOKUP_VERSION}"

    progress = SyncProgress()  # already checked with this lyrics model: nothing to do next time
    sync.run([track("a"), track("b"), track("c")], progress)
    assert progress.lyrics_total == 0


def test_unavailable_song_is_skipped_and_retried_next_sync(tmp_path):
    from src.lyrics import LyricsUnavailable
    sync, store, calls = lyrics_sync(tmp_path, {"a": LyricsUnavailable("get"), "b": {"lyrics": "x", "instrumental": False}})
    progress = SyncProgress()
    sync.run([track("a"), track("b")], progress)
    assert len(calls) == 2 and progress.lyrics_paused and progress.running is False
    assert store.ids_with_lyrics("l1", LOOKUP_VERSION) == {"b"}  # "a" is tried again next sync


def test_lyrics_wait_for_next_sync_when_the_sites_stay_down(tmp_path):
    from src.library_sync import MAX_UNAVAILABLE_IN_A_ROW
    from src.lyrics import LyricsUnavailable
    names = [f"s{i}" for i in range(MAX_UNAVAILABLE_IN_A_ROW + 3)]
    sync, store, calls = lyrics_sync(tmp_path, {n: LyricsUnavailable("get") for n in names})
    sync.run([track(n) for n in names], SyncProgress())
    assert len(calls) == MAX_UNAVAILABLE_IN_A_ROW


def test_missing_lyrics_model_skips_lyrics(tmp_path):
    def broken():
        raise FileNotFoundError("lyrics_ridge.json")
    sync, store, calls = lyrics_sync(tmp_path, {}, scorer=broken)
    progress = SyncProgress()
    sync.run([track("a")], progress)
    assert progress.analyzed == 1 and progress.lyrics_total == 0 and calls == []
