import numpy as np
import pytest

from src.store import TrackStore
from src.web.app import create_app

ANALYSIS = {"genre": "disco", "genre_conf": 0.9, "danceability": 0.8, "energy": 0.8,
            "valence": 0.7, "acousticness": 0.1, "tempo": 122.0}


class FakeAuth:
    logged_in = True

    def __init__(self, cache_handler):
        pass

    def validate_token(self, token):
        return {"access_token": "x"} if FakeAuth.logged_in else None

    @property
    def cache_handler(self):
        return self

    def get_cached_token(self):
        return None

    def get_authorize_url(self):
        return "https://accounts.spotify.com/authorize?fake"

    def get_access_token(self, code):
        return {"access_token": "x"}


class FakeSpotify:
    """In-memory Spotify: the user "me" and their playlists {id: {"name", "owner", "items": [track ids]}}."""

    def __init__(self):
        self.posts = []
        self.playlists = {}

    def current_user_saved_tracks(self, limit, offset):
        return {"items": [{"track": {"id": "t1", "uri": "spotify:track:t1", "name": "Song", "artists": [{"name": "A"}]}}],
                "next": None}

    def add_playlist(self, pid, name, items=(), owner="me", collaborative=False):
        self.playlists[pid] = {"name": name, "owner": owner, "items": list(items), "collaborative": collaborative}

    def _obj(self, pid):
        p = self.playlists[pid]
        return {"id": pid, "name": p["name"], "owner": {"id": p["owner"]}, "collaborative": p["collaborative"],
                "items": {"total": len(p["items"])},
                "external_urls": {"spotify": f"https://open.spotify.com/playlist/{pid}"}}

    def _get(self, url, **params):
        if url == "me":
            return {"id": "me"}
        if url == "me/playlists":
            return {"items": [self._obj(pid) for pid in self.playlists], "next": None}
        pid = url.split("/")[1]
        return {"items": [{"item": {"id": t, "uri": f"spotify:track:{t}", "name": f"Song {t}", "artists": [{"name": "A"}]}}
                          for t in self.playlists[pid]["items"]], "next": None}

    def _post(self, url, payload=None):
        self.posts.append((url, payload))
        if url == "me/playlists":
            pid = f"pl{len(self.playlists) + 1}"
            self.add_playlist(pid, payload["name"])
            return self._obj(pid)
        self.playlists[url.split("/")[1]]["items"] += [u.rsplit(":", 1)[1] for u in payload["uris"]]
        return {"snapshot_id": "s"}


class FakeAnalyzer:
    def analyze_audio(self, y):
        return dict(ANALYSIS)


@pytest.fixture
def env(tmp_path):
    FakeAuth.logged_in = True
    sp = FakeSpotify()
    store = TrackStore(tmp_path / "t.db")
    app = create_app({
        "TESTING": True, "SECRET_KEY": "test", "STORE": store,
        "OAUTH_FACTORY": FakeAuth, "SPOTIFY_FACTORY": lambda auth: sp,
        "ANALYZER_FACTORY": FakeAnalyzer,
        "SYNC_FIND_PREVIEW": lambda n, a: "http://p",
        "SYNC_DOWNLOAD": lambda url: np.ones(10, dtype="float32"),
        "DATASET": None,
    })
    return app, app.test_client(), store, sp


def seed(store, sid="t1"):
    store.upsert({"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": "Song", "artists": "A",
                  "preview_found": 1, "model_version": "v1", **ANALYSIS})


def test_index_shows_login_when_logged_out(env):
    _, client, _, _ = env
    FakeAuth.logged_in = False
    assert b"Log in with Spotify" in client.get("/").data


def test_sync_runs_and_reports_status(env):
    app, client, store, _ = env
    assert client.post("/sync").status_code == 202
    app.extensions["playlist"].sync_thread.join(timeout=5)
    status = client.get("/sync/status").get_json()
    assert status["analyzed"] == 1 and status["running"] is False
    assert status["counts"]["analyzed"] == 1
    assert store.get("t1")["genre"] == "disco"


def test_sync_selected_sources_records_membership(env):
    app, client, store, sp = env
    sp.add_playlist("mine", "Road trip", items=["t1", "t2"])
    assert client.post("/sync", json={"sources": ["liked", "mine"]}).status_code == 202
    app.extensions["playlist"].sync_thread.join(timeout=5)
    assert client.get("/sync/status").get_json()["analyzed"] == 2  # t1 is in both, analyzed once
    assert [t["spotify_id"] for t in store.analyzed_tracks(["liked"])] == ["t1"]
    assert [t["spotify_id"] for t in store.analyzed_tracks(["mine"])] == ["t1", "t2"]


def test_sources_lists_liked_and_readable_playlists(env):
    _, client, store, sp = env
    sp.add_playlist("mine", "Road trip", items=["t1", "t2"])
    sp.add_playlist("collab", "Party", owner="friend", collaborative=True)
    sp.add_playlist("followed", "Top 50", owner="spotify")
    store.set_source_tracks("mine", ["t1"])
    data = client.get("/sources").get_json()["sources"]
    assert [s["id"] for s in data] == ["liked", "mine", "collab"]
    assert data[1] == {"id": "mine", "name": "Road trip", "size": 2, "total": 1, "analyzed": 0, "image": None}


def test_generate_uses_only_selected_sources(env):
    _, client, store, _ = env
    seed(store, "t1")
    seed(store, "t2")
    store.set_source_tracks("liked", ["t1"])
    store.set_source_tracks("mine", ["t2"])
    ids = lambda srcs: [t["spotify_id"] for t in client.post("/generate", json={"sources": srcs}).get_json()["tracks"]]
    assert ids(["liked"]) == ["t1"]
    assert sorted(ids(["liked", "mine"])) == ["t1", "t2"]
    assert ids([]) == []


def test_sync_conflict_while_running(env):
    app, client, _, _ = env
    app.extensions["playlist"].progress.running = True
    assert client.post("/sync").status_code == 409


def test_sync_requires_login(env):
    _, client, _, _ = env
    FakeAuth.logged_in = False
    assert client.post("/sync").status_code == 401


def test_generate_returns_matching_tracks(env):
    _, client, store, _ = env
    seed(store)
    data = client.post("/generate", json={"mood": "dance", "size": 10}).get_json()
    assert [t["spotify_id"] for t in data["tracks"]] == ["t1"]
    assert data["matched"] == 1


def test_generate_applies_slider_adjustments(env):
    _, client, store, _ = env
    seed(store)  # danceability 0.8
    assert len(client.post("/generate", json={"adjust": {"danceability": -1}}).get_json()["tracks"]) == 0
    assert len(client.post("/generate", json={"adjust": {"danceability": 1}}).get_json()["tracks"]) == 1
    assert client.post("/generate", json={"adjust": {"danceability": "x"}}).status_code == 400


def test_generate_rejects_unknown_mood(env):
    _, client, _, _ = env
    assert client.post("/generate", json={"mood": "angry"}).status_code == 400


def test_save_creates_playlist_named_after_mood(env):
    _, client, store, sp = env
    seed(store)
    data = client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]}).get_json()
    assert data["name"] == "Dance" and data["url"].endswith("pl1")
    assert sp.posts[0][0] == "me/playlists" and sp.posts[0][1]["name"] == "Dance"
    assert sp.posts[1] == ("playlists/pl1/items", {"uris": ["spotify:track:t1"]})


def test_save_name_includes_selected_genres(env):
    _, client, store, sp = env
    seed(store)
    data = client.post("/playlist/save", json={"mood": "happy", "genres": ["disco"], "track_ids": ["t1"]}).get_json()
    assert data["name"] == "Happy · disco"


def test_save_adds_only_new_songs_to_existing_playlist(env):
    _, client, store, sp = env
    seed(store, "t1")
    seed(store, "t2")
    client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]})
    data = client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1", "t2"]}).get_json()
    assert data["created"] is False and data["added"] == 1 and data["total"] == 2
    assert len(sp.playlists) == 1 and sp.playlists["pl1"]["items"] == ["t1", "t2"]


def test_save_finds_existing_playlist_by_name(env):
    _, client, store, sp = env
    seed(store)
    sp.add_playlist("old", "Dance", items=["x"])
    sp.add_playlist("theirs", "Happy", owner="someone-else")
    assert client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]}).get_json()["id"] == "old"
    assert client.post("/playlist/save", json={"mood": "happy", "track_ids": ["t1"]}).get_json()["created"] is True


def test_save_follows_renamed_playlist_and_recreates_deleted_one(env):
    _, client, store, sp = env
    seed(store)
    client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]})
    sp.playlists["pl1"]["name"] = "My dance songs"  # renamed in Spotify: still found by its id
    assert client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]}).get_json()["id"] == "pl1"
    del sp.playlists["pl1"]  # deleted (unfollowed) in Spotify: a new one is created
    data = client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]}).get_json()
    assert data["created"] is True and data["name"] == "Dance"


def test_save_new_playlist_gets_unique_name_and_becomes_target(env):
    _, client, store, sp = env
    seed(store, "t1")
    seed(store, "t2")
    client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]})
    data = client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"], "new_playlist": True}).get_json()
    assert data["created"] is True and data["name"] == "Dance (2)"
    assert client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t2"]}).get_json()["id"] == data["id"]


def test_generate_skips_songs_already_in_target_playlist(env):
    _, client, store, sp = env
    seed(store, "t1")
    seed(store, "t2")
    sp.add_playlist("pl9", "Dance", items=["t1"])
    data = client.post("/generate", json={"mood": "dance"}).get_json()
    assert [t["spotify_id"] for t in data["tracks"]] == ["t2"]
    assert data["target"] == {"name": "Dance", "url": "https://open.spotify.com/playlist/pl9", "exists": True, "already_in": 1}
    data = client.post("/generate", json={"mood": "dance", "new_playlist": True}).get_json()
    assert len(data["tracks"]) == 2 and data["target"]["exists"] is False


def test_save_requires_tracks(env):
    _, client, _, _ = env
    assert client.post("/playlist/save", json={"mood": "dance", "track_ids": []}).status_code == 400


def test_dashboard_renders_controls(env):
    _, client, store, _ = env
    seed(store)
    html = client.get("/").data.decode()
    for needle in ['id="sync-btn"', 'data-mood="dance"', 'value="disco"', 'id="generate-btn"', "app.js"]:
        assert needle in html


def test_server_address_follows_redirect_uri(monkeypatch):
    from src.web.app import server_address
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    assert server_address() == ("127.0.0.1", 8888)
    monkeypatch.delenv("SPOTIFY_REDIRECT_URI")
    assert server_address() == ("127.0.0.1", 5000)


def test_server_address_https_redirect_uses_port_setting(monkeypatch):
    from src.web.app import server_address
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "https://playlist.example.ts.net/callback")
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert server_address() == ("0.0.0.0", 5000)
    monkeypatch.setenv("PORT", "6000")
    assert server_address() == ("0.0.0.0", 6000)


def test_session_cookie_secure_only_behind_https(monkeypatch, tmp_path):
    cfg = {"TESTING": True, "STORE": TrackStore(tmp_path / "t.db"), "DATASET": None}
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:5000/callback")
    assert create_app(cfg).config["SESSION_COOKIE_SECURE"] is False
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "https://playlist.example.ts.net/callback")
    assert create_app(cfg).config["SESSION_COOKIE_SECURE"] is True
