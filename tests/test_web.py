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
    def __init__(self):
        self.posts = []

    def current_user_saved_tracks(self, limit, offset):
        return {"items": [{"track": {"id": "t1", "uri": "spotify:track:t1", "name": "Song", "artists": [{"name": "A"}]}}],
                "next": None}

    def _post(self, url, payload=None):
        self.posts.append((url, payload))
        return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}


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
        "LLM": lambda mood, genres, artists, **kw: {"name": "Disco Fever", "description": "d", "source": "ollama"},
        "SYNC_FIND_PREVIEW": lambda n, a: "http://p",
        "SYNC_DOWNLOAD": lambda url: np.ones(10, dtype="float32"),
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


def test_generate_rejects_unknown_mood(env):
    _, client, _, _ = env
    assert client.post("/generate", json={"mood": "angry"}).status_code == 400


def test_save_creates_playlist_with_llm_name(env):
    _, client, store, sp = env
    seed(store)
    data = client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]}).get_json()
    assert data["name"] == "Disco Fever" and data["url"].endswith("pl1")
    assert sp.posts[0][0] == "me/playlists"
    assert sp.posts[1] == ("playlists/pl1/items", {"uris": ["spotify:track:t1"]})


def test_save_requires_tracks(env):
    _, client, _, _ = env
    assert client.post("/playlist/save", json={"mood": "dance", "track_ids": []}).status_code == 400


def test_dashboard_renders_controls(env):
    _, client, store, _ = env
    seed(store)
    html = client.get("/").data.decode()
    for needle in ['id="sync-btn"', 'data-mood="dance"', 'value="disco"', 'id="generate-btn"', "app.js"]:
        assert needle in html


def test_server_port_follows_redirect_uri(monkeypatch):
    from src.web.app import server_port
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    assert server_port() == 8888
    monkeypatch.delenv("SPOTIFY_REDIRECT_URI")
    assert server_port() == 5000
