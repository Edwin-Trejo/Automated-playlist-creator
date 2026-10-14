from src.spotify_client import SCOPES, create_playlist, make_oauth


class FakeSp:
    def __init__(self):
        self.posts = []

    def _post(self, url, payload=None):
        self.posts.append((url, payload))
        if url == "me/playlists":
            return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}
        return {"snapshot_id": "s"}


def test_create_playlist_uses_new_endpoints_and_chunks():
    sp = FakeSp()
    uris = [f"spotify:track:{i}" for i in range(250)]
    out = create_playlist(sp, "Name", "Desc", uris)
    assert out == {"id": "pl1", "url": "https://open.spotify.com/playlist/pl1"}
    assert sp.posts[0] == ("me/playlists", {"name": "Name", "description": "Desc", "public": False})
    item_posts = sp.posts[1:]
    assert [p[0] for p in item_posts] == ["playlists/pl1/items"] * 3
    assert [len(p[1]["uris"]) for p in item_posts] == [100, 100, 50]


def test_make_oauth_reads_env(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    monkeypatch.delenv("SPOTIFY_REDIRECT_URI", raising=False)
    auth = make_oauth(cache_handler=None)
    assert auth.client_id == "cid"
    assert auth.redirect_uri == "http://127.0.0.1:5000/callback"
    assert set(auth.scope.split()) == set(SCOPES.split())
