from src.spotify_client import (SCOPES, create_playlist, find_playlist, make_oauth, playlist_track_ids, unique_name,
                                user_playlists)


class PagedSp:
    def __init__(self, pages):
        self.pages = pages
        self.urls = []

    def _get(self, url, **params):
        self.urls.append(url)
        return self.pages[url]


def test_playlist_track_ids_follows_next_and_reads_item_or_track():
    sp = PagedSp({
        "playlists/p/items": {"items": [{"item": {"id": "a"}}, {"item": None}], "next": "https://api/next"},
        "https://api/next": {"items": [{"track": {"id": "b"}}, {"item": {"id": None}}], "next": None},
    })
    assert playlist_track_ids(sp, "p") == ["a", "b"]


def test_user_playlists_skips_empty_entries():
    sp = PagedSp({"me/playlists": {"items": [{"id": "1"}, None], "next": None}})
    assert user_playlists(sp) == [{"id": "1"}]


def test_find_playlist_prefers_id_then_own_name():
    lists = [{"id": "a", "name": "Happy", "owner": {"id": "other"}}, {"id": "b", "name": "Happy", "owner": {"id": "me"}},
             {"id": "c", "name": "Renamed", "owner": {"id": "me"}}]
    assert find_playlist(lists, "me", "c", "Happy")["id"] == "c"
    assert find_playlist(lists, "me", "gone", "Happy")["id"] == "b"
    assert find_playlist(lists, "me", None, "Sad") is None


def test_unique_name():
    lists = [{"name": "Happy"}, {"name": "Happy (2)"}]
    assert unique_name("Happy", lists) == "Happy (3)"
    assert unique_name("Sad", lists) == "Sad"


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
