import os

from spotipy.oauth2 import SpotifyOAuth

SCOPES = "user-library-read playlist-modify-private playlist-modify-public"
DEFAULT_REDIRECT = "http://127.0.0.1:5000/callback"
MAX_ITEMS_PER_REQUEST = 100

# Raw endpoints on purpose: Spotify's Feb 2026 API removed /users/{id}/playlists and
# /playlists/{id}/tracks, which spotipy's helper methods may still call.


def make_oauth(cache_handler):
    return SpotifyOAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", DEFAULT_REDIRECT),
        scope=SCOPES,
        cache_handler=cache_handler,
    )


def _pages(sp, url, **params):
    page = sp._get(url, **params)
    while page:
        yield from page.get("items") or []
        page = sp._get(page["next"]) if page.get("next") else None


def current_user_id(sp):
    return sp._get("me")["id"]


def user_playlists(sp):
    """Playlists the user owns or follows (deleting a playlist in Spotify only unfollows it)."""
    return [p for p in _pages(sp, "me/playlists", limit=50) if p]


def find_playlist(playlists, user_id, playlist_id=None, name=None):
    """The user's playlist with this id, else one they own with exactly this name, else None."""
    for p in playlists:
        if playlist_id and p["id"] == playlist_id:
            return p
    for p in playlists:
        if name and p.get("name") == name and (p.get("owner") or {}).get("id") == user_id:
            return p
    return None


def unique_name(name, playlists):
    taken = {p.get("name") for p in playlists}
    n = 2
    candidate = name
    while candidate in taken:
        candidate, n = f"{name} ({n})", n + 1
    return candidate


def playlist_track_ids(sp, playlist_id):
    ids = []
    for entry in _pages(sp, f"playlists/{playlist_id}/items", limit=100):
        track = entry.get("item") or entry.get("track") or {}
        if track.get("id"):
            ids.append(track["id"])
    return ids


def add_items(sp, playlist_id, uris):
    for i in range(0, len(uris), MAX_ITEMS_PER_REQUEST):
        sp._post(f"playlists/{playlist_id}/items", payload={"uris": uris[i:i + MAX_ITEMS_PER_REQUEST]})


def create_playlist(sp, name, description, uris, public=False):
    playlist = sp._post("me/playlists", payload={"name": name, "description": description, "public": public})
    add_items(sp, playlist["id"], uris)
    return {"id": playlist["id"], "url": playlist["external_urls"]["spotify"]}
