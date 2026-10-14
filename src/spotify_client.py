import os

from spotipy.oauth2 import SpotifyOAuth

SCOPES = "user-library-read playlist-modify-private playlist-modify-public"
DEFAULT_REDIRECT = "http://127.0.0.1:5000/callback"
MAX_ITEMS_PER_REQUEST = 100


def make_oauth(cache_handler):
    return SpotifyOAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", DEFAULT_REDIRECT),
        scope=SCOPES,
        cache_handler=cache_handler,
    )


def create_playlist(sp, name, description, uris, public=False):
    # Raw endpoints on purpose: Spotify's Feb 2026 API removed /users/{id}/playlists and
    # /playlists/{id}/tracks, which spotipy's helper methods may still call.
    playlist = sp._post("me/playlists", payload={"name": name, "description": description, "public": public})
    for i in range(0, len(uris), MAX_ITEMS_PER_REQUEST):
        sp._post(f"playlists/{playlist['id']}/items", payload={"uris": uris[i:i + MAX_ITEMS_PER_REQUEST]})
    return {"id": playlist["id"], "url": playlist["external_urls"]["spotify"]}
