import spotipy
from spotipy.oauth2 import SpotifyOAuth

# You can reuse the same auth manager to avoid re-authenticating
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="f501fe6ff14d4743b67067ec3eccab06",
    client_secret="3ef165730bb44a69b739356a8d034799",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-library-read playlist-modify-private playlist-modify-public",
    show_dialog=True
))

def create_playlist_if_not_exists(user_id, genre_name):
    """
    Checks if a playlist for the genre already exists; creates it if it doesn't.
    Returns the playlist ID.
    """
    playlists = sp.current_user_playlists(limit=5)
    for playlist in playlists['items']:
        if playlist['name'].lower() == genre_name.lower():
            return playlist['id']
    
    new_playlist = sp.user_playlist_create(
        user=user_id,
        name=genre_name,
        public=False,
        description=f"Auto-created playlist for {genre_name} songs"
    )
    return new_playlist['id']

def add_track_to_playlist(playlist_id, track_id):
    """
    Adds a track to the given playlist.
    """
    sp.playlist_add_items(playlist_id, [track_id])
