import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Set up Spotify API credentials
auth_manager = SpotifyOAuth(
    client_id="f501fe6ff14d4743b67067ec3eccab06",
    client_secret="3ef165730bb44a69b739356a8d034799",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-library-read playlist-modify-private playlist-modify-public",
    show_dialog=True
)

sp = spotipy.Spotify(auth_manager=auth_manager)
token_info = auth_manager.get_cached_token()
if token_info:
    print("Granted scopes:", token_info.get("scope"))
else:
    print("No token found.")


#print("Authenticated scopes:", sp.current_user()['product'])
#print("Token info:", auth_manager.get_cached_token())

def get_liked_tracks(limit=50):
    results = sp.current_user_saved_tracks(limit=limit)
    tracks = []
    for item in results['items']:
        track = item['track']
        tracks.append({
            'id': track['id'],
            'name': track['name'],
            'artists': [artist['name'] for artist in track['artists']]
        })
    return tracks

def get_audio_features(track_id):
    features = sp.audio_features([track_id])
    if features and features[0]:
        return features[0]
    else:
        print(f"No audio features for track ID: {track_id}")
        return {}


def get_current_user_id():
    return sp.current_user()['id']


def create_playlist_if_not_exists(user_id, genre_name):
    playlists = sp.current_user_playlists(limit=5)
    for playlist in playlists['items']:
        if playlist['name'].lower() == genre_name.lower():
            return playlist['id']
    new_playlist = sp.user_playlist_create(user=user_id, name=genre_name, public=False)
    return new_playlist['id']

def add_track_to_playlist(playlist_id, track_id):
    sp.playlist_add_items(playlist_id, [track_id])   


