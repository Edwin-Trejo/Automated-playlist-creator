import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from deezer_utils import get_deezer_preview


# === Load environment variables from .env ===
load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")

# === Spotify authentication ===
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-library-read"  # permission to read liked songs
))
# === Fetch first 20 liked songs ===
results = sp.current_user_saved_tracks(limit=20)

print("Your liked songs:")
for idx, item in enumerate(results['items']):
    track = item['track']
    artist_names = ", ".join([artist['name'] for artist in track['artists']])
    print(f"{idx+1}. {track['name']} — {artist_names}")

print("\n🎧 Fetching your liked songs and preview links...\n")

results = sp.current_user_saved_tracks(limit=5)

for item in results["items"]:
    track = item["track"]
    name = track["name"]
    artist = ", ".join([a["name"] for a in track["artists"]])
    print(f"Searching preview for: {name} — {artist}")
    preview_url = get_deezer_preview(name, artist)
    print("Preview:", preview_url if preview_url else "❌ None found")
    print("-" * 60)