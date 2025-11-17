import os
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tensorflow.keras.models import load_model
import joblib

from deezer_utils import get_deezer_preview
from audio_utils import download_audio, audio_to_mel

# === 1. Load environment and authenticate ===
load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
    scope="user-library-read playlist-modify-public playlist-modify-private"
))

# === 2. Load model and encoder ===
print("🧠 Loading model and label encoder...")
model = load_model("../models/genre_cnn_model.keras")
le = joblib.load("../models/label_encoder_cnn.pkl")

# === 3. Ask user how many songs to classify ===
n = int(input("🎧 How many liked songs would you like to classify? (e.g., 5): "))
print(f"\nFetching {n} liked songs from Spotify...\n")

# === 4. Fetch liked songs ===
results = sp.current_user_saved_tracks(limit=n)
songs = results["items"]

# === 5. Process each song ===
for i, item in enumerate(songs, start=1):
    track = item["track"]
    name = track["name"]
    artist = ", ".join(a["name"] for a in track["artists"])
    print(f"\n[{i}/{len(songs)}] 🎵 {name} — {artist}")

    # Fetch preview from Deezer
    preview_url = get_deezer_preview(name, artist)
    if not preview_url:
        print("❌ No preview found, skipping.")
        continue

    print(f"🔗 Preview: {preview_url}")

    # Download and preprocess
    y, sr = download_audio(preview_url)
    mel = audio_to_mel(y, sr)

    if mel is None:
        print("⚠️ Audio processing failed.")
        continue

    # Predict genre
    preds = model.predict(mel)
    genre_index = np.argmax(preds)
    genre_conf = np.max(preds)
    genre = le.inverse_transform([genre_index])[0]

    print(f"✅ Predicted genre: {genre} ({genre_conf:.2f} confidence)")
    print("-" * 60)

print("\n✅ Done! All requested songs have been classified.")
