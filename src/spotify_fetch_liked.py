import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from deezer_utils import get_deezer_preview
from audio_utils import download_audio, audio_to_mel
from tensorflow.keras.models import load_model
from pathlib import Path
import joblib
import numpy as np


# Resolve model paths relative to project root
base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "models" / "genre_cnn_model.keras"
encoder_path = base_dir / "models" / "label_encoder_cnn.pkl"

# Load CNN model and label encoder
model = load_model(model_path)
le = joblib.load(encoder_path)

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
# results = sp.current_user_saved_tracks(limit=20)

# print("Your liked songs:")
# for idx, item in enumerate(results['items']):
#     track = item['track']
#     artist_names = ", ".join([artist['name'] for artist in track['artists']])
#     print(f"{idx+1}. {track['name']} — {artist_names}")

# print("\n🎧 Fetching your liked songs and preview links...\n")

# results = sp.current_user_saved_tracks(limit=5)

# for item in results["items"]:
#     track = item["track"]
#     name = track["name"]
#     artist = ", ".join([a["name"] for a in track["artists"]])
#     print(f"Searching preview for: {name} — {artist}")
#     preview_url = get_deezer_preview(name, artist)
#     print("Preview:", preview_url if preview_url else "❌ None found")
#     print("-" * 60)

# Fetch one liked song
results = sp.current_user_saved_tracks(limit=1)
track = results["items"][0]["track"]
name = track["name"]
artist = ", ".join(a["name"] for a in track["artists"])
print(f"\n🎧 Classifying: {name} — {artist}")

# Get Deezer preview
preview_url = get_deezer_preview(name, artist)
if not preview_url:
    print("❌ No preview found.")
else:
    print(f"🔗 Preview: {preview_url}")

    # Download + preprocess audio
    y, sr = download_audio(preview_url)
    mel = audio_to_mel(y, sr)

    if mel is not None:
        # Predict genre
        preds = model.predict(mel)
        pred_class = np.argmax(preds)
        confidence = np.max(preds)
        genre = le.inverse_transform([pred_class])[0]

        print(f"✅ Predicted genre: {genre} (confidence: {confidence:.2f})\n" + "-"*60)

    else:
        print("⚠️ Audio processing failed.")


#Fetch your 5 most recent liked songs
results = sp.current_user_saved_tracks(limit=5)

print("\n🎧 Classifying your 5 most recent liked songs...\n")

for item in results["items"]:
    track = item["track"]
    name = track["name"]
    artist = ", ".join(a["name"] for a in track["artists"])
    print(f"🎵 Track: {name} — {artist}")

    # Get preview link from Deezer
    preview_url = get_deezer_preview(name, artist)
    if not preview_url:
        print("❌ No preview found.\n")
        continue

    print(f"🔗 Preview: {preview_url}")

    # Download + process audio
    y, sr = download_audio(preview_url)
    mel = audio_to_mel(y, sr)

    if mel is None:
        print("⚠️ Audio processing failed.\n")
        continue

    # Predict genre
    preds = model.predict(mel)
    confidence = np.max(preds)

    genre = le.inverse_transform([np.argmax(preds)])[0]

    #print(f"✅ Predicted genre: {genre}\n" + "-"*60)
    print(f"✅ Predicted genre: {genre} (confidence: {confidence:.2f})\n" + "-"*60)
