# 🎧 Automated Playlist Creator

A local Flask web app that reads your Spotify **Liked Songs**, analyzes every song with machine-learning models trained for this project, and builds new playlists by **mood, danceability, energy, tempo and genre**. A local LLM (Ollama) names each playlist and writes its description, and the playlist is saved straight to your Spotify account.

## How it works

```
Liked Songs (Spotify API)
   │  track name + artist
   ▼
Deezer 30 s preview (16 kHz)
   ▼
YAMNet, a pretrained CNN (frozen) ─► 1,024-number embedding every 0.48 s ─┐
Librosa ─► tempo, loudness, beat strength ────────────────────────────────┴─► 2,053 features per song
   ├─► LightGBM classifier ──────────► genre (10 GTZAN genres)
   └─► 4 LightGBM regressors ────────► danceability, energy, valence, acousticness
   ▼
SQLite cache (each song analyzed once)
   ▼
Mood preset + sliders + genre filter ─► selector ─► preview table
   ▼
Ollama (name + description) ─► new Spotify playlist
```

- **Why train mood models?** Spotify stopped serving `audio-features` (danceability, energy, valence…) to new apps in Nov 2024. This project re-creates them: `data/dataset.csv` holds Spotify's original scores for ~114k songs, and the models learn to predict those scores from each song's audio.
- **Transfer learning:** YAMNet (Google, trained on millions of YouTube clips) is used as a frozen feature extractor, the "ears". Only the small LightGBM models on top are trained, which reaches far higher accuracy than training a CNN from scratch on ~1,000 clips.

## Model results

All scores are on a held-out test set of **artists the models never saw** during training (artist-grouped split).

**Genre:** GTZAN, 999 clips, 10 genres

| Model | Test accuracy |
|---|---|
| **LightGBM on YAMNet features (used by the app)** | **81.6%** |
| Neural-network head on YAMNet features | 77.1% |
| Average of both | 76.5% |

**Mood:** 4,275 songs from `dataset.csv` with Deezer previews

| Target | MAE | Baseline MAE (always predict the mean) | R² |
|---|---|---|---|
| Danceability | 0.083 | 0.138 | 0.60 |
| Energy | 0.098 | 0.200 | 0.69 |
| Valence | 0.145 | 0.221 | 0.52 |
| Acousticness | 0.133 | 0.282 | 0.67 |

## Setup

1. **Python 3.11** virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
   pip install -r requirements.txt
   ```
2. **Spotify app** at https://developer.spotify.com/dashboard
   - Add a redirect URI such as `http://127.0.0.1:5000/callback` (Spotify rejects `localhost`). The app serves on whatever port this URI uses.
   - Development Mode requires the app owner to have Spotify Premium. Other listeners must be added under *User Management* (max 5).
3. **`.env`**: copy `.env.example` to `.env` and fill in `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI` and a random `FLASK_SECRET_KEY`.
4. **Ollama** (free, local LLM): install from https://ollama.com/download, then `ollama pull llama3.2:3b` (or set `OLLAMA_MODEL` to a model you already have, e.g. `gemma2:9b`). Without Ollama the app still works and uses a default playlist name.

## Training

The trained models are committed in `models/`, so this is only needed to retrain.

1. Open https://colab.research.google.com, choose **File → Upload notebook**, and pick `notebooks/train_models_colab.ipynb`.
2. Choose **Runtime → Run all** (a CPU runtime is fine). All data is streamed by link: GTZAN from Hugging Face, `dataset.csv` from this repo, and previews from Deezer. Progress is saved to Google Drive, so a disconnect just resumes. Expect about 1–1.5 h.
3. Unzip the downloaded `playlist_models.zip` into `models/`.

## Run

```
python -m src.web.app
```

Open the redirect URI's host (e.g. http://127.0.0.1:5000), log in with Spotify and click **Sync liked songs**. The first sync analyzes every song once (a few seconds per song); later syncs only process new likes. Then pick a mood, adjust the sliders, **Generate**, and **Save to Spotify**.

## Tests

```
python -m pytest
```

## Project layout

```
notebooks/      Colab training notebook (YAMNet features + LightGBM models)
models/         trained models (yamnet.onnx, genre_lgbm.txt, mood_*.txt, *_meta.json)
src/audio/      Deezer preview lookup, decoding, YAMNet + Librosa feature extraction
src/ml/         Analyzer: features -> genre + mood predictions
src/web/        Flask app, routes, templates, static JS/CSS
src/store.py    SQLite cache · src/library_sync.py incremental sync
src/moods.py    mood presets · src/selector.py track ranking · src/llm.py Ollama naming
src/legacy/     the original model trained on Spotify audio-features (reference only)
```

## Roadmap

- **Phase 2:** ReccoBeats hybrid. Use ReccoBeats audio features as the primary source, keep these models as the fallback, and benchmark them against it.
- **Phase 2:** modern genres. Train genre on Spotify's labels in `dataset.csv` (reggaeton, Latin, techno, house, EDM, K-pop…), grouped into ~15–20 broad genres, instead of only GTZAN's 10.
- **Phase 2:** dataset lookup. If a liked song is already in `dataset.csv`, use its real Spotify mood scores and only predict the rest.
- **Later:** free-text mood search ("rainy-day study music") with sentence-transformer embeddings.
