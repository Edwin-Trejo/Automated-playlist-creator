# Automated Playlist Creator

A local Flask web app that reads your Spotify **Liked Songs and your own playlists**, analyzes every song with machine-learning models trained for this project, and builds playlists by **mood, danceability, energy, tempo and genre**. Each mood gets its own playlist in your Spotify account ("Happy", or "Happy · Reggaeton / Urbano" with a genre filter), and new matching songs are added to it over time.

## How it works

```
Liked Songs + selected playlists (Spotify API)
   │  track name + artist
   ▼
Deezer 30 s preview (16 kHz)
   ▼
YAMNet, a pretrained CNN (frozen) ─► 1,024-number embedding every 0.48 s ─┐
Librosa ─► tempo, loudness, beat strength ────────────────────────────────┴─► 2,053 features per song
   ├─► LightGBM classifier ──────────► genre (19 broad genres)
   └─► 4 LightGBM regressors ────────► danceability, energy, valence, acousticness
   ▼
SQLite cache (each song analyzed once, remembers which source it came from)
   ▼
Mood (ranked against your whole library)  or  Genre ─► selector ─► preview
   ▼
Add new songs to that mood's or genre's playlist (or create it)
```

- **Why train mood models?** Spotify stopped serving `audio-features` (danceability, energy, valence…) to new apps in Nov 2024. This project re-creates them: `data/dataset.csv` holds Spotify's original scores for ~114k songs, and the models learn to predict those scores from each song's audio.
- **Transfer learning:** YAMNet (Google, trained on millions of YouTube clips) is used as a frozen feature extractor, the "ears". Only the small LightGBM models on top are trained.
- **Genres:** Spotify's 114 genre labels in `dataset.csv` are grouped into 19 broad genres (Reggaeton / Urbano, Latin, House, Techno, K-pop / J-pop / C-pop, Rock…) by `data/genre_groups.json`. The genre is used as a filter for mood playlists, so only the broad genres matter.

## Model results

All scores are on a held-out test set of **artists the models never saw** during training (artist-grouped split).

**Genre:** 4,955 songs from `dataset.csv`, 19 broad genres

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| **LightGBM on YAMNet features (used by the app)** | **43.8%** | **0.44** |
| Same, with GTZAN clips added to training | 41.5% | 0.42 |

Random guessing would get about 5%. Most mistakes are between similar-sounding genres (e.g. House / Techno / EDM). The earlier GTZAN-only model reached 81.6%, but on only 10 very different genres, and it couldn't recognize reggaeton, house, K-pop and others.

**Mood:** 6,211 songs from `dataset.csv` with Deezer previews

| Target | MAE | Baseline MAE (always predict the mean) | R² |
|---|---|---|---|
| Danceability | 0.078 | 0.138 | 0.64 |
| Energy | 0.090 | 0.198 | 0.75 |
| Valence | 0.142 | 0.220 | 0.52 |
| Acousticness | 0.125 | 0.276 | 0.70 |

## Setup

1. **Python 3.11** virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
   pip install -r requirements.txt
   ```
2. **Spotify app** at https://developer.spotify.com/dashboard
   - Add a redirect URI such as `http://127.0.0.1:5000/callback` (Spotify rejects `localhost`, and other hosts must use HTTPS; see Deploy). Locally, the app serves on whatever port this URI uses.
   - Development Mode requires the app owner to have Spotify Premium. Other listeners must be added under *User Management* (max 5).
   - Development Mode only lets the app read songs from playlists you **own or collaborate on**, so playlists you only follow can't be used as sources.
3. **`.env`**: copy `.env.example` to `.env` and fill in `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI` and a random `FLASK_SECRET_KEY`.

## Training

The trained models are committed in `models/`, so this is only needed to retrain.

1. Open https://colab.research.google.com, choose **File → Upload notebook**, and pick `notebooks/train_models_colab.ipynb`.
2. Choose **Runtime → Run all** (a CPU runtime is fine). All data is streamed by link: `dataset.csv` and `genre_groups.json` from this repo's `main` branch, and previews from Deezer. Progress is saved to Google Drive, so a disconnect just resumes, and songs processed in earlier runs are reused. A first run takes about 1–2 h.
3. Unzip the downloaded `playlist_models.zip` into `models/`, and bump `MODEL_VERSION` in `src/ml/analyzer.py` so the next sync re-analyzes your songs with the new models.

## Run

```
python -m src.web.app
```

Open the redirect URI's host (e.g. http://127.0.0.1:5000) and log in with Spotify.

1. **Sources:** Liked Songs is selected by default. Check any of your own playlists to use them too, then click **Sync selected**. The first sync analyzes every song once (a few seconds per song); later syncs only process new songs.
2. **Make a playlist by Mood or Genre:**
   - **Mood:** Happy, Sad, Dance, Chill or Energetic. Songs are ranked by how far they lean toward the mood *compared with every analyzed song* (percentiles of valence, energy, danceability, acousticness and tempo), and the ones that fit well enough are kept. Fixed thresholds didn't work: the models' scores cluster in the middle, so in a dance-heavy library almost nothing was "low energy". Sad went from 1 to about 100 matches in a 521-song library.
   - **Genre:** one or more of the 19 genres, ranked by how confident the genre model is.
3. **Generate**, then **Save**:
   - By default, songs are **added to your existing playlist for that mood or genre** ("Sad", or "House · Techno"). The preview leaves out songs already in it, so only new ones are added. The app remembers the playlist, so renaming it in Spotify is fine; if you delete it, the next save creates a new one.
   - Check **Create a new playlist** to start a separate one ("Sad (2)"). Future songs are then added to that one.

## Deploy (Proxmox container + Tailscale)

The app runs with Docker Compose and is reached from your phone over Tailscale. Spotify only allows `http://` redirect URIs on `127.0.0.1`, so `compose.yml` runs two containers:
- **`tailscale`** joins your tailnet as its own machine (`TS_HOSTNAME`, default `playlist`) and serves `https://playlist.<tailnet>.ts.net` with a valid certificate (`deploy/serve.json`). Only devices on your tailnet can open it.
- **`app`** (built from the `Dockerfile`: Python 3.11, ffmpeg, the models) shares the tailscale container's network and listens on `127.0.0.1:5000`, so nothing is exposed on the LAN. The song cache is kept in the `app-data` volume, so rebuilds keep it.

1. **Server:** any Linux host with Docker (e.g. a Proxmox container with nesting enabled), 2 cores, 2 GB RAM, 16 GB disk. The image is about 1.7 GB.
2. **Tailscale:** in the admin console, turn on **MagicDNS** and **HTTPS certificates** (DNS page), and create an **auth key** (Settings → Keys).
3. **Secrets:** clone the repo, copy `.env.example` to `.env` (`chmod 600`), and fill in `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, a random `FLASK_SECRET_KEY`, `TS_AUTHKEY`, and `SPOTIFY_REDIRECT_URI=https://playlist.<tailnet>.ts.net/callback`. The `.env` stays on the server and is never committed or copied into the image.
4. **Spotify dashboard:** add the same `https://…ts.net/callback` redirect URI. Keep the `127.0.0.1` one for local development.
5. **Start:** `docker compose up -d --build`. **Update:** `git pull && docker compose up -d --build`.

Logs: `docker compose logs -f app`. To skip re-analyzing your library, copy your local cache in: `docker compose cp data/app.db app:/data/app.db`, then `docker compose restart app`.

## Tests

```
python -m pytest
```

## Project layout

```
notebooks/      Colab training notebook (YAMNet features + LightGBM models)
models/         trained models (yamnet.onnx, genre_lgbm.txt, mood_*.txt, *_meta.json)
data/           dataset.csv (Spotify scores + genres), genre_groups.json (114 -> 19 genres), app.db (local cache)
src/audio/      Deezer preview lookup, decoding, YAMNet + Librosa feature extraction
src/ml/         Analyzer: features -> genre + mood predictions
src/web/        Flask app, routes, templates, static JS/CSS
src/store.py    SQLite cache (songs, their sources, mood playlists) · src/library_sync.py incremental sync
src/moods.py    mood recipes (feature weights) · src/selector.py percentile ranking · src/playlists.py playlist names
src/spotify_client.py   Spotify login and playlist endpoints
src/legacy/     the original model trained on Spotify audio-features (reference only)
Dockerfile, compose.yml, deploy/serve.json   server deploy (app + Tailscale HTTPS)
```

## Roadmap

- **Next:** lyrics as a mood signal (LRCLIB lyrics + a multilingual sentiment model), to catch sad songs with danceable beats.
- **Multiple users:** per-user sources and playlists, sharing the analyzed-song cache.
- **Better genres:** merge the genres the model confuses most (e.g. House / Techno / EDM), or add more songs per genre.
- **Dataset lookup:** if a song is already in `dataset.csv`, use its real Spotify mood scores and only predict the rest.
- **ReccoBeats hybrid:** use ReccoBeats audio features as the primary source, keep these models as the fallback, and benchmark them against it.
- **Later:** free-text mood search ("rainy-day study music") with sentence-transformer embeddings.
