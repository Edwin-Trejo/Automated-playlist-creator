# Spotify Playlist Automation — Phase 1 Design

## Context
The repo started as a learning project: a CNN trained on GTZAN (10 genres) and a LightGBM trained on Spotify audio-features, plus CLI scripts (`src/app.py`, `src/spotify_fetch_liked.py`) that classify a few liked songs via Deezer previews. Goal now: a Flask web app that reads the user's Liked Songs and **creates new Spotify playlists by mood / danceability / energy / genre**, powered by the user's own trained models, with an LLM naming the playlist.

Constraints discovered:
- Spotify removed `audio-features` (danceability, energy, valence…) for new apps (Nov 2024) → we must **predict these ourselves**. The existing LightGBM depends on them, so it can't serve real songs.
- `src/app.py` is a CLI (`input()`), not Flask; README overstates what exists.
- Bug in `src/audio_utils.py:download_audio`: pads/trims to `22050*30` samples *before* resampling from 44.1 kHz → model sees ~15 s, not 30 s.
- `create_playlist.py`, `test_with_spotify.py` empty; `__pycache__/` is committed; GTZAN audio (`data/genres/`, `data/jazz/`) is untracked and should be git-ignored.

## Agreed scope
**Phase 1 (this plan):** own trained models only · mood preset buttons (Happy, Sad, Dance, Chill, Energetic) + optional sliders + genre filter · Ollama-generated name/description (template fallback) · save playlist to Spotify. **No free-text box / embeddings.**
**Phase 2 (later):**
- ReccoBeats hybrid — ReccoBeats as primary feature source, own models as fallback and benchmarked against it.
- Modern genre coverage — replace/extend GTZAN's 10 genres (no reggaeton, techno, house, Latin…) by training genre on `dataset.csv`'s Spotify genre labels (114, incl. reggaeton, latin, salsa, techno, house, deep-house, edm, dancehall, k-pop, r-n-b), grouped into ~15–20 broad genres (e.g. Reggaeton/Urbano, Latin, House, Techno, EDM, Hip-Hop, R&B, Pop, Rock, Metal, Jazz, Classical, Country, Reggae, K-pop), ~150–250 songs each via the same Deezer-preview + YAMNet pipeline (optionally keep GTZAN as extra examples). Evaluate with artist-grouped splits; expect a lower headline accuracy than GTZAN but better real-world fit.
- Dataset lookup — when a liked song exists in `dataset.csv` (match by Spotify `track_id`, fallback normalized name + first artist), use its real Spotify danceability/energy/valence/acousticness directly; run the trained models only for songs not in the dataset. Store which source each value came from.
**Later:** free-text mood → sentence-transformers embeddings → song-level retrieval.

Runs locally for the user (Spotify dev mode).

**Revision (2026-10-14): training architecture.** Training moved to one Colab notebook (`notebooks/train_models_colab.ipynb`) that streams all data by link. Features are now a frozen pretrained CNN (YAMNet, ONNX) embedding plus 5 Librosa rhythm/loudness numbers (2,053 per song, 16 kHz audio); genre = LightGBM classifier (81.6% on unseen artists, beat the NN head and the ensemble), mood = 4 LightGBM regressors (R² 0.52–0.69, artist-grouped test). This replaces the from-scratch CNN and 72-feature Librosa vectors described in the Architecture section below; the app no longer depends on TensorFlow.

## Architecture

```
src/
  audio/
    preview.py        # Deezer search (reuse/upgrade deezer_utils.get_deezer_preview; use artist:"" track:"" query)
    loader.py         # download_audio (fixed: resample → then trim/pad 30 s)
    features.py       # mel_spectrogram() for CNN (reuse audio_to_mel logic) + summary_vector() for LightGBM
                      #   (MFCC mean/std, chroma, spectral centroid/contrast/rolloff, ZCR, RMS, onset strength, tempo)
  training/
    collect_mood_dataset.py  # sample ~5k tracks from data/dataset.csv (stratified by genre), find Deezer previews,
                             # extract summary_vector in parallel (ThreadPool for downloads, ProcessPool for librosa),
                             # save data/features/mood_train.parquet (resumable cache)
    train_mood_models.py     # LightGBM regressors: danceability, energy, valence, acousticness; report MAE/R² vs
                             # mean baseline; save models/mood_regressors.pkl
    train_genre_lgbm.py      # LightGBM genre classifier on GTZAN summary vectors → models/genre_lgbm_audio.pkl
    train_genre_cnn.py       # cleaned-up train_model_cnn.py (path-safe, early-stopping actually wired in)
  notebooks/train_cnn_colab.ipynb   # same CNN training on Colab GPU
  legacy/train_model_spotify_features.py  # old train_model.py, kept for reference
  ml/analyzer.py      # analyze(track) → {genre, genre_conf (CNN+LGBM averaged probs), danceability, energy,
                      #   valence, acousticness, tempo (librosa beat_track)}; models loaded once
  store.py            # SQLite data/app.db: tracks(spotify_id PK, name, artists, preview_found, genre, genre_conf,
                      #   danceability, energy, valence, acousticness, tempo, model_version, analyzed_at)
  library_sync.py     # paginate current_user_saved_tracks, analyze only uncached songs; background thread + progress
  moods.py            # preset → target ranges; merge with slider overrides
  selector.py         # filter by ranges/genres, rank by distance to range centers, top-N
  llm.py              # Ollama POST localhost:11434/api/generate (format=json, model from OLLAMA_MODEL,
                      #   default llama3.2:3b) → {name, description}; template fallback on any failure
  web/
    app.py            # Flask factory; spotipy OAuth w/ FlaskSessionCacheHandler
    routes: /  /login  /callback  /logout  POST /sync  GET /sync/status  GET /library
            POST /generate (preview list)  POST /playlist/save
    templates/ (Jinja) + static/ (vanilla JS + CSS)
tests/                # pytest
```

### Data flow
1. Login → OAuth (`user-library-read playlist-modify-private playlist-modify-public`).
2. **Sync**: fetch all liked songs → for each uncached: Deezer preview → audio → features → analyzer → SQLite. UI polls `/sync/status` (done/total/skipped-no-preview).
3. **Generate**: user picks mood preset, tweaks sliders/genres/size → `selector` returns ranked tracks → preview table (remove tracks if desired).
4. **Save**: `llm.py` gets name+description from mood + top genres/artists → create playlist + add tracks via spotipy → link shown.

### Mood presets (starting values, tunable in `moods.py`)
| Mood | Dance | Energy | Valence | Tempo |
|---|---|---|---|---|
| Happy | .5–1 | .5–1 | .6–1 | any |
| Sad | 0–.5 | 0–.5 | 0–.35 | <110 |
| Dance | .65–1 | .6–1 | any | 110–135 |
| Chill | 0–.6 | 0–.45 | .3–.7 | <115 |
| Energetic | any | .7–1 | any | >120 |

### Error handling
- No Deezer preview → stored with `preview_found=0`, excluded from generation, counted in UI.
- Deezer/Spotify 429 or timeout → retry w/ backoff; one failed song never aborts sync.
- Ollama down → template name ("Happy Mix · Oct 2026") + generic description.
- Models missing → clear startup error telling which training script to run.
- Selector returns < requested size → show what matched + suggest loosening sliders.

## Milestones (each independently runnable)
1. Repo hygiene: `.gitignore` (GTZAN audio, `data/features/`, `app.db`, `__pycache__`), untrack pycache, move legacy script, fix `download_audio` bug, path-safe model loading.
2. Audio + features module with tests.
3. Training data collection + mood regressors (report metrics).
4. Genre LightGBM on GTZAN + CNN script/Colab notebook; ensemble in `analyzer.py`.
5. SQLite store + library sync (CLI-callable first).
6. Moods + selector.
7. Flask app: auth, sync w/ progress, library view, generate, save to Spotify.
8. Ollama integration + fallback; README rewrite (incl. Ollama install: `ollama pull llama3.2:3b`).

## Dependencies
Add: `pyarrow`, `pytest`, `requests` (exists). Ollama is an external install. Remove `seaborn/matplotlib` from runtime requirements → `requirements-train.txt`.

## Verification
- `pytest` — unit tests for features (shapes/finite values on synthetic sine/noise audio), `moods` merging, `selector` ranking, `llm` JSON parsing + fallback (mocked HTTP), `store` caching, Flask routes with mocked spotipy.
- Training scripts print MAE/R² (mood) and accuracy/confusion (genre); expect mood regressors to beat mean baseline clearly (energy/acousticness easiest, valence hardest).
- End-to-end: `python -m src.web.app` → log in → sync liked songs → generate "Dance" (20 songs) → save → confirm playlist appears in Spotify with Ollama name; stop Ollama and confirm fallback name.

## Process after approval
Save this design to `docs/superpowers/specs/2026-10-13-playlist-automation-phase1-design.md`, then write a detailed step-by-step implementation plan (writing-plans) for your review before coding.
