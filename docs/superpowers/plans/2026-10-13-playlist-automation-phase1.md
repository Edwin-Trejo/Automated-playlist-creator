# Spotify Playlist Automation — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Flask web app that syncs the user's Spotify Liked Songs, analyzes each song with the user's own trained models (genre + danceability/energy/valence/acousticness + tempo), and creates new Spotify playlists from mood presets, sliders and genre filters, named by a local Ollama LLM.

**Architecture:** A `src` package with focused modules: `audio/` (Deezer preview lookup, decoding, Librosa features), `ml/` (model training helpers + `Analyzer` that serves CNN+LightGBM genre ensemble and LightGBM mood regressors), `store.py` (SQLite cache), `library_sync.py` (incremental background analysis), `moods.py` + `selector.py` (playlist building), `llm.py` (Ollama with template fallback), `spotify_client.py`, and `web/` (Flask factory, routes, Jinja + vanilla JS UI). Training scripts live in `src/training/` and are run once from the CLI.

**Tech Stack:** Python 3.11, Flask 2.3.3, spotipy, librosa, pydub+ffmpeg, LightGBM, TensorFlow/Keras, scikit-learn 1.4.2, pandas 2.2.2, pyarrow, SQLite, Ollama (`llama3.2:3b`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-playlist-automation-phase1-design.md`

## Global Constraints

- Python 3.11 (existing `__pycache__` is cpython-311). Keep pins `Flask==2.3.3`, `scikit-learn==1.4.2`, `pandas==2.2.2`.
- All commands run from the repo root; modules run as `python -m src....`. All paths come from `src/config.py` (no `../` relative paths).
- Env var names: `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI` (default `http://127.0.0.1:5000/callback` — Spotify rejects `localhost` redirect URIs), `FLASK_SECRET_KEY`, `OLLAMA_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.2:3b`).
- Spotify scopes: `user-library-read playlist-modify-private playlist-modify-public`. Playlist creation uses `POST /me/playlists` and `POST /playlists/{id}/items` (Feb 2026 API; old `/users/{id}/playlists` and `/tracks` endpoints are removed), max 100 URIs per add request.
- CNN input stays `(1, 128, 640, 1)` mel-dB so the committed `models/genre_cnn_model.keras` keeps working.
- Audio pipeline: mono, peak-normalized, 22,050 Hz, exactly 30 s (`22050*30` samples) **after** resampling.
- Mood features predicted: `danceability`, `energy`, `valence`, `acousticness` (0–1). `tempo` comes from Librosa beat tracking.
- Mood presets (exact): Happy, Sad, Dance, Chill, Energetic, with ranges from the spec table. No free-text input, no embeddings, no ReccoBeats in Phase 1.
- Ollama failure of any kind → template name `"<Mood> Mix · <Mon YYYY>"`; never fail the save because of the LLM.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Silent / undecodable previews** — a preview that decodes to all zeros (or fails to decode) must not produce NaN features or crash sync; it's counted as failed. (Task 3 tests)
2. **Local files and unavailable tracks in Liked Songs** — items with `track: null`, `is_local: true` or no `id` must be skipped, not crash pagination. (Task 8 test)
3. **Playlists over 100 songs** — adding items must chunk at 100 URIs per request. (Task 11 test)
4. **Ollama returns prose, invalid JSON, empty name, or a very long name with newlines** — parse the JSON object out, collapse whitespace, truncate (name ≤100, description ≤300), else fall back to the template. (Task 10 tests)
5. **Pressing "Sync" while a sync is running** — second request returns HTTP 409 instead of starting a parallel thread that double-writes. (Task 12 test)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/__init__.py`, `src/audio/__init__.py`, `src/ml/__init__.py`, `src/training/__init__.py`, `src/web/__init__.py` | Package markers (empty) |
| `src/config.py` | Paths and audio constants |
| `src/http_utils.py` | `get_with_retry` (429/backoff) |
| `src/audio/preview.py` | `find_preview` via Deezer search |
| `src/audio/loader.py` | `decode_mp3`, `prepare`, `download_audio` (fixed) |
| `src/audio/features.py` | `mel_spectrogram`, `summary_vector`, `SUMMARY_FEATURE_NAMES` |
| `src/ml/mood_model.py` | `TARGETS`, `train_regressors`, `evaluate_regressors`, `predict_mood` |
| `src/ml/genre_model.py` | `train_genre_classifier` |
| `src/ml/analyzer.py` | `Analyzer`, `ModelsMissingError`, `MODEL_VERSION` |
| `src/training/collect_mood_dataset.py` | Build `data/features/mood_train.parquet` from `data/dataset.csv` |
| `src/training/train_mood_models.py` | Train/save `models/mood_regressors.pkl` |
| `src/training/train_genre_lgbm.py` | Train/save `models/genre_lgbm_audio.pkl` on GTZAN |
| `src/training/train_genre_cnn.py` | Train/save CNN (replaces `src/train_model_cnn.py`) |
| `notebooks/train_cnn_colab.ipynb` | Run CNN training on Colab GPU |
| `src/legacy/train_model_spotify_features.py` | Old `src/train_model.py`, reference only |
| `src/store.py` | `TrackStore` SQLite cache |
| `src/library_sync.py` | `iter_liked_tracks`, `SyncProgress`, `LibrarySync`, `start_background_sync` |
| `src/moods.py` | `MOODS`, `FULL_RANGE`, `build_targets` |
| `src/selector.py` | `select_tracks` |
| `src/llm.py` | `generate_playlist_text`, `parse_llm_json`, `template_text` |
| `src/spotify_client.py` | `SCOPES`, `make_oauth`, `create_playlist` |
| `src/web/app.py` | `create_app`, `AppState`, `__main__` runner |
| `src/web/routes.py` | Blueprint with all routes |
| `src/web/templates/{base,index,dashboard}.html`, `src/web/static/{app.js,style.css}` | UI |
| `tests/fakes.py`, `tests/test_*.py`, `pytest.ini` | Tests |

Deleted: `src/app.py`, `src/spotify_fetch_liked.py`, `src/audio_utils.py`, `src/deezer_utils.py`, `src/create_playlist.py`, `src/test_with_spotify.py`, tracked `__pycache__/`.

---

### Task 1: Repo hygiene, package skeleton, config, pytest

**Files:**
- Modify: `.gitignore`
- Create: `src/__init__.py`, `src/audio/__init__.py`, `src/ml/__init__.py`, `src/training/__init__.py`, `src/web/__init__.py`, `src/legacy/__init__.py` (all empty), `src/config.py`, `pytest.ini`, `tests/__init__.py` (empty), `tests/test_config.py`
- Move: `src/train_model.py` → `src/legacy/train_model_spotify_features.py`
- Untrack: `__pycache__/`

**Interfaces:**
- Produces: `src.config` constants `ROOT: Path`, `DATA_DIR`, `MODELS_DIR`, `FEATURES_DIR`, `GTZAN_DIR`, `DATASET_CSV`, `DB_PATH`, `SAMPLE_RATE = 22050`, `CLIP_SECONDS = 30`, `CLIP_SAMPLES = 661500`.

- [ ] **Step 1: Verify environment**

Run: `python --version && python -c "import librosa, lightgbm, tensorflow, spotipy, flask, pydub; print('ok')" && ffmpeg -version | head -1`
Expected: Python 3.11.x, `ok`, an ffmpeg version line. If ffmpeg missing: `pip install ffmpeg-downloader && ffdl install --add-path`, then reopen the shell. If imports fail: `pip install -r requirements.txt`. Also `pip install pytest pyarrow`.

- [ ] **Step 2: Replace `.gitignore`**

```gitignore
.env
.cache
*.wav
__pycache__/
.pytest_cache/

# GTZAN audio (large, downloaded separately)
data/genres/
data/jazz/

# Generated
data/features/
data/app.db
models/*.png
```

- [ ] **Step 3: Untrack pycache and move the legacy script**

```bash
git rm -r --cached __pycache__
mkdir -p src/legacy
git mv src/train_model.py src/legacy/train_model_spotify_features.py
```

Add this as the first line of `src/legacy/train_model_spotify_features.py`:

```python
# LEGACY: trained on Spotify audio-features, which Spotify no longer serves to new apps (Nov 2024). Kept for reference only.
```

- [ ] **Step 4: Write the failing test** — `tests/test_config.py`

```python
from src import config


def test_paths_are_absolute_and_rooted():
    assert config.ROOT.is_absolute()
    assert (config.ROOT / "README.md").exists()
    assert config.MODELS_DIR == config.ROOT / "models"
    assert config.DB_PATH == config.ROOT / "data" / "app.db"


def test_audio_constants():
    assert config.SAMPLE_RATE == 22050
    assert config.CLIP_SAMPLES == 22050 * 30
```

`pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 5: Run to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'config'`.

- [ ] **Step 6: Implement** — `src/config.py` (and create the empty `__init__.py` files listed above)

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURES_DIR = DATA_DIR / "features"
GTZAN_DIR = DATA_DIR / "genres"
DATASET_CSV = DATA_DIR / "dataset.csv"
DB_PATH = DATA_DIR / "app.db"

SAMPLE_RATE = 22050
CLIP_SECONDS = 30
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
```

- [ ] **Step 7: Run to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pytest.ini src/__init__.py src/audio/__init__.py src/ml/__init__.py src/training/__init__.py src/web/__init__.py src/legacy src/config.py tests/__init__.py tests/test_config.py
git commit -m "chore: package skeleton, config paths, gitignore, legacy model" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: HTTP retry helper + Deezer preview lookup

**Files:**
- Create: `src/http_utils.py`, `src/audio/preview.py`, `tests/fakes.py`, `tests/test_http_utils.py`, `tests/test_preview.py`
- Delete: `src/deezer_utils.py` (replaced)

**Interfaces:**
- Produces: `get_with_retry(url, *, params=None, session=None, retries=3, backoff=1.0, timeout=10, sleep=time.sleep) -> requests.Response | None`
- Produces: `find_preview(track_name: str, artist_name: str, *, session=None, sleep=time.sleep) -> str | None`
- Produces (tests): `tests.fakes.FakeResp(payload=None, status=200, headers=None, content=b"")`, `tests.fakes.FakeSession(responses)` with `.calls: list[dict]` (each item a response or an Exception to raise).

- [ ] **Step 1: Write test fakes** — `tests/fakes.py`

```python
class FakeResp:
    def __init__(self, payload=None, status=200, headers=None, content=b""):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.ok = 200 <= status < 300
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Returns queued responses in order; Exception items are raised."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
```

- [ ] **Step 2: Write failing tests** — `tests/test_http_utils.py`

```python
import requests

from src.http_utils import get_with_retry
from tests.fakes import FakeResp, FakeSession


def test_returns_ok_response():
    s = FakeSession([FakeResp({"a": 1})])
    assert get_with_retry("http://x", session=s, sleep=lambda _: None).json() == {"a": 1}


def test_honors_retry_after_on_429():
    waits = []
    s = FakeSession([FakeResp(status=429, headers={"Retry-After": "2"}), FakeResp({"ok": True})])
    resp = get_with_retry("http://x", session=s, sleep=waits.append)
    assert resp.json() == {"ok": True}
    assert waits == [2.0]


def test_client_error_returns_none_without_retry():
    s = FakeSession([FakeResp(status=404)])
    assert get_with_retry("http://x", session=s, sleep=lambda _: None) is None
    assert len(s.calls) == 1


def test_network_errors_exhaust_retries():
    s = FakeSession([requests.ConnectionError()] * 4)
    assert get_with_retry("http://x", session=s, retries=3, sleep=lambda _: None) is None
    assert len(s.calls) == 4
```

`tests/test_preview.py`:

```python
from src.audio.preview import find_preview
from tests.fakes import FakeResp, FakeSession

NO_SLEEP = lambda _: None


def test_strict_query_hit():
    s = FakeSession([FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "Daft Punk"}}]})])
    assert find_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) == "http://p/1.mp3"
    assert s.calls[0]["params"]["q"] == 'artist:"Daft Punk" track:"One More Time"'


def test_falls_back_to_loose_query_with_artist_match():
    s = FakeSession([
        FakeResp({"data": []}),
        FakeResp({"data": [
            {"preview": "http://p/wrong.mp3", "artist": {"name": "Someone Else"}},
            {"preview": "http://p/right.mp3", "artist": {"name": "Daft Punk"}},
        ]}),
    ])
    assert find_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) == "http://p/right.mp3"


def test_loose_query_rejects_other_artists():
    s = FakeSession([
        FakeResp({"data": []}),
        FakeResp({"data": [{"preview": "http://p/wrong.mp3", "artist": {"name": "Someone Else"}}]}),
    ])
    assert find_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) is None


def test_retries_on_deezer_quota_error():
    waits = []
    s = FakeSession([
        FakeResp({"error": {"code": 4, "message": "Quota limit exceeded"}}),
        FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "A"}}]}),
    ])
    assert find_preview("T", "A", session=s, sleep=waits.append) == "http://p/1.mp3"
    assert waits


def test_quotes_in_names_are_stripped():
    s = FakeSession([FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "A"}}]})])
    find_preview('Say "Hi"', "A", session=s, sleep=NO_SLEEP)
    assert s.calls[0]["params"]["q"] == 'artist:"A" track:"Say Hi"'
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_http_utils.py tests/test_preview.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.http_utils'`.

- [ ] **Step 4: Implement** — `src/http_utils.py`

```python
import time

import requests


def get_with_retry(url, *, params=None, session=None, retries=3, backoff=1.0, timeout=10, sleep=time.sleep):
    """GET with retries on 429/5xx/network errors. Returns the response, or None on failure."""
    http = session or requests
    for attempt in range(retries + 1):
        try:
            resp = http.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            resp = None

        if resp is not None and resp.ok:
            return resp
        if resp is not None and resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", backoff * 2 ** attempt))
        elif resp is not None and 400 <= resp.status_code < 500:
            return None
        else:
            wait = backoff * 2 ** attempt

        if attempt < retries:
            sleep(wait)
    return None
```

`src/audio/preview.py`:

```python
import time

from src.http_utils import get_with_retry

DEEZER_SEARCH = "https://api.deezer.com/search"
QUOTA_ERROR = 4  # Deezer returns HTTP 200 with this error code when rate limited


def _search(query, session, sleep):
    for attempt in range(3):
        resp = get_with_retry(DEEZER_SEARCH, params={"q": query}, session=session, sleep=sleep)
        if resp is None:
            return []
        data = resp.json() or {}
        error = data.get("error")
        if error and error.get("code") == QUOTA_ERROR:
            sleep(2.0 * (attempt + 1))
            continue
        return data.get("data", [])
    return []


def find_preview(track_name, artist_name, *, session=None, sleep=time.sleep):
    """Return a Deezer 30 s preview URL for the song, or None."""
    track = track_name.replace('"', "")
    artist = artist_name.replace('"', "")

    for item in _search(f'artist:"{artist}" track:"{track}"', session, sleep):
        if item.get("preview"):
            return item["preview"]

    # Loose search can return covers/other songs, so require the artist to match.
    wanted = artist.lower()
    for item in _search(f"{track} {artist}", session, sleep):
        found = (item.get("artist") or {}).get("name", "").lower()
        if item.get("preview") and found and (found in wanted or wanted in found):
            return item["preview"]
    return None
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest tests/test_http_utils.py tests/test_preview.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git rm src/deezer_utils.py
git add src/http_utils.py src/audio/preview.py tests/fakes.py tests/test_http_utils.py tests/test_preview.py
git commit -m "feat: Deezer preview lookup with retry and artist matching" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Audio loader (bug fix) + Librosa features

**Files:**
- Create: `src/audio/loader.py`, `src/audio/features.py`, `tests/test_loader.py`, `tests/test_features.py`
- Delete: `src/audio_utils.py`, `src/app.py`, `src/spotify_fetch_liked.py`, `src/create_playlist.py`, `src/test_with_spotify.py` (old CLI scripts; replaced by the web app)

**Interfaces:**
- Consumes: `get_with_retry` (Task 2), `SAMPLE_RATE`, `CLIP_SAMPLES` (Task 1).
- Produces: `decode_mp3(data: bytes) -> tuple[np.ndarray, int]`; `prepare(y: np.ndarray, sr: int) -> np.ndarray` (float32, len `CLIP_SAMPLES`); `download_audio(url: str, *, session=None) -> np.ndarray | None`.
- Produces: `mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=128, fixed_frames=640) -> np.ndarray` shape `(128, 640)`; `summary_vector(y, sr=SAMPLE_RATE) -> np.ndarray` shape `(72,)` float32, always finite; `SUMMARY_FEATURE_NAMES: list[str]` (len 72, includes `"tempo"`).

- [ ] **Step 1: Write failing tests** — `tests/test_loader.py`

```python
import numpy as np

from src.audio import loader
from src.config import CLIP_SAMPLES
from tests.fakes import FakeResp


def test_prepare_pads_short_audio():
    y = loader.prepare(np.ones(1000, dtype=np.float32), 22050)
    assert y.shape == (CLIP_SAMPLES,)
    assert y.dtype == np.float32


def test_prepare_resamples_before_trimming():
    # 40 s at 44.1 kHz must become exactly 30 s at 22.05 kHz (old bug kept only ~15 s of music)
    sr = 44100
    t = np.arange(sr * 40) / sr
    y = loader.prepare(np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    assert y.shape == (CLIP_SAMPLES,)
    assert np.abs(y[-1000:]).max() > 0.1  # end of clip is still music, not padding


def test_prepare_peak_normalizes():
    y = loader.prepare(np.full(CLIP_SAMPLES, 5000.0, dtype=np.float32), 22050)
    assert np.isclose(np.abs(y).max(), 1.0)


def test_prepare_silence_stays_finite():
    y = loader.prepare(np.zeros(CLIP_SAMPLES, dtype=np.float32), 22050)
    assert np.all(np.isfinite(y))


def test_prepare_empty_input():
    assert loader.prepare(np.zeros(0, dtype=np.float32), 44100).shape == (CLIP_SAMPLES,)


def test_download_audio_rejects_silent_preview(monkeypatch):
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"mp3"))
    monkeypatch.setattr(loader, "decode_mp3", lambda data: (np.zeros(44100, dtype=np.float32), 44100))
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_handles_decode_error(monkeypatch):
    def boom(data):
        raise ValueError("bad mp3")
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"x"))
    monkeypatch.setattr(loader, "decode_mp3", boom)
    assert loader.download_audio("http://p/1.mp3") is None


def test_download_audio_returns_prepared_clip(monkeypatch):
    monkeypatch.setattr(loader, "get_with_retry", lambda url, session=None: FakeResp(content=b"mp3"))
    monkeypatch.setattr(loader, "decode_mp3", lambda data: (np.random.rand(44100 * 30).astype(np.float32), 44100))
    assert loader.download_audio("http://p/1.mp3").shape == (CLIP_SAMPLES,)
```

`tests/test_features.py`:

```python
import numpy as np
import pytest

from src.audio.features import SUMMARY_FEATURE_NAMES, mel_spectrogram, summary_vector
from src.config import CLIP_SAMPLES, SAMPLE_RATE


@pytest.fixture(scope="module")
def clicks():
    # 120 BPM click track over a quiet tone
    y = 0.05 * np.sin(2 * np.pi * 220 * np.arange(CLIP_SAMPLES) / SAMPLE_RATE)
    for start in range(0, CLIP_SAMPLES, SAMPLE_RATE // 2):
        y[start:start + 500] += 0.9
    return y.astype(np.float32)


def test_mel_shape_matches_cnn_input(clicks):
    assert mel_spectrogram(clicks).shape == (128, 640)


def test_summary_vector_shape_and_names(clicks):
    vec = summary_vector(clicks)
    assert vec.shape == (len(SUMMARY_FEATURE_NAMES),) == (72,)
    assert np.all(np.isfinite(vec))
    assert len(set(SUMMARY_FEATURE_NAMES)) == 72


def test_tempo_feature_is_detected(clicks):
    tempo = summary_vector(clicks)[SUMMARY_FEATURE_NAMES.index("tempo")]
    assert 110 <= tempo <= 130 or 55 <= tempo <= 65 or 230 <= tempo <= 250


def test_silence_is_finite():
    silent = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    assert np.all(np.isfinite(summary_vector(silent)))
    assert np.all(np.isfinite(mel_spectrogram(silent)))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_loader.py tests/test_features.py -v`
Expected: FAIL — `ImportError` for `src.audio.loader` / `src.audio.features`.

- [ ] **Step 3: Implement** — `src/audio/loader.py`

```python
import io

import librosa
import numpy as np
from pydub import AudioSegment

from src.config import CLIP_SAMPLES, SAMPLE_RATE
from src.http_utils import get_with_retry


def decode_mp3(data):
    """Decode MP3 bytes (needs ffmpeg on PATH) to mono float32 samples + sample rate."""
    audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    return samples, audio.frame_rate


def prepare(y, sr):
    """Peak-normalize, resample to 22,050 Hz, then pad/trim to exactly 30 s."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)


def download_audio(url, *, session=None):
    """Download a preview MP3 and return a prepared 30 s clip, or None if unusable."""
    resp = get_with_retry(url, session=session)
    if resp is None:
        return None
    try:
        samples, sr = decode_mp3(resp.content)
    except Exception:
        return None
    if samples.size == 0 or not np.any(samples):
        return None
    return prepare(samples, sr)
```

`src/audio/features.py`:

```python
import librosa
import numpy as np

from src.config import SAMPLE_RATE

SUMMARY_FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(20)]
    + [f"mfcc_std_{i}" for i in range(20)]
    + [f"chroma_{i}" for i in range(12)]
    + ["centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean"]
    + [f"contrast_{i}" for i in range(7)]
    + ["zcr_mean", "zcr_std", "rms_mean", "rms_std", "onset_mean", "onset_std",
       "tempo", "pulse_clarity", "percussive_ratio"]
)


def mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=128, fixed_frames=640):
    """Mel-dB spectrogram identical to the CNN's training preprocessing."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < fixed_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, fixed_frames - mel_db.shape[1])), mode="constant")
    return mel_db[:, :fixed_frames].astype(np.float32)


def summary_vector(y, sr=SAMPLE_RATE):
    """Fixed-length audio descriptor used by the LightGBM models."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    pulse = float(onset_env[beats].mean() / (onset_env.mean() + 1e-8)) if len(beats) else 0.0
    _, percussive = librosa.effects.hpss(y)
    perc_ratio = float(np.sum(percussive ** 2) / (np.sum(y ** 2) + 1e-8))

    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1), chroma.mean(axis=1),
        [centroid.mean(), centroid.std(), bandwidth.mean(), rolloff.mean()],
        contrast.mean(axis=1),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std(), onset_env.mean(), onset_env.std(),
         tempo, pulse, perc_ratio],
    ]
    vec = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_loader.py tests/test_features.py -v`
Expected: 12 passed. If `test_tempo_feature_is_detected` fails, print the detected tempo and widen to the octave it reports — the point is "near 120 or an octave of it".

- [ ] **Step 5: Delete replaced scripts and commit**

```bash
git rm src/audio_utils.py src/app.py src/spotify_fetch_liked.py src/create_playlist.py src/test_with_spotify.py
git add src/audio/loader.py src/audio/features.py tests/test_loader.py tests/test_features.py
git commit -m "feat: audio loader (fix resample-before-trim bug) and Librosa feature extraction" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Mood regressors — dataset collection + training

**Files:**
- Create: `src/ml/mood_model.py`, `src/training/collect_mood_dataset.py`, `src/training/train_mood_models.py`, `tests/test_mood_model.py`, `tests/test_collect_mood_dataset.py`
- Output (git-ignored data, committed model): `data/features/mood_train.parquet`, `data/features/mood_failed.csv`, `models/mood_regressors.pkl`

**Interfaces:**
- Consumes: `find_preview`, `download_audio`, `summary_vector`, `SUMMARY_FEATURE_NAMES`, config paths.
- Produces: `TARGETS = ("danceability", "energy", "valence", "acousticness")`; `train_regressors(X: pd.DataFrame, y: pd.DataFrame, seed=42) -> dict` bundle with keys `models`, `feature_names`, `train_means`, `version`; `evaluate_regressors(bundle, X, y) -> dict[str, dict]` (`mae`, `r2`, `baseline_mae`); `predict_mood(bundle, vec: np.ndarray) -> dict[str, float]` clipped to [0, 1].
- Produces: `sample_tracks(df, per_genre, seed=42) -> pd.DataFrame`; `process_row(row: dict) -> dict` (always has `track_id`, `ok`).

- [ ] **Step 1: Write failing tests** — `tests/test_mood_model.py`

```python
import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.ml.mood_model import TARGETS, evaluate_regressors, predict_mood, train_regressors


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(SUMMARY_FEATURE_NAMES))), columns=SUMMARY_FEATURE_NAMES)
    y = pd.DataFrame({t: 1 / (1 + np.exp(-X.iloc[:, i])) for i, t in enumerate(TARGETS)})
    return X, y


def test_regressors_learn_signal_and_beat_baseline():
    X, y = make_data()
    bundle = train_regressors(X.iloc[:300], y.iloc[:300])
    metrics = evaluate_regressors(bundle, X.iloc[300:], y.iloc[300:])
    for t in TARGETS:
        assert metrics[t]["mae"] < metrics[t]["baseline_mae"]


def test_predict_mood_returns_clipped_values():
    X, y = make_data()
    bundle = train_regressors(X, y)
    out = predict_mood(bundle, X.iloc[0].to_numpy() * 100)
    assert set(out) == set(TARGETS)
    assert all(0.0 <= v <= 1.0 for v in out.values())
```

`tests/test_collect_mood_dataset.py`:

```python
import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.training import collect_mood_dataset as cmd


def test_sample_tracks_dedupes_and_caps_per_genre():
    df = pd.DataFrame({
        "track_id": ["a", "a", "b", "c", "d", "e"],
        "track_genre": ["pop", "rock", "pop", "pop", "rock", "rock"],
        "track_name": list("123456"), "artists": list("xyzuvw"),
    })
    out = cmd.sample_tracks(df, per_genre=2)
    assert out["track_id"].is_unique
    assert out.groupby("track_genre").size().max() <= 2


ROW = {"track_id": "t1", "track_name": "Song", "artists": "Main Artist;Feat", "track_genre": "pop",
       "danceability": 0.5, "energy": 0.6, "valence": 0.7, "acousticness": 0.1}


def test_process_row_uses_first_artist_and_returns_features(monkeypatch):
    seen = {}
    monkeypatch.setattr(cmd, "find_preview", lambda name, artist: seen.setdefault("artist", artist) and "http://p")
    monkeypatch.setattr(cmd, "download_audio", lambda url: np.zeros(10, dtype=np.float32))
    monkeypatch.setattr(cmd, "summary_vector", lambda y: np.arange(72, dtype=np.float32))
    out = cmd.process_row(ROW)
    assert seen["artist"] == "Main Artist"
    assert out["ok"] is True
    assert out[SUMMARY_FEATURE_NAMES[5]] == 5.0
    assert out["energy"] == 0.6


def test_process_row_marks_missing_preview(monkeypatch):
    monkeypatch.setattr(cmd, "find_preview", lambda name, artist: None)
    assert cmd.process_row(ROW) == {"track_id": "t1", "ok": False}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_mood_model.py tests/test_collect_mood_dataset.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement** — `src/ml/mood_model.py`

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.audio.features import SUMMARY_FEATURE_NAMES

TARGETS = ("danceability", "energy", "valence", "acousticness")
VERSION = "mood-v1"


def train_regressors(X, y, seed=42):
    models = {}
    for target in TARGETS:
        model = LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        )
        model.fit(X[SUMMARY_FEATURE_NAMES], y[target])
        models[target] = model
    return {
        "models": models,
        "feature_names": list(SUMMARY_FEATURE_NAMES),
        "train_means": {t: float(y[t].mean()) for t in TARGETS},
        "version": VERSION,
    }


def evaluate_regressors(bundle, X, y):
    metrics = {}
    for target, model in bundle["models"].items():
        pred = model.predict(X[bundle["feature_names"]])
        baseline = np.full(len(y), bundle["train_means"][target])
        metrics[target] = {
            "mae": float(mean_absolute_error(y[target], pred)),
            "r2": float(r2_score(y[target], pred)),
            "baseline_mae": float(mean_absolute_error(y[target], baseline)),
        }
    return metrics


def predict_mood(bundle, vec):
    X = pd.DataFrame([np.asarray(vec, dtype=np.float32)], columns=bundle["feature_names"])
    return {t: float(np.clip(m.predict(X)[0], 0.0, 1.0)) for t, m in bundle["models"].items()}
```

`src/training/collect_mood_dataset.py`:

```python
"""Build mood training data: Deezer previews for dataset.csv tracks -> Librosa features + Spotify labels.

Usage: python -m src.training.collect_mood_dataset --per-genre 45 [--workers 8]
Resumable: re-running skips tracks already saved or already failed.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES, summary_vector
from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.config import DATASET_CSV, FEATURES_DIR
from src.ml.mood_model import TARGETS

OUT_PATH = FEATURES_DIR / "mood_train.parquet"
FAILED_PATH = FEATURES_DIR / "mood_failed.csv"
FLUSH_EVERY = 100


def sample_tracks(df, per_genre, seed=42):
    df = df.dropna(subset=["track_name", "artists"]).drop_duplicates("track_id")
    return df.sample(frac=1, random_state=seed).groupby("track_genre").head(per_genre).reset_index(drop=True)


def process_row(row):
    artist = str(row["artists"]).split(";")[0]
    url = find_preview(str(row["track_name"]), artist)
    if not url:
        return {"track_id": row["track_id"], "ok": False}
    y = download_audio(url)
    if y is None:
        return {"track_id": row["track_id"], "ok": False}
    out = {"track_id": row["track_id"], "ok": True, "track_genre": row["track_genre"]}
    out.update(dict(zip(SUMMARY_FEATURE_NAMES, summary_vector(y).tolist())))
    out.update({t: float(row[t]) for t in TARGETS})
    return out


def _flush(rows, failed):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        new = pd.DataFrame(rows)
        if OUT_PATH.exists():
            new = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True)
        new.to_parquet(OUT_PATH, index=False)
    if failed:
        pd.DataFrame({"track_id": failed}).to_csv(FAILED_PATH, mode="a", header=not FAILED_PATH.exists(), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-genre", type=int, default=45)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    sampled = sample_tracks(pd.read_csv(DATASET_CSV), args.per_genre)
    done = set(pd.read_parquet(OUT_PATH)["track_id"]) if OUT_PATH.exists() else set()
    if FAILED_PATH.exists():
        done |= set(pd.read_csv(FAILED_PATH)["track_id"])
    pending = [r for r in sampled.to_dict("records") if r["track_id"] not in done]
    print(f"{len(sampled)} sampled, {len(done)} already processed, {len(pending)} to go")

    rows, failed, finished = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_row, r) for r in pending]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as exc:  # one bad track never stops the run
                print(f"worker error: {exc}")
                continue
            (rows if res["ok"] else failed).append(res if res["ok"] else res["track_id"])
            finished += 1
            if finished % FLUSH_EVERY == 0:
                _flush(rows, failed)
                rows, failed = [], []
                print(f"{finished}/{len(pending)} processed")
    _flush(rows, failed)
    total = len(pd.read_parquet(OUT_PATH)) if OUT_PATH.exists() else 0
    print(f"Done. {total} usable tracks in {OUT_PATH}")


if __name__ == "__main__":
    main()
```

`src/training/train_mood_models.py`:

```python
"""Train LightGBM mood regressors. Usage: python -m src.training.train_mood_models"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.config import MODELS_DIR
from src.ml.mood_model import TARGETS, evaluate_regressors, train_regressors
from src.training.collect_mood_dataset import OUT_PATH


def main():
    df = pd.read_parquet(OUT_PATH)
    print(f"Training on {len(df)} tracks")
    X, y = df[SUMMARY_FEATURE_NAMES], df[list(TARGETS)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bundle = train_regressors(X_train, y_train)
    metrics = evaluate_regressors(bundle, X_test, y_test)
    bundle["metrics"] = metrics

    print(f"\n{'target':<14}{'MAE':>8}{'baseline':>10}{'R2':>8}")
    for t, m in metrics.items():
        print(f"{t:<14}{m['mae']:>8.3f}{m['baseline_mae']:>10.3f}{m['r2']:>8.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "mood_regressors.pkl")
    print(f"\nSaved {MODELS_DIR / 'mood_regressors.pkl'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify tests pass**

Run: `python -m pytest tests/test_mood_model.py tests/test_collect_mood_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit code**

```bash
git add src/ml/mood_model.py src/training/collect_mood_dataset.py src/training/train_mood_models.py tests/test_mood_model.py tests/test_collect_mood_dataset.py
git commit -m "feat: mood dataset collection and LightGBM mood regressors" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Smoke-run collection, then full run**

Run: `python -m src.training.collect_mood_dataset --per-genre 1`
Expected: ~114 sampled; finishes in a few minutes; prints `Done. N usable tracks` with N > 50. If N is near 0, stop and debug preview lookup/ffmpeg before the full run.

Then (long, ~30–45 min, resumable — run in background): `python -m src.training.collect_mood_dataset --per-genre 45`
Expected: ~5,100 sampled, roughly 3,000–4,500 usable tracks.

- [ ] **Step 7: Train and record metrics**

Run: `python -m src.training.train_mood_models`
Expected: table where every target's MAE < baseline. Typical: energy/acousticness R² ≈ 0.5–0.7, danceability ≈ 0.3–0.5, valence ≈ 0.2–0.4. Paste the table into the commit message.

```bash
git add models/mood_regressors.pkl
git commit -m "model: train mood regressors" -m "<paste metrics table>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Genre models — LightGBM on GTZAN + CNN script + Colab notebook

**Files:**
- Create: `src/ml/genre_model.py`, `src/training/train_genre_lgbm.py`, `tests/test_genre_model.py`, `notebooks/train_cnn_colab.ipynb`
- Move + rewrite: `src/train_model_cnn.py` → `src/training/train_genre_cnn.py`
- Output: `data/features/gtzan_summary.parquet`, `models/genre_lgbm_audio.pkl`

**Interfaces:**
- Consumes: `prepare`, `mel_spectrogram`, `summary_vector`, `SUMMARY_FEATURE_NAMES`, `GTZAN_DIR`, `MODELS_DIR`.
- Produces: `train_genre_classifier(X: pd.DataFrame, y: pd.Series, seed=42) -> dict` bundle with keys `model` (fitted `LGBMClassifier`), `classes: list[str]`, `feature_names`.
- Produces files: `models/genre_lgbm_audio.pkl` (bundle above), `models/genre_cnn_model.keras` + `models/label_encoder_cnn.pkl` (existing format: sklearn `LabelEncoder`).

- [ ] **Step 1: Write failing test** — `tests/test_genre_model.py`

```python
import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.ml.genre_model import train_genre_classifier


def test_genre_classifier_learns_separable_classes():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, len(SUMMARY_FEATURE_NAMES))), columns=SUMMARY_FEATURE_NAMES)
    y = pd.Series(np.where(X.iloc[:, 0] > 0, "rock", "jazz"))
    bundle = train_genre_classifier(X, y)
    assert bundle["classes"] == ["jazz", "rock"]
    assert (bundle["model"].predict(X) == y).mean() > 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_genre_model.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** — `src/ml/genre_model.py`

```python
from lightgbm import LGBMClassifier

from src.audio.features import SUMMARY_FEATURE_NAMES


def train_genre_classifier(X, y, seed=42):
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )
    model.fit(X[SUMMARY_FEATURE_NAMES], y)
    return {"model": model, "classes": [str(c) for c in model.classes_], "feature_names": list(SUMMARY_FEATURE_NAMES)}
```

`src/training/train_genre_lgbm.py`:

```python
"""Train LightGBM genre classifier on GTZAN audio. Usage: python -m src.training.train_genre_lgbm"""
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import librosa
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.audio.features import SUMMARY_FEATURE_NAMES, summary_vector
from src.audio.loader import prepare
from src.config import CLIP_SECONDS, FEATURES_DIR, GTZAN_DIR, MODELS_DIR
from src.ml.genre_model import train_genre_classifier

CACHE = FEATURES_DIR / "gtzan_summary.parquet"


def _extract(path_and_genre):
    path, genre = path_and_genre
    try:
        y, sr = librosa.load(path, sr=None, duration=CLIP_SECONDS)
    except Exception as exc:  # GTZAN has a known corrupt file (jazz.00054.wav)
        print(f"skip {path}: {exc}")
        return None
    return {"genre": genre, **dict(zip(SUMMARY_FEATURE_NAMES, summary_vector(prepare(y, sr)).tolist()))}


def build_features():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    jobs = [(str(p), p.parent.name) for p in sorted(GTZAN_DIR.glob("*/*.wav"))]
    print(f"Extracting features from {len(jobs)} GTZAN clips...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = [r for r in pool.map(_extract, jobs, chunksize=8) if r]
    df = pd.DataFrame(rows)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def main():
    df = build_features()
    X, y = df[SUMMARY_FEATURE_NAMES], df["genre"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    bundle = train_genre_classifier(X_train, y_train)
    print(classification_report(y_test, bundle["model"].predict(X_test)))
    joblib.dump(bundle, MODELS_DIR / "genre_lgbm_audio.pkl")
    print(f"Saved {MODELS_DIR / 'genre_lgbm_audio.pkl'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify test passes**

Run: `python -m pytest tests/test_genre_model.py -v`
Expected: 1 passed.

- [ ] **Step 5: Rewrite CNN training script**

```bash
git mv src/train_model_cnn.py src/training/train_genre_cnn.py
```

Replace its contents with:

```python
"""Train the CNN genre classifier on GTZAN mel-spectrograms.

Usage: python -m src.training.train_genre_cnn
Tip: much faster on a GPU -- see notebooks/train_cnn_colab.ipynb.
"""
import joblib
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

from src.audio.features import mel_spectrogram
from src.config import CLIP_SECONDS, GTZAN_DIR, MODELS_DIR, SAMPLE_RATE


def load_dataset():
    X, y = [], []
    for path in sorted(GTZAN_DIR.glob("*/*.wav")):
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=CLIP_SECONDS)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        X.append(mel_spectrogram(audio))
        y.append(path.parent.name)
    return np.array(X)[..., np.newaxis], np.array(y)


def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y = load_dataset()
    print(f"Loaded {len(X)} clips")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Genres:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
    model = build_cnn_model(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.3f}")
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "genre_cnn_model.keras")
    joblib.dump(le, MODELS_DIR / "label_encoder_cnn.pkl")
    print("Saved models/genre_cnn_model.keras and models/label_encoder_cnn.pkl")


if __name__ == "__main__":
    main()
```

Verify import only (do not train locally unless you want to): `python -c "import src.training.train_genre_cnn"` → no error.

- [ ] **Step 6: Create the Colab notebook** — `notebooks/train_cnn_colab.ipynb`

Write this JSON exactly:

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Train the genre CNN on a free GPU\n", "1. Runtime → Change runtime type → **T4 GPU**.\n", "2. Zip your local `data/genres/` folder as `genres.zip` and upload it when prompted below.\n", "3. Run all cells, then download the two model files at the end and put them in your repo's `models/` folder."]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["!git clone https://github.com/Edwin-Trejo/Automated-playlist-creator.git repo\n", "%cd repo\n", "!pip -q install librosa joblib"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["from google.colab import files\n", "uploaded = files.upload()  # choose genres.zip\n", "!mkdir -p data && unzip -q -o genres.zip -d data/\n", "!ls data/genres"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["import tensorflow as tf\n", "print('GPU:', tf.config.list_physical_devices('GPU'))\n", "!python -m src.training.train_genre_cnn"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["files.download('models/genre_cnn_model.keras')\n", "files.download('models/label_encoder_cnn.pkl')"]}
 ]
}
```

(The clone cell only works after this branch is pushed to GitHub; until then, upload the repo as a zip instead.)

- [ ] **Step 7: Train the LightGBM genre model**

Run: `python -m src.training.train_genre_lgbm`
Expected: extracts ~999 clips (a few minutes), prints a classification report with overall accuracy roughly 0.65–0.80, saves `models/genre_lgbm_audio.pkl`.

- [ ] **Step 8: Commit**

```bash
git add src/ml/genre_model.py src/training/train_genre_lgbm.py src/training/train_genre_cnn.py tests/test_genre_model.py notebooks/train_cnn_colab.ipynb models/genre_lgbm_audio.pkl
git commit -m "feat: LightGBM genre model on GTZAN audio features, CNN training script and Colab notebook" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Analyzer (serves all models)

**Files:**
- Create: `src/ml/analyzer.py`, `tests/test_analyzer.py`

**Interfaces:**
- Consumes: `mel_spectrogram`, `summary_vector`, `SUMMARY_FEATURE_NAMES`, `predict_mood`, model files from Tasks 4–5.
- Produces: `MODEL_VERSION = "v1"`; `class ModelsMissingError(RuntimeError)`; `Analyzer(cnn, cnn_classes: list[str], genre_lgbm: dict | None, mood_bundle: dict)`; `Analyzer.load(models_dir=MODELS_DIR) -> Analyzer`; `Analyzer.genre_probs(y, vec) -> dict[str, float]`; `Analyzer.analyze_audio(y: np.ndarray) -> dict` with keys `genre, genre_conf, danceability, energy, valence, acousticness, tempo`.

- [ ] **Step 1: Write failing tests** — `tests/test_analyzer.py`

```python
import numpy as np
import pytest

from src.config import CLIP_SAMPLES
from src.ml import analyzer as an


class FakeCNN:
    def predict(self, x, verbose=0):
        assert x.shape == (1, 128, 640, 1)
        return np.array([[0.2, 0.8]])  # classes: jazz, rock


class FakeLGBM:
    classes_ = np.array(["rock", "jazz"])  # different order on purpose

    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])  # rock 0.4, jazz 0.6


class FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value])


MOOD = {
    "models": {"danceability": FakeRegressor(0.7), "energy": FakeRegressor(1.4),
               "valence": FakeRegressor(0.3), "acousticness": FakeRegressor(-0.2)},
    "feature_names": list(an.SUMMARY_FEATURE_NAMES),
}


@pytest.fixture
def clip():
    return (np.random.default_rng(0).random(CLIP_SAMPLES).astype(np.float32) - 0.5)


def test_ensemble_averages_aligned_probabilities(clip):
    a = an.Analyzer(FakeCNN(), ["jazz", "rock"], {"model": FakeLGBM(), "classes": ["rock", "jazz"],
                                                  "feature_names": list(an.SUMMARY_FEATURE_NAMES)}, MOOD)
    out = a.analyze_audio(clip)
    # rock = (0.8 + 0.4) / 2 = 0.6 ; jazz = (0.2 + 0.6) / 2 = 0.4
    assert out["genre"] == "rock"
    assert out["genre_conf"] == pytest.approx(0.6)


def test_cnn_only_when_lgbm_missing(clip):
    out = an.Analyzer(FakeCNN(), ["jazz", "rock"], None, MOOD).analyze_audio(clip)
    assert out["genre"] == "rock" and out["genre_conf"] == pytest.approx(0.8)


def test_mood_values_clipped_and_tempo_present(clip):
    out = an.Analyzer(FakeCNN(), ["jazz", "rock"], None, MOOD).analyze_audio(clip)
    assert out["energy"] == 1.0 and out["acousticness"] == 0.0 and out["danceability"] == pytest.approx(0.7)
    assert out["tempo"] >= 0


def test_load_reports_missing_models(tmp_path):
    with pytest.raises(an.ModelsMissingError, match="train_mood_models"):
        an.Analyzer.load(tmp_path)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_analyzer.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** — `src/ml/analyzer.py`

```python
import joblib
import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES, mel_spectrogram, summary_vector
from src.config import MODELS_DIR
from src.ml.mood_model import predict_mood

MODEL_VERSION = "v1"
TEMPO_INDEX = SUMMARY_FEATURE_NAMES.index("tempo")


class ModelsMissingError(RuntimeError):
    pass


class Analyzer:
    def __init__(self, cnn, cnn_classes, genre_lgbm, mood_bundle):
        self.cnn = cnn
        self.cnn_classes = [str(c) for c in cnn_classes]
        self.genre_lgbm = genre_lgbm
        self.mood_bundle = mood_bundle

    @classmethod
    def load(cls, models_dir=MODELS_DIR):
        required = {
            "mood_regressors.pkl": "python -m src.training.collect_mood_dataset && python -m src.training.train_mood_models",
            "genre_cnn_model.keras": "python -m src.training.train_genre_cnn (or notebooks/train_cnn_colab.ipynb)",
            "label_encoder_cnn.pkl": "python -m src.training.train_genre_cnn",
        }
        missing = [f"models/{name} -> run: {cmd}" for name, cmd in required.items() if not (models_dir / name).exists()]
        if missing:
            raise ModelsMissingError("Missing trained models:\n" + "\n".join(missing))

        from tensorflow.keras.models import load_model  # slow import; only when actually serving

        lgbm_path = models_dir / "genre_lgbm_audio.pkl"
        return cls(
            cnn=load_model(models_dir / "genre_cnn_model.keras"),
            cnn_classes=list(joblib.load(models_dir / "label_encoder_cnn.pkl").classes_),
            genre_lgbm=joblib.load(lgbm_path) if lgbm_path.exists() else None,
            mood_bundle=joblib.load(models_dir / "mood_regressors.pkl"),
        )

    def genre_probs(self, y, vec):
        mel = mel_spectrogram(y)[np.newaxis, ..., np.newaxis]
        cnn = self.cnn.predict(mel, verbose=0)[0]
        probs = {c: float(p) for c, p in zip(self.cnn_classes, cnn)}
        if self.genre_lgbm is not None:
            X = pd.DataFrame([vec], columns=self.genre_lgbm["feature_names"])
            lgbm = self.genre_lgbm["model"].predict_proba(X)[0]
            lgbm_probs = {str(c): float(p) for c, p in zip(self.genre_lgbm["model"].classes_, lgbm)}
            probs = {c: (p + lgbm_probs.get(c, 0.0)) / 2 for c, p in probs.items()}
        return probs

    def analyze_audio(self, y):
        vec = summary_vector(y)
        probs = self.genre_probs(y, vec)
        genre = max(probs, key=probs.get)
        return {
            "genre": genre,
            "genre_conf": round(probs[genre], 4),
            **predict_mood(self.mood_bundle, vec),
            "tempo": round(float(vec[TEMPO_INDEX]), 1),
        }
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_analyzer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Real-model smoke check**

Run: `python -c "from src.ml.analyzer import Analyzer; import numpy as np; a=Analyzer.load(); print(a.analyze_audio((np.random.rand(661500).astype('float32')-0.5)))"`
Expected: a dict with a GTZAN genre, values in [0, 1], a tempo number.

- [ ] **Step 6: Commit**

```bash
git add src/ml/analyzer.py tests/test_analyzer.py
git commit -m "feat: Analyzer serving CNN+LightGBM genre ensemble and mood regressors" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: SQLite track store

**Files:**
- Create: `src/store.py`, `tests/test_store.py`

**Interfaces:**
- Produces: `TrackStore(path: str | Path)` with `upsert(row: dict) -> None`, `get(spotify_id) -> dict | None`, `get_many(ids: list[str]) -> list[dict]` (input order, unknown skipped), `ids_with_version(version: str) -> set[str]`, `analyzed_tracks() -> list[dict]` (preview_found = 1), `counts() -> dict` (`total`, `analyzed`, `no_preview`). Row keys = `COLUMNS`.

- [ ] **Step 1: Write failing tests** — `tests/test_store.py`

```python
import pytest

from src.store import TrackStore


@pytest.fixture
def store(tmp_path):
    return TrackStore(tmp_path / "t.db")


def row(sid, **kw):
    base = {"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": f"Song {sid}", "artists": "A",
            "preview_found": 1, "genre": "pop", "genre_conf": 0.9, "danceability": 0.5, "energy": 0.5,
            "valence": 0.5, "acousticness": 0.5, "tempo": 120.0, "model_version": "v1", "analyzed_at": "now"}
    base.update(kw)
    return base


def test_upsert_and_get(store):
    store.upsert(row("a"))
    assert store.get("a")["genre"] == "pop"
    store.upsert(row("a", genre="rock"))
    assert store.get("a")["genre"] == "rock"
    assert store.get("missing") is None


def test_partial_row_fills_nulls(store):
    store.upsert({"spotify_id": "x", "uri": "u", "name": "n", "artists": "a", "preview_found": 0, "model_version": "v1"})
    assert store.get("x")["energy"] is None


def test_versions_analyzed_and_counts(store):
    store.upsert(row("a"))
    store.upsert(row("b", preview_found=0, genre=None))
    store.upsert(row("c", model_version="v0"))
    assert store.ids_with_version("v1") == {"a", "b"}
    assert [t["spotify_id"] for t in store.analyzed_tracks()] == ["a", "c"]
    assert store.counts() == {"total": 3, "analyzed": 2, "no_preview": 1}


def test_get_many_preserves_order(store):
    for sid in "abc":
        store.upsert(row(sid))
    assert [t["spotify_id"] for t in store.get_many(["c", "zz", "a"])] == ["c", "a"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** — `src/store.py`

```python
import sqlite3
from pathlib import Path

COLUMNS = ("spotify_id", "uri", "name", "artists", "preview_found", "genre", "genre_conf",
           "danceability", "energy", "valence", "acousticness", "tempo", "model_version", "analyzed_at")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    spotify_id TEXT PRIMARY KEY,
    uri TEXT NOT NULL,
    name TEXT NOT NULL,
    artists TEXT NOT NULL,
    preview_found INTEGER NOT NULL DEFAULT 0,
    genre TEXT, genre_conf REAL,
    danceability REAL, energy REAL, valence REAL, acousticness REAL, tempo REAL,
    model_version TEXT, analyzed_at TEXT
)
"""


class TrackStore:
    """SQLite cache of analyzed liked songs. Opens a connection per call so it is safe across threads."""

    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, row):
        values = [row.get(c) for c in COLUMNS]
        updates = ", ".join(f"{c}=excluded.{c}" for c in COLUMNS[1:])
        sql = (f"INSERT INTO tracks ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))}) "
               f"ON CONFLICT(spotify_id) DO UPDATE SET {updates}")
        with self._connect() as conn:
            conn.execute(sql, values)

    def get(self, spotify_id):
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM tracks WHERE spotify_id = ?", (spotify_id,)).fetchone()
        return dict(r) if r else None

    def get_many(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM tracks WHERE spotify_id IN ({', '.join('?' * len(ids))})", ids).fetchall()
        by_id = {r["spotify_id"]: dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def ids_with_version(self, version):
        with self._connect() as conn:
            rows = conn.execute("SELECT spotify_id FROM tracks WHERE model_version = ?", (version,)).fetchall()
        return {r["spotify_id"] for r in rows}

    def analyzed_tracks(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tracks WHERE preview_found = 1 ORDER BY spotify_id").fetchall()
        return [dict(r) for r in rows]

    def counts(self):
        with self._connect() as conn:
            total, analyzed = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(preview_found), 0) FROM tracks").fetchone()
        return {"total": total, "analyzed": analyzed, "no_preview": total - analyzed}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_store.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/store.py tests/test_store.py
git commit -m "feat: SQLite track store" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Library sync

**Files:**
- Create: `src/library_sync.py`, `tests/test_library_sync.py`

**Interfaces:**
- Consumes: `TrackStore` (Task 7), `Analyzer.analyze_audio` + `MODEL_VERSION` (Task 6), `find_preview` (Task 2), `download_audio` (Task 3).
- Produces: `iter_liked_tracks(sp, page_size=50) -> Iterator[dict]` (keys `spotify_id, uri, name, artists`); `SyncProgress` dataclass (`running, total, done, analyzed, no_preview, failed, error`, `to_dict()`); `LibrarySync(store, analyzer, find_preview=find_preview, download=download_audio)` with `analyze_one(track) -> str` (`"analyzed" | "no_preview" | "failed"`) and `run(tracks, progress) -> None`; `start_background_sync(tracks, syncer, progress) -> threading.Thread`.

- [ ] **Step 1: Write failing tests** — `tests/test_library_sync.py`

```python
import numpy as np

from src.library_sync import LibrarySync, SyncProgress, iter_liked_tracks, start_background_sync
from src.ml.analyzer import MODEL_VERSION
from src.store import TrackStore

ANALYSIS = {"genre": "pop", "genre_conf": 0.9, "danceability": 0.6, "energy": 0.7,
            "valence": 0.8, "acousticness": 0.1, "tempo": 120.0}


class FakeAnalyzer:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)

    def analyze_audio(self, y):
        if y[0] in self.fail_for:
            raise RuntimeError("model blew up")
        return dict(ANALYSIS)


def track(sid, name=None):
    return {"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": name or sid, "artists": "Main, Feat"}


class FakeSpotify:
    def __init__(self, pages):
        self.pages = pages
        self.offsets = []

    def current_user_saved_tracks(self, limit, offset):
        self.offsets.append(offset)
        return self.pages[len(self.offsets) - 1]


def item(sid, local=False):
    return {"track": {"id": sid, "uri": f"spotify:track:{sid}", "name": sid, "is_local": local,
                      "artists": [{"name": "A"}, {"name": "B"}]}}


def test_iter_liked_tracks_paginates_and_skips_local_and_null():
    sp = FakeSpotify([
        {"items": [item("a"), {"track": None}, item(None, local=True)], "next": "more"},
        {"items": [item("b")], "next": None},
    ])
    tracks = list(iter_liked_tracks(sp, page_size=3))
    assert [t["spotify_id"] for t in tracks] == ["a", "b"]
    assert tracks[0]["artists"] == "A, B"
    assert sp.offsets == [0, 3]


def make_sync(tmp_path, previews, audio, analyzer=None):
    store = TrackStore(tmp_path / "t.db")
    seen_artists = []

    def fake_find(name, artist):
        seen_artists.append(artist)
        return previews.get(name)

    sync = LibrarySync(store, analyzer or FakeAnalyzer(), find_preview=fake_find, download=lambda url: audio.get(url))
    return sync, store, seen_artists


def test_run_records_outcomes(tmp_path):
    previews = {"ok": "u-ok", "bad-audio": "u-bad"}
    audio = {"u-ok": np.array([1.0], dtype=np.float32)}  # u-bad -> None
    sync, store, artists = make_sync(tmp_path, previews, audio)
    progress = SyncProgress()
    sync.run([track("ok"), track("nopreview"), track("bad-audio")], progress)

    assert progress.to_dict()["analyzed"] == 1
    assert (progress.done, progress.analyzed, progress.no_preview, progress.failed) == (3, 1, 1, 1)
    assert progress.running is False
    assert store.get("ok")["energy"] == 0.7 and store.get("ok")["model_version"] == MODEL_VERSION
    assert store.get("nopreview")["preview_found"] == 0
    assert store.get("bad-audio") is None  # failed downloads are retried next sync
    assert artists[0] == "Main"


def test_run_skips_already_analyzed(tmp_path):
    sync, store, _ = make_sync(tmp_path, {"ok": "u"}, {"u": np.array([1.0], dtype=np.float32)})
    sync.run([track("ok")], SyncProgress())
    progress = SyncProgress()
    sync.run([track("ok")], progress)
    assert progress.total == 0


def test_analyzer_exception_does_not_abort(tmp_path):
    previews = {"boom": "u1", "ok": "u2"}
    audio = {"u1": np.array([9.0], dtype=np.float32), "u2": np.array([1.0], dtype=np.float32)}
    sync, store, _ = make_sync(tmp_path, previews, audio, FakeAnalyzer(fail_for=[9.0]))
    progress = SyncProgress()
    sync.run([track("boom"), track("ok")], progress)
    assert progress.failed == 1 and progress.analyzed == 1


def test_background_sync_runs_and_finishes(tmp_path):
    sync, store, _ = make_sync(tmp_path, {"ok": "u"}, {"u": np.array([1.0], dtype=np.float32)})
    progress = SyncProgress()
    thread = start_background_sync([track("ok")], sync, progress)
    assert progress.running is True or progress.done == 1
    thread.join(timeout=5)
    assert progress.running is False and progress.analyzed == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_library_sync.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** — `src/library_sync.py`

```python
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.ml.analyzer import MODEL_VERSION


def iter_liked_tracks(sp, page_size=50):
    """Yield the user's liked songs, skipping local files and unavailable tracks."""
    offset = 0
    while True:
        page = sp.current_user_saved_tracks(limit=page_size, offset=offset)
        for entry in page.get("items", []):
            t = entry.get("track") or {}
            if not t.get("id") or t.get("is_local"):
                continue
            yield {
                "spotify_id": t["id"],
                "uri": t["uri"],
                "name": t["name"],
                "artists": ", ".join(a["name"] for a in t.get("artists", [])),
            }
        if not page.get("next"):
            return
        offset += page_size


@dataclass
class SyncProgress:
    running: bool = False
    total: int = 0
    done: int = 0
    analyzed: int = 0
    no_preview: int = 0
    failed: int = 0
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class LibrarySync:
    def __init__(self, store, analyzer, find_preview=find_preview, download=download_audio):
        self.store = store
        self.analyzer = analyzer
        self.find_preview = find_preview
        self.download = download

    def analyze_one(self, track):
        main_artist = track["artists"].split(", ")[0]
        url = self.find_preview(track["name"], main_artist)
        if not url:
            self.store.upsert({**track, "preview_found": 0, "model_version": MODEL_VERSION})
            return "no_preview"
        y = self.download(url)
        if y is None:
            return "failed"  # not stored, so the next sync retries it
        analysis = self.analyzer.analyze_audio(y)
        self.store.upsert({
            **track, **analysis, "preview_found": 1, "model_version": MODEL_VERSION,
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return "analyzed"

    def run(self, tracks, progress):
        known = self.store.ids_with_version(MODEL_VERSION)
        todo = [t for t in tracks if t["spotify_id"] not in known]
        progress.total, progress.running = len(todo), True
        try:
            for t in todo:
                try:
                    outcome = self.analyze_one(t)
                except Exception:
                    traceback.print_exc()
                    outcome = "failed"
                setattr(progress, outcome, getattr(progress, outcome) + 1)
                progress.done += 1
        finally:
            progress.running = False


def start_background_sync(tracks, syncer, progress):
    progress.running = True  # set before the thread starts so concurrent requests see it

    def target():
        try:
            syncer.run(tracks, progress)
        except Exception as exc:
            progress.error = str(exc)
            progress.running = False

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_library_sync.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/library_sync.py tests/test_library_sync.py
git commit -m "feat: incremental liked-songs sync with background progress" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Mood presets + selector

**Files:**
- Create: `src/moods.py`, `src/selector.py`, `tests/test_moods.py`, `tests/test_selector.py`

**Interfaces:**
- Produces: `FULL_RANGE: dict[str, tuple[float, float]]` (`danceability, energy, valence` → `(0.0, 1.0)`, `tempo` → `(40.0, 220.0)`); `MOODS: dict[str, dict[str, tuple]]` keys `happy, sad, dance, chill, energetic`; `build_targets(mood: str | None = None, overrides: dict | None = None) -> dict[str, tuple[float, float]]` (raises `ValueError` on unknown mood/feature).
- Produces: `select_tracks(tracks: list[dict], targets: dict, genres: list[str] | None = None, size: int = 25) -> list[dict]` — each returned dict is the track plus `score` (0–1, higher = closer to range centers).

- [ ] **Step 1: Write failing tests** — `tests/test_moods.py`

```python
import pytest

from src.moods import FULL_RANGE, MOODS, build_targets


def test_presets_exist():
    assert set(MOODS) == {"happy", "sad", "dance", "chill", "energetic"}


def test_no_mood_is_full_range():
    assert build_targets() == FULL_RANGE


def test_mood_fills_unspecified_features_with_full_range():
    t = build_targets("energetic")
    assert t["energy"] == (0.7, 1.0)
    assert t["valence"] == FULL_RANGE["valence"]


def test_overrides_win_are_swapped_and_clamped():
    t = build_targets("happy", {"valence": [0.9, 0.2], "tempo": [0, 500]})
    assert t["valence"] == (0.2, 0.9)
    assert t["tempo"] == (40.0, 220.0)


def test_unknown_mood_or_feature_raises():
    with pytest.raises(ValueError):
        build_targets("angry")
    with pytest.raises(ValueError):
        build_targets(None, {"loudness": [0, 1]})
```

`tests/test_selector.py`:

```python
from src.moods import build_targets
from src.selector import select_tracks


def t(sid, **kw):
    base = {"spotify_id": sid, "name": sid, "genre": "pop", "genre_conf": 0.5,
            "danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0}
    base.update(kw)
    return base


def test_filters_by_ranges_and_ranks_by_closeness():
    tracks = [t("edge", energy=0.71), t("center", energy=0.85), t("low", energy=0.3)]
    out = select_tracks(tracks, build_targets(None, {"energy": [0.7, 1.0]}))
    assert [x["spotify_id"] for x in out] == ["center", "edge"]
    assert out[0]["score"] > out[1]["score"]


def test_genre_filter_is_case_insensitive():
    tracks = [t("a", genre="rock"), t("b", genre="pop")]
    assert [x["spotify_id"] for x in select_tracks(tracks, build_targets(), genres=["Rock"])] == ["a"]


def test_tracks_missing_a_constrained_feature_are_excluded():
    tracks = [t("a", valence=None), t("b")]
    assert [x["spotify_id"] for x in select_tracks(tracks, build_targets(None, {"valence": [0.4, 0.6]}))] == ["b"]


def test_half_time_tempo_counts_as_double():
    tracks = [t("half", tempo=62.0), t("slow", tempo=90.0)]
    out = select_tracks(tracks, build_targets(None, {"tempo": [110, 135]}))
    assert [x["spotify_id"] for x in out] == ["half"]


def test_size_limits_results():
    tracks = [t(str(i)) for i in range(10)]
    assert len(select_tracks(tracks, build_targets(), size=3)) == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_moods.py tests/test_selector.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement** — `src/moods.py`

```python
FULL_RANGE = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (40.0, 220.0),
}

MOODS = {
    "happy": {"danceability": (0.5, 1.0), "energy": (0.5, 1.0), "valence": (0.6, 1.0)},
    "sad": {"danceability": (0.0, 0.5), "energy": (0.0, 0.5), "valence": (0.0, 0.35), "tempo": (40.0, 110.0)},
    "dance": {"danceability": (0.65, 1.0), "energy": (0.6, 1.0), "tempo": (110.0, 135.0)},
    "chill": {"danceability": (0.0, 0.6), "energy": (0.0, 0.45), "valence": (0.3, 0.7), "tempo": (40.0, 115.0)},
    "energetic": {"energy": (0.7, 1.0), "tempo": (120.0, 220.0)},
}


def build_targets(mood=None, overrides=None):
    """Start from full ranges, apply the mood preset, then slider overrides."""
    if mood is not None and mood not in MOODS:
        raise ValueError(f"Unknown mood: {mood}")
    targets = dict(FULL_RANGE)
    targets.update(MOODS.get(mood, {}))
    for feature, (lo, hi) in (overrides or {}).items():
        if feature not in FULL_RANGE:
            raise ValueError(f"Unknown feature: {feature}")
        lo, hi = sorted((float(lo), float(hi)))
        full_lo, full_hi = FULL_RANGE[feature]
        targets[feature] = (max(lo, full_lo), min(hi, full_hi))
    return targets
```

`src/selector.py`:

```python
from src.moods import FULL_RANGE


def _tempo_candidates(tempo):
    # Beat trackers often report half or double time; accept the likely octave too.
    candidates = [tempo]
    if tempo < 75:
        candidates.append(tempo * 2)
    if tempo > 190:
        candidates.append(tempo / 2)
    return candidates


def _distance(value, lo, hi):
    """0 at the range center, 1 at its edges."""
    half = max((hi - lo) / 2, 1e-9)
    return min(abs(value - (lo + hi) / 2) / half, 1.0)


def select_tracks(tracks, targets, genres=None, size=25):
    active = {f: r for f, r in targets.items() if tuple(r) != FULL_RANGE[f]}
    wanted = {g.lower() for g in genres} if genres else None
    scored = []
    for track in tracks:
        if wanted and (track.get("genre") or "").lower() not in wanted:
            continue
        distances = []
        for feature, (lo, hi) in active.items():
            value = track.get(feature)
            if value is None:
                break
            values = _tempo_candidates(value) if feature == "tempo" else [value]
            inside = [v for v in values if lo <= v <= hi]
            if not inside:
                break
            distances.append(min(_distance(v, lo, hi) for v in inside))
        else:
            score = 1 - sum(distances) / len(distances) if distances else 1.0
            scored.append({**track, "score": round(score, 4)})
    scored.sort(key=lambda x: (-x["score"], -(x.get("genre_conf") or 0), x["name"]))
    return scored[:size]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_moods.py tests/test_selector.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moods.py src/selector.py tests/test_moods.py tests/test_selector.py
git commit -m "feat: mood presets and playlist track selector" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Ollama naming with template fallback

**Files:**
- Create: `src/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Produces: `generate_playlist_text(mood: str | None, genres: list[str], artists: list[str], *, base_url="http://localhost:11434", model="llama3.2:3b", timeout=60, post=None, today=None) -> dict` with keys `name`, `description`, `source` (`"ollama"` | `"template"`); `parse_llm_json(text: str) -> dict | None`; `template_text(mood, genres, today: date) -> dict`.

- [ ] **Step 1: Write failing tests** — `tests/test_llm.py`

```python
from datetime import date

import requests

from src.llm import generate_playlist_text, parse_llm_json, template_text
from tests.fakes import FakeResp

TODAY = date(2026, 10, 13)


def test_parse_plain_json():
    assert parse_llm_json('{"name": "Sunny Days", "description": "Bright tunes."}') == {
        "name": "Sunny Days", "description": "Bright tunes."}


def test_parse_json_wrapped_in_prose_and_cleans_whitespace():
    text = 'Sure! Here you go:\n{"name": "Neon\\nNights", "description": "x"}\nEnjoy'
    assert parse_llm_json(text)["name"] == "Neon Nights"


def test_parse_truncates_long_fields():
    out = parse_llm_json('{"name": "%s", "description": "%s"}' % ("n" * 300, "d" * 900))
    assert len(out["name"]) == 100 and len(out["description"]) == 300


def test_parse_rejects_garbage_and_empty_name():
    assert parse_llm_json("no json here") is None
    assert parse_llm_json('{"name": "  ", "description": "x"}') is None


def test_template_text():
    out = template_text("happy", ["pop", "disco"], TODAY)
    assert out["name"] == "Happy Mix · Oct 2026"
    assert "pop" in out["description"]


def test_uses_ollama_response():
    calls = []

    def post(url, json, timeout):
        calls.append((url, json))
        return FakeResp({"response": '{"name": "Gym Fuel", "description": "Loud and fast."}'})

    out = generate_playlist_text("energetic", ["metal"], ["A"], post=post, today=TODAY)
    assert out == {"name": "Gym Fuel", "description": "Loud and fast.", "source": "ollama"}
    assert calls[0][0] == "http://localhost:11434/api/generate"
    assert calls[0][1]["format"] == "json" and calls[0][1]["stream"] is False


def test_falls_back_when_ollama_down():
    def post(url, json, timeout):
        raise requests.ConnectionError("refused")

    out = generate_playlist_text("sad", [], [], post=post, today=TODAY)
    assert out["source"] == "template" and out["name"] == "Sad Mix · Oct 2026"


def test_falls_back_on_bad_model_output():
    out = generate_playlist_text(None, [], [], post=lambda url, json, timeout: FakeResp({"response": "oops"}), today=TODAY)
    assert out["source"] == "template" and out["name"] == "Custom Mix · Oct 2026"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** — `src/llm.py`

```python
import json
import re
from datetime import date

import requests

PROMPT = """You name Spotify playlists.
Mood: {mood}
Top genres: {genres}
Featured artists: {artists}
Reply with JSON only, exactly this shape:
{{"name": "<catchy playlist name, at most 6 words>", "description": "<one or two sentences, under 250 characters>"}}"""


def _clean(text, limit):
    return " ".join(str(text).split())[:limit]


def parse_llm_json(text):
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    name = _clean(data.get("name", ""), 100)
    if not name:
        return None
    return {"name": name, "description": _clean(data.get("description", ""), 300)}


def template_text(mood, genres, today):
    label = (mood or "custom").title()
    featuring = f", featuring {', '.join(genres[:3])}" if genres else ""
    return {
        "name": f"{label} Mix · {today:%b %Y}",
        "description": f"{label} picks from your Liked Songs{featuring}. Made by Automated Playlist Creator.",
    }


def generate_playlist_text(mood, genres, artists, *, base_url="http://localhost:11434",
                           model="llama3.2:3b", timeout=60, post=None, today=None):
    post = post or requests.post
    today = today or date.today()
    prompt = PROMPT.format(mood=mood or "custom mix", genres=", ".join(genres) or "mixed",
                           artists=", ".join(artists) or "various")
    try:
        resp = post(f"{base_url}/api/generate", json={
            "model": model, "prompt": prompt, "format": "json", "stream": False,
            "options": {"temperature": 0.8},
        }, timeout=timeout)
        resp.raise_for_status()
        parsed = parse_llm_json(resp.json().get("response", ""))
    except (requests.RequestException, ValueError, AttributeError):
        parsed = None
    if parsed:
        return {**parsed, "source": "ollama"}
    return {**template_text(mood, genres, today), "source": "template"}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 8 passed.

- [ ] **Step 5: Live check (optional, needs Ollama)**

Install Ollama from https://ollama.com/download, then `ollama pull llama3.2:3b`.
Run: `python -c "from src.llm import generate_playlist_text as g; print(g('dance', ['disco','pop'], ['Daft Punk']))"`
Expected: `source: 'ollama'` and a sensible name.

- [ ] **Step 6: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: Ollama playlist naming with template fallback" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Spotify client (OAuth + playlist creation)

**Files:**
- Create: `src/spotify_client.py`, `tests/test_spotify_client.py`

**Interfaces:**
- Produces: `SCOPES: str`; `make_oauth(cache_handler) -> SpotifyOAuth`; `create_playlist(sp, name: str, description: str, uris: list[str], public: bool = False) -> dict` with `id`, `url`.

- [ ] **Step 1: Write failing tests** — `tests/test_spotify_client.py`

```python
from src.spotify_client import SCOPES, create_playlist, make_oauth


class FakeSp:
    def __init__(self):
        self.posts = []

    def _post(self, url, payload=None):
        self.posts.append((url, payload))
        if url == "me/playlists":
            return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}
        return {"snapshot_id": "s"}


def test_create_playlist_uses_new_endpoints_and_chunks():
    sp = FakeSp()
    uris = [f"spotify:track:{i}" for i in range(250)]
    out = create_playlist(sp, "Name", "Desc", uris)
    assert out == {"id": "pl1", "url": "https://open.spotify.com/playlist/pl1"}
    assert sp.posts[0] == ("me/playlists", {"name": "Name", "description": "Desc", "public": False})
    item_posts = sp.posts[1:]
    assert [p[0] for p in item_posts] == ["playlists/pl1/items"] * 3
    assert [len(p[1]["uris"]) for p in item_posts] == [100, 100, 50]


def test_make_oauth_reads_env(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    monkeypatch.delenv("SPOTIFY_REDIRECT_URI", raising=False)
    auth = make_oauth(cache_handler=None)
    assert auth.client_id == "cid"
    assert auth.redirect_uri == "http://127.0.0.1:5000/callback"
    assert set(auth.scope.split()) == set(SCOPES.split())
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_spotify_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** — `src/spotify_client.py`

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_spotify_client.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/spotify_client.py tests/test_spotify_client.py
git commit -m "feat: Spotify OAuth and playlist creation on Feb 2026 endpoints" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Flask backend (app factory + routes)

**Files:**
- Create: `src/web/app.py`, `src/web/routes.py`, `src/web/templates/base.html`, `src/web/templates/index.html`, `src/web/templates/dashboard.html` (minimal versions here; styled in Task 13), `tests/test_web.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `create_app(test_config: dict | None = None) -> Flask`. Injectable config keys: `STORE`, `ANALYZER_FACTORY` (`() -> Analyzer`), `OAUTH_FACTORY` (`(cache_handler) -> auth manager`), `SPOTIFY_FACTORY` (`(auth) -> spotipy-like`), `LLM` (same signature as `generate_playlist_text`). `AppState` at `app.extensions["playlist"]` with `store, progress: SyncProgress, sync_thread, get_analyzer()`.
- Routes: `GET /`, `GET /login`, `GET /callback`, `GET /logout`, `POST /sync` (202 / 401 / 409 / 503), `GET /sync/status`, `GET /library`, `POST /generate` (400 on bad mood/feature), `POST /playlist/save` (400 / 401 / 502).

- [ ] **Step 1: Write failing tests** — `tests/test_web.py`

```python
import pytest

from src.store import TrackStore
from src.web.app import create_app

ANALYSIS = {"genre": "disco", "genre_conf": 0.9, "danceability": 0.8, "energy": 0.8,
            "valence": 0.7, "acousticness": 0.1, "tempo": 122.0}


class FakeAuth:
    logged_in = True

    def __init__(self, cache_handler):
        pass

    def validate_token(self, token):
        return {"access_token": "x"} if FakeAuth.logged_in else None

    @property
    def cache_handler(self):
        return self

    def get_cached_token(self):
        return None

    def get_authorize_url(self):
        return "https://accounts.spotify.com/authorize?fake"

    def get_access_token(self, code):
        return {"access_token": "x"}


class FakeSpotify:
    def __init__(self):
        self.posts = []

    def current_user_saved_tracks(self, limit, offset):
        return {"items": [{"track": {"id": "t1", "uri": "spotify:track:t1", "name": "Song", "artists": [{"name": "A"}]}}],
                "next": None}

    def _post(self, url, payload=None):
        self.posts.append((url, payload))
        return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}


class FakeAnalyzer:
    def analyze_audio(self, y):
        return dict(ANALYSIS)


@pytest.fixture
def env(tmp_path):
    FakeAuth.logged_in = True
    sp = FakeSpotify()
    store = TrackStore(tmp_path / "t.db")
    app = create_app({
        "TESTING": True, "SECRET_KEY": "test", "STORE": store,
        "OAUTH_FACTORY": FakeAuth, "SPOTIFY_FACTORY": lambda auth: sp,
        "ANALYZER_FACTORY": FakeAnalyzer,
        "LLM": lambda mood, genres, artists, **kw: {"name": "Disco Fever", "description": "d", "source": "ollama"},
        "SYNC_FIND_PREVIEW": lambda n, a: "http://p",
        "SYNC_DOWNLOAD": lambda url: __import__("numpy").ones(10, dtype="float32"),
    })
    return app, app.test_client(), store, sp


def seed(store, sid="t1"):
    store.upsert({"spotify_id": sid, "uri": f"spotify:track:{sid}", "name": "Song", "artists": "A",
                  "preview_found": 1, "model_version": "v1", **ANALYSIS})


def test_index_shows_login_when_logged_out(env):
    _, client, _, _ = env
    FakeAuth.logged_in = False
    assert b"Log in with Spotify" in client.get("/").data


def test_sync_runs_and_reports_status(env):
    app, client, store, _ = env
    assert client.post("/sync").status_code == 202
    app.extensions["playlist"].sync_thread.join(timeout=5)
    status = client.get("/sync/status").get_json()
    assert status["analyzed"] == 1 and status["running"] is False
    assert status["counts"]["analyzed"] == 1
    assert store.get("t1")["genre"] == "disco"


def test_sync_conflict_while_running(env):
    app, client, _, _ = env
    app.extensions["playlist"].progress.running = True
    assert client.post("/sync").status_code == 409


def test_sync_requires_login(env):
    _, client, _, _ = env
    FakeAuth.logged_in = False
    assert client.post("/sync").status_code == 401


def test_generate_returns_matching_tracks(env):
    _, client, store, _ = env
    seed(store)
    data = client.post("/generate", json={"mood": "dance", "size": 10}).get_json()
    assert [t["spotify_id"] for t in data["tracks"]] == ["t1"]
    assert data["matched"] == 1


def test_generate_rejects_unknown_mood(env):
    _, client, _, _ = env
    assert client.post("/generate", json={"mood": "angry"}).status_code == 400


def test_save_creates_playlist_with_llm_name(env):
    _, client, store, sp = env
    seed(store)
    data = client.post("/playlist/save", json={"mood": "dance", "track_ids": ["t1"]}).get_json()
    assert data["name"] == "Disco Fever" and data["url"].endswith("pl1")
    assert sp.posts[0][0] == "me/playlists"
    assert sp.posts[1] == ("playlists/pl1/items", {"uris": ["spotify:track:t1"]})


def test_save_requires_tracks(env):
    _, client, _, _ = env
    assert client.post("/playlist/save", json={"mood": "dance", "track_ids": []}).status_code == 400
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_web.py -v`
Expected: FAIL — `ModuleNotFoundError: src.web.app`.

- [ ] **Step 3: Implement** — `src/web/app.py`

```python
import os
import threading
from dataclasses import dataclass, field

import spotipy
from dotenv import load_dotenv
from flask import Flask

from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.config import DB_PATH
from src.library_sync import SyncProgress
from src.llm import generate_playlist_text
from src.ml.analyzer import Analyzer
from src.spotify_client import make_oauth
from src.store import TrackStore


@dataclass
class AppState:
    store: TrackStore
    analyzer_factory: object
    oauth_factory: object
    spotify_factory: object
    llm: object
    find_preview: object
    download: object
    progress: SyncProgress = field(default_factory=SyncProgress)
    sync_thread: threading.Thread | None = None
    _analyzer: Analyzer | None = None

    def get_analyzer(self):
        if self._analyzer is None:
            self._analyzer = self.analyzer_factory()
        return self._analyzer


def create_app(test_config=None):
    load_dotenv()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or os.urandom(24).hex(),
        OLLAMA_URL=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    )
    if test_config:
        app.config.update(test_config)

    cfg = app.config
    app.extensions["playlist"] = AppState(
        store=cfg.get("STORE") or TrackStore(DB_PATH),
        analyzer_factory=cfg.get("ANALYZER_FACTORY", Analyzer.load),
        oauth_factory=cfg.get("OAUTH_FACTORY", make_oauth),
        spotify_factory=cfg.get("SPOTIFY_FACTORY", lambda auth: spotipy.Spotify(auth_manager=auth)),
        llm=cfg.get("LLM", generate_playlist_text),
        find_preview=cfg.get("SYNC_FIND_PREVIEW", find_preview),
        download=cfg.get("SYNC_DOWNLOAD", download_audio),
    )

    from src.web.routes import bp
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
```

`src/web/routes.py`:

```python
from collections import Counter
from functools import wraps

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from spotipy.cache_handler import FlaskSessionCacheHandler
from spotipy.exceptions import SpotifyException

from src.library_sync import LibrarySync, SyncProgress, iter_liked_tracks, start_background_sync
from src.ml.analyzer import ModelsMissingError
from src.moods import FULL_RANGE, MOODS, build_targets
from src.selector import select_tracks
from src.spotify_client import create_playlist

bp = Blueprint("main", __name__)
TRACK_FIELDS = ("spotify_id", "name", "artists", "genre", "genre_conf", "danceability",
                "energy", "valence", "acousticness", "tempo", "score")


def _state():
    return current_app.extensions["playlist"]


def _auth():
    return _state().oauth_factory(FlaskSessionCacheHandler(session))


def _spotify():
    auth = _auth()
    if not auth.validate_token(auth.cache_handler.get_cached_token()):
        return None
    return _state().spotify_factory(auth)


def spotify_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        sp = _spotify()
        if sp is None:
            return jsonify(error="Please log in with Spotify first."), 401
        return view(sp, *args, **kwargs)
    return wrapper


@bp.get("/")
def index():
    if _spotify() is None:
        return render_template("index.html")
    store = _state().store
    genres = sorted({t["genre"] for t in store.analyzed_tracks() if t["genre"]})
    app_data = {"moods": {m: {f: list(r) for f, r in v.items()} for m, v in MOODS.items()},
                "full": {f: list(r) for f, r in FULL_RANGE.items()}}
    return render_template("dashboard.html", counts=store.counts(), moods=list(MOODS), genres=genres, app_data=app_data)


@bp.get("/login")
def login():
    return redirect(_auth().get_authorize_url())


@bp.get("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return render_template("index.html", error=request.args.get("error", "Login was cancelled.")), 400
    _auth().get_access_token(code)
    return redirect(url_for("main.index"))


@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


@bp.post("/sync")
@spotify_required
def sync(sp):
    state = _state()
    if state.progress.running:
        return jsonify(error="A sync is already running.", **state.progress.to_dict()), 409
    try:
        analyzer = state.get_analyzer()
    except ModelsMissingError as exc:
        return jsonify(error=str(exc)), 503
    # Fetch inside the request: the Spotify token lives in the Flask session, which a background thread can't read.
    try:
        tracks = list(iter_liked_tracks(sp))
    except SpotifyException as exc:
        return jsonify(error=f"Spotify error: {exc.msg}"), 502
    state.progress = SyncProgress()
    syncer = LibrarySync(state.store, analyzer, find_preview=state.find_preview, download=state.download)
    state.sync_thread = start_background_sync(tracks, syncer, state.progress)
    return jsonify(state.progress.to_dict()), 202


@bp.get("/sync/status")
def sync_status():
    state = _state()
    return jsonify(**state.progress.to_dict(), counts=state.store.counts())


@bp.get("/library")
def library():
    return jsonify(tracks=_state().store.analyzed_tracks())


@bp.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}
    try:
        targets = build_targets(body.get("mood"), body.get("overrides"))
        size = max(1, min(int(body.get("size", 25)), 100))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    tracks = select_tracks(_state().store.analyzed_tracks(), targets, body.get("genres") or None, size)
    return jsonify(tracks=[{k: t.get(k) for k in TRACK_FIELDS} for t in tracks],
                   matched=len(tracks), requested=size)


@bp.post("/playlist/save")
@spotify_required
def save_playlist(sp):
    body = request.get_json(silent=True) or {}
    rows = _state().store.get_many(list(body.get("track_ids") or []))
    if not rows:
        return jsonify(error="No songs to save."), 400
    genres = [g for g, _ in Counter(r["genre"] for r in rows if r["genre"]).most_common(3)]
    artists = [a for a, _ in Counter(r["artists"].split(", ")[0] for r in rows).most_common(5)]
    text = _state().llm(body.get("mood"), genres, artists,
                        base_url=current_app.config["OLLAMA_URL"], model=current_app.config["OLLAMA_MODEL"])
    try:
        result = create_playlist(sp, text["name"], text["description"], [r["uri"] for r in rows])
    except SpotifyException as exc:
        return jsonify(error=f"Spotify error: {exc.msg}"), 502
    return jsonify(**result, **text)
```

Minimal templates (Task 13 fills them in):

`src/web/templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Playlist Creator</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <main>{% block content %}{% endblock %}</main>
</body>
</html>
```

`src/web/templates/index.html`:

```html
{% extends "base.html" %}
{% block content %}
<section class="hero">
  <h1>Playlist Creator</h1>
  <p>Build playlists from your Liked Songs by mood, energy and genre.</p>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
  <a class="button primary" href="{{ url_for('main.login') }}">Log in with Spotify</a>
</section>
{% endblock %}
```

`src/web/templates/dashboard.html`: `{% extends "base.html" %}{% block content %}<p>{{ counts.analyzed }} analyzed</p>{% endblock %}` (replaced in Task 13). Also create an empty `src/web/static/style.css`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_web.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/web tests/test_web.py
git commit -m "feat: Flask backend with Spotify auth, sync, generate and save routes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Web UI (dashboard, sliders, results, save)

**Files:**
- Modify: `src/web/templates/base.html`, `src/web/templates/dashboard.html`, `src/web/static/style.css`
- Create: `src/web/static/app.js`
- Test: `tests/test_web.py` (add one render test)

**Interfaces:**
- Consumes: routes from Task 12; `window.APP = {moods: {mood: {feature: [lo, hi]}}, full: {feature: [lo, hi]}}`.

- [ ] **Step 1: Write failing test** — append to `tests/test_web.py`

```python
def test_dashboard_renders_controls(env):
    _, client, store, _ = env
    seed(store)
    html = client.get("/").data.decode()
    for needle in ['id="sync-btn"', 'data-mood="dance"', 'value="disco"', 'id="generate-btn"', "app.js"]:
        assert needle in html
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web.py::test_dashboard_renders_controls -v`
Expected: FAIL — `id="sync-btn"` not in html.

- [ ] **Step 3: Implement** — `src/web/templates/base.html` (add header with logout)

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Playlist Creator</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <header class="topbar">
    <span class="brand">Playlist Creator</span>
    {% block nav %}{% endblock %}
  </header>
  <main>{% block content %}{% endblock %}</main>
</body>
</html>
```

`src/web/templates/dashboard.html`:

```html
{% extends "base.html" %}
{% block nav %}<a href="{{ url_for('main.logout') }}">Log out</a>{% endblock %}
{% block content %}
<section class="card">
  <div class="row">
    <h2>Your library</h2>
    <button id="sync-btn">Sync liked songs</button>
  </div>
  <p id="counts" class="muted">{{ counts.analyzed }} analyzed · {{ counts.no_preview }} without preview</p>
  <div id="progress" hidden>
    <progress id="progress-bar" max="1" value="0"></progress>
    <span id="progress-text" class="muted"></span>
  </div>
  <p id="sync-error" class="error" hidden></p>
</section>

<section class="card">
  <h2>Mood</h2>
  <div id="moods" class="chips">
    {% for m in moods %}<button class="chip" data-mood="{{ m }}">{{ m|title }}</button>{% endfor %}
  </div>

  <h3>Fine-tune</h3>
  <div id="sliders"></div>

  <h3>Genres</h3>
  <div id="genres" class="chips">
    {% for g in genres %}<label class="chip"><input type="checkbox" value="{{ g }}"> {{ g }}</label>{% endfor %}
    {% if not genres %}<span class="muted">Sync your library to see genres.</span>{% endif %}
  </div>

  <div class="row">
    <label>Songs <input type="number" id="size" min="1" max="100" value="25"></label>
    <button id="generate-btn" class="primary">Generate</button>
  </div>
</section>

<section class="card" id="results" hidden>
  <div class="row">
    <h2>Preview</h2>
    <button id="save-btn" class="primary">Save to Spotify</button>
  </div>
  <p id="results-note" class="muted"></p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Song</th><th>Artist</th><th>Genre</th><th>Dance</th><th>Energy</th><th>Valence</th><th>BPM</th><th></th></tr></thead>
      <tbody id="results-body"></tbody>
    </table>
  </div>
  <div id="save-result"></div>
</section>

<script>window.APP = {{ app_data|tojson }};</script>
<script src="{{ url_for('static', filename='app.js') }}"></script>
{% endblock %}
```

`src/web/static/app.js`:

```js
const { moods, full } = window.APP;
const FEATURES = Object.keys(full);
const state = { mood: null, tracks: [] };
const $ = (id) => document.getElementById(id);

async function api(url, body) {
  const opts = body === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// ---- sliders ----
function buildSliders() {
  const box = $("sliders");
  for (const f of FEATURES) {
    const [lo, hi] = full[f];
    const step = f === "tempo" ? 1 : 0.01;
    box.insertAdjacentHTML("beforeend", `
      <div class="slider" data-feature="${f}">
        <span class="label">${f}</span>
        <input type="range" class="min" min="${lo}" max="${hi}" step="${step}" value="${lo}" aria-label="${f} minimum">
        <input type="range" class="max" min="${lo}" max="${hi}" step="${step}" value="${hi}" aria-label="${f} maximum">
        <span class="value"></span>
      </div>`);
  }
  box.addEventListener("input", (e) => {
    const row = e.target.closest(".slider");
    if (row) updateLabel(row);
  });
  document.querySelectorAll(".slider").forEach(updateLabel);
}

function updateLabel(row) {
  const min = row.querySelector(".min");
  const max = row.querySelector(".max");
  if (Number(min.value) > Number(max.value)) [min.value, max.value] = [max.value, min.value];
  row.querySelector(".value").textContent = `${min.value} – ${max.value}`;
}

function setSliders(ranges) {
  document.querySelectorAll(".slider").forEach((row) => {
    const [lo, hi] = ranges[row.dataset.feature] || full[row.dataset.feature];
    row.querySelector(".min").value = lo;
    row.querySelector(".max").value = hi;
    updateLabel(row);
  });
}

function readSliders() {
  const out = {};
  document.querySelectorAll(".slider").forEach((row) => {
    out[row.dataset.feature] = [Number(row.querySelector(".min").value), Number(row.querySelector(".max").value)];
  });
  return out;
}

// ---- mood chips ----
function selectMood(mood) {
  state.mood = state.mood === mood ? null : mood;
  document.querySelectorAll("#moods .chip").forEach((b) => b.classList.toggle("active", b.dataset.mood === state.mood));
  setSliders(state.mood ? moods[state.mood] : full);
}

// ---- sync ----
function showSyncError(msg) {
  $("sync-error").textContent = msg;
  $("sync-error").hidden = false;
  $("sync-btn").disabled = false;
}

async function startSync() {
  $("sync-error").hidden = true;
  $("sync-btn").disabled = true;
  $("progress").hidden = false;
  $("progress-text").textContent = "Fetching your liked songs…";
  try {
    await api("/sync", {});
    pollSync();
  } catch (e) {
    showSyncError(e.message);
  }
}

async function pollSync() {
  let p;
  try {
    p = await api("/sync/status");
  } catch (e) {
    return showSyncError(e.message);
  }
  $("progress-bar").max = p.total || 1;
  $("progress-bar").value = p.done;
  $("progress-text").textContent =
    `${p.done}/${p.total} · ${p.analyzed} analyzed · ${p.no_preview} no preview · ${p.failed} failed`;
  $("counts").textContent = `${p.counts.analyzed} analyzed · ${p.counts.no_preview} without preview`;
  if (p.running) {
    $("sync-btn").disabled = true;
    $("progress").hidden = false;
    setTimeout(pollSync, 1500);
  } else {
    $("sync-btn").disabled = false;
    if (p.error) showSyncError(p.error);
    else if (p.total) $("progress-text").textContent += " · done (reload to refresh genres)";
  }
}

// ---- generate + results ----
const fmt = (v, digits = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));

function renderResults(data) {
  $("results").hidden = false;
  $("save-result").textContent = "";
  const body = $("results-body");
  body.replaceChildren();
  state.tracks.forEach((t, i) => {
    const tr = document.createElement("tr");
    const cells = [t.name, t.artists, t.genre, fmt(t.danceability), fmt(t.energy), fmt(t.valence), fmt(t.tempo, 0)];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    const rm = document.createElement("button");
    rm.className = "icon";
    rm.textContent = "×";
    rm.title = "Remove from playlist";
    rm.addEventListener("click", () => {
      state.tracks.splice(i, 1);
      renderResults({ matched: state.tracks.length, requested: data.requested });
    });
    td.appendChild(rm);
    tr.appendChild(td);
    body.appendChild(tr);
  });
  $("results-note").textContent = data.matched < data.requested
    ? `Only ${state.tracks.length} songs match. Try widening the sliders or removing genre filters.`
    : `${state.tracks.length} songs`;
  $("save-btn").disabled = state.tracks.length === 0;
}

async function generate() {
  const genres = [...document.querySelectorAll("#genres input:checked")].map((i) => i.value);
  $("generate-btn").disabled = true;
  try {
    const data = await api("/generate", { mood: state.mood, overrides: readSliders(), genres, size: Number($("size").value) });
    state.tracks = data.tracks;
    renderResults(data);
  } catch (e) {
    $("results").hidden = false;
    $("results-note").textContent = e.message;
  } finally {
    $("generate-btn").disabled = false;
  }
}

async function save() {
  $("save-btn").disabled = true;
  const out = $("save-result");
  out.textContent = "Naming your playlist and saving…";
  try {
    const d = await api("/playlist/save", { mood: state.mood, track_ids: state.tracks.map((t) => t.spotify_id) });
    out.replaceChildren();
    const a = document.createElement("a");
    a.href = d.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `Open “${d.name}” in Spotify`;
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = d.description + (d.source === "template" ? " (Ollama unavailable: used a default name)" : "");
    out.append(a, p);
  } catch (e) {
    out.textContent = e.message;
  } finally {
    $("save-btn").disabled = false;
  }
}

// ---- wire up ----
buildSliders();
document.querySelectorAll("#moods .chip").forEach((b) => b.addEventListener("click", () => selectMood(b.dataset.mood)));
$("sync-btn").addEventListener("click", startSync);
$("generate-btn").addEventListener("click", generate);
$("save-btn").addEventListener("click", save);
pollSync(); // resume progress display if a sync is already running
```

`src/web/static/style.css`:

```css
:root {
  --bg: #0f1115; --card: #181b22; --text: #e8eaf0; --muted: #9aa1b2;
  --accent: #1db954; --accent-text: #04140a; --border: #2a2f3a; --error: #ff6b6b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 16px; }
.topbar { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.topbar a { color: var(--muted); }
.brand { font-weight: 700; }
.hero { text-align: center; padding: 80px 0; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; }
.row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
h2 { margin: 0 0 8px; font-size: 18px; } h3 { margin: 16px 0 8px; font-size: 14px; color: var(--muted); }
.muted { color: var(--muted); } .error { color: var(--error); }
button, .button { background: var(--border); color: var(--text); border: 0; border-radius: 999px; padding: 8px 16px; font: inherit; cursor: pointer; text-decoration: none; display: inline-block; }
button:disabled { opacity: .5; cursor: default; }
.primary { background: var(--accent); color: var(--accent-text); font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { background: transparent; border: 1px solid var(--border); border-radius: 999px; padding: 6px 14px; }
.chip.active { background: var(--accent); color: var(--accent-text); border-color: var(--accent); }
.slider { display: grid; grid-template-columns: 110px 1fr 1fr 100px; gap: 8px; align-items: center; margin: 6px 0; }
.slider .label { text-transform: capitalize; } .slider .value { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
input[type=range] { accent-color: var(--accent); width: 100%; }
input[type=number] { width: 70px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px; }
progress { width: 100%; accent-color: var(--accent); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; }
.icon { background: transparent; padding: 2px 8px; color: var(--muted); }
#save-result a { color: var(--accent); font-weight: 600; }
@media (max-width: 600px) {
  .slider { grid-template-columns: 1fr 1fr; }
  .slider .label, .slider .value { grid-column: span 1; }
}
```

- [ ] **Step 4: Run to verify tests pass**

Run: `python -m pytest tests/test_web.py -v`
Expected: 9 passed.

- [ ] **Step 5: Manual UI check**

Run: `python -m src.web.app` and open http://127.0.0.1:5000. Verify: login page renders; after login, mood chip sets sliders; Generate renders a table (needs synced data); × removes a row; layout works at ~375 px width.

- [ ] **Step 6: Commit**

```bash
git add src/web tests/test_web.py
git commit -m "feat: dashboard UI with mood presets, sliders, preview table and save" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Requirements, README, end-to-end verification

**Files:**
- Modify: `requirements.txt`, `README.md`
- Create: `requirements-train.txt`, `.env.example`

- [ ] **Step 1: `requirements.txt`**

```text
Flask==2.3.3
spotipy
python-dotenv
requests
numpy
pandas==2.2.2
pyarrow
scikit-learn==1.4.2
lightgbm
joblib
librosa
pydub
ffmpeg-downloader
tensorflow
pytest
```

`requirements-train.txt`:

```text
-r requirements.txt
matplotlib
seaborn
```

`.env.example`:

```text
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/callback
FLASK_SECRET_KEY=change-me
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

- [ ] **Step 2: Rewrite `README.md`**

Write a README with these sections and concrete content:
1. **What it does** — sync Liked Songs → analyze with own models → mood/slider/genre playlists → Ollama names → saved to Spotify.
2. **How it works** — diagram-like list: Deezer preview → Librosa (mel + 72-dim summary) → CNN + LightGBM genre ensemble, LightGBM mood regressors trained on `data/dataset.csv` labels → SQLite cache → selector → Ollama → Spotify.
3. **Model results** — paste the mood metrics table (Task 4 Step 7) and genre accuracy (Task 5 Step 7, plus CNN if retrained).
4. **Setup** — venv, `pip install -r requirements.txt`, ffmpeg (`ffdl install --add-path`), Spotify app (dashboard → redirect URI `http://127.0.0.1:5000/callback`; note Dev Mode requires the owner to have Premium and allow-listing users under *User Management*), copy `.env.example` → `.env`, Ollama install + `ollama pull llama3.2:3b`.
5. **Training (one time)** — GTZAN into `data/genres/<genre>/*.wav`; commands from Tasks 4–5 with timing notes; Colab notebook for the CNN.
6. **Run** — `python -m src.web.app`, open http://127.0.0.1:5000. First sync analyzes every song once (~2–4 s per song); later syncs only process new likes.
7. **Tests** — `python -m pytest`.
8. **Roadmap** — Phase 2 ReccoBeats hybrid; later free-text mood search with sentence-transformer embeddings.

Remove claims that aren't true (Spotify audio-features usage).

- [ ] **Step 3: Full test suite**

Run: `python -m pytest -q`
Expected: all pass, 0 failures.

- [ ] **Step 4: End-to-end manual run**

1. `ollama serve` running (or the Ollama app open); `python -m src.web.app`.
2. Open http://127.0.0.1:5000 → Log in with Spotify → dashboard.
3. Click **Sync liked songs** → progress bar advances; counts update; no crash on songs without previews.
4. Reload → genres appear. Pick **Dance**, size 20 → **Generate** → table of songs with dance/energy ≥ preset minimums.
5. **Save to Spotify** → link opens a new private playlist with an Ollama-written name/description and the shown songs.
6. Stop Ollama → save again → playlist named `Dance Mix · Oct 2026` and the note "(Ollama unavailable…)".
7. Click Sync twice quickly → second click shows "A sync is already running."

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements-train.txt .env.example README.md
git commit -m "docs: README, requirements and env example for Phase 1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
