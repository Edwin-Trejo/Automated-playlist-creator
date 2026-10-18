import logging
import os
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse

import spotipy
from dotenv import load_dotenv
from flask import Flask

from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.config import DB_PATH
from src.fallback import DatasetLookup
from src.library_sync import SyncProgress
from src.ml.analyzer import Analyzer
from src.spotify_client import DEFAULT_REDIRECT, make_oauth
from src.store import TrackStore

log = logging.getLogger(__name__)

@dataclass
class AppState:
    store: TrackStore
    analyzer_factory: object
    oauth_factory: object
    spotify_factory: object
    find_preview: object
    download: object
    dataset: object = None
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
    secret = os.getenv("FLASK_SECRET_KEY")
    if not secret:
        log.warning("FLASK_SECRET_KEY is not set: using a random key, so every restart logs you out.")
    app.config.update(
        SECRET_KEY=secret or os.urandom(24).hex(),
        # Behind HTTPS (tailscale serve), only send the login cookie over HTTPS.
        SESSION_COOKIE_SECURE=redirect_uri().startswith("https://"),
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)

    cfg = app.config
    app.extensions["playlist"] = AppState(
        store=cfg.get("STORE") or TrackStore(DB_PATH),
        analyzer_factory=cfg.get("ANALYZER_FACTORY", Analyzer.load),
        oauth_factory=cfg.get("OAUTH_FACTORY", make_oauth),
        spotify_factory=cfg.get("SPOTIFY_FACTORY", lambda auth: spotipy.Spotify(auth_manager=auth)),
        find_preview=cfg.get("SYNC_FIND_PREVIEW", find_preview),
        download=cfg.get("SYNC_DOWNLOAD", download_audio),
        dataset=cfg["DATASET"] if "DATASET" in cfg else DatasetLookup(),  # loads dataset.csv on first use
    )

    from src.web.routes import bp
    app.register_blueprint(bp)
    return app


def redirect_uri():
    return os.getenv("SPOTIFY_REDIRECT_URI", DEFAULT_REDIRECT)


def server_address():
    """HOST and PORT from .env. Without PORT, use the redirect URI's port so the OAuth callback reaches
    this app locally. An HTTPS redirect (tailscale serve) has no port, so it falls back to 5000."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT") or urlparse(redirect_uri()).port or 5000)
    return host, port


if __name__ == "__main__":
    from waitress import serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    host, port = server_address()
    log.info("Serving on http://%s:%d", host, port)
    # One process with threads: the background sync and its progress live in this process's memory.
    serve(create_app(), host=host, port=port, threads=8)
