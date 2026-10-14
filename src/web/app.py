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
