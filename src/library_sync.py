import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.ml.analyzer import MODEL_VERSION
from src.spotify_client import playlist_entries


LIKED = "liked"  # source id for Liked Songs; any other source id is a Spotify playlist id


def _track(entry):
    """Track dict from a liked-songs or playlist entry; None for local files, podcasts and unavailable tracks."""
    t = entry.get("item") or entry.get("track") or {}
    if not t.get("id") or t.get("is_local") or t.get("type", "track") != "track":
        return None
    return {
        "spotify_id": t["id"],
        "uri": t["uri"],
        "name": t["name"],
        "artists": ", ".join(a["name"] for a in t.get("artists", [])),
    }


def iter_liked_tracks(sp, page_size=50):
    """Yield the user's liked songs, skipping local files and unavailable tracks."""
    offset = 0
    while True:
        page = sp.current_user_saved_tracks(limit=page_size, offset=offset)
        for entry in page.get("items", []):
            if t := _track(entry):
                yield t
        if not page.get("next"):
            return
        offset += page_size


def iter_playlist_tracks(sp, playlist_id):
    for entry in playlist_entries(sp, playlist_id):
        if t := _track(entry):
            yield t


def iter_source_tracks(sp, source_id):
    return iter_liked_tracks(sp) if source_id == LIKED else iter_playlist_tracks(sp, source_id)


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
