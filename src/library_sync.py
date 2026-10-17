import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.fallback import artist_estimate
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
    from_dataset: int = 0  # no preview, but Spotify's real scores were in dataset.csv
    estimated: int = 0     # no preview, estimated from the artist's other songs
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class LibrarySync:
    """Analyzes songs: preview audio (Deezer, then iTunes) -> dataset.csv -> artist estimate -> no preview."""

    def __init__(self, store, analyzer, find_preview=find_preview, download=download_audio, dataset=None):
        self.store = store
        self.analyzer = analyzer
        self.find_preview = find_preview
        self.download = download
        self.dataset = dataset

    def _save(self, track, analysis, source):
        self.store.upsert({
            **track, **analysis, "preview_found": 1, "analysis_source": source, "model_version": MODEL_VERSION,
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    def analyze_one(self, track):
        """Returns "analyzed", "from_dataset", or "pending" (no audio and not in the dataset)."""
        main_artist = track["artists"].split(", ")[0]
        url = self.find_preview(track["name"], main_artist)
        y = self.download(url) if url else None
        if y is not None:
            self._save(track, self.analyzer.analyze_audio(y), "audio")
            return "analyzed"
        found = self.dataset.find(track) if self.dataset else None
        if found:
            self._save(track, found, "dataset")
            return "from_dataset"
        return "pending"

    def finish_one(self, track, analyzed):
        """Last resort for a song with no audio: estimate it from the artist, else mark it as having no preview."""
        estimate = artist_estimate(track, analyzed)
        if estimate:
            self._save(track, estimate, "artist")
            return "estimated"
        self.store.upsert({**track, "preview_found": 0, "analysis_source": None, "model_version": MODEL_VERSION})
        return "no_preview"

    def _record(self, progress, outcome):
        setattr(progress, outcome, getattr(progress, outcome) + 1)
        progress.done += 1

    def run(self, tracks, progress):
        known = self.store.ids_done(MODEL_VERSION)
        todo = [t for t in tracks if t["spotify_id"] not in known]
        progress.total, progress.running = len(todo), True
        pending = []
        try:
            for t in todo:
                try:
                    outcome = self.analyze_one(t)
                except Exception:
                    traceback.print_exc()
                    outcome = "failed"
                if outcome == "pending":
                    pending.append(t)  # estimated at the end, once the artist's other songs are analyzed
                else:
                    self._record(progress, outcome)
            analyzed = self.store.analyzed_tracks() if pending else []
            for t in pending:
                self._record(progress, self.finish_one(t, analyzed))
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
