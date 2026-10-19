import logging
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.audio.loader import download_audio
from src.audio.preview import find_preview
from src.fallback import artist_estimate
from src.lyrics import LOOKUP_VERSION, LyricsUnavailable, find_lyrics
from src.ml.analyzer import MODEL_VERSION
from src.spotify_client import playlist_entries

log = logging.getLogger(__name__)


LIKED = "liked"  # source id for Liked Songs; any other source id is a Spotify playlist id
MAX_UNAVAILABLE_IN_A_ROW = 5  # songs in a row the lyrics sites couldn't answer before lyrics wait for the next sync


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
        # Not stored; they let the lyrics lookup ask LRCLIB for the exact recording.
        "album": (t.get("album") or {}).get("name"),
        "duration_ms": t.get("duration_ms"),
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
    lyrics_total: int = 0  # songs whose lyrics are checked this sync (new songs, or a new lyrics model)
    lyrics_done: int = 0
    lyrics_found: int = 0
    lyrics_paused: bool = False  # LRCLIB stayed busy; the rest are checked on the next sync
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class LibrarySync:
    """Analyzes songs: preview audio (Deezer, then iTunes) -> dataset.csv -> artist estimate -> no preview.
    Then scores the lyrics of analyzed songs (LRCLIB + the local lyrics model), when a lyrics model is given."""

    def __init__(self, store, analyzer, find_preview=find_preview, download=download_audio, dataset=None,
                 lyrics=None, find_lyrics=find_lyrics):
        self.store = store
        self.analyzer = analyzer
        self.find_preview = find_preview
        self.download = download
        self.dataset = dataset
        self.lyrics = lyrics  # callable returning a LyricsMood (loaded lazily), or None to skip lyrics
        self.find_lyrics = find_lyrics

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
            self.run_lyrics(tracks, progress)
        finally:
            progress.running = False

    def run_lyrics(self, tracks, progress):
        """Score the lyrics of analyzed songs not yet checked with the current lyrics model."""
        if self.lyrics is None:
            return
        try:
            scorer = self.lyrics()
        except Exception:
            log.exception("Lyrics model unavailable; songs are ranked by audio only this sync")
            return
        checked = self.store.ids_with_lyrics(scorer.version, LOOKUP_VERSION)
        usable = {t["spotify_id"] for t in self.store.analyzed_tracks()}
        todo = [t for t in tracks if t["spotify_id"] in usable and t["spotify_id"] not in checked]
        progress.lyrics_total = len(todo)
        unavailable_in_a_row = 0
        for t in todo:
            try:
                found = self.find_lyrics(t["name"], t["artists"].split(", ")[0], t.get("album"), t.get("duration_ms"))
                unavailable_in_a_row = 0
                if found["lyrics"]:
                    self.store.set_lyrics(t["spotify_id"], "found", scorer.score(found["lyrics"]), scorer.version)
                    progress.lyrics_found += 1
                elif found["instrumental"]:
                    self.store.set_lyrics(t["spotify_id"], "instrumental", None, scorer.version)
                else:
                    # Remembered per lookup version, so adding a lyrics source looks these songs up again.
                    self.store.set_lyrics(t["spotify_id"], "none", None, f"{scorer.version}+{LOOKUP_VERSION}")
            except LyricsUnavailable:
                # LRCLIB is busy and lyrics.ovh doesn't have this song: left unchecked for the next sync.
                progress.lyrics_paused = True
                unavailable_in_a_row += 1
                if unavailable_in_a_row >= MAX_UNAVAILABLE_IN_A_ROW:
                    log.warning("Lyrics sites are unavailable; the remaining lyrics are tried on the next sync")
                    return
            except Exception:
                traceback.print_exc()  # left unchecked, so the next sync tries it again
            progress.lyrics_done += 1


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
