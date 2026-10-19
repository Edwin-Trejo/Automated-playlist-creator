"""Song lyrics from LRCLIB (free, no key). Lyrics are only used to score moods; the text is never stored."""
import threading
import time

import requests

from src.audio.preview import clean_title

LRCLIB = "https://lrclib.net/api"
USER_AGENT = "Automated-playlist-creator (github.com/Edwin-Trejo/Automated-playlist-creator)"
MIN_INTERVAL = 1.0  # seconds between requests; LRCLIB starts answering 429 at ~1.6 requests/s
BACKOFF = (5, 15, 30, 60, 120, 300)  # waits after a 429/5xx (~9 min in total) before leaving it for the next sync

_lock = threading.Lock()
_last_call = 0.0


class LyricsUnavailable(Exception):
    """LRCLIB could not be reached (rate limit or outage): try the song again on a later sync."""


def _get(path, params, session, sleep, clock):
    """GET that returns the response for 200 and 404, and raises LyricsUnavailable if LRCLIB keeps failing."""
    global _last_call
    http = session or requests
    for wait in (*BACKOFF, None):
        with _lock:
            gap = MIN_INTERVAL - (clock() - _last_call)
            if gap > 0:
                sleep(gap)
            try:
                resp = http.get(f"{LRCLIB}/{path}", params=params, timeout=20, headers={"User-Agent": USER_AGENT})
            except requests.RequestException:
                resp = None
            _last_call = clock()
        if resp is not None and resp.status_code in (200, 404):
            return resp
        if wait is None:
            break
        retry = resp.headers.get("Retry-After") if resp is not None else None
        sleep(float(retry) if retry and str(retry).isdigit() else wait)
    raise LyricsUnavailable(path)


def find_lyrics(name, artist, album=None, duration_ms=None, *, session=None, sleep=time.sleep, clock=time.monotonic):
    """{"lyrics": text or None, "instrumental": bool}. Exact lookup by album and length first, then search."""
    title = clean_title(name)
    hit = None
    if album and duration_ms:
        resp = _get("get", {"track_name": title, "artist_name": artist, "album_name": album,
                            "duration": round(duration_ms / 1000)}, session, sleep, clock)
        hit = resp.json() if resp.status_code == 200 else None
    if not hit or not hit.get("plainLyrics"):
        resp = _get("search", {"track_name": title, "artist_name": artist}, session, sleep, clock)
        hits = (resp.json() if resp.status_code == 200 else None) or []
        hit = (next((h for h in hits if h.get("plainLyrics")), None)
               or next((h for h in hits if h.get("instrumental")), None) or hit)
    hit = hit or {}
    return {"lyrics": hit.get("plainLyrics") or None, "instrumental": bool(hit.get("instrumental"))}
