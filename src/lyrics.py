"""Song lyrics from LRCLIB, then lyrics.ovh (both free, no key). Lyrics are only used to score moods; the text
is never stored."""
import re
import threading
import time
from urllib.parse import quote

import requests

from src.audio.preview import clean_title

LRCLIB = "https://lrclib.net/api"
LYRICS_OVH = "https://api.lyrics.ovh/v1"
LOOKUP_VERSION = "2"  # bump when a lyrics source is added, so songs with no lyrics found are looked up again
USER_AGENT = "Automated-playlist-creator (github.com/Edwin-Trejo/Automated-playlist-creator)"
MIN_INTERVAL = 1.0  # seconds between LRCLIB requests; it starts answering 429 at ~1.6 requests/s
BACKOFF = (5, 15, 30, 60)  # waits after an LRCLIB 429/5xx (~2 min in total) before falling back to lyrics.ovh
LRCLIB_COOLDOWN = 600  # after LRCLIB stays busy, skip it (lyrics.ovh only) for this many seconds
OVH_BACKOFF = (5,)  # lyrics.ovh is only a fallback: one retry, then the song counts as not found there
_OVH_HEADER = re.compile(r"^Paroles de la chanson .* par .*\r?\n")  # some lyrics.ovh texts start with this line

_lock = threading.Lock()
_last_call = 0.0
_lrclib_paused_until = 0.0


class LyricsUnavailable(Exception):
    """LRCLIB could not be reached (rate limit or outage): try the song again on a later sync."""


def _get(url, params, session, sleep, clock, backoff=BACKOFF):
    """GET that returns the response for 200 and 404, and raises LyricsUnavailable if the site keeps failing."""
    global _last_call
    http = session or requests
    for wait in (*backoff, None):
        with _lock:
            gap = MIN_INTERVAL - (clock() - _last_call)
            if gap > 0:
                sleep(gap)
            try:
                resp = http.get(url, params=params, timeout=20, headers={"User-Agent": USER_AGENT})
            except requests.RequestException:
                resp = None
            _last_call = clock()
        if resp is not None and resp.status_code in (200, 404):
            return resp
        if wait is None:
            break
        retry = resp.headers.get("Retry-After") if resp is not None else None
        sleep(float(retry) if retry and str(retry).isdigit() else wait)
    raise LyricsUnavailable(url)


def _lrclib(title, artist, album, duration_ms, session, sleep, clock):
    """LRCLIB: exact lookup by album and length first, then search. Returns its best hit (or {})."""
    hit = None
    if album and duration_ms:
        resp = _get(f"{LRCLIB}/get", {"track_name": title, "artist_name": artist, "album_name": album,
                                      "duration": round(duration_ms / 1000)}, session, sleep, clock)
        hit = resp.json() if resp.status_code == 200 else None
    if not hit or not hit.get("plainLyrics"):
        resp = _get(f"{LRCLIB}/search", {"track_name": title, "artist_name": artist}, session, sleep, clock)
        hits = (resp.json() if resp.status_code == 200 else None) or []
        hit = (next((h for h in hits if h.get("plainLyrics")), None)
               or next((h for h in hits if h.get("instrumental")), None) or hit)
    return hit or {}


def _lyrics_ovh(title, artist, session, sleep, clock):
    """lyrics.ovh: lyrics text, or None if it doesn't have the song (or is down)."""
    try:
        resp = _get(f"{LYRICS_OVH}/{quote(artist, safe='')}/{quote(title, safe='')}", None, session, sleep, clock,
                    backoff=OVH_BACKOFF)
    except LyricsUnavailable:
        return None
    text = ((resp.json() or {}).get("lyrics") if resp.status_code == 200 else None) or ""
    return _OVH_HEADER.sub("", text).strip() or None


def find_lyrics(name, artist, album=None, duration_ms=None, *, session=None, sleep=time.sleep, clock=time.monotonic):
    """{"lyrics": text or None, "instrumental": bool}.

    LRCLIB first; lyrics.ovh when LRCLIB doesn't have the lyrics or is rate limiting. Raises LyricsUnavailable
    only when LRCLIB is unreachable and lyrics.ovh doesn't have the song either, so it is tried next sync.
    """
    global _lrclib_paused_until
    title = clean_title(name)
    hit, lrclib_down = {}, clock() < _lrclib_paused_until
    if not lrclib_down:
        try:
            hit = _lrclib(title, artist, album, duration_ms, session, sleep, clock)
        except LyricsUnavailable:
            lrclib_down = True
            _lrclib_paused_until = clock() + LRCLIB_COOLDOWN
    if hit.get("plainLyrics") or hit.get("instrumental"):
        return {"lyrics": hit.get("plainLyrics") or None, "instrumental": bool(hit.get("instrumental"))}
    text = _lyrics_ovh(title, artist, session, sleep, clock)
    if text:
        return {"lyrics": text, "instrumental": False}
    if lrclib_down:
        raise LyricsUnavailable("lrclib")
    return {"lyrics": None, "instrumental": False}
