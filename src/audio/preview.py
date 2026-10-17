import re
import threading
import time

from src.http_utils import get_with_retry

DEEZER_SEARCH = "https://api.deezer.com/search"
ITUNES_SEARCH = "https://itunes.apple.com/search"
QUOTA_ERROR = 4  # Deezer returns HTTP 200 with this error code when rate limited
ITUNES_MIN_INTERVAL = 3.1  # iTunes Search allows roughly 20 requests per minute

_FEATURING = re.compile(r"\s*[\(\[](?:feat\.?|ft\.?|with)\s[^\)\]]*[\)\]]", re.I)
_DASH_SUFFIX = re.compile(r"\s+-\s+.*$")  # " - 1998 Remaster", " - Sped Up", " - Odd Mob Club Mix"
_itunes_lock = threading.Lock()
_itunes_last_call = 0.0


def clean_title(title):
    """Spotify title without "(with …)"/"(feat. …)" or a " - …" suffix, which Deezer and iTunes titles often lack."""
    cleaned = _DASH_SUFFIX.sub("", _FEATURING.sub("", title)).strip()
    return cleaned or title


def _artist_matches(wanted, found):
    wanted, found = wanted.lower(), (found or "").lower()
    return bool(found) and (found in wanted or wanted in found)


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


def find_deezer_preview(track_name, artist_name, *, session=None, sleep=time.sleep):
    """Return a Deezer 30 s preview URL for the song, or None."""
    track = track_name.replace('"', "")
    artist = artist_name.replace('"', "")

    for item in _search(f'artist:"{artist}" track:"{track}"', session, sleep):
        if item.get("preview"):
            return item["preview"]

    # Loose search can return covers/other songs, so require the artist to match.
    for item in _search(f"{track} {artist}", session, sleep):
        if item.get("preview") and _artist_matches(artist, (item.get("artist") or {}).get("name")):
            return item["preview"]
    return None


def find_itunes_preview(track_name, artist_name, *, session=None, sleep=time.sleep, clock=time.monotonic):
    """Return an iTunes 30 s preview URL (.m4a) for the song, or None."""
    global _itunes_last_call
    with _itunes_lock:
        wait = ITUNES_MIN_INTERVAL - (clock() - _itunes_last_call)
        if wait > 0:
            sleep(wait)
        resp = get_with_retry(ITUNES_SEARCH, params={"term": f"{artist_name} {track_name}", "entity": "song", "limit": 10},
                              session=session, sleep=sleep)
        _itunes_last_call = clock()
    if resp is None:
        return None
    for item in (resp.json() or {}).get("results", []):
        if item.get("previewUrl") and _artist_matches(artist_name, item.get("artistName")):
            return item["previewUrl"]
    return None


def find_preview(track_name, artist_name, *, session=None, sleep=time.sleep):
    """Deezer, then Deezer with a cleaned title, then iTunes. Returns a preview URL or None."""
    url = find_deezer_preview(track_name, artist_name, session=session, sleep=sleep)
    cleaned = clean_title(track_name)
    if not url and cleaned != track_name:
        url = find_deezer_preview(cleaned, artist_name, session=session, sleep=sleep)
    return url or find_itunes_preview(cleaned, artist_name, session=session, sleep=sleep)
