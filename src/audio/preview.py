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
