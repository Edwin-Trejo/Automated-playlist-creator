import requests
import urllib.parse

def get_deezer_preview(song_name: str, artist_name: str):
    """Search Deezer for a song and return its 30-second preview URL."""
    # Build query: "song_name artist_name"
    query = f"{song_name} {artist_name}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.deezer.com/search?q={encoded_query}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("data", [])

        if not results:
            return None

        # Pick the top result that has a preview link
        for track in results:
            preview = track.get("preview")
            if preview:
                return preview

        return None  # if none have preview links
    except requests.RequestException as e:
        print(f"⚠️  Deezer API error for '{query}': {e}")
        return None
