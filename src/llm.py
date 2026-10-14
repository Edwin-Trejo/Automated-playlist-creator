import json
import re
from datetime import date

import requests

PROMPT = """You name Spotify playlists.
Mood: {mood}
Top genres: {genres}
Featured artists: {artists}
Reply with JSON only, exactly this shape:
{{"name": "<catchy playlist name, at most 6 words>", "description": "<one or two sentences, under 250 characters>"}}"""


def _clean(text, limit):
    return " ".join(str(text).split())[:limit]


def parse_llm_json(text):
    try:
        data = json.loads(text, strict=False)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0), strict=False)
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    name = _clean(data.get("name", ""), 100)
    if not name:
        return None
    return {"name": name, "description": _clean(data.get("description", ""), 300)}


def template_text(mood, genres, today):
    label = (mood or "custom").title()
    featuring = f", featuring {', '.join(genres[:3])}" if genres else ""
    return {
        "name": f"{label} Mix · {today:%b %Y}",
        "description": f"{label} picks from your Liked Songs{featuring}. Made by Automated Playlist Creator.",
    }


def generate_playlist_text(mood, genres, artists, *, base_url="http://localhost:11434",
                           model="llama3.2:3b", timeout=60, post=None, today=None):
    post = post or requests.post
    today = today or date.today()
    prompt = PROMPT.format(mood=mood or "custom mix", genres=", ".join(genres) or "mixed",
                           artists=", ".join(artists) or "various")
    try:
        resp = post(f"{base_url}/api/generate", json={
            "model": model, "prompt": prompt, "format": "json", "stream": False,
            "options": {"temperature": 0.8},
        }, timeout=timeout)
        resp.raise_for_status()
        parsed = parse_llm_json(resp.json().get("response", ""))
    except (requests.RequestException, ValueError, AttributeError):
        parsed = None
    if parsed:
        return {**parsed, "source": "ollama"}
    return {**template_text(mood, genres, today), "source": "template"}
