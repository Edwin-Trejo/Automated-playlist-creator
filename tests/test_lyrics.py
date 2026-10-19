import itertools

import pytest

from src.lyrics import BACKOFF, LyricsUnavailable, find_lyrics
from tests.fakes import FakeResp, FakeSession

NO_SLEEP = lambda _: None


def run(responses, **kw):
    s = FakeSession(responses)
    clock = itertools.count(1000, 1000).__next__  # always "long after" the last call: no pacing waits
    return find_lyrics(session=s, sleep=kw.pop("sleep", NO_SLEEP), clock=clock, **kw), s


def test_exact_lookup_with_album_and_length():
    out, s = run([FakeResp({"plainLyrics": "la la", "instrumental": False})],
                 name="Song (feat. X)", artist="A", album="Album", duration_ms=201_400)
    assert out == {"lyrics": "la la", "instrumental": False}
    assert s.calls[0]["url"].endswith("/get")
    assert s.calls[0]["params"] == {"track_name": "Song", "artist_name": "A", "album_name": "Album", "duration": 201}
    assert "User-Agent" in s.calls[0]["headers"]


def test_falls_back_to_search_when_exact_lookup_misses():
    out, s = run([FakeResp(None, status=404),
                  FakeResp([{"plainLyrics": None, "instrumental": False}, {"plainLyrics": "found it"}])],
                 name="Song", artist="A", album="Album", duration_ms=1000)
    assert out["lyrics"] == "found it" and s.calls[1]["url"].endswith("/search")


def test_without_album_only_searches():
    out, s = run([FakeResp([])], name="Song", artist="A")
    assert out == {"lyrics": None, "instrumental": False} and len(s.calls) == 1


def test_instrumental():
    out, _ = run([FakeResp([{"plainLyrics": None, "instrumental": True}])], name="Song", artist="A")
    assert out == {"lyrics": None, "instrumental": True}


def test_rate_limit_is_retried():
    waits = []
    out, _ = run([FakeResp(None, status=429, headers={"Retry-After": "3"}), FakeResp([{"plainLyrics": "ok"}])],
                 name="Song", artist="A", sleep=waits.append)
    assert out["lyrics"] == "ok" and 3.0 in waits


def test_keeps_failing_raises_so_the_song_is_tried_next_sync():
    with pytest.raises(LyricsUnavailable):
        run([FakeResp(None, status=503)] * (len(BACKOFF) + 1), name="Song", artist="A")
