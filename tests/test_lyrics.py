import itertools

import pytest

from src import lyrics
from src.lyrics import BACKOFF, LRCLIB_COOLDOWN, OVH_BACKOFF, LyricsUnavailable, find_lyrics
from tests.fakes import FakeResp, FakeSession

NO_SLEEP = lambda _: None
OVH_MISS = FakeResp({"error": "No lyrics found"}, status=404)


@pytest.fixture(autouse=True)
def lrclib_not_paused(monkeypatch):
    monkeypatch.setattr(lyrics, "_lrclib_paused_until", 0.0)


def run(responses, session=None, clock=None, **kw):
    s = session or FakeSession(responses)
    clock = clock or itertools.count(1000, 1000).__next__  # always "long after" the last call: no pacing waits
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


def test_without_album_searches_lrclib_then_lyrics_ovh():
    out, s = run([FakeResp([]), OVH_MISS], name="Song - Remix", artist="Bad Bunny")
    assert out == {"lyrics": None, "instrumental": False}
    assert [c["url"].split("/")[2] for c in s.calls] == ["lrclib.net", "api.lyrics.ovh"]
    assert s.calls[1]["url"].endswith("/v1/Bad%20Bunny/Song")


def test_lyrics_ovh_fills_songs_lrclib_lacks():
    ovh = FakeResp({"lyrics": "Paroles de la chanson Efecto par Bad Bunny\r\nNo sé si es casualidad"})
    out, _ = run([FakeResp([]), ovh], name="Efecto", artist="Bad Bunny")
    assert out == {"lyrics": "No sé si es casualidad", "instrumental": False}  # site header line removed


def test_instrumental_does_not_ask_lyrics_ovh():
    out, s = run([FakeResp([{"plainLyrics": None, "instrumental": True}])], name="Song", artist="A")
    assert out == {"lyrics": None, "instrumental": True} and len(s.calls) == 1


def test_rate_limit_is_retried():
    waits = []
    out, _ = run([FakeResp(None, status=429, headers={"Retry-After": "3"}), FakeResp([{"plainLyrics": "ok"}])],
                 name="Song", artist="A", sleep=waits.append)
    assert out["lyrics"] == "ok" and 3.0 in waits


def test_busy_lrclib_falls_back_to_lyrics_ovh_and_is_skipped_for_a_while():
    t = [0.0]
    clock = lambda: t[0]
    sleep = lambda s: t.__setitem__(0, t[0] + s)
    busy = [FakeResp(None, status=429)] * (len(BACKOFF) + 1)
    s = FakeSession([*busy, FakeResp({"lyrics": "first"}), FakeResp({"lyrics": "second"})])
    first, _ = run(None, session=s, clock=clock, sleep=sleep, name="One", artist="A")
    second, _ = run(None, session=s, clock=clock, sleep=sleep, name="Two", artist="A")
    assert (first["lyrics"], second["lyrics"]) == ("first", "second")
    assert s.calls[-1]["url"].startswith("https://api.lyrics.ovh") and len(s.calls) == len(busy) + 2  # LRCLIB skipped
    t[0] += LRCLIB_COOLDOWN
    s.responses = [FakeResp([{"plainLyrics": "back"}])]
    assert run(None, session=s, clock=clock, sleep=sleep, name="Three", artist="A")[0]["lyrics"] == "back"


def test_both_sites_failing_raises_so_the_song_is_tried_next_sync():
    with pytest.raises(LyricsUnavailable):
        run([FakeResp(None, status=503)] * (len(BACKOFF) + 1 + len(OVH_BACKOFF) + 1), name="Song", artist="A")
