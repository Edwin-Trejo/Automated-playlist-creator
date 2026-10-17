from src.audio import preview
from src.audio.preview import clean_title, find_deezer_preview, find_itunes_preview
from tests.fakes import FakeResp, FakeSession

NO_SLEEP = lambda _: None


def test_strict_query_hit():
    s = FakeSession([FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "Daft Punk"}}]})])
    assert find_deezer_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) == "http://p/1.mp3"
    assert s.calls[0]["params"]["q"] == 'artist:"Daft Punk" track:"One More Time"'


def test_falls_back_to_loose_query_with_artist_match():
    s = FakeSession([
        FakeResp({"data": []}),
        FakeResp({"data": [
            {"preview": "http://p/wrong.mp3", "artist": {"name": "Someone Else"}},
            {"preview": "http://p/right.mp3", "artist": {"name": "Daft Punk"}},
        ]}),
    ])
    assert find_deezer_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) == "http://p/right.mp3"


def test_loose_query_rejects_other_artists():
    s = FakeSession([
        FakeResp({"data": []}),
        FakeResp({"data": [{"preview": "http://p/wrong.mp3", "artist": {"name": "Someone Else"}}]}),
    ])
    assert find_deezer_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) is None


def test_retries_on_deezer_quota_error():
    waits = []
    s = FakeSession([
        FakeResp({"error": {"code": 4, "message": "Quota limit exceeded"}}),
        FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "A"}}]}),
    ])
    assert find_deezer_preview("T", "A", session=s, sleep=waits.append) == "http://p/1.mp3"
    assert waits


def test_clean_title():
    assert clean_title("2 Minutes to Midnight - 1998 Remaster") == "2 Minutes to Midnight"
    assert clean_title("Labios Mordidos (with KAROL G)") == "Labios Mordidos"
    assert clean_title("La Temperatura (feat. Eli Palacios)") == "La Temperatura"
    assert clean_title("Mirada (Original mix)") == "Mirada (Original mix)"
    assert clean_title("Plain") == "Plain"


def test_itunes_requires_matching_artist():
    s = FakeSession([FakeResp({"results": [
        {"previewUrl": "http://a/wrong.m4a", "artistName": "Cover Band"},
        {"previewUrl": "http://a/right.m4a", "artistName": "J Balvin"},
    ]})])
    assert find_itunes_preview("Azul", "J Balvin", session=s, sleep=NO_SLEEP, clock=lambda: 1e9) == "http://a/right.m4a"
    assert s.calls[0]["params"]["term"] == "J Balvin Azul"


def test_find_preview_tries_cleaned_title_then_itunes(monkeypatch):
    calls = []
    monkeypatch.setattr(preview, "find_deezer_preview", lambda t, a, **kw: calls.append(("deezer", t)) or None)
    monkeypatch.setattr(preview, "find_itunes_preview", lambda t, a, **kw: calls.append(("itunes", t)) or "http://a/1.m4a")
    assert preview.find_preview("Azul - Remix", "J Balvin") == "http://a/1.m4a"
    assert calls == [("deezer", "Azul - Remix"), ("deezer", "Azul"), ("itunes", "Azul")]


def test_find_preview_stops_at_first_hit(monkeypatch):
    monkeypatch.setattr(preview, "find_deezer_preview", lambda t, a, **kw: "http://d/1.mp3")
    monkeypatch.setattr(preview, "find_itunes_preview", lambda t, a, **kw: 1 / 0)
    assert preview.find_preview("Song", "A") == "http://d/1.mp3"


def test_quotes_in_names_are_stripped():
    s = FakeSession([FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "A"}}]})])
    find_deezer_preview('Say "Hi"', "A", session=s, sleep=NO_SLEEP)
    assert s.calls[0]["params"]["q"] == 'artist:"A" track:"Say Hi"'
