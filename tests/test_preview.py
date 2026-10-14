from src.audio.preview import find_preview
from tests.fakes import FakeResp, FakeSession

NO_SLEEP = lambda _: None


def test_strict_query_hit():
    s = FakeSession([FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "Daft Punk"}}]})])
    assert find_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) == "http://p/1.mp3"
    assert s.calls[0]["params"]["q"] == 'artist:"Daft Punk" track:"One More Time"'


def test_falls_back_to_loose_query_with_artist_match():
    s = FakeSession([
        FakeResp({"data": []}),
        FakeResp({"data": [
            {"preview": "http://p/wrong.mp3", "artist": {"name": "Someone Else"}},
            {"preview": "http://p/right.mp3", "artist": {"name": "Daft Punk"}},
        ]}),
    ])
    assert find_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) == "http://p/right.mp3"


def test_loose_query_rejects_other_artists():
    s = FakeSession([
        FakeResp({"data": []}),
        FakeResp({"data": [{"preview": "http://p/wrong.mp3", "artist": {"name": "Someone Else"}}]}),
    ])
    assert find_preview("One More Time", "Daft Punk", session=s, sleep=NO_SLEEP) is None


def test_retries_on_deezer_quota_error():
    waits = []
    s = FakeSession([
        FakeResp({"error": {"code": 4, "message": "Quota limit exceeded"}}),
        FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "A"}}]}),
    ])
    assert find_preview("T", "A", session=s, sleep=waits.append) == "http://p/1.mp3"
    assert waits


def test_quotes_in_names_are_stripped():
    s = FakeSession([FakeResp({"data": [{"preview": "http://p/1.mp3", "artist": {"name": "A"}}]})])
    find_preview('Say "Hi"', "A", session=s, sleep=NO_SLEEP)
    assert s.calls[0]["params"]["q"] == 'artist:"A" track:"Say Hi"'
