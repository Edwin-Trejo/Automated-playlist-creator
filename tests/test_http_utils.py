import requests

from src.http_utils import get_with_retry
from tests.fakes import FakeResp, FakeSession


def test_returns_ok_response():
    s = FakeSession([FakeResp({"a": 1})])
    assert get_with_retry("http://x", session=s, sleep=lambda _: None).json() == {"a": 1}


def test_honors_retry_after_on_429():
    waits = []
    s = FakeSession([FakeResp(status=429, headers={"Retry-After": "2"}), FakeResp({"ok": True})])
    resp = get_with_retry("http://x", session=s, sleep=waits.append)
    assert resp.json() == {"ok": True}
    assert waits == [2.0]


def test_client_error_returns_none_without_retry():
    s = FakeSession([FakeResp(status=404)])
    assert get_with_retry("http://x", session=s, sleep=lambda _: None) is None
    assert len(s.calls) == 1


def test_network_errors_exhaust_retries():
    s = FakeSession([requests.ConnectionError()] * 4)
    assert get_with_retry("http://x", session=s, retries=3, sleep=lambda _: None) is None
    assert len(s.calls) == 4
