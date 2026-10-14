class FakeResp:
    def __init__(self, payload=None, status=200, headers=None, content=b""):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.ok = 200 <= status < 300
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Returns queued responses in order; Exception items are raised."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
