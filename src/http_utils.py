import time

import requests


def get_with_retry(url, *, params=None, session=None, retries=3, backoff=1.0, timeout=10, sleep=time.sleep):
    """GET with retries on 429/5xx/network errors. Returns the response, or None on failure."""
    http = session or requests
    for attempt in range(retries + 1):
        try:
            resp = http.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            resp = None

        if resp is not None and resp.ok:
            return resp
        if resp is not None and resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", backoff * 2 ** attempt))
        elif resp is not None and 400 <= resp.status_code < 500:
            return None
        else:
            wait = backoff * 2 ** attempt

        if attempt < retries:
            sleep(wait)
    return None
