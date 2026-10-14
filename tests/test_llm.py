from datetime import date

import requests

from src.llm import generate_playlist_text, parse_llm_json, template_text
from tests.fakes import FakeResp

TODAY = date(2026, 10, 13)


def test_parse_plain_json():
    assert parse_llm_json('{"name": "Sunny Days", "description": "Bright tunes."}') == {
        "name": "Sunny Days", "description": "Bright tunes."}


def test_parse_json_wrapped_in_prose_and_cleans_whitespace():
    text = 'Sure! Here you go:\n{"name": "Neon\nNights", "description": "x"}\nEnjoy'
    assert parse_llm_json(text)["name"] == "Neon Nights"


def test_parse_truncates_long_fields():
    out = parse_llm_json('{"name": "%s", "description": "%s"}' % ("n" * 300, "d" * 900))
    assert len(out["name"]) == 100 and len(out["description"]) == 300


def test_parse_rejects_garbage_and_empty_name():
    assert parse_llm_json("no json here") is None
    assert parse_llm_json('{"name": "  ", "description": "x"}') is None


def test_template_text():
    out = template_text("happy", ["pop", "disco"], TODAY)
    assert out["name"] == "Happy Mix · Oct 2026"
    assert "pop" in out["description"]


def test_uses_ollama_response():
    calls = []

    def post(url, json, timeout):
        calls.append((url, json))
        return FakeResp({"response": '{"name": "Gym Fuel", "description": "Loud and fast."}'})

    out = generate_playlist_text("energetic", ["metal"], ["A"], post=post, today=TODAY)
    assert out == {"name": "Gym Fuel", "description": "Loud and fast.", "source": "ollama"}
    assert calls[0][0] == "http://localhost:11434/api/generate"
    assert calls[0][1]["format"] == "json" and calls[0][1]["stream"] is False


def test_falls_back_when_ollama_down():
    def post(url, json, timeout):
        raise requests.ConnectionError("refused")

    out = generate_playlist_text("sad", [], [], post=post, today=TODAY)
    assert out["source"] == "template" and out["name"] == "Sad Mix · Oct 2026"


def test_falls_back_on_bad_model_output():
    out = generate_playlist_text(None, [], [], post=lambda url, json, timeout: FakeResp({"response": "oops"}), today=TODAY)
    assert out["source"] == "template" and out["name"] == "Custom Mix · Oct 2026"
