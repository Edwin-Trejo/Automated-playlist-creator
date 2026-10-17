from src.playlists import playlist_description, playlist_key, playlist_name


def test_mood_only():
    assert playlist_key("happy") == "happy"
    assert playlist_name("happy") == "Happy"
    assert playlist_key("happy", []) == playlist_key("happy", None) == "happy"


def test_mood_with_genres_is_sorted_and_case_insensitive():
    assert playlist_name("happy", ["Rock", "Reggaeton / Urbano"]) == "Happy · Reggaeton / Urbano · Rock"
    assert playlist_key("happy", ["Rock", "Reggaeton / Urbano"]) == playlist_key("happy", ["reggaeton / urbano", "rock"])
    assert playlist_key("happy", ["Rock"]) == "happy|rock"


def test_no_mood_is_custom():
    assert playlist_key(None) == "custom"
    assert playlist_name(None, ["Pop"]) == "Custom mix · Pop"


def test_description():
    assert playlist_description("sad") == "Sad songs from your library, picked by Automated Playlist Creator."
    assert "(House, Techno)" in playlist_description("dance", ["Techno", "House"])
