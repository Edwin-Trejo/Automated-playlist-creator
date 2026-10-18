from src.playlists import playlist_description, playlist_key, playlist_name


def test_mood_playlist():
    assert playlist_key("happy") == "happy"
    assert playlist_name("happy") == "Happy"
    assert playlist_key("happy", ["Rock"]) == "happy"  # a mood playlist never has a genre filter


def test_genre_playlist_is_sorted_and_case_insensitive():
    assert playlist_name(None, ["Techno", "House"]) == "House · Techno"
    assert playlist_key(None, ["Techno", "House"]) == playlist_key(None, ["house", "techno"]) == "genre|house|techno"
    assert playlist_name(None, ["Reggaeton / Urbano"]) == "Reggaeton / Urbano"


def test_description():
    assert playlist_description("sad") == "Sad songs from your library, picked by Automated Playlist Creator."
    assert playlist_description(None, ["Techno", "House"]).startswith("House, Techno songs")
