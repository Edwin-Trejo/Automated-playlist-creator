import pytest

from src.selector import ESTIMATE_PENALTY, MIN_SCORE, Percentiles, mood_score, select_tracks


def t(sid, **kw):
    base = {"spotify_id": sid, "name": sid, "genre": "pop", "genre_conf": 0.5, "danceability": 0.5,
            "energy": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120.0}
    base.update(kw)
    return base


def library(n=11):
    """Songs spread evenly from all-low (s0) to all-high (s10) features."""
    return [t(f"s{i}", danceability=i / 10, energy=i / 10, valence=i / 10, acousticness=i / 10) for i in range(n)]


def ids(tracks):
    return [x["spotify_id"] for x in tracks]


def test_percentiles():
    pct = Percentiles(library())
    assert pct("valence", 0.0) == 0.0
    assert pct("valence", 1.0) == 1.0
    assert pct("valence", 0.5) == pytest.approx(0.5)
    assert pct("valence", 5.0) == 1.0  # above every reference song


def test_sad_ranks_relative_to_the_library_even_when_nothing_is_absolutely_low():
    # A dance-heavy library: nothing is below 0.6, but the lowest songs are still the "saddest".
    songs = [t(f"s{i}", danceability=0.6 + i / 40, energy=0.6 + i / 40, valence=0.6 + i / 40) for i in range(11)]
    out = select_tracks(songs, "sad", size=50)
    assert ids(out)[:3] == ["s0", "s1", "s2"]
    assert all(x["score"] >= MIN_SCORE for x in out)
    assert "s10" not in ids(out)


def test_happy_is_the_opposite_end():
    assert ids(select_tracks(library(), "happy", size=3)) == ["s10", "s9", "s8"]


def test_reference_sets_the_scale():
    # Against a reference of much sadder songs, this song is "happy", not "sad".
    song = [t("x", danceability=0.5, energy=0.5, valence=0.5)]
    sadder = [t(f"r{i}", danceability=0.1, energy=0.1, valence=0.1) for i in range(5)]
    assert select_tracks(song, "sad", reference=sadder + song) == []
    assert ids(select_tracks(song, "happy", reference=sadder + song)) == ["x"]


def test_dance_tempo_counts_half_time():
    pct = Percentiles(library())
    weights = {"tempo_near": 1.0}
    assert mood_score(t("a", tempo=122.0), weights, pct) == 1.0
    assert mood_score(t("half", tempo=61.0), weights, pct) == 1.0
    assert mood_score(t("far", tempo=170.0), weights, pct) == 0.0


def test_missing_feature_is_excluded_from_mood_playlists():
    songs = library() + [t("none", valence=None, danceability=0.0, energy=0.0)]
    assert "none" not in ids(select_tracks(songs, "sad", size=50))


def test_estimated_songs_are_penalized():
    songs = library() + [t("est", danceability=0.0, energy=0.0, valence=0.0, analysis_source="artist")]
    out = {x["spotify_id"]: x["score"] for x in select_tracks(songs, "sad", size=50)}
    assert out["est"] == pytest.approx(out["s0"] * ESTIMATE_PENALTY, abs=1e-3)


def test_genre_playlist_keeps_genre_and_ranks_by_confidence():
    songs = [t("a", genre="Rock", genre_conf=0.4), t("b", genre="rock", genre_conf=0.9), t("c", genre="Pop")]
    assert ids(select_tracks(songs, genres=["Rock"])) == ["b", "a"]


def test_size_limits_results():
    assert len(select_tracks([t(str(i)) for i in range(10)], genres=["pop"], size=3)) == 3
