from src.moods import FULL_RANGE, build_targets
from src.selector import select_tracks


def t(sid, **kw):
    base = {"spotify_id": sid, "name": sid, "genre": "pop", "genre_conf": 0.5,
            "danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0}
    base.update(kw)
    return base


def test_filters_by_ranges_and_ranks_by_closeness():
    tracks = [t("edge", energy=0.71), t("center", energy=0.85), t("low", energy=0.3)]
    out = select_tracks(tracks, {**FULL_RANGE, "energy": (0.7, 1.0)})
    assert [x["spotify_id"] for x in out] == ["center", "edge"]
    assert out[0]["score"] > out[1]["score"]


def test_genre_filter_is_case_insensitive():
    tracks = [t("a", genre="rock"), t("b", genre="pop")]
    assert [x["spotify_id"] for x in select_tracks(tracks, build_targets(), genres=["Rock"])] == ["a"]


def test_tracks_missing_a_constrained_feature_are_excluded():
    tracks = [t("a", valence=None), t("b")]
    assert [x["spotify_id"] for x in select_tracks(tracks, {**FULL_RANGE, "valence": (0.4, 0.6)})] == ["b"]


def test_half_time_tempo_counts_as_double():
    tracks = [t("half", tempo=62.0), t("slow", tempo=90.0)]
    out = select_tracks(tracks, {**FULL_RANGE, "tempo": (110.0, 135.0)})
    assert [x["spotify_id"] for x in out] == ["half"]


def test_size_limits_results():
    tracks = [t(str(i)) for i in range(10)]
    assert len(select_tracks(tracks, build_targets(), size=3)) == 3
