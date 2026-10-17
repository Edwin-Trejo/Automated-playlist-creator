import pytest

from src.moods import FULL_RANGE, MOODS, build_targets


def test_presets_exist():
    assert set(MOODS) == {"happy", "sad", "dance", "chill", "energetic"}


def test_no_mood_is_full_range():
    assert build_targets() == FULL_RANGE


def test_mood_fills_unspecified_features_with_full_range():
    t = build_targets("energetic")
    assert t["energy"] == (0.7, 1.0)
    assert t["valence"] == FULL_RANGE["valence"]


def test_zero_adjust_keeps_the_mood():
    assert build_targets("happy", {"valence": 0, "energy": 0}) == build_targets("happy")


def test_adjust_shifts_mood_range():
    assert build_targets("sad", {"energy": 0.5})["energy"] == pytest.approx((0.2, 0.7))
    assert build_targets("dance", {"tempo": -1})["tempo"] == pytest.approx((70.0, 95.0))


def test_adjust_without_mood_drops_the_low_or_high_end():
    assert build_targets(None, {"danceability": 0.5})["danceability"] == pytest.approx((0.2, 1.0))
    assert build_targets(None, {"danceability": -1})["danceability"] == pytest.approx((0.0, 0.6))


def test_adjust_keeps_a_minimum_width_at_the_edges():
    lo, hi = build_targets("happy", {"valence": 1})["valence"]  # happy valence is already 0.6-1.0
    assert hi == 1.0 and lo == pytest.approx(0.85)


def test_adjust_is_clamped_to_plus_minus_one():
    assert build_targets(None, {"energy": 5}) == build_targets(None, {"energy": 1})


def test_unknown_mood_or_feature_or_bad_value_raises():
    with pytest.raises(ValueError):
        build_targets("angry")
    with pytest.raises(ValueError):
        build_targets(None, {"loudness": 0.5})
    with pytest.raises(ValueError):
        build_targets(None, {"energy": "lots"})
