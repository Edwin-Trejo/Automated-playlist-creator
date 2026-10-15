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


def test_overrides_win_are_swapped_and_clamped():
    t = build_targets("happy", {"valence": [0.9, 0.2], "tempo": [0, 500]})
    assert t["valence"] == (0.2, 0.9)
    assert t["tempo"] == (40.0, 220.0)


def test_unknown_mood_or_feature_raises():
    with pytest.raises(ValueError):
        build_targets("angry")
    with pytest.raises(ValueError):
        build_targets(None, {"loudness": [0, 1]})
