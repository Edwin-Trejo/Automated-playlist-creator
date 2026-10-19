import pytest

from src.moods import LYRIC_FEATURES, MOOD_FEATURES, MOODS, mood_weights


def test_presets_exist():
    assert set(MOODS) == {"happy", "sad", "dance", "chill", "energetic"}


def test_weights_use_known_features():
    for weights in MOODS.values():
        assert set(weights) <= {*MOOD_FEATURES, *LYRIC_FEATURES, "tempo_near"}
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)


def test_sad_prefers_low_valence_and_happy_high():
    assert mood_weights("sad")["valence"] < 0 < mood_weights("happy")["valence"]


def test_unknown_mood_raises():
    with pytest.raises(ValueError):
        mood_weights("angry")
