"""Mood recipes: how much each feature counts, and in which direction.

Songs are ranked by percentile within the analyzed library, not by fixed thresholds. The models'
predictions cluster in the middle (and a dance-heavy library has few "low energy" songs), so fixed
ranges like "valence below 0.35" matched almost nothing. A positive weight means "higher is better",
negative means "lower is better". "tempo_near" scores closeness to a BPM instead of a direction.

lyric_* features are what a song's lyrics mean (0..1, from src/ml/lyrics_mood.py). They are ranked as
percentiles too: the lyrics model's scores are squeezed toward the middle (a clear heartbreak song gets ~0.7),
so raw scores would mostly pull songs with lyrics down. A song without lyric scores is ranked by its audio
weights alone. Lyrics count most for sad (heartbreak over a party beat is still sad); the happy and calm lyric
models are less accurate.
"""

MOODS = {
    "happy": {"valence": 0.4, "energy": 0.2, "danceability": 0.2, "lyric_happy": 0.2},
    "sad": {"valence": -0.3, "energy": -0.18, "danceability": -0.12, "lyric_sad": 0.4},
    "dance": {"danceability": 0.5, "energy": 0.3, "tempo_near": 0.2},
    "chill": {"energy": -0.4, "acousticness": 0.24, "danceability": -0.16, "lyric_calm": 0.2},
    "energetic": {"energy": 0.6, "tempo": 0.4},
}

DANCE_TEMPO = 122.0  # BPM that "tempo_near" scores best
TEMPO_TOLERANCE = 40.0  # BPM away from DANCE_TEMPO that scores 0

MOOD_FEATURES = ("danceability", "energy", "valence", "acousticness", "tempo")
LYRIC_FEATURES = ("lyric_sad", "lyric_happy", "lyric_calm")


def mood_weights(mood):
    if mood not in MOODS:
        raise ValueError(f"Unknown mood: {mood}")
    return MOODS[mood]
