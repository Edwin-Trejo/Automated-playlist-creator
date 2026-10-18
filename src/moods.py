"""Mood recipes: how much each feature counts, and in which direction.

Songs are ranked by percentile within the analyzed library, not by fixed thresholds. The models'
predictions cluster in the middle (and a dance-heavy library has few "low energy" songs), so fixed
ranges like "valence below 0.35" matched almost nothing. A positive weight means "higher is better",
negative means "lower is better". "tempo_near" scores closeness to a BPM instead of a direction.
"""

MOODS = {
    "happy": {"valence": 0.5, "energy": 0.25, "danceability": 0.25},
    "sad": {"valence": -0.5, "energy": -0.3, "danceability": -0.2},
    "dance": {"danceability": 0.5, "energy": 0.3, "tempo_near": 0.2},
    "chill": {"energy": -0.5, "acousticness": 0.3, "danceability": -0.2},
    "energetic": {"energy": 0.6, "tempo": 0.4},
}

DANCE_TEMPO = 122.0  # BPM that "tempo_near" scores best
TEMPO_TOLERANCE = 40.0  # BPM away from DANCE_TEMPO that scores 0

MOOD_FEATURES = ("danceability", "energy", "valence", "acousticness", "tempo")


def mood_weights(mood):
    if mood not in MOODS:
        raise ValueError(f"Unknown mood: {mood}")
    return MOODS[mood]
