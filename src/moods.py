FULL_RANGE = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (40.0, 220.0),
}

MOODS = {
    "happy": {"danceability": (0.5, 1.0), "energy": (0.5, 1.0), "valence": (0.6, 1.0)},
    "sad": {"danceability": (0.0, 0.5), "energy": (0.0, 0.5), "valence": (0.0, 0.35), "tempo": (40.0, 110.0)},
    "dance": {"danceability": (0.65, 1.0), "energy": (0.6, 1.0), "tempo": (110.0, 135.0)},
    "chill": {"danceability": (0.0, 0.6), "energy": (0.0, 0.45), "valence": (0.3, 0.7), "tempo": (40.0, 115.0)},
    "energetic": {"energy": (0.7, 1.0), "tempo": (120.0, 220.0)},
}


def build_targets(mood=None, overrides=None):
    """Start from full ranges, apply the mood preset, then slider overrides."""
    if mood is not None and mood not in MOODS:
        raise ValueError(f"Unknown mood: {mood}")
    targets = dict(FULL_RANGE)
    targets.update(MOODS.get(mood, {}))
    for feature, (lo, hi) in (overrides or {}).items():
        if feature not in FULL_RANGE:
            raise ValueError(f"Unknown feature: {feature}")
        lo, hi = sorted((float(lo), float(hi)))
        full_lo, full_hi = FULL_RANGE[feature]
        targets[feature] = (max(lo, full_lo), min(hi, full_hi))
    return targets
