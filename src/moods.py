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


# A slider at +1/-1 moves a feature's range by this much; a range never gets narrower than MIN_WIDTH.
MAX_SHIFT = {"danceability": 0.4, "energy": 0.4, "valence": 0.4, "tempo": 40.0}
MIN_WIDTH = {"danceability": 0.15, "energy": 0.15, "valence": 0.15, "tempo": 25.0}


def _shift(feature, lo, hi, amount):
    full_lo, full_hi = FULL_RANGE[feature]
    d = amount * MAX_SHIFT[feature]
    lo = max(full_lo, min(lo + d, full_hi - MIN_WIDTH[feature]))
    hi = min(full_hi, max(hi + d, full_lo + MIN_WIDTH[feature]))
    return (lo, hi)


def build_targets(mood=None, adjust=None):
    """Start from full ranges, apply the mood preset, then shift features by the sliders.

    `adjust` maps a feature to a value in [-1, 1]: negative means less (slower), positive means more (faster).
    """
    if mood is not None and mood not in MOODS:
        raise ValueError(f"Unknown mood: {mood}")
    targets = dict(FULL_RANGE)
    targets.update(MOODS.get(mood, {}))
    for feature, amount in (adjust or {}).items():
        if feature not in FULL_RANGE:
            raise ValueError(f"Unknown feature: {feature}")
        amount = max(-1.0, min(float(amount), 1.0))
        if amount:
            targets[feature] = _shift(feature, *targets[feature], amount)
    return targets
