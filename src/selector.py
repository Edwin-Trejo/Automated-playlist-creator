from bisect import bisect_left, bisect_right

from src.moods import DANCE_TEMPO, MOOD_FEATURES, TEMPO_TOLERANCE, mood_weights

ESTIMATE_PENALTY = 0.8  # songs estimated from their artist (never heard) rank below analyzed songs
MIN_SCORE = 0.7  # a mood playlist keeps songs at least this typical of the mood (1 = the most extreme in the library)


def _tempo_candidates(tempo):
    # Beat trackers often report half or double time; accept the likely octave too.
    candidates = [tempo]
    if tempo < 75:
        candidates.append(tempo * 2)
    if tempo > 190:
        candidates.append(tempo / 2)
    return candidates


class Percentiles:
    """Where a value sits among the reference songs: 0 = lowest, 1 = highest."""

    def __init__(self, reference):
        self.sorted = {f: sorted(t[f] for t in reference if t.get(f) is not None) for f in MOOD_FEATURES}

    def __call__(self, feature, value):
        values = self.sorted[feature]
        if len(values) < 2:
            return 0.5
        # Average rank of ties, scaled to 0..1.
        rank = (bisect_left(values, value) + bisect_right(values, value) - 1) / 2
        return min(max(rank / (len(values) - 1), 0.0), 1.0)


def _tempo_near(tempo):
    best = min(abs(t - DANCE_TEMPO) for t in _tempo_candidates(tempo))
    return max(0.0, 1 - best / TEMPO_TOLERANCE)


def mood_score(track, weights, pct):
    """Weighted mean of how far toward the mood each feature is, 0..1. None if a feature is missing."""
    total = 0.0
    for feature, weight in weights.items():
        value = track.get("tempo" if feature == "tempo_near" else feature)
        if value is None:
            return None
        if feature == "tempo_near":
            part = _tempo_near(value)
        else:
            p = pct(feature, value)
            part = p if weight > 0 else 1 - p
        total += abs(weight) * part
    return total / sum(abs(w) for w in weights.values())


def select_tracks(tracks, mood=None, genres=None, size=25, reference=None):
    """A mood playlist (ranked against `reference`, default `tracks`) or a genre playlist (ranked by genre confidence)."""
    wanted = {g.lower() for g in genres} if genres else None
    weights = mood_weights(mood) if mood else None
    pct = Percentiles(tracks if reference is None else reference) if weights else None
    scored = []
    for track in tracks:
        if wanted and (track.get("genre") or "").lower() not in wanted:
            continue
        if weights:
            score = mood_score(track, weights, pct)
            if score is None or score < MIN_SCORE:
                continue
        else:
            score = track.get("genre_conf") or 0.0
        if track.get("analysis_source") == "artist":
            score *= ESTIMATE_PENALTY
        scored.append({**track, "score": round(score, 4)})
    scored.sort(key=lambda x: (-x["score"], -(x.get("genre_conf") or 0), x["name"]))
    return scored[:size]
