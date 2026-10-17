from src.moods import FULL_RANGE

ESTIMATE_PENALTY = 0.8  # songs estimated from their artist (never heard) rank below analyzed songs


def _tempo_candidates(tempo):
    # Beat trackers often report half or double time; accept the likely octave too.
    candidates = [tempo]
    if tempo < 75:
        candidates.append(tempo * 2)
    if tempo > 190:
        candidates.append(tempo / 2)
    return candidates


def _distance(value, lo, hi):
    """0 at the range center, 1 at its edges."""
    half = max((hi - lo) / 2, 1e-9)
    return min(abs(value - (lo + hi) / 2) / half, 1.0)


def select_tracks(tracks, targets, genres=None, size=25):
    active = {f: r for f, r in targets.items() if tuple(r) != FULL_RANGE[f]}
    wanted = {g.lower() for g in genres} if genres else None
    scored = []
    for track in tracks:
        if wanted and (track.get("genre") or "").lower() not in wanted:
            continue
        distances = []
        for feature, (lo, hi) in active.items():
            value = track.get(feature)
            if value is None:
                break
            values = _tempo_candidates(value) if feature == "tempo" else [value]
            inside = [v for v in values if lo <= v <= hi]
            if not inside:
                break
            distances.append(min(_distance(v, lo, hi) for v in inside))
        else:
            score = 1 - sum(distances) / len(distances) if distances else 1.0
            if track.get("analysis_source") == "artist":
                score *= ESTIMATE_PENALTY
            scored.append({**track, "score": round(score, 4)})
    scored.sort(key=lambda x: (-x["score"], -(x.get("genre_conf") or 0), x["name"]))
    return scored[:size]
