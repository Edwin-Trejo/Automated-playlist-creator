"""Fixed playlist names: the mood, plus the genres when a genre filter is used."""


def _genres(genres):
    return sorted({g.strip() for g in genres or [] if g and g.strip()}, key=str.lower)


def playlist_key(mood, genres=None):
    """Identifies "the" playlist for a mood + genre filter, e.g. "happy" or "happy|reggaeton / urbano"."""
    return "|".join([mood or "custom", *(g.lower() for g in _genres(genres))])


def playlist_name(mood, genres=None):
    label = mood.title() if mood else "Custom mix"
    return " · ".join([label, *_genres(genres)])


def playlist_description(mood, genres=None):
    genres = _genres(genres)
    label = f"{mood.title()} songs" if mood else "A custom mix"
    featuring = f" ({', '.join(genres)})" if genres else ""
    return f"{label}{featuring} from your library, picked by Automated Playlist Creator."
