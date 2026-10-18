"""Fixed playlist names: a mood playlist is named after the mood, a genre playlist after its genres."""


def _genres(genres):
    return sorted({g.strip() for g in genres or [] if g and g.strip()}, key=str.lower)


def playlist_key(mood=None, genres=None):
    """Identifies "the" playlist for a mood ("happy") or a set of genres ("genre|house|techno")."""
    if mood:
        return mood
    return "|".join(["genre", *(g.lower() for g in _genres(genres))])


def playlist_name(mood=None, genres=None):
    if mood:
        return mood.title()
    return " · ".join(_genres(genres)) or "Custom mix"


def playlist_description(mood=None, genres=None):
    label = mood.title() if mood else ", ".join(_genres(genres))
    return f"{label} songs from your library, picked by Automated Playlist Creator."
