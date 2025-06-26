def predict_genre(features):
    """
    Predicts a genre based on Spotify audio features using basic rule-based logic.
    You can replace this with an ML model later.

    Parameters:
        features (dict): A dict containing Spotify's audio features.

    Returns:
        str: Predicted genre label (e.g., "Rock", "Pop", "Hip-Hop").
    """
    danceability = features.get('danceability', 0)
    energy = features.get('energy', 0)
    valence = features.get('valence', 0)
    speechiness = features.get('speechiness', 0)
    acousticness = features.get('acousticness', 0)

    # Very simple logic – just to start
    if speechiness > 0.4:
        return "Hip-Hop"
    elif energy > 0.7 and danceability > 0.6:
        return "Pop"
    elif acousticness > 0.7 and valence > 0.5:
        return "Folk"
    elif energy > 0.6 and valence < 0.4:
        return "Rock"
    elif energy < 0.3 and acousticness > 0.6:
        return "Classical"
    else:
        return "Indie"
