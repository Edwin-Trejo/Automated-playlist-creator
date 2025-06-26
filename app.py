from flask import Flask, render_template, request, redirect
import spotify_api
from genre_classifier import predict_genre
from playlist_manager import create_playlist_if_not_exists, add_track_to_playlist
from logger import logger

app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Index page accessed")
    return "Smart Playlist Sorter is running!"

@app.route('/sort', methods=['GET'])
def sort_liked_songs():
    logger.info("Starting /sort route")
    liked_tracks = spotify_api.get_liked_tracks()
    user_id = spotify_api.get_current_user_id()

    for track in liked_tracks:
        logger.debug(f"Analyzing: {track['name']} by {', '.join(track['artists'])}")
        #print(f"Track: {track['name']} | ID: {track['id']}")

        features = spotify_api.get_audio_features(track['id'])
        if not features:
            logger.warning(f"No features found for {track['name']} - skipping.")
            continue

        #genre = predict_genre(features) TODO: REPLACE WITH ML
        genre = "ROCK"
        logger.debug(f"Predicted genre: {genre}")
        playlist_id = create_playlist_if_not_exists(user_id, genre)
        add_track_to_playlist(playlist_id, track['id'])
       
        logger.success(f"Added to '{genre}' playlist.\n")

    return "Liked songs sorted into playlists."

if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(debug=True)
