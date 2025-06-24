from flask import Flask, render_template, request, redirect
import spotify_api
from genre_classifier import predict_genre
from playlist_manager import create_playlist_if_not_exists, add_track_to_playlist

app = Flask(__name__)

@app.route('/')
def index():
    return "Smart Playlist Sorter is running!"

@app.route('/sort', methods=['GET'])
def sort_liked_songs():
    liked_tracks = spotify_api.get_liked_tracks()
    user_id = spotify_api.get_current_user_id()
    for track in liked_tracks:
        features = spotify_api.get_audio_features(track['id'])
        genre = predict_genre(features)
        playlist_id = create_playlist_if_not_exists(user_id, genre)
        add_track_to_playlist(playlist_id, track['id'])
    return "Liked songs sorted into playlists."

if __name__ == '__main__':
    app.run(debug=True)
