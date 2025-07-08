from flask import Flask, render_template, request, redirect, jsonify
import spotify_api
from genre_classifier import predict_genre
from logger import logger
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Index page accessed")
    return """
    <h1>Smart Playlist Sorter</h1>
    <p>The application is running!</p>
    <p><a href="/sort">Sort your liked songs into playlists</a></p>
    <p><a href="/status">Check connection status</a></p>
    <p><a href="/debug">Debug information</a></p>
    <p><a href="/clear_cache">Clear token cache (if having auth issues)</a></p>
    <p><a href="/test_audio">Test audio features access</a></p>
    """

@app.route('/clear_cache')
def clear_cache():
    """Clear the Spotify token cache and redirect to re-authenticate."""
    try:
        spotify_api.clear_token_cache()
        return """
        <h2>Token Cache Cleared</h2>
        <p>Your authentication token has been cleared. Please click the link below to re-authenticate:</p>
        <p><a href="/status">Re-authenticate and check status</a></p>
        <p><a href="/">Back to home</a></p>
        """
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return f"Error clearing cache: {str(e)}", 500

@app.route('/test_audio')
def test_audio():
    """Test audio features access."""
    try:
        success = spotify_api.test_audio_features()
        if success:
            return """
            <h2>✅ Audio Features Test Successful</h2>
            <p>Audio features access is working properly!</p>
            <p><a href="/sort">Try sorting your music</a></p>
            <p><a href="/">Back to home</a></p>
            """
        else:
            return """
            <h2>❌ Audio Features Test Failed</h2>
            <p>There's still an issue with audio features access.</p>
            <p><a href="/clear_cache">Try clearing the token cache</a></p>
            <p><a href="/debug">Check debug information</a></p>
            <p><a href="/">Back to home</a></p>
            """
    except Exception as e:
        logger.error(f"Error testing audio features: {e}")
        return f"Error testing audio features: {str(e)}", 500

@app.route('/status')
def status():
    """Check if Spotify connection is working and show detailed info."""
    try:
        user_id = spotify_api.get_current_user_id()
        
        # Get token info to see granted scopes
        token_info = spotify_api.spotify_manager.auth_manager.get_cached_token()
        granted_scopes = token_info.get('scope', 'No scopes found') if token_info else 'No token found'
        
        # Get user profile info
        user_profile = spotify_api.spotify_manager.sp.current_user()
        
        # Test audio features access with improved error handling
        audio_features_status = "❌ Not tested yet"
        try:
            success = spotify_api.test_audio_features()
            if success:
                audio_features_status = "✅ Audio features accessible"
            else:
                audio_features_status = "❌ Audio features access failed"
        except Exception as af_error:
            audio_features_status = f"❌ Audio features error: {str(af_error)}"
        
        status_info = {
            "status": "connected",
            "user_id": user_id,
            "user_display_name": user_profile.get('display_name', 'N/A'),
            "user_country": user_profile.get('country', 'N/A'),
            "granted_scopes": granted_scopes,
            "audio_features_test": audio_features_status,
            "message": "Spotify API connection is working"
        }
        
        # Log detailed info
        logger.info(f"Status check - User: {user_id}")
        logger.info(f"Granted scopes: {granted_scopes}")
        logger.info(f"Audio features test: {audio_features_status}")
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Connection status check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/sort', methods=['GET'])
def sort_liked_songs():
    logger.info("Starting /sort route")
    
    try:
        # Get user info first
        user_id = spotify_api.get_current_user_id()
        logger.info(f"Processing for user: {user_id}")
        
        # Get liked tracks
        logger.info("Fetching liked tracks...")
        liked_tracks = spotify_api.get_liked_tracks(limit=20)  # Limit for testing
        
        if not liked_tracks:
            logger.warning("No liked tracks found")
            return "No liked tracks found to sort."
        
        logger.info(f"Found {len(liked_tracks)} liked tracks to process")
        
        # Process tracks
        processed_count = 0
        error_count = 0
        
        for i, track in enumerate(liked_tracks):
            try:
                logger.debug(f"Processing track {i+1}/{len(liked_tracks)}: {track['name']} by {', '.join(track['artists'])}")
                
                # Get audio features
                features = spotify_api.get_audio_features(track['id'])
                if not features:
                    logger.warning(f"No features found for {track['name']} - skipping.")
                    error_count += 1
                    continue
                
                # Predict genre (using your current rule-based system)
                genre = predict_genre(features)
                logger.debug(f"Predicted genre: {genre}")
                
                # Create playlist if needed
                playlist_id = spotify_api.create_playlist_if_not_exists(user_id, genre)
                
                # Add track to playlist
                success = spotify_api.add_track_to_playlist(playlist_id, track['id'])
                
                if success:
                    processed_count += 1
                    logger.info(f"✓ Added '{track['name']}' to '{genre}' playlist")
                else:
                    error_count += 1
                    logger.error(f"✗ Failed to add '{track['name']}' to playlist")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing track '{track['name']}': {e}")
                logger.debug(traceback.format_exc())
                continue
        
        # Return summary
        summary = f"""
        <h2>Playlist Sorting Complete!</h2>
        <p><strong>Total tracks processed:</strong> {processed_count}</p>
        <p><strong>Errors encountered:</strong> {error_count}</p>
        <p><strong>Total tracks analyzed:</strong> {len(liked_tracks)}</p>
        <p><a href="/">Back to home</a></p>
        """
        
        logger.info(f"Sorting complete: {processed_count} processed, {error_count} errors")
        return summary
        
    except Exception as e:
        logger.error(f"Critical error in /sort route: {e}")
        logger.debug(traceback.format_exc())
        return f"An error occurred: {str(e)}", 500

@app.route('/debug')
def debug_info():
    """Detailed debug information about Spotify API access."""
    try:
        # Get token information
        token_info = spotify_api.spotify_manager.auth_manager.get_cached_token()
        
        debug_data = {
            "token_exists": token_info is not None,
            "token_info": {},
            "api_tests": {}
        }
        
        if token_info:
            debug_data["token_info"] = {
                "access_token_present": bool(token_info.get('access_token')),
                "refresh_token_present": bool(token_info.get('refresh_token')),
                "granted_scopes": token_info.get('scope', 'No scopes'),
                "expires_at": token_info.get('expires_at', 'Unknown'),
                "token_type": token_info.get('token_type', 'Unknown')
            }
        
        # Test different API endpoints
        sp = spotify_api.spotify_manager.sp
        
        # Test 1: User profile
        try:
            user_profile = sp.current_user()
            debug_data["api_tests"]["user_profile"] = "✅ Success"
        except Exception as e:
            debug_data["api_tests"]["user_profile"] = f"❌ Error: {str(e)}"
        
        # Test 2: Liked tracks
        try:
            liked_tracks = sp.current_user_saved_tracks(limit=1)
            debug_data["api_tests"]["liked_tracks"] = "✅ Success"
        except Exception as e:
            debug_data["api_tests"]["liked_tracks"] = f"❌ Error: {str(e)}"
        
        # Test 3: Audio features (the problematic one)
        try:
            success = spotify_api.test_audio_features()
            if success:
                debug_data["api_tests"]["audio_features"] = "✅ Success"
            else:
                debug_data["api_tests"]["audio_features"] = "❌ Failed (check logs for details)"
        except Exception as e:
            debug_data["api_tests"]["audio_features"] = f"❌ Error: {str(e)}"
        
        # Test 4: Playlists
        try:
            playlists = sp.current_user_playlists(limit=1)
            debug_data["api_tests"]["playlists"] = "✅ Success"
        except Exception as e:
            debug_data["api_tests"]["playlists"] = f"❌ Error: {str(e)}"
        
        return jsonify(debug_data)
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/debug_html')
def debug_html():
    """HTML version of debug info for easier reading."""
    import json
    debug_data = debug_info().get_json()
    
    html = "<h1>Spotify API Debug Information</h1>"
    html += f"<pre>{json.dumps(debug_data, indent=2)}</pre>"
    html += '<p><a href="/">Back to home</a></p>'
    
    return html

@app.route('/sort_batch')
def sort_liked_songs_batch():
    """More efficient batch processing version."""
    logger.info("Starting batch sort")
    
    try:
        user_id = spotify_api.get_current_user_id()
        liked_tracks = spotify_api.get_liked_tracks(limit=20)
        
        if not liked_tracks:
            return "No liked tracks found."
        
        # Get all audio features in batches for efficiency
        track_ids = [track['id'] for track in liked_tracks]
        features_dict = spotify_api.spotify_manager.get_audio_features_batch(track_ids)
        
        # Process tracks
        genre_tracks = {}  # Group tracks by genre
        
        for track in liked_tracks:
            features = features_dict.get(track['id'])
            if not features:
                continue
                
            genre = predict_genre(features)
            
            if genre not in genre_tracks:
                genre_tracks[genre] = []
            genre_tracks[genre].append(track)
        
        # Create playlists and add tracks
        results = {}
        for genre, tracks in genre_tracks.items():
            try:
                playlist_id = spotify_api.create_playlist_if_not_exists(user_id, genre)
                success_count = 0
                
                for track in tracks:
                    if spotify_api.add_track_to_playlist(playlist_id, track['id']):
                        success_count += 1
                
                results[genre] = {
                    'total': len(tracks),
                    'success': success_count
                }
                
            except Exception as e:
                logger.error(f"Error processing genre {genre}: {e}")
                results[genre] = {'error': str(e)}
        
        # Create summary
        summary_html = "<h2>Batch Processing Complete!</h2>"
        for genre, result in results.items():
            if 'error' in result:
                summary_html += f"<p><strong>{genre}:</strong> Error - {result['error']}</p>"
            else:
                summary_html += f"<p><strong>{genre}:</strong> {result['success']}/{result['total']} tracks added</p>"
        
        summary_html += '<p><a href="/">Back to home</a></p>'
        
        return summary_html
        
    except Exception as e:
        logger.error(f"Critical error in batch sort: {e}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(debug=True, port=5000)