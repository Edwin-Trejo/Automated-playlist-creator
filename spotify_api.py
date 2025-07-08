import spotipy
from spotipy.oauth2 import SpotifyOAuth
from loguru import logger
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables (no fallbacks for security)
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')  
REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8888/callback')

# Validate that credentials are provided
if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "Missing Spotify credentials! Please set SPOTIFY_CLIENT_ID and "
        "SPOTIFY_CLIENT_SECRET in your .env file"
    )

class SpotifyManager:
    def __init__(self):
        # Updated scopes to include everything needed for audio features
        self.auth_manager = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-library-read playlist-modify-private playlist-modify-public user-read-private user-read-playback-state user-read-recently-played",
            cache_path=".cache",
            show_dialog=True
        )
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        self._validate_connection()

    def _validate_connection(self):
        """Validate Spotify connection and log user info."""
        try:
            user_info = self.sp.current_user()
            logger.info(f"Connected to Spotify as: {user_info['id']}")
            
            token_info = self.auth_manager.get_cached_token()
            if token_info:
                logger.info(f"Granted scopes: {token_info.get('scope')}")
                logger.debug(f"Full token: {token_info}")
            else:
                logger.warning("No cached token found")
        except Exception as e:
            logger.error(f"Failed to validate Spotify connection: {e}")
            raise

    def clear_token_cache(self):
        """Clear the token cache to force re-authentication."""
        try:
            cache_path = ".cache"
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info("Token cache cleared. Please re-authenticate.")
            else:
                logger.info("No token cache found to clear.")
        except Exception as e:
            logger.error(f"Error clearing token cache: {e}")

    def get_liked_tracks(self, limit: int = None) -> List[Dict]:
        """
        Fetch all liked tracks from user's library.
        
        Args:
            limit: Maximum number of tracks to fetch (None for all)
            
        Returns:
            List of track dictionaries
        """
        tracks = []
        offset = 0
        batch_size = 50
        
        try:
            while True:
                logger.debug(f"Fetching liked tracks batch: offset={offset}")
                results = self.sp.current_user_saved_tracks(limit=batch_size, offset=offset)
                
                if not results['items']:
                    break
                    
                for item in results['items']:
                    track = item['track']
                    tracks.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']],
                        'album': track['album']['name'],
                        'popularity': track['popularity'],
                        'explicit': track['explicit']
                    })
                
                offset += batch_size
                
                # Break if we've reached the limit or no more tracks
                if limit and len(tracks) >= limit:
                    tracks = tracks[:limit]
                    break
                    
                if len(results['items']) < batch_size:
                    break
                    
            logger.info(f"Fetched {len(tracks)} liked tracks")
            return tracks
            
        except Exception as e:
            logger.error(f"Error fetching liked tracks: {e}")
            return []

    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a specific track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary of audio features or None if not found
        """
        try:
            logger.debug(f"Fetching audio features for track: {track_id}")
            features = self.sp.audio_features([track_id])
            
            if features and features[0]:
                logger.debug(f"Successfully fetched audio features for {track_id}")
                return features[0]
            else:
                logger.warning(f"No audio features found for track ID: {track_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting audio features for {track_id}: {e}")
            # Log the full exception for debugging
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            return None

    def get_audio_features_batch(self, track_ids: List[str]) -> Dict[str, Dict]:
        """
        Get audio features for multiple tracks in batches.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            Dictionary mapping track IDs to their audio features
        """
        features_dict = {}
        batch_size = 100  # Spotify API limit
        
        try:
            for i in range(0, len(track_ids), batch_size):
                batch = track_ids[i:i + batch_size]
                logger.debug(f"Fetching audio features batch {i//batch_size + 1}")
                
                features = self.sp.audio_features(batch)
                
                for track_id, feature in zip(batch, features):
                    if feature:
                        features_dict[track_id] = feature
                    else:
                        logger.warning(f"No features for track: {track_id}")
                        
            logger.info(f"Fetched audio features for {len(features_dict)} tracks")
            return features_dict
            
        except Exception as e:
            logger.error(f"Error fetching audio features batch: {e}")
            return {}

    def test_audio_features_access(self):
        """Test audio features access with a well-known track."""
        try:
            # Use a very popular track that should definitely exist
            test_track_id = "4iV5W9uYEdYUVa79Axb7Rh"  # Shape of You by Ed Sheeran
            
            logger.info(f"Testing audio features access with track: {test_track_id}")
            
            # First, let's try to get track info to make sure the track exists
            track_info = self.sp.track(test_track_id)
            logger.info(f"Track found: {track_info['name']} by {track_info['artists'][0]['name']}")
            
            # Now try to get audio features
            features = self.sp.audio_features([test_track_id])
            
            if features and features[0]:
                logger.info("✅ Audio features access successful!")
                logger.info(f"Sample features: danceability={features[0]['danceability']}, energy={features[0]['energy']}")
                return True
            else:
                logger.error("❌ Audio features returned None or empty")
                return False
                
        except Exception as e:
            logger.error(f"❌ Audio features test failed: {e}")
            return False

    def get_current_user_id(self) -> str:
        """Get the current user's Spotify ID."""
        try:
            return self.sp.current_user()['id']
        except Exception as e:
            logger.error(f"Error getting current user ID: {e}")
            raise

    def get_user_playlists(self) -> List[Dict]:
        """
        Get all user playlists.
        
        Returns:
            List of playlist dictionaries
        """
        playlists = []
        offset = 0
        batch_size = 50
        
        try:
            while True:
                results = self.sp.current_user_playlists(limit=batch_size, offset=offset)
                
                if not results['items']:
                    break
                    
                playlists.extend(results['items'])
                offset += batch_size
                
                if len(results['items']) < batch_size:
                    break
                    
            logger.info(f"Fetched {len(playlists)} playlists")
            return playlists
            
        except Exception as e:
            logger.error(f"Error fetching playlists: {e}")
            return []

    def create_playlist_if_not_exists(self, user_id: str, genre_name: str) -> str:
        """
        Create a playlist for the genre if it doesn't exist.
        
        Args:
            user_id: Spotify user ID
            genre_name: Name of the genre/playlist
            
        Returns:
            Playlist ID
        """
        try:
            # Check existing playlists
            playlists = self.get_user_playlists()
            for playlist in playlists:
                if playlist['name'].lower() == genre_name.lower():
                    logger.debug(f"Found existing playlist: {genre_name}")
                    return playlist['id']
            
            # Create new playlist
            logger.info(f"Creating new playlist: {genre_name}")
            new_playlist = self.sp.user_playlist_create(
                user=user_id,
                name=genre_name,
                public=False,
                description=f"Auto-generated playlist for {genre_name} songs"
            )
            return new_playlist['id']
            
        except Exception as e:
            logger.error(f"Error creating playlist '{genre_name}': {e}")
            raise

    def add_track_to_playlist(self, playlist_id: str, track_id: str) -> bool:
        """
        Add a track to a playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            track_id: Spotify track ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if track is already in playlist
            playlist_tracks = self.sp.playlist_tracks(playlist_id, fields="items(track(id))")
            existing_track_ids = [item['track']['id'] for item in playlist_tracks['items']]
            
            if track_id in existing_track_ids:
                logger.debug(f"Track {track_id} already in playlist")
                return True
                
            self.sp.playlist_add_items(playlist_id, [track_id])
            logger.debug(f"Added track {track_id} to playlist {playlist_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding track {track_id} to playlist {playlist_id}: {e}")
            return False

# Create a global instance
spotify_manager = SpotifyManager()

# Convenience functions for backward compatibility
def get_liked_tracks(limit=None):
    return spotify_manager.get_liked_tracks(limit)

def get_audio_features(track_id):
    return spotify_manager.get_audio_features(track_id)

def get_current_user_id():
    return spotify_manager.get_current_user_id()

def create_playlist_if_not_exists(user_id, genre_name):
    return spotify_manager.create_playlist_if_not_exists(user_id, genre_name)

def add_track_to_playlist(playlist_id, track_id):
    return spotify_manager.add_track_to_playlist(playlist_id, track_id)

def clear_token_cache():
    return spotify_manager.clear_token_cache()

def test_audio_features():
    return spotify_manager.test_audio_features_access()