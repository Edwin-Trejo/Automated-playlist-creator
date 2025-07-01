import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class GenreClassifier:
    def __init__(self, model_path: str = "models/genre_classifier.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo'
        ]
        self.genres = [
            'Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Folk', 
            'Classical', 'Jazz', 'R&B', 'Country', 'Indie'
        ]
        
        # Try to load existing model
        self.load_model()
    
    def extract_features(self, audio_features: Dict) -> np.ndarray:
        """
        Extract and normalize features from Spotify audio features.
        
        Args:
            audio_features: Dictionary from Spotify API
            
        Returns:
            Numpy array of features
        """
        features = []
        for feature_name in self.feature_names:
            value = audio_features.get(feature_name, 0)
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def predict_genre_rule_based(self, features: Dict) -> str:
        """
        Rule-based genre prediction (your current approach).
        This serves as a fallback and baseline.
        """
        danceability = features.get('danceability', 0)
        energy = features.get('energy', 0)
        valence = features.get('valence', 0)
        speechiness = features.get('speechiness', 0)
        acousticness = features.get('acousticness', 0)
        instrumentalness = features.get('instrumentalness', 0)
        tempo = features.get('tempo', 120)
        
        # Enhanced rule-based logic
        if speechiness > 0.4:
            return "Hip-Hop"
        elif instrumentalness > 0.7 and acousticness > 0.6:
            return "Classical"
        elif energy > 0.8 and danceability > 0.7 and tempo > 120:
            return "Electronic"
        elif energy > 0.7 and danceability > 0.6 and valence > 0.6:
            return "Pop"
        elif acousticness > 0.7 and energy < 0.4:
            return "Folk"
        elif energy > 0.6 and valence < 0.4 and acousticness < 0.3:
            return "Rock"
        elif energy < 0.3 and acousticness > 0.6:
            return "Classical"
        elif danceability > 0.6 and valence > 0.7:
            return "R&B"
        elif acousticness > 0.5 and energy < 0.5:
            return "Country"
        else:
            return "Indie"
    
    def predict_genre_ml(self, features: Dict) -> str:
        """
        ML-based genre prediction.
        """
        if self.model is None or self.scaler is None:
            logger.warning("ML model not loaded, falling back to rule-based prediction")
            return self.predict_genre_rule_based(features)
        
        try:
            # Extract and scale features
            feature_vector = self.extract_features(features)
            scaled_features = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = self.model.predict(scaled_features)[0]
            confidence = np.max(self.model.predict_proba(scaled_features)[0])
            
            logger.debug(f"ML prediction: {prediction} (confidence: {confidence:.2f})")
            
            # If confidence is too low, fall back to rule-based
            if confidence < 0.4:
                logger.debug("Low confidence, using rule-based fallback")
                return self.predict_genre_rule_based(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self.predict_genre_rule_based(features)
    
    def train_model(self, training_data: pd.DataFrame):
        """
        Train the ML model on provided data.
        
        Args:
            training_data: DataFrame with audio features and genre labels
        """
        logger.info("Training genre classification model...")
        
        # Prepare features and labels
        X = training_data[self.feature_names].values
        y = training_data['genre'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        self.save_model()
        
    def save_model(self):
        """Save the trained model and scaler."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'genres': self.genres
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load a pre-trained model and scaler."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.genres = model_data.get('genres', self.genres)
                
                logger.info(f"Model loaded from {self.model_path}")
                
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Will use rule-based prediction.")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        
        try:
            importance_dict = {}
            importances = self.model.feature_importances_
            
            for feature, importance in zip(self.feature_names, importances):
                importance_dict[feature] = float(importance)
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

# Global classifier instance
genre_classifier = GenreClassifier()

def predict_genre(features: Dict) -> str:
    """
    Main function to predict genre - tries ML first, then falls back to rules.
    
    Args:
        features: Dictionary of Spotify audio features
        
    Returns:
        Predicted genre string
    """
    return genre_classifier.predict_genre_ml(features)

def train_genre_model(data_path: str):
    """
    Convenience function to train the model from a CSV file.
    
    CSV should have columns for audio features plus a 'genre' column.
    """
    try:
        data = pd.read_csv(data_path)
        genre_classifier.train_model(data)
        logger.info("Model training completed")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")

def get_model_info() -> Dict:
    """Get information about the current model."""
    return {
        'model_loaded': genre_classifier.model is not None,
        'model_path': genre_classifier.model_path,
        'supported_genres': genre_classifier.genres,
        'feature_names': genre_classifier.feature_names,
        'feature_importance': genre_classifier.get_feature_importance()
    }