import json

import lightgbm as lgb
import numpy as np

from src.audio.embeddings import FEATURE_NAMES, FeatureExtractor
from src.config import MODELS_DIR

MODEL_VERSION = "v2"  # bump when models change so synced songs get re-analyzed
TEMPO_INDEX = FEATURE_NAMES.index("tempo")
MOOD_TARGETS = ("danceability", "energy", "valence", "acousticness")
REQUIRED_FILES = ("yamnet.onnx", "genre_lgbm.txt", "genre_meta.json", "mood_meta.json",
                  *(f"mood_{t}.txt" for t in MOOD_TARGETS))


class ModelsMissingError(RuntimeError):
    pass


class Analyzer:
    """Audio -> YAMNet features -> LightGBM genre classifier + 4 LightGBM mood regressors."""

    def __init__(self, features, genre_model, genres, mood_models):
        self.features = features
        self.genre_model = genre_model
        self.genres = list(genres)
        self.mood_models = mood_models

    @classmethod
    def load(cls, models_dir=MODELS_DIR):
        missing = [name for name in REQUIRED_FILES if not (models_dir / name).exists()]
        if missing:
            raise ModelsMissingError(
                f"Missing in models/: {', '.join(missing)}. Run notebooks/train_models_colab.ipynb in Google Colab "
                "and unzip playlist_models.zip into models/.")
        genre_meta = json.loads((models_dir / "genre_meta.json").read_text())
        mood_meta = json.loads((models_dir / "mood_meta.json").read_text())
        for meta in (genre_meta, mood_meta):
            if meta.get("feature_names") != FEATURE_NAMES:
                raise ModelsMissingError(
                    "The models in models/ were trained on different features than this app computes. "
                    "Retrain with notebooks/train_models_colab.ipynb.")
        return cls(
            features=FeatureExtractor(models_dir / "yamnet.onnx"),
            genre_model=lgb.Booster(model_file=str(models_dir / "genre_lgbm.txt")),
            genres=genre_meta["genres"],
            mood_models={t: lgb.Booster(model_file=str(models_dir / f"mood_{t}.txt")) for t in MOOD_TARGETS},
        )

    def analyze_audio(self, y):
        vec = self.features(y)[np.newaxis, :]
        probs = np.asarray(self.genre_model.predict(vec))[0]
        best = int(np.argmax(probs))
        return {
            "genre": self.genres[best],
            "genre_conf": round(float(probs[best]), 4),
            **{t: float(np.clip(np.asarray(m.predict(vec)).ravel()[0], 0.0, 1.0)) for t, m in self.mood_models.items()},
            "tempo": round(float(vec[0, TEMPO_INDEX]), 1),
        }
