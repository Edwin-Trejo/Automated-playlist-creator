import joblib
import numpy as np
import pandas as pd

from src.audio.features import SUMMARY_FEATURE_NAMES, mel_spectrogram, summary_vector
from src.config import MODELS_DIR
from src.ml.mood_model import predict_mood

MODEL_VERSION = "v1"
TEMPO_INDEX = SUMMARY_FEATURE_NAMES.index("tempo")


class ModelsMissingError(RuntimeError):
    pass


class Analyzer:
    def __init__(self, cnn, cnn_classes, genre_lgbm, mood_bundle):
        self.cnn = cnn
        self.cnn_classes = [str(c) for c in cnn_classes]
        self.genre_lgbm = genre_lgbm
        self.mood_bundle = mood_bundle

    @classmethod
    def load(cls, models_dir=MODELS_DIR):
        required = {
            "mood_regressors.pkl": "python -m src.training.collect_mood_dataset && python -m src.training.train_mood_models",
            "genre_cnn_model.keras": "python -m src.training.train_genre_cnn (or notebooks/train_cnn_colab.ipynb)",
            "label_encoder_cnn.pkl": "python -m src.training.train_genre_cnn",
        }
        missing = [f"models/{name} -> run: {cmd}" for name, cmd in required.items() if not (models_dir / name).exists()]
        if missing:
            raise ModelsMissingError("Missing trained models:\n" + "\n".join(missing))

        from tensorflow.keras.models import load_model  # slow import; only when actually serving

        lgbm_path = models_dir / "genre_lgbm_audio.pkl"
        return cls(
            cnn=load_model(models_dir / "genre_cnn_model.keras"),
            cnn_classes=list(joblib.load(models_dir / "label_encoder_cnn.pkl").classes_),
            genre_lgbm=joblib.load(lgbm_path) if lgbm_path.exists() else None,
            mood_bundle=joblib.load(models_dir / "mood_regressors.pkl"),
        )

    def genre_probs(self, y, vec):
        mel = mel_spectrogram(y)[np.newaxis, ..., np.newaxis]
        cnn = self.cnn.predict(mel, verbose=0)[0]
        probs = {c: float(p) for c, p in zip(self.cnn_classes, cnn)}
        if self.genre_lgbm is not None:
            X = pd.DataFrame([vec], columns=self.genre_lgbm["feature_names"])
            lgbm = self.genre_lgbm["model"].predict_proba(X)[0]
            lgbm_probs = {str(c): float(p) for c, p in zip(self.genre_lgbm["model"].classes_, lgbm)}
            # Union of classes: a genre one model wasn't trained on still counts via the other.
            classes = list(dict.fromkeys([*probs, *lgbm_probs]))
            probs = {c: (probs.get(c, 0.0) + lgbm_probs.get(c, 0.0)) / 2 for c in classes}
        return probs

    def analyze_audio(self, y):
        vec = summary_vector(y)
        probs = self.genre_probs(y, vec)
        genre = max(probs, key=probs.get)
        return {
            "genre": genre,
            "genre_conf": round(probs[genre], 4),
            **predict_mood(self.mood_bundle, vec),
            "tempo": round(float(vec[TEMPO_INDEX]), 1),
        }
