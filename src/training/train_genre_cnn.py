"""Train the CNN genre classifier on GTZAN mel-spectrograms.

Usage: python -m src.training.train_genre_cnn
Tip: much faster on a GPU -- see notebooks/train_cnn_colab.ipynb.
"""
import joblib
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

from src.audio.features import mel_spectrogram
from src.config import CLIP_SECONDS, GTZAN_DIR, MODELS_DIR, SAMPLE_RATE


def load_dataset():
    X, y = [], []
    for path in sorted(GTZAN_DIR.glob("*/*.wav")):
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=CLIP_SECONDS)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        X.append(mel_spectrogram(audio))
        y.append(path.parent.name)
    return np.array(X)[..., np.newaxis], np.array(y)


def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y = load_dataset()
    print(f"Loaded {len(X)} clips")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Genres:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
    model = build_cnn_model(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.3f}")
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "genre_cnn_model.keras")
    joblib.dump(le, MODELS_DIR / "label_encoder_cnn.pkl")
    print("Saved models/genre_cnn_model.keras and models/label_encoder_cnn.pkl")


if __name__ == "__main__":
    main()
