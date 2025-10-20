# train_model_cnn.py
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# === 1. Configurations ===
DATASET_PATH = "../data/genres"
MODEL_PATH = "../models/genre_cnn_model.h5"
LABEL_ENCODER_PATH = "../models/label_encoder_cnn.pkl"

SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# === 2. Feature extraction ===
def extract_features(file_path, n_mels=128, fixed_frames=640):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=DURATION)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ensure consistent shape: pad or trim time dimension
        if mel_db.shape[1] < fixed_frames:
            pad_width = fixed_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :fixed_frames]

        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# === 3. Load dataset ===
def load_dataset(data_path):
    X, y = [], []
    genres = [g for g in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, g))]
    for genre in genres:
        genre_dir = os.path.join(data_path, genre)
        if genre.lower() == "archive":  # skip archive or other folders
            continue
        print(f"Processing genre: {genre}")
        for filename in os.listdir(genre_dir):
            if not filename.lower().endswith((".wav", ".mp3")):
                continue  # skip non-audio files
            file_path = os.path.join(genre_dir, filename)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(genre)
    return np.array(X), np.array(y)


# === 4. Build CNN model ===
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === 5. Main training logic ===
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)
    print(f"Loaded {len(X)} samples.")

    # Prepare input
    X = np.array(X)
    X = X[..., np.newaxis]  # add channel dimension for CNN

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print("Genres:", le.classes_)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Build and train
    model = build_cnn_model(X_train.shape[1:], num_classes)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.3f}")

    # Save model and label encoder
    os.makedirs("../models", exist_ok=True)
    model.save("../models/genre_cnn_model.keras", save_format="keras")
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

    # Plot training curves
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
