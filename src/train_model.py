# train_model.py
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# === 1. Load Dataset ===
df = pd.read_csv("../data/dataset.csv")

# === Select genres ===
df['track_genre'] = df['track_genre'].str.lower().str.strip()

common_genres = ['pop', 'rock', 'hip-hop', 'country', 'latin', 'jazz', 'electronic', 'classical', 'metal','chill']
df = df[df['track_genre'].isin(common_genres)]

# Select features
features = [
    'duration_ms','explicit','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
]
X = df[features]
y = df['track_genre']


# === 2. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 3. Train LGBMClassifier ===
model = LGBMClassifier(
    num_leaves=128,          # more complex trees (default ~31)
    learning_rate=0.03,      # smaller steps for better generalization
    n_estimators=800,        # more boosting rounds
    max_depth=-1,            # let LightGBM choose depth automatically
    min_data_in_leaf=20,     # small but prevents overfitting
    feature_fraction=0.9,    # use 90% of features per iteration
    bagging_fraction=0.8,    # use 80% of samples per iteration
    bagging_freq=5,          # perform bagging every 5 iterations
    class_weight='balanced', # handle imbalanced classes
    random_state=42,
    verbose=-1               # keep logs clean
)
model.fit(X_train, y_train)

# === 4. Evaluate ===
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# === 5. Confusion Matrix Visualization ===
plt.figure(figsize=(10,6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Genre Classification Confusion Matrix')
plt.show()

# === 6. Feature Importance ===
plt.figure(figsize=(8,5))
sns.barplot(x=model.feature_importances_, y=features)
plt.title('Feature Importance')
plt.show()

# === 7. Save Model ===
joblib.dump(model, "../models/genre_classifier_lgbm.pkl")
print("\n✅ Model saved as models/genre_classifier_lgbm.pkl")

