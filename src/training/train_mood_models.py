"""Train LightGBM mood regressors. Usage: python -m src.training.train_mood_models"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.audio.features import SUMMARY_FEATURE_NAMES
from src.config import MODELS_DIR
from src.ml.mood_model import TARGETS, evaluate_regressors, train_regressors
from src.training.collect_mood_dataset import OUT_PATH


def main():
    df = pd.read_parquet(OUT_PATH)
    print(f"Training on {len(df)} tracks")
    X, y = df[SUMMARY_FEATURE_NAMES], df[list(TARGETS)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bundle = train_regressors(X_train, y_train)
    metrics = evaluate_regressors(bundle, X_test, y_test)
    bundle["metrics"] = metrics

    print(f"\n{'target':<14}{'MAE':>8}{'baseline':>10}{'R2':>8}")
    for t, m in metrics.items():
        print(f"{t:<14}{m['mae']:>8.3f}{m['baseline_mae']:>10.3f}{m['r2']:>8.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "mood_regressors.pkl")
    print(f"\nSaved {MODELS_DIR / 'mood_regressors.pkl'}")


if __name__ == "__main__":
    main()
