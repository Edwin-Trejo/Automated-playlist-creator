from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURES_DIR = DATA_DIR / "features"
GTZAN_DIR = DATA_DIR / "genres"
DATASET_CSV = DATA_DIR / "dataset.csv"
DB_PATH = DATA_DIR / "app.db"

SAMPLE_RATE = 16000  # YAMNet input rate; must match notebooks/train_models_colab.ipynb
CLIP_SECONDS = 30
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
