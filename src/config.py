import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURES_DIR = DATA_DIR / "features"
GTZAN_DIR = DATA_DIR / "genres"
DATASET_CSV = DATA_DIR / "dataset.csv"
GENRE_GROUPS_JSON = DATA_DIR / "genre_groups.json"
DB_PATH = Path(os.getenv("DB_PATH") or DATA_DIR / "app.db")  # Docker keeps it in a volume (/data/app.db)
LYRICS_MODEL_JSON = MODELS_DIR / "lyrics_ridge.json"  # small trained part, committed
# The multilingual embedding model (~130 MB) is too big for git: downloaded here on first use.
LYRICS_EMBEDDER_DIR = Path(os.getenv("LYRICS_EMBEDDER_DIR") or DATA_DIR / "lyrics_embedder")

SAMPLE_RATE = 16000  # YAMNet input rate; must match notebooks/train_models_colab.ipynb
CLIP_SECONDS = 30
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
