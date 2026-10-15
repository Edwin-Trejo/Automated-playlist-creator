from src import config


def test_paths_are_absolute_and_rooted():
    assert config.ROOT.is_absolute()
    assert (config.ROOT / "README.md").exists()
    assert config.MODELS_DIR == config.ROOT / "models"
    assert config.DB_PATH == config.ROOT / "data" / "app.db"


def test_audio_constants_match_training_notebook():
    # notebooks/train_models_colab.ipynb trained every model on 16 kHz, 30 s clips
    assert config.SAMPLE_RATE == 16000
    assert config.CLIP_SAMPLES == 16000 * 30
