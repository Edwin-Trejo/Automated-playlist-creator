from src import config


def test_paths_are_absolute_and_rooted():
    assert config.ROOT.is_absolute()
    assert (config.ROOT / "README.md").exists()
    assert config.MODELS_DIR == config.ROOT / "models"
    assert config.DB_PATH == config.ROOT / "data" / "app.db"


def test_audio_constants():
    assert config.SAMPLE_RATE == 22050
    assert config.CLIP_SAMPLES == 22050 * 30
