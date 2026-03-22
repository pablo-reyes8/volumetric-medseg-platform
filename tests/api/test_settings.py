from pathlib import Path

import pytest

from src.api.settings import Settings


def test_settings_defaults_include_supported_extensions():
    settings = Settings()
    assert settings.supported_extensions == (".nii", ".nii.gz")
    assert settings.server_port == 8000
    assert settings.preload_model is True


def test_settings_validates_class_names_length():
    with pytest.raises(ValueError, match="class_names"):
        Settings(num_classes=2, class_names=["background"])


def test_settings_accepts_custom_model_card_path(tmp_path: Path):
    model_card = tmp_path / "model.yaml"
    model_card.write_text("name: demo\n", encoding="utf-8")

    settings = Settings(model_card_path=model_card, preload_model=False)

    assert settings.model_card_path == model_card
    assert settings.preload_model is False
