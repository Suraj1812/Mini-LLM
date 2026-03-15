from pathlib import Path

import config


def test_railway_env_defaults_are_supported(monkeypatch, tmp_path):
    monkeypatch.delenv("MINI_LLM_PORT", raising=False)
    monkeypatch.delenv("MINI_LLM_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path / "volume"))

    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.port == 9123
    assert settings.artifacts_dir == Path(tmp_path / "volume").resolve()

    config.get_settings.cache_clear()


def test_port_env_takes_priority_over_local_port(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("MINI_LLM_PORT", "8000")

    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.port == 9999

    config.get_settings.cache_clear()
