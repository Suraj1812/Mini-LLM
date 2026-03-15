from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app


def _build_client(tmp_path):
    with patch("app.get_settings") as mocked_settings:
        from config import Settings

        mocked_settings.return_value = Settings(
            host="127.0.0.1",
            port=8000,
            artifacts_dir=tmp_path,
            default_batch_size=4,
            default_block_size=32,
            default_epochs=2,
            default_generation_length=10,
            learning_rate=3e-4,
            max_epochs=5,
            max_generation_length=20,
            max_training_chars=5000,
            max_request_bytes=1048576,
            log_level="INFO",
        )
        mocked_settings.return_value.ensure_artifact_dir()
        return TestClient(create_app())


def test_status_endpoint_reports_idle(tmp_path):
    client = _build_client(tmp_path)

    response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"idle", "ready"}
    assert payload["model_ready"] is False


def test_generate_requires_model_artifacts(tmp_path):
    client = _build_client(tmp_path)

    response = client.post("/api/generate", json={"prompt": "hello", "length": 5})

    assert response.status_code == 400
    assert response.json()["detail"] == "Train the model before generating text."


def test_train_starts_background_job(tmp_path):
    client = _build_client(tmp_path)

    with patch("app.MiniLLMService.start_training") as mocked_start:
        response = client.post(
            "/api/train",
            json={"text": "mini llm " * 20, "epochs": 1},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "running"
    mocked_start.assert_called_once()


def test_generate_uses_backend_service(tmp_path):
    client = _build_client(tmp_path)
    for artifact in ("mini_llm.pt", "vocab.json", "merges.txt"):
        (tmp_path / artifact).write_text("stub", encoding="utf-8")

    with patch("service._generate_text", return_value="hello world"):
        response = client.post(
            "/api/generate",
            json={"prompt": "hello", "length": 5},
        )

    assert response.status_code == 200
    assert response.json()["output"] == "hello world"
