import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

from config import (
    BASE_DIR,
    DATA_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_GENERATION_LENGTH,
    MERGES_PATH,
    MODEL_PATH,
    VOCAB_PATH,
)
from generate import generate_text
from train import train_model


STATIC_DIR = BASE_DIR / "web"


def _artifacts_ready():
    return all(path.exists() for path in (MODEL_PATH, VOCAB_PATH, MERGES_PATH))


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "idle"
    message: str = "Paste training text and start training."
    last_error: Optional[str] = None
    last_loss: Optional[float] = None
    trained_at: Optional[str] = None

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "last_error": self.last_error,
                "last_loss": self.last_loss,
                "trained_at": self.trained_at,
                "model_ready": _artifacts_ready(),
                "has_training_data": DATA_PATH.exists(),
            }

    def update(self, **changes):
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)


STATE = AppState()
if _artifacts_ready():
    STATE.update(
        status="ready",
        message="Model artifacts found. You can generate text now.",
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )


def _read_json(request):
    raw_length = request.headers.get("Content-Length", "0")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header.") from exc

    payload = request.rfile.read(content_length) if content_length > 0 else b"{}"
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc


def _write_json(handler, payload, status=HTTPStatus.OK):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_file(handler, path, content_type):
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _run_training(text, epochs):
    STATE.update(status="running", message="Preparing training run.", last_error=None)

    def on_progress(message, loss):
        STATE.update(status="running", message=message, last_loss=loss, last_error=None)

    try:
        result = train_model(text, epochs=epochs, progress_callback=on_progress)
        STATE.update(
            status="ready",
            message="Training complete. Model is ready for generation.",
            last_loss=result["last_loss"],
            last_error=None,
            trained_at=datetime.now().isoformat(timespec="seconds"),
        )
    except Exception as exc:
        STATE.update(
            status="error",
            message="Training failed. Check the error details.",
            last_error=str(exc),
        )


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path).path

        if route == "/":
            return _write_file(self, STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if route == "/static/styles.css":
            return _write_file(self, STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        if route == "/static/app.js":
            return _write_file(
                self, STATIC_DIR / "app.js", "application/javascript; charset=utf-8"
            )
        if route == "/api/status":
            return _write_json(self, STATE.snapshot())

        return _write_json(
            self,
            {"error": f"Unknown route: {route}"},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self):
        route = urlparse(self.path).path

        try:
            payload = _read_json(self)
        except ValueError as exc:
            return _write_json(
                self,
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

        if route == "/api/train":
            return self._handle_train(payload)
        if route == "/api/generate":
            return self._handle_generate(payload)

        return _write_json(
            self,
            {"error": f"Unknown route: {route}"},
            status=HTTPStatus.NOT_FOUND,
        )

    def log_message(self, format, *args):
        return

    def _handle_train(self, payload):
        status = STATE.snapshot()
        if status["status"] == "running":
            return _write_json(
                self,
                {"error": "Training is already running."},
                status=HTTPStatus.CONFLICT,
            )

        text = str(payload.get("text") or "").strip()
        if not text and DATA_PATH.exists():
            text = DATA_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return _write_json(
                self,
                {"error": "Training text is required."},
                status=HTTPStatus.BAD_REQUEST,
            )

        try:
            epochs = int(payload.get("epochs", DEFAULT_EPOCHS))
        except (TypeError, ValueError):
            return _write_json(
                self,
                {"error": "Epochs must be an integer."},
                status=HTTPStatus.BAD_REQUEST,
            )
        if epochs < 1:
            return _write_json(
                self,
                {"error": "Epochs must be at least 1."},
                status=HTTPStatus.BAD_REQUEST,
            )

        worker = threading.Thread(target=_run_training, args=(text, epochs), daemon=True)
        worker.start()

        return _write_json(
            self,
            {"message": "Training started.", "status": "running"},
            status=HTTPStatus.ACCEPTED,
        )

    def _handle_generate(self, payload):
        if not _artifacts_ready():
            return _write_json(
                self,
                {"error": "Train the model before generating text."},
                status=HTTPStatus.BAD_REQUEST,
            )

        prompt = str(payload.get("prompt") or "")
        try:
            length = int(payload.get("length", DEFAULT_GENERATION_LENGTH))
        except (TypeError, ValueError):
            return _write_json(
                self,
                {"error": "Length must be an integer."},
                status=HTTPStatus.BAD_REQUEST,
            )
        if length < 1:
            return _write_json(
                self,
                {"error": "Length must be at least 1."},
                status=HTTPStatus.BAD_REQUEST,
            )

        try:
            output = generate_text(prompt, length=length)
        except (ValueError, FileNotFoundError) as exc:
            return _write_json(
                self,
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            return _write_json(
                self,
                {"error": f"Generation failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        return _write_json(
            self,
            {
                "prompt": prompt,
                "output": output,
                "length": length,
            },
        )


def run_server(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving Mini LLM UI at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
