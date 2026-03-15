import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

def _timestamp():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class AppState:
    settings: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "idle"
    message: str = "Paste training text and start training."
    last_error: Optional[str] = None
    last_loss: Optional[float] = None
    trained_at: Optional[str] = None

    def artifact_paths_ready(self):
        return all(
            path.exists()
            for path in (
                self.settings.model_path,
                self.settings.vocab_path,
                self.settings.merges_path,
            )
        )

    def bootstrap(self):
        if self.artifact_paths_ready():
            self.update(
                status="ready",
                message="Model artifacts found. You can generate text now.",
                trained_at=_timestamp(),
            )

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "last_error": self.last_error,
                "last_loss": self.last_loss,
                "trained_at": self.trained_at,
                "model_ready": self.artifact_paths_ready(),
                "has_training_data": self.settings.data_path.exists(),
            }

    def update(self, **changes):
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def mark_running(self):
        self.update(
            status="running",
            message="Preparing training run.",
            last_error=None,
        )

    def mark_ready(self, last_loss):
        self.update(
            status="ready",
            message="Training complete. Model is ready for generation.",
            last_loss=last_loss,
            last_error=None,
            trained_at=_timestamp(),
        )

    def mark_error(self, error_message):
        self.update(
            status="error",
            message="Training failed. Check the error details.",
            last_error=error_message,
        )
