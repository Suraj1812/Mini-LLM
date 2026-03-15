import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_path(name, default):
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _default_artifacts_dir():
    # Railway injects the mounted volume path at runtime when a volume is attached.
    railway_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    if railway_volume_path:
        return Path(railway_volume_path).expanduser().resolve()
    return BASE_DIR / "artifacts"


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    artifacts_dir: Path
    default_batch_size: int
    default_block_size: int
    default_epochs: int
    default_generation_length: int
    learning_rate: float
    max_epochs: int
    max_generation_length: int
    max_training_chars: int
    max_request_bytes: int
    log_level: str
    model_embed_dim: int = 256
    model_num_heads: int = 4
    model_num_layers: int = 6
    model_dropout: float = 0.1
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    validation_split: float = 0.1
    generation_temperature: float = 0.8
    generation_top_k: int = 20
    generation_top_p: float = 0.9
    repetition_penalty: float = 1.1
    tokenizer_vocab_size: int = 2048

    @property
    def data_path(self):
        return self.artifacts_dir / "data.txt"

    @property
    def model_path(self):
        return self.artifacts_dir / "mini_llm.pt"

    @property
    def vocab_path(self):
        return self.artifacts_dir / "vocab.json"

    @property
    def merges_path(self):
        return self.artifacts_dir / "merges.txt"

    def ensure_artifact_dir(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings():
    settings = Settings(
        host=os.getenv("MINI_LLM_HOST", "0.0.0.0"),
        port=_env_int("PORT", _env_int("MINI_LLM_PORT", 8000)),
        artifacts_dir=_env_path("MINI_LLM_ARTIFACTS_DIR", _default_artifacts_dir()),
        default_batch_size=_env_int("MINI_LLM_BATCH_SIZE", 32),
        default_block_size=_env_int("MINI_LLM_BLOCK_SIZE", 128),
        default_epochs=_env_int("MINI_LLM_DEFAULT_EPOCHS", 5),
        default_generation_length=_env_int("MINI_LLM_DEFAULT_GENERATION_LENGTH", 50),
        learning_rate=_env_float("MINI_LLM_LEARNING_RATE", 3e-4),
        max_epochs=_env_int("MINI_LLM_MAX_EPOCHS", 20),
        max_generation_length=_env_int("MINI_LLM_MAX_GENERATION_LENGTH", 200),
        max_training_chars=_env_int("MINI_LLM_MAX_TRAINING_CHARS", 100000),
        max_request_bytes=_env_int("MINI_LLM_MAX_REQUEST_BYTES", 1048576),
        log_level=os.getenv("MINI_LLM_LOG_LEVEL", "INFO"),
        model_embed_dim=_env_int("MINI_LLM_MODEL_EMBED_DIM", 256),
        model_num_heads=_env_int("MINI_LLM_MODEL_NUM_HEADS", 4),
        model_num_layers=_env_int("MINI_LLM_MODEL_NUM_LAYERS", 6),
        model_dropout=_env_float("MINI_LLM_MODEL_DROPOUT", 0.1),
        weight_decay=_env_float("MINI_LLM_WEIGHT_DECAY", 0.1),
        gradient_clip=_env_float("MINI_LLM_GRADIENT_CLIP", 1.0),
        validation_split=_env_float("MINI_LLM_VALIDATION_SPLIT", 0.1),
        generation_temperature=_env_float("MINI_LLM_GENERATION_TEMPERATURE", 0.8),
        generation_top_k=_env_int("MINI_LLM_GENERATION_TOP_K", 20),
        generation_top_p=_env_float("MINI_LLM_GENERATION_TOP_P", 0.9),
        repetition_penalty=_env_float("MINI_LLM_REPETITION_PENALTY", 1.1),
        tokenizer_vocab_size=_env_int("MINI_LLM_TOKENIZER_VOCAB_SIZE", 2048),
    )
    settings.ensure_artifact_dir()
    return settings


SETTINGS = get_settings()

DATA_PATH = SETTINGS.data_path
MODEL_PATH = SETTINGS.model_path
VOCAB_PATH = SETTINGS.vocab_path
MERGES_PATH = SETTINGS.merges_path

DEFAULT_BATCH_SIZE = SETTINGS.default_batch_size
DEFAULT_BLOCK_SIZE = SETTINGS.default_block_size
DEFAULT_EPOCHS = SETTINGS.default_epochs
DEFAULT_GENERATION_LENGTH = SETTINGS.default_generation_length
DEFAULT_LEARNING_RATE = SETTINGS.learning_rate
