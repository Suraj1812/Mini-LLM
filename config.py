from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.txt"
MODEL_PATH = BASE_DIR / "mini_llm.pt"
VOCAB_PATH = BASE_DIR / "vocab.json"
MERGES_PATH = BASE_DIR / "merges.txt"

DEFAULT_BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_GENERATION_LENGTH = 50
DEFAULT_LEARNING_RATE = 3e-4
