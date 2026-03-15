from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GENERATION_LENGTH,
    MERGES_PATH,
    MODEL_PATH,
    VOCAB_PATH,
)
from model import GPTModel


def _require_artifact(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def load_runtime(
    model_path=MODEL_PATH,
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    block_size=DEFAULT_BLOCK_SIZE,
    device=None,
):
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    _require_artifact(model_path)
    _require_artifact(vocab_path)
    _require_artifact(merges_path)

    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    model = GPTModel(tokenizer.get_vocab_size(), block_size=block_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(target_device)))
    model.to(target_device)
    model.eval()

    return tokenizer, model, target_device


def generate_text(
    prompt,
    length=DEFAULT_GENERATION_LENGTH,
    model_path=MODEL_PATH,
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    block_size=DEFAULT_BLOCK_SIZE,
    device=None,
):
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("Prompt cannot be empty.")

    tokenizer, model, target_device = load_runtime(
        model_path=model_path,
        vocab_path=vocab_path,
        merges_path=merges_path,
        block_size=block_size,
        device=device,
    )

    tokens = tokenizer.encode(clean_prompt).ids
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(target_device)

    with torch.no_grad():
        for _ in range(length):
            context = x[:, -model.block_size :]
            logits = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    print(generate_text("Artificial intelligence"))


if __name__ == "__main__":
    main()
