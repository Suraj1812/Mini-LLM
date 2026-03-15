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
from model import GPTConfig, GPTModel


def _detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require_artifact(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def _load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Saved model format is outdated. Please retrain the model with the current code."
        )
    if "model_config" not in checkpoint:
        raise ValueError("Model checkpoint is missing config metadata. Please retrain.")
    return checkpoint


def _apply_sampling_controls(logits, generated, temperature, top_k, top_p, repetition_penalty):
    if repetition_penalty > 1.0:
        unique_tokens = torch.unique(generated)
        penalized = logits[:, unique_tokens]
        penalized = torch.where(
            penalized < 0,
            penalized * repetition_penalty,
            penalized / repetition_penalty,
        )
        logits[:, unique_tokens] = penalized

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _decode_output(tokenizer, token_ids, special_token_ids):
    special_ids = {value for value in special_token_ids.values() if value is not None}
    filtered_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    decoded = tokenizer.decode(filtered_ids).strip()
    return decoded


def load_runtime(
    model_path=MODEL_PATH,
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    block_size=DEFAULT_BLOCK_SIZE,
    device=None,
):
    target_device = device or _detect_device()

    _require_artifact(model_path)
    _require_artifact(vocab_path)
    _require_artifact(merges_path)

    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    checkpoint = _load_checkpoint(model_path, target_device)
    model_config = GPTConfig.from_dict(checkpoint["model_config"])
    model = GPTModel(config=model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()

    return tokenizer, model, target_device, checkpoint.get("special_token_ids", {})


def generate_text(
    prompt,
    length=DEFAULT_GENERATION_LENGTH,
    model_path=MODEL_PATH,
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    block_size=DEFAULT_BLOCK_SIZE,
    device=None,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.1,
):
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("Prompt cannot be empty.")

    tokenizer, model, target_device, special_token_ids = load_runtime(
        model_path=model_path,
        vocab_path=vocab_path,
        merges_path=merges_path,
        block_size=block_size,
        device=device,
    )

    bos_id = special_token_ids.get("bos")
    eos_id = special_token_ids.get("eos")

    tokens = []
    if bos_id is not None:
        tokens.append(bos_id)
    tokens.extend(tokenizer.encode(clean_prompt).ids)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(target_device)
    generated = 0

    with torch.no_grad():
        for _ in range(length):
            context = x[:, -model.block_size :]
            logits = model(context)
            logits = logits[:, -1, :]
            next_token = _apply_sampling_controls(
                logits,
                x,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            x = torch.cat([x, next_token], dim=1)
            generated += 1
            if eos_id is not None and next_token.item() == eos_id:
                break

    return _decode_output(tokenizer, x[0].tolist(), special_token_ids)


def main():
    print(generate_text("Artificial intelligence"))


if __name__ == "__main__":
    main()
