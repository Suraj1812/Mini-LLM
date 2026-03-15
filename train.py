import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (
    DATA_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    MODEL_PATH,
)
from dataset import TextDataset, build_training_tokens
from model import GPTConfig, GPTModel
from tokenizer import train_tokenizer


def _detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_scheduler(optimizer, total_steps, warmup_steps=20):
    def lr_lambda(step):
        if total_steps <= 1:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _evaluate(model, loader, device):
    if loader is None:
        return None

    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())
    model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def train_model(
    text,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    block_size=DEFAULT_BLOCK_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    device=None,
    data_path=DATA_PATH,
    output_dir=None,
    model_path=MODEL_PATH,
    progress_callback=None,
    embed_dim=256,
    num_heads=4,
    num_layers=6,
    dropout=0.1,
    weight_decay=0.1,
    gradient_clip=1.0,
    validation_split=0.1,
    tokenizer_vocab_size=2048,
):
    training_text = text.strip()
    if not training_text:
        raise ValueError("Training text cannot be empty.")

    _set_seed()
    target_device = device or _detect_device()
    data_file = Path(data_path)
    model_file = Path(model_path)
    output_dir = Path(output_dir or model_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file.write_text(training_text, encoding="utf-8")

    tokenizer = train_tokenizer(
        [str(data_file)],
        output_dir=output_dir,
        vocab_size=tokenizer_vocab_size,
    )
    tokens = build_training_tokens(training_text, tokenizer)
    dataset = TextDataset(tokens, block_size=block_size)
    if len(dataset) == 0:
        raise ValueError("Training text is too short for the selected block size.")

    val_size = 0
    if len(dataset) >= 10 and validation_split > 0:
        val_size = max(1, int(len(dataset) * validation_split))
        val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    vocab_size = tokenizer.get_vocab_size()

    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = GPTModel(config=model_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        total_steps=max(1, epochs * len(train_loader)),
    )

    last_loss = None
    best_val_loss = None

    for epoch in range(epochs):
        pbar = tqdm(train_loader, disable=progress_callback is not None)

        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(target_device)
            y = y.to(target_device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            last_loss = loss.item()
            val_loss = _evaluate(model, val_loader, target_device) if step == len(train_loader) else None
            if val_loss is not None:
                best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)
            message = f"Epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} train_loss {last_loss:.4f}"
            if val_loss is not None:
                message += f" val_loss {val_loss:.4f}"
            if progress_callback:
                progress_callback(message, last_loss)
            else:
                pbar.set_description(message)

    checkpoint = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
        },
        "special_token_ids": {
            "bos": tokenizer.token_to_id("<s>"),
            "eos": tokenizer.token_to_id("</s>"),
            "pad": tokenizer.token_to_id("<pad>"),
            "unk": tokenizer.token_to_id("<unk>"),
        },
        "metrics": {
            "last_train_loss": last_loss,
            "best_val_loss": best_val_loss,
            "dataset_size": len(dataset),
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset) if val_dataset is not None else 0,
        },
    }
    torch.save(checkpoint, model_file)

    return {
        "device": target_device,
        "epochs": epochs,
        "last_loss": last_loss,
        "model_path": str(model_file),
        "dataset_size": len(dataset),
        "best_val_loss": best_val_loss,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    text = DATA_PATH.read_text(encoding="utf-8")
    result = train_model(text)
    print(
        "Training complete. "
        f"Saved model to {result['model_path']} with loss {result['last_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
