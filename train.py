from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DATA_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    MODEL_PATH,
)
from dataset import TextDataset
from model import GPTModel
from tokenizer import train_tokenizer


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
):
    training_text = text.strip()
    if not training_text:
        raise ValueError("Training text cannot be empty.")

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_file = Path(data_path)
    model_file = Path(model_path)
    output_dir = Path(output_dir or model_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file.write_text(training_text, encoding="utf-8")

    tokenizer = train_tokenizer([str(data_file)], output_dir=output_dir)
    dataset = TextDataset(training_text, tokenizer, block_size=block_size)
    if len(dataset) == 0:
        raise ValueError("Training text is too short for the selected block size.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = tokenizer.get_vocab_size()

    model = GPTModel(vocab_size, block_size=block_size).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    last_loss = None

    for epoch in range(epochs):
        pbar = tqdm(loader, disable=progress_callback is not None)

        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(target_device)
            y = y.to(target_device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            message = (
                f"Epoch {epoch + 1}/{epochs} "
                f"step {step}/{len(loader)} loss {last_loss:.4f}"
            )
            if progress_callback:
                progress_callback(message, last_loss)
            else:
                pbar.set_description(f"loss {last_loss:.4f}")

    torch.save(model.state_dict(), model_file)

    return {
        "device": target_device,
        "epochs": epochs,
        "last_loss": last_loss,
        "model_path": str(model_file),
        "dataset_size": len(dataset),
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
