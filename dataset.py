import torch
from torch.utils.data import Dataset


def build_training_tokens(text, tokenizer):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text.strip()]

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    tokens = []
    for line in lines:
        if bos_id is not None:
            tokens.append(bos_id)
        tokens.extend(tokenizer.encode(line).ids)
        if eos_id is not None:
            tokens.append(eos_id)

    return tokens


class TextDataset(Dataset):
    def __init__(self, tokens, block_size=128, stride=None):
        self.tokens = list(tokens)

        if len(self.tokens) < 2:
            raise ValueError("Training text must contain at least two tokens.")

        self.window_size = min(block_size, len(self.tokens) - 1)
        self.stride = max(1, stride or max(1, self.window_size // 2))
        last_start = len(self.tokens) - self.window_size - 1
        self.starts = list(range(0, max(1, last_start + 1), self.stride))
        if self.starts[-1] != last_start:
            self.starts.append(last_start)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        chunk = self.tokens[start : start + self.window_size + 1]
        x = chunk[:-1]
        y = chunk[1:]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
