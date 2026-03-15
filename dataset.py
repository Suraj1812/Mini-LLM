import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        self.tokens = tokenizer.encode(text).ids

        if len(self.tokens) < 2:
            raise ValueError("Training text must contain at least two tokens.")

        self.window_size = min(block_size, len(self.tokens) - 1)

    def __len__(self):
        return len(self.tokens) - self.window_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.window_size + 1]
        x = chunk[:-1]
        y = chunk[1:]

        return (torch.tensor(x), torch.tensor(y))
