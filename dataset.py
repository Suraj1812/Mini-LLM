import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        tokens = tokenizer.encode(text).ids

        if len(tokens) < 2:
            raise ValueError("Training text must contain at least two tokens.")

        window_size = min(block_size, len(tokens) - 1)

        self.examples = []

        for i in range(0, len(tokens) - window_size):
            chunk = tokens[i : i + window_size + 1]

            x = chunk[:-1]
            y = chunk[1:]

            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]

        return (torch.tensor(x), torch.tensor(y))
