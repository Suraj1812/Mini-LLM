import torch
import torch.nn as nn


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        block_size=128,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        _, T = x.shape
        if T > self.block_size:
            raise ValueError(
                f"Input sequence length {T} exceeds block size {self.block_size}."
            )

        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        tok = self.token_embedding(x)
        pos = self.pos_embedding(pos)

        x = tok + pos
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)

        x = self.ln(x)

        logits = self.head(x)

        return logits
