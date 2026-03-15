import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = self.head_dim ** -0.5
            attn_scores = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights,
                p=self.dropout,
                training=self.training,
            )
            attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            channels,
        )
        return self.resid_dropout(self.out_proj(attn_output))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.embed_dim
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        embed_dim=256,
        num_heads=4,
        num_layers=6,
        block_size=128,
        dropout=0.1,
        config=None,
    ):
        super().__init__()

        self.config = config or GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.block_size = self.config.block_size

        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embed_dim,
        )
        self.pos_embedding = nn.Embedding(
            self.config.block_size,
            self.config.embed_dim,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.ln = nn.LayerNorm(self.config.embed_dim)
        self.head = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        _, T = x.shape
        if T > self.block_size:
            raise ValueError(
                f"Input sequence length {T} exceeds block size {self.block_size}."
            )

        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        tok = self.token_embedding(x)
        pos = self.pos_embedding(pos)

        x = self.dropout(tok + pos)

        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)
        return self.head(x)
