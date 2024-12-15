import torch
import torch.nn as nn

from core.config import device
from core.components.embedding import TokenEmbedding
from core.components.transformer import Decoder, Decoder


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        max_len: int,
        layers: int,
        heads: int,
        device: torch.device
    ):
        super().__init__()
        self.emb = TokenEmbedding(vocab_size, embedding_size, max_len, device)
        self.encoder = Decoder(embedding_size, layers, heads)
        self.linear = nn.Linear(embedding_size, vocab_size)
        super().to(device)

    def forward(self, x):
        x_mask = self.get_mask(x)
        x_emb = self.emb(x)
        out = self.encoder(x_emb, x_mask)
        return self.linear(out)

    def get_mask(self, x):
        seq_len = x.shape[1]
        lh_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.long, device=device)
        )
        return lh_mask
    