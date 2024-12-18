import torch
import torch.nn as nn

from torch.nn.functional import cross_entropy
from core.config import device
from core.components.embedding import TokenEmbedding
from core.components.transformer import Decoder, Decoder
from typing import Optional


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        max_len: int,
        layers: int,
        heads: int,
        pad_id: int,
        device: torch.device,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.emb = TokenEmbedding(vocab_size, embedding_size, max_len, device)
        self.encoder = Decoder(embedding_size, layers, heads)
        self.linear = nn.Linear(embedding_size, vocab_size)
        super().to(device)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_mask = self.get_mask(x)
        x_emb = self.emb(x)
        out = self.encoder(x_emb, x_mask)
        logits = self.linear(out)
        loss = None
        if y is not None:
            reshaped_logits = logits.reshape(-1, logits.shape[-1])
            loss = cross_entropy(reshaped_logits, y.reshape(-1))
        return logits, loss

    def get_mask(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        lookahead_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.long, device=device)
        )
        pad_mask = (x != self.pad_id).unsqueeze(1)
        return lookahead_mask & pad_mask
