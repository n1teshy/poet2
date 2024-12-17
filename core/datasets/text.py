import torch
import glob
import os.path as path

from core.config import TOKEN_BOS, TOKEN_EOS, TOKEN_PAD, device
from core.tokenizers.base import Tokenizer


class GnerativeDataset:
    def __init__(
        self,
        source: str,
        tokenizer: Tokenizer,
        batch_size: int,
        context: int,
        device: torch.device,
        indefinite: bool = False,
    ):
        super().__init__()
        self.tokens = []
        self.accumulate_tokens(source, tokenizer)
        self.batch_size = batch_size
        self.context = context
        self.current_idx = 0
        self.device = device
        self.indefinite = indefinite

    def __len__(self):
        return len(self.tokens) // (self.batch_size * self.context)

    def accumulate_tokens(self, source: str, tokenizer: Tokenizer):
        if path.isfile(source):
            files = [source]
        else:
            files = glob.glob(path.join(source, "**/.*txt"))
        for file in files:
            with open(file, encoding="utf-8") as f:
                self.tokens += (
                    [tokenizer.special_tokens[TOKEN_BOS]]
                    + tokenizer.encode(f.read())
                    + [tokenizer.special_tokens[TOKEN_EOS]]
                )

    def next_batch(self) -> torch.Tensor:
        end_idx = self.current_idx + self.batch_size * self.context
        if end_idx >= len(self.tokens):
            return None
        tokens = self.tokens[self.current_idx: end_idx]
        self.current_idx = end_idx
        return tokens
    
    def reset(self):
        self.current_idx = 0
