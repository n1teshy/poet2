import torch
import pickle
import glob
import os.path as path

from core.config import TOKEN_BOS, TOKEN_EOS, TOKEN_PAD, device
from core.tokenizers.base import Tokenizer
from core.utils import ProgressBar
from typing import Optional


class GenerativeDataset:
    def __init__(
        self,
        source: str,
        batch_size: int,
        device: torch.device,
        tokenizer: Optional[Tokenizer] = None,
        context: Optional[int] = None,
        window_size: float = 0.5,
        from_pickle: bool = False,
    ):
        super().__init__()
        assert from_pickle or (
            None not in (tokenizer, context)
        ), "tokenizer and context are required"
        self.samples = []
        if from_pickle:
            self.load_from(source)
        else:
            self._accumulate_tokens(source, tokenizer, window_size)
        self.batch_size = batch_size
        self._current_sample_idx = 0
        self.device = device

    def __len__(self):
        return len(self.samples)

    def load_from(self, source: str):
        self.samples = pickle.load(open(source, "rb"))
        self.reset()

    def dump_to(self, file: str):
        pickle.dump(self.samples, open(file, "wb"))

    def _accumulate_tokens(
        self, source: str, tokenizer: Tokenizer, context: int, window_size: float
    ):
        window_size = context - int(context * window_size)
        if path.isfile(source):
            files = [source]
        else:
            files = glob.glob(path.join(source, "**/*.txt"), recursive=True)
        print("tokenizing text...")
        with ProgressBar(total=len(files)) as pbar:
            for file in files:
                with open(file, encoding="utf-8") as f:
                    sample_tokens = (
                        [tokenizer.special_tokens[TOKEN_BOS]]
                        + tokenizer.encode(f.read())
                        + [tokenizer.special_tokens[TOKEN_EOS]]
                    )
                    if len(sample_tokens) <= context:
                        self.samples.append(sample_tokens)
                    else:
                        # use sliding window to split into
                        # context sized samples
                        self.samples.extend(
                            self._split_sample(sample_tokens, context, window_size)
                        )
                pbar.update()
            self.samples.sort(key=lambda batch: len(batch))

    def _split_sample(
        self, tokens: list[int], context: int, window_size: int
    ) -> list[list[int]]:
        positions = [(i, i + context) for i in range(0, len(tokens), context)]
        positions += [
            (i, i + context) for i in range(window_size, len(tokens), context)
        ]
        return [tokens[p[0] : p[1]] for p in positions]

    def next_batch(self) -> list[list[int]]:
        if self._current_sample_idx >= len(self.samples):
            return
        batch = self.samples[
            self._current_sample_idx : (self._current_sample_idx + self.batch_size)
        ]
        if len(batch) < self.batch_size:
            batch += self.samples[: self.batch_size - len(batch)]
            self._current_sample_idx = float("inf")
        else:
            self._current_sample_idx += self.batch_size
        return batch

    def reset(self):
        self._current_sample_idx = 0
