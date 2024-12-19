import math
import torch
import pickle
import glob
import os.path as path

from torch.nn.utils.rnn import pad_sequence
from core.tokenizers.base import Tokenizer
from core.utils import ProgressBar
from typing import Optional


class GenerativeDataset:
    def __init__(
        self,
        source: str,
        batch_size: int,
        device: torch.device,
        tokenizer: Tokenizer,
        context: Optional[int] = None,
        window_size: float = 0.5,
        from_pickle: bool = False,
    ):
        super().__init__()
        assert from_pickle or (
            None not in (tokenizer, context)
        ), "tokenizer and context are required"
        self.tokenizer = tokenizer
        if from_pickle:
            self.load_from(source)
        else:
            self.samples = []
            self._accumulate_tokens(source, context, window_size)
        self.batch_size = batch_size
        self._current_sample_idx = 0
        self.device = device

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def load_from(self, source: str):
        self.samples = pickle.load(open(source, "rb"))
        self.reset()

    def dump_to(self, file: str):
        pickle.dump(self.samples, open(file, "wb"))

    def _accumulate_tokens(self, source: str, context: int, window_size: float):
        window_size = context - int(context * window_size)
        if path.isfile(source):
            files = [source]
        else:
            files = glob.glob(path.join(source, "**/*.txt"), recursive=True)
        print("tokenizing text...")
        with ProgressBar(total=len(files)) as pbar:
            for file in files:
                with open(file, encoding="utf-8") as f:
                    tokens = self.tokenizer.encode(f.read())
                    x_tokens = [self.tokenizer.bos_id] + tokens
                    y_tokens = tokens + [self.tokenizer.eos_id]
                    if len(x_tokens) <= context:
                        self.samples.append((x_tokens, y_tokens))
                    else:
                        # use sliding window to split into
                        # context sized samples
                        self.samples.extend(
                            self._split_sample(x_tokens, y_tokens, context, window_size)
                        )
                pbar.update()
            self.samples.sort(key=lambda batch: len(batch), reverse=True)

    def _split_sample(
        self, x_tokens: list[int], y_tokens: list[int], context: int, window_size: int
    ) -> list[list[int]]:
        positions = [(i, i + context) for i in range(0, len(x_tokens), context)]
        positions += [
            (i, i + context) for i in range(window_size, len(x_tokens), context)
        ]
        return [(x_tokens[p[0] : p[1]], y_tokens[p[0] : p[1]]) for p in positions]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._current_sample_idx >= len(self.samples):
            return
        samples = self.samples[
            self._current_sample_idx : (self._current_sample_idx + self.batch_size)
        ]
        if len(samples) < self.batch_size:
            samples += self.samples[: self.batch_size - len(samples)]
            self._current_sample_idx = float("inf")
        else:
            self._current_sample_idx += self.batch_size
        Xs, Ys = map(
            lambda batch: [torch.tensor(sample) for sample in batch], zip(*samples)
        )
        Xs = pad_sequence(Xs, batch_first=True, padding_value=self.tokenizer.pad_id).to(
            self.device
        )
        Ys = pad_sequence(Ys, batch_first=True, padding_value=self.tokenizer.pad_id).to(
            self.device
        )
        return Xs, Ys

    def reset(self):
        self._current_sample_idx = 0
