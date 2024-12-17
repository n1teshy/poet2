import os
import torch
import shutil

from pathlib import Path
from typing import Optional, Type

# 2 chars are used to show boundary of progress bar
USABLE_TERM_WIDTH = shutil.get_terminal_size().columns - 2


def get_param_count(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def get_root():
    return Path(os.path.abspath(__file__ + "/../.."))


def kaiming_init(model: torch.nn.Module):
    def init(m):
        if hasattr(m, "weight") and m.weight.dim() > 1:
            torch.nn.init.kaiming_uniform(m.weight.data)

    model.apply(init)


class ProgressBar:
    def __init__(self, total: int):
        self.done = 0
        self.total = total
        self.last_bar_length = -1

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], *args):
        print("")
        return exc_type is None

    def update(self, increment: int | float = 1):
        self.done += increment
        bar_length = int(USABLE_TERM_WIDTH * (self.done / self.total))
        if bar_length != self.last_bar_length:
            print(f"\r[{'=' * bar_length}{' ' * (USABLE_TERM_WIDTH - bar_length)}]", end="")
            self.last_bar_length = bar_length
        
