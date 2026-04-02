"""Utility helpers for training baselines."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    mode : str
        "max" if higher is better, "min" if lower is better.
    """

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self._is_best = False

    def step(self, value: float) -> None:
        """Update with the latest metric value."""
        if self.best is None:
            self.best = value
            self._is_best = True
            self.counter = 0
            return

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self._is_best = True
            self.counter = 0
        else:
            self._is_best = False
            self.counter += 1

    @property
    def is_best(self) -> bool:
        return self._is_best

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def save_results(data: dict, path: str | Path) -> None:
    """Save results dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
