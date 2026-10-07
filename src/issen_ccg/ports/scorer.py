from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SentenceScores:
    categories: np.ndarray
    heads: np.ndarray
    pos: tuple[str, ...]


class Scorer(Protocol):
    categories: tuple[str, ...]

    def score(self, sentences: Sequence[Sequence[str]]) -> list[SentenceScores]: ...
