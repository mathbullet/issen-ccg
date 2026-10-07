from typing import Protocol

import numpy as np

from issen_ccg.domain.grammar import Grammar
from issen_ccg.domain.result import ParseResult


class Decoder(Protocol):
    grammar: Grammar

    def parse(
        self,
        tokens: tuple[str, ...],
        lexical_scores: np.ndarray,
        head_scores: np.ndarray,
        **options,
    ) -> ParseResult: ...
