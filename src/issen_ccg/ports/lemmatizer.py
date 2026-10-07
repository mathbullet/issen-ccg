from collections.abc import Sequence
from typing import Protocol


class Lemmatizer(Protocol):
    def lemmatize(self, tokens: Sequence[str]) -> tuple[str, ...]: ...
