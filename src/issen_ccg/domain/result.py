from dataclasses import dataclass

from issen_ccg.domain.tree import CcgTree


@dataclass(frozen=True, slots=True)
class ParseResult:
    tree: CcgTree | None
    score: float
    status: str
    expanded: int = 0
    pushed: int = 0

    @property
    def success(self):
        return self.tree is not None
