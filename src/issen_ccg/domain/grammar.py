from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnaryRule:
    child: int
    parent: int
    name: str = "lex"


@dataclass(frozen=True, slots=True)
class BinaryRule:
    left: int
    right: int
    parent: int
    name: str
    head_child: int


@dataclass(frozen=True, slots=True)
class Grammar:
    categories: tuple[str, ...]
    unary: tuple[UnaryRule, ...]
    binary: tuple[BinaryRule, ...]
    roots: tuple[int, ...]
    lexical_categories: tuple[int, ...] = ()

    def __post_init__(self):
        size = len(self.categories)
        if not size or len(set(self.categories)) != size or not self.roots:
            raise ValueError("a grammar needs unique categories and root categories")
        for rule in self.unary:
            if not all(0 <= value < size for value in (rule.child, rule.parent)):
                raise ValueError("invalid unary category index")
        for rule in self.binary:
            if not all(0 <= value < size for value in (rule.left, rule.right, rule.parent)):
                raise ValueError("invalid binary category index")
            if rule.head_child not in (0, 1):
                raise ValueError("invalid binary head")
        if not all(0 <= value < size for value in self.roots):
            raise ValueError("invalid root category index")
        if not all(0 <= value < size for value in self.lexical_categories):
            raise ValueError("invalid lexical category index")
