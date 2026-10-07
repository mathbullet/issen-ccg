from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Category:
    atom: str = ""
    feature: str = ""
    result: "Category | None" = None
    argument: "Category | None" = None
    slash: str = ""
    coordinated: bool = False

    def __str__(self):
        if self.atom:
            text = self.atom + (f"[{self.feature}]" if self.feature else "")
        else:

            def wrap(cat):
                return str(cat) if cat.atom else f"({cat})"

            text = wrap(self.result) + self.slash + wrap(self.argument)
        return text + ("[conj]" if self.coordinated else "")


@lru_cache(maxsize=8192)
def parse_category(text: str) -> Category:
    original = text
    coordinated = text.endswith("[conj]")
    if coordinated:
        text = text[:-6]
    position = 0

    def part():
        nonlocal position
        if position < len(text) and text[position] == "(":
            position += 1
            node = expression()
            if position >= len(text) or text[position] != ")":
                raise ValueError(f"unclosed category: {original}")
            position += 1
            return node
        begin = position
        while position < len(text) and text[position] not in "()/\\[]":
            position += 1
        atom = text[begin:position]
        if not atom or any(c.isspace() for c in atom):
            raise ValueError(f"invalid category: {original}")
        feature = ""
        if position < len(text) and text[position] == "[":
            end = text.find("]", position)
            if end < 0:
                raise ValueError(f"unclosed feature: {original}")
            feature = text[position + 1 : end]
            position = end + 1
        return Category(atom=atom, feature=feature)

    def expression():
        nonlocal position
        node = part()
        while position < len(text) and text[position] in "/\\":
            slash = text[position]
            position += 1
            node = Category(result=node, argument=part(), slash=slash)
        return node

    result = expression()
    if position != len(text):
        raise ValueError(f"trailing category input: {original}")
    return replace(result, coordinated=coordinated)


def same_shape(left: Category, right: Category):
    if left.coordinated != right.coordinated:
        return False
    if left.atom or right.atom:
        return left.atom == right.atom
    return (
        left.slash == right.slash
        and same_shape(left.result, right.result)
        and same_shape(left.argument, right.argument)
    )


def explicit_category(text: str) -> str:
    """Expand CCGBank's whole-category [conj] annotation for CCG consumers.

    X[conj] denotes X\\X, including when X is itself a complex category.
    Treating it as a feature of the last atom loses coordinated arguments.
    """
    if not text.endswith("[conj]"):
        return text
    category = replace(parse_category(text), coordinated=False)
    return str(Category(result=category, slash="\\", argument=category))


def rule_name(left: str, right: str, parent: str) -> str:
    try:
        lhs, rhs, result = map(parse_category, (left, right, parent))
    except ValueError:
        # Preserve malformed treebank labels in diagnostics without inventing repairs.
        return "other"
    if rhs.coordinated:
        return "conj"
    if lhs.atom == "conj" or result.coordinated:
        return "conj"
    punctuation = {",", ".", ":", ";", "LRB", "RRB", "LQU", "RQU"}
    if lhs.atom in punctuation or rhs.atom in punctuation:
        return "lp" if lhs.atom in punctuation else "rp"
    if lhs.slash == "/" and same_shape(lhs.argument, rhs) and same_shape(lhs.result, result):
        return "fa"
    if rhs.slash == "\\" and same_shape(rhs.argument, lhs) and same_shape(rhs.result, result):
        return "ba"
    if lhs.slash == "/" and rhs.slash == "/":
        return "fc"
    if lhs.slash == "\\" and rhs.slash == "\\":
        return "bc"
    if lhs.slash == "/" and rhs.slash == "\\":
        return "bx"
    return "other"
