from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CcgTree:
    category: str
    children: tuple["CcgTree", ...] = ()
    token: str = ""
    pos: str = ""
    head_child: int = 0
    combinator: str = ""

    def __post_init__(self):
        if not self.category or len(self.children) > 2:
            raise ValueError("invalid CCG node")
        if bool(self.children) == bool(self.token):
            raise ValueError("a CCG node must have either a token or children")
        if not 0 <= self.head_child < max(1, len(self.children)):
            raise ValueError("invalid head child")

    @property
    def terminals(self) -> tuple["CcgTree", ...]:
        pending = [self]
        output = []
        while pending:
            node = pending.pop()
            if node.token:
                output.append(node)
            else:
                pending.extend(reversed(node.children))
        return tuple(output)

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(leaf.token for leaf in self.terminals)

    @property
    def lexical_categories(self) -> tuple[str, ...]:
        return tuple(leaf.category for leaf in self.terminals)

    @property
    def pos_tags(self) -> tuple[str, ...]:
        return tuple(leaf.pos for leaf in self.terminals)

    @property
    def leftmost_dependencies(self) -> tuple[int, ...]:
        heads = [0] * len(self.terminals)

        def visit(node, start):
            if node.token:
                return start + 1
            middle = visit(node.children[0], start)
            if len(node.children) == 1:
                return middle
            heads[middle] = start
            return visit(node.children[1], middle)

        visit(self, 0)
        return tuple(heads)
