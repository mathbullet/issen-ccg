import re

from issen_ccg.domain.tree import CcgTree

_NODE = re.compile(r"\s*\(<([^>]+)>")


def read_derivation(text: str) -> CcgTree:
    offset = 0

    def read_node():
        nonlocal offset
        header = _NODE.match(text, offset)
        if header is None:
            raise ValueError(f"expected a CCG node at character {offset}")
        fields = header[1].split()
        offset = header.end()
        if len(fields) == 6 and fields[0] == "L":
            result = CcgTree(fields[1], token=fields[4], pos=fields[2])
        elif len(fields) == 4 and fields[0] == "T":
            children = int(fields[3])
            if children not in (1, 2):
                raise ValueError("CCG trees must be unary or binary")
            result = CcgTree(
                fields[1], tuple(read_node() for _ in range(children)), head_child=int(fields[2])
            )
        else:
            raise ValueError("invalid AUTO node header")
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if text[offset : offset + 1] != ")":
            raise ValueError(f"missing node terminator at character {offset}")
        offset += 1
        return result

    tree = read_node()
    if text[offset:].strip():
        raise ValueError("unexpected trailing content")
    return tree


def normalize_word(word: str) -> str:
    return {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}.get(
        word, word
    )
