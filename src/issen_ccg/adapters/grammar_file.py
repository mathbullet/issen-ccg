import json
from dataclasses import asdict
from pathlib import Path

from issen_ccg.domain.grammar import BinaryRule, Grammar, UnaryRule


def load_grammar(path: Path):
    value = json.loads(path.read_text())
    return Grammar(
        tuple(value["categories"]),
        tuple(UnaryRule(**r) for r in value["unary"]),
        tuple(BinaryRule(**r) for r in value["binary"]),
        tuple(value["roots"]),
        tuple(value.get("lexical_categories", ())),
    )


def save_grammar(grammar: Grammar, path: Path, metadata=None):
    value = asdict(grammar)
    value["metadata"] = metadata or {}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
