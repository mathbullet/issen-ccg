import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence

from issen_ccg.domain.category import explicit_category
from issen_ccg.domain.tree import CcgTree


def to_auto(tree: CcgTree):
    category = explicit_category(tree.category)
    if tree.token:
        if any(c.isspace() for c in tree.token):
            raise ValueError("AUTO tokens cannot contain whitespace")
        word = {"(": "-LRB-", ")": "-RRB-", "{": "-LCB-", "}": "-RCB-"}.get(tree.token, tree.token)
        pos = tree.pos or "XX"
        return f"(<L {category} {pos} {pos} {word} {category}>)"
    children = " ".join(to_auto(child) for child in tree.children)
    return f"(<T {category} {tree.head_child} {len(tree.children)}> {children} )"


def to_json(identifier, tokens, result):
    return json.dumps(
        {
            "id": identifier,
            "tokens": list(tokens),
            "status": result.status,
            "score": result.score if result.success else None,
            "expanded": result.expanded,
            "pushed": result.pushed,
            "auto": to_auto(result.tree) if result.success else None,
        },
        ensure_ascii=False,
    )


# ccg2lambda templates give these punctuated type changes their own meanings.
_TYPE_CHANGING_PUNCTUATION = {
    (",", "NP", r"(S\NP)\(S\NP)"): "ltc",
    ("NP", ",", "S/S"): "rtc",
    ("S[dcl]/S[dcl]", ",", r"(S\NP)\(S\NP)"): "rtc",
    ("S[dcl]/S[dcl]", ",", r"(S\NP)/(S\NP)"): "rtc",
}


def _jigg_rule(tree: CcgTree) -> str:
    if tree.combinator in ("lp", "rp") and len(tree.children) == 2:
        left, right = (child.category for child in tree.children)
        renamed = _TYPE_CHANGING_PUNCTUATION.get((left, right, tree.category))
        if renamed:
            return renamed
    # Jigg expands X[conj] to X\X, so completing coordination is backward application.
    if (
        tree.combinator == "conj"
        and len(tree.children) == 2
        and tree.children[1].category.endswith("[conj]")
        and not tree.category.endswith("[conj]")
    ):
        return "ba"
    return tree.combinator


def _checked_lemmas(tokens, result, lemmas) -> tuple[str, ...]:
    if isinstance(lemmas, str):
        raise ValueError("pass a sequence of lemmas for each sentence, not a string")
    lemmas = tuple(lemmas)
    if len(lemmas) != len(tokens):
        raise ValueError(f"{len(tokens)} token(s) but {len(lemmas)} lemma(s)")
    if not all(isinstance(lemma, str) and lemma for lemma in lemmas):
        raise ValueError("lemmas must be nonempty strings")
    if result.success and tuple(result.tree.tokens) != tuple(tokens):
        raise ValueError("parse tree tokens differ from the input tokens")
    return lemmas


def to_jigg(sentences, results, *, lemmas_by_sentence: Sequence[Sequence[str]]) -> str:
    sentences, results = list(sentences), list(results)
    lemmas_by_sentence = [
        _checked_lemmas(tokens, result, lemmas)
        for tokens, result, lemmas in zip(sentences, results, lemmas_by_sentence, strict=True)
    ]
    root = ET.Element("root")
    document = ET.SubElement(root, "document", id="d0")
    container = ET.SubElement(document, "sentences")
    for number, (tokens, result, lemmas) in enumerate(
        zip(sentences, results, lemmas_by_sentence, strict=True)
    ):
        sid = f"s{number}"
        sentence = ET.SubElement(container, "sentence", id=sid)
        terminals = ET.SubElement(sentence, "tokens")
        leaves = result.tree.terminals if result.success else ()
        for index, (token, lemma) in enumerate(zip(tokens, lemmas, strict=True)):
            ET.SubElement(
                terminals,
                "token",
                id=f"{sid}_{index}",
                surf=token,
                base=lemma,
                pos=leaves[index].pos if leaves else "XX",
            )
        if not result.success:
            ET.SubElement(sentence, "ccg", id=f"{sid}_ccg0", status=result.status)
            continue
        ccg = ET.SubElement(sentence, "ccg", id=f"{sid}_ccg0", score=str(result.score))
        spans = []

        def visit(tree, begin, sid=sid, spans=spans):
            if tree.token:
                end = begin + 1
                attributes = {"terminal": f"{sid}_{begin}"}
            else:
                child_ids, end = [], begin
                for child in tree.children:
                    child_id, end = visit(child, end)
                    child_ids.append(child_id)
                attributes = {"child": " ".join(child_ids), "rule": _jigg_rule(tree)}
            span_id = f"{sid}_sp{len(spans)}"
            spans.append(
                {
                    "id": span_id,
                    "category": explicit_category(tree.category),
                    "begin": str(begin),
                    "end": str(end),
                    **attributes,
                }
            )
            return span_id, end

        root_id, _ = visit(result.tree, 0)
        ccg.set("root", root_id)
        for attributes in spans:
            if attributes["id"] == root_id:
                attributes["root"] = "true"
            ET.SubElement(ccg, "span", attributes)
    return ET.tostring(root, encoding="unicode")
