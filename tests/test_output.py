import xml.etree.ElementTree as ET

import pytest

from issen_ccg.adapters.tree_output import to_auto, to_jigg
from issen_ccg.adapters.treebank import read_derivation
from issen_ccg.domain.category import explicit_category
from issen_ccg.domain.result import ParseResult
from issen_ccg.domain.tree import CcgTree


def jigg(sentences, results):
    lemmas = [tuple(token.lower() for token in tokens) for tokens in sentences]
    return to_jigg(sentences, results, lemmas_by_sentence=lemmas)


def test_coordination_annotation_is_explicit_for_consumers():
    assert explicit_category("NP[conj]") == r"NP\NP"
    assert explicit_category("S[em][conj]") == r"S[em]\S[em]"
    assert explicit_category(r"S[dcl]\NP[conj]") == r"(S[dcl]\NP)\(S[dcl]\NP)"


def test_auto_roundtrip_and_jigg_terminal_offsets():
    tree = CcgTree(
        "NP",
        (
            CcgTree("NP", token="Mary", pos="NNP"),
            CcgTree(
                "NP[conj]",
                (CcgTree("conj", token="and", pos="CC"), CcgTree("NP", token="John", pos="NNP")),
                combinator="conj",
            ),
        ),
        combinator="conj",
    )
    recovered = read_derivation(to_auto(tree))
    assert recovered.tokens == tree.tokens
    assert recovered.children[1].category == r"NP\NP"
    document = ET.fromstring(jigg([tree.tokens], [ParseResult(tree, -1, "success")]))
    ccg = document.find(".//ccg")
    root = ccg.find(f"span[@id='{ccg.get('root')}']")
    assert (root.get("begin"), root.get("end")) == ("0", "3")
    spans = ccg.findall("span")
    token_ids = {token.get("id") for token in document.findall(".//token")}
    assert {span.get("terminal") for span in spans if span.get("terminal")} == token_ids


@pytest.mark.parametrize(
    "tree",
    [
        CcgTree("NP", token="John", pos="NNP"),
        CcgTree("NP", (CcgTree("N", token="John", pos="NNP"),), combinator="lex"),
        CcgTree(
            "S[dcl]",
            (
                CcgTree("NP", token="John", pos="NNP"),
                CcgTree(r"S[dcl]\NP", token="runs", pos="VBZ"),
            ),
            combinator="ba",
        ),
    ],
    ids=["terminal", "unary", "binary"],
)
def test_jigg_marks_only_the_root_span_of_each_successful_parse(tree):
    results = [
        ParseResult(tree, -1, "success"),
        ParseResult(None, float("-inf"), "node_limit"),
        ParseResult(tree, -1, "success"),
    ]
    document = ET.fromstring(jigg([tree.tokens] * len(results), results))
    ccgs = document.findall(".//ccg")
    for ccg in (ccgs[0], ccgs[2]):
        marked = ccg.findall("span[@root='true']")
        assert len(marked) == 1
        assert marked[0].get("id") == ccg.get("root")
        assert len(ccg.findall("span[@root]")) == 1
    assert ccgs[1].findall("span") == []
    assert ccgs[1].get("root") is None


@pytest.mark.parametrize(
    ("category", "modifier", "words"),
    [
        ("NP", r"NP\NP", ("John", "Mary")),
        ("N", r"N\N", ("apples", "oranges")),
        (r"S[dcl]\NP", r"(S[dcl]\NP)\(S[dcl]\NP)", ("runs", "walks")),
        ("S[dcl]", r"S[dcl]\S[dcl]", ("yes", "no")),
    ],
)
def test_jigg_distinguishes_coordination_from_its_backward_application(category, modifier, words):
    tree = CcgTree(
        category,
        (
            CcgTree(category, token=words[0]),
            CcgTree(
                f"{category}[conj]",
                (CcgTree("conj", token="and"), CcgTree(category, token=words[1])),
                combinator="conj",
            ),
        ),
        combinator="conj",
    )
    auto = to_auto(tree)
    document = ET.fromstring(jigg([tree.tokens], [ParseResult(tree, -1, "success")]))
    ccg = document.find(".//ccg")
    spans = {span.get("id"): span for span in ccg.findall("span")}
    root = spans[ccg.get("root")]
    right = spans[root.get("child").split()[1]]
    assert root.get("rule") == "ba"
    assert right.get("rule") == "conj"
    assert right.get("category") == modifier
    assert tree.combinator == "conj"
    assert to_auto(tree) == auto


@pytest.mark.parametrize("marker", ["and", ","])
def test_jigg_preserves_coordination_when_a_modifier_is_still_being_built(marker):
    category = "conj" if marker == "and" else ","
    coordinated = CcgTree(
        "NP[conj]",
        (CcgTree(category, token=marker), CcgTree("NP", token="Mary")),
        combinator="conj",
    )
    tree = CcgTree(
        "NP[conj]",
        (CcgTree("conj", token="and"), coordinated),
        combinator="conj",
    )
    document = ET.fromstring(jigg([tree.tokens], [ParseResult(tree, -1, "success")]))
    rules = [span.get("rule") for span in document.findall(".//span[@child]")]
    assert rules == ["conj", "conj"]


WOMEN = CcgTree(
    "S[dcl]",
    (
        CcgTree("NP", (CcgTree("N", token="women", pos="NNS"),), combinator="lex"),
        CcgTree(r"S[dcl]\NP", token="sketch", pos="VBP"),
    ),
    combinator="ba",
)


def test_jigg_writes_lemmas_as_token_bases():
    tokens = ("women", "sketch")
    results = [ParseResult(WOMEN, -1, "success"), ParseResult(None, float("-inf"), "timeout")]
    xml = to_jigg(
        [tokens, tokens], results, lemmas_by_sentence=[("woman", "sketch"), ("woman", "sketch")]
    )
    sentences = ET.fromstring(xml).findall(".//sentence")
    parsed, failed = (
        [(t.get("surf"), t.get("base"), t.get("pos")) for t in sentence.findall(".//token")]
        for sentence in sentences
    )
    assert parsed == [("women", "woman", "NNS"), ("sketch", "sketch", "VBP")]
    assert failed == [("women", "woman", "XX"), ("sketch", "sketch", "XX")]
    assert sentences[1].find("ccg").get("status") == "timeout"
    lemma_free = ET.fromstring(jigg([tokens, tokens], results))
    assert [ET.tostring(ccg) for ccg in ET.fromstring(xml).iter("ccg")] == [
        ET.tostring(ccg) for ccg in lemma_free.iter("ccg")
    ]


@pytest.mark.parametrize(
    ("sentences", "results", "lemmas"),
    [
        ([("women",)], [ParseResult(None, float("-inf"), "timeout")], []),
        ([("women",)], [ParseResult(None, float("-inf"), "timeout")], [("woman", "x")]),
        ([("women",)], [ParseResult(None, float("-inf"), "timeout")], ["woman"]),
        ([("women",)], [ParseResult(None, float("-inf"), "timeout")], [(1,)]),
        ([("women",)], [ParseResult(None, float("-inf"), "timeout")], [("",)]),
        ([("men", "sketch")], [ParseResult(WOMEN, -1, "success")], [("man", "sketch")]),
    ],
    ids=[
        "sentence-count",
        "token-count",
        "string-lemmas",
        "non-string",
        "empty",
        "tree-tokens",
    ],
)
def test_jigg_rejects_lemmas_that_do_not_match_tokens(sentences, results, lemmas):
    with pytest.raises(ValueError):
        to_jigg(sentences, results, lemmas_by_sentence=lemmas)


def test_jigg_requires_lemmas():
    with pytest.raises(TypeError):
        to_jigg([("women",)], [ParseResult(None, float("-inf"), "timeout")])


@pytest.mark.parametrize(
    ("left", "right", "parent", "combinator", "rule"),
    [
        (",", "NP", r"(S\NP)\(S\NP)", "lp", "ltc"),
        ("NP", ",", "S/S", "rp", "rtc"),
        ("S[dcl]/S[dcl]", ",", r"(S\NP)\(S\NP)", "rp", "rtc"),
        ("S[dcl]/S[dcl]", ",", r"(S\NP)/(S\NP)", "rp", "rtc"),
        ("S[dcl]/S[dcl]", ",", "S/S", "rp", "rp"),
        ("S[dcl]/S[dcl]", ",", r"S\S", "rp", "rp"),
        (r"S[dcl]\S[dcl]", ",", "S/S", "rp", "rp"),
        ("S[dcl]", ".", "S[dcl]", "rp", "rp"),
        ("NP", ",", "NP", "rp", "rp"),
        (",", "NP", "NP", "lp", "lp"),
        ("LRB", "NP", "NP", "lp", "lp"),
    ],
)
def test_jigg_names_punctuated_type_changes_apart_from_punctuation_removal(
    left, right, parent, combinator, rule
):
    tree = CcgTree(
        parent,
        (CcgTree(left, token="a"), CcgTree(right, token="b")),
        combinator=combinator,
    )
    document = ET.fromstring(jigg([tree.tokens], [ParseResult(tree, -1, "success")]))
    assert document.find(".//span[@child]").get("rule") == rule
