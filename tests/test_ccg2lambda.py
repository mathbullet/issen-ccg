"""Fixed CCG trees through Jigg export to ccg2lambda logical forms.

Runs only when NC2L_CONFIG names the neo-ccg2lambda config directory.
"""

import os
import re
from pathlib import Path

import pytest

from issen_ccg.adapters.tree_output import to_jigg
from issen_ccg.domain.result import ParseResult
from issen_ccg.domain.tree import CcgTree

CONFIG = os.environ.get("NC2L_CONFIG")
pytestmark = pytest.mark.skipif(not CONFIG, reason="NC2L_CONFIG is not set")


LEMMAS = {
    "is": "be",
    "sketching": "sketch",
    "sketches": "sketch",
    "runs": "run",
    "ran": "run",
    "sings": "sing",
    "dances": "dance",
    "walks": "walk",
}


def leaf(category, token, pos):
    return CcgTree(category, token=token, pos=pos)


def node(category, combinator, *children):
    return CcgTree(category, children, combinator=combinator)


def np(word, pos="NNP"):
    return node("NP", "lex", leaf("N", word, pos))


def lemmas(tokens):
    return tuple(LEMMAS.get(token, token.lower()) for token in tokens)


def jigg(*results):
    sentences = [result.tree.tokens if result.success else tokens for tokens, result in results]
    return to_jigg(
        sentences,
        [result for _, result in results],
        lemmas_by_sentence=[lemmas(tokens) for tokens in sentences],
    )


def formula(tree):
    from neo_ccg2lambda.adapters.jigg.reading import read_sentences
    from neo_ccg2lambda.adapters.templates.loader import load_semantic_index
    from neo_ccg2lambda.application.semparse.parsing import parse_sentence_semantics
    from neo_ccg2lambda.domain.logic.printer import to_string

    index = load_semantic_index(Path(CONFIG) / "semantics" / "semantic_templates_en_event.yaml")
    (sentence,) = read_sentences(jigg(((), ParseResult(tree, -1, "success"))))
    (parse,) = parse_sentence_semantics(sentence, index, nbest=1)
    assert parse.root is not None, parse.error
    text = to_string(parse.root.semantics)
    assert "\\" not in text, text
    return text


def variable_of(predicate, text):
    return re.search(rf"_{predicate}\((\w+)\)", text).group(1)


VP = r"S[dcl]\NP"
A_WOMAN = node("NP", "fa", leaf("NP[nb]/N", "A", "DT"), leaf("N", "woman", "NN"))


def test_lemmas_choose_the_copula_and_share_verb_predicates():
    progressive = node(
        "S[dcl]",
        "rp",
        node(
            "S[dcl]",
            "ba",
            A_WOMAN,
            node(
                VP,
                "fa",
                leaf(r"(S[dcl]\NP)/(S[ng]\NP)", "is", "VBZ"),
                leaf(r"S[ng]\NP", "sketching", "VBG"),
            ),
        ),
        leaf(".", ".", "."),
    )
    simple = node(
        "S[dcl]",
        "rp",
        node("S[dcl]", "ba", A_WOMAN, leaf(VP, "sketches", "VBZ")),
        leaf(".", ".", "."),
    )
    assert "_is(" not in formula(progressive)
    assert formula(progressive) == formula(simple)
    assert "_sketch(" in formula(simple)


def test_the_marked_root_completes_the_proposition():
    tree = node("S[dcl]", "ba", np("John"), leaf(VP, "runs", "VBZ"))
    assert formula(tree) == "exists x.(_john(x) & True & exists e.(_run(e) & (Subj(e) = x) & True))"


def coordinate(category, left, marker, right, marker_category="conj"):
    conjunct = node(f"{category}[conj]", "conj", leaf(marker_category, marker, "CC"), right)
    return node(category, "conj", left, conjunct)


def sentence(subject, predicate):
    return node("S[dcl]", "rp", node("S[dcl]", "ba", subject, predicate), leaf(".", ".", "."))


def verb(word):
    return leaf(VP, word, "VBZ")


COORDINATIONS = {
    "noun phrases": sentence(
        coordinate("NP", np("John"), "and", np("Mary")), leaf(r"S[dcl]\NP", "run", "VBP")
    ),
    "verb phrases": sentence(np("John"), coordinate(VP, verb("sings"), "and", verb("dances"))),
    "sentences": node(
        "S[dcl]",
        "rp",
        coordinate(
            "S[dcl]",
            node("S[dcl]", "ba", np("John"), verb("runs")),
            "and",
            node("S[dcl]", "ba", np("Mary"), verb("walks")),
        ),
        leaf(".", ".", "."),
    ),
    "comma list": sentence(
        np("John"),
        coordinate(
            VP,
            verb("sings"),
            ",",
            coordinate(VP, verb("dances"), "and", verb("runs")),
            marker_category=",",
        ),
    ),
}


def event(verb, subject="x"):
    return f"exists e.(_{verb}(e) & (Subj(e) = {subject}) & True)"


def test_coordinated_noun_phrases_and_sentences_conjoin_propositions():
    assert formula(COORDINATIONS["noun phrases"]) == (
        f"(exists x.(_john(x) & True & {event('run')})"
        f" & exists x.(_mary(x) & True & {event('run')}))"
    )
    assert formula(COORDINATIONS["sentences"]) == (
        f"(exists x.(_john(x) & True & {event('run')})"
        f" & exists x.(_mary(x) & True & {event('walk')}))"
    )


@pytest.mark.parametrize(
    ("name", "verbs"),
    [("verb phrases", ("sing", "dance")), ("comma list", ("sing", "dance", "run"))],
)
def test_coordinated_verb_phrases_share_their_subject(name, verbs):
    events = " & ".join(event(verb) for verb in verbs)
    assert formula(COORDINATIONS[name]) == f"exists x.(_john(x) & True & {events})"


def test_punctuated_type_changes_use_their_own_templates():
    fronted = node(
        "S[dcl]",
        "rp",
        node(
            "S[dcl]",
            "fa",
            node("S/S", "rp", node("NP", "lex", leaf("N", "Yesterday", "NN")), leaf(",", ",", ",")),
            node("S[dcl]", "ba", np("John"), leaf(VP, "ran", "VBD")),
        ),
        leaf(".", ".", "."),
    )
    trailing = sentence(
        np("John"),
        node(
            VP,
            "ba",
            leaf(VP, "ran", "VBD"),
            node(r"(S\NP)\(S\NP)", "lp", leaf(",", ",", ","), np("Mary")),
        ),
    )
    assert formula(fronted) == f"exists x.(_john(x) & True & {event('run')})"
    assert formula(trailing) == (
        f"(exists x.(_mary(x) & True & True) & exists x.(_john(x) & True & {event('run')}))"
    )


def test_failed_parses_keep_their_place_between_parsed_sentences():
    from neo_ccg2lambda.adapters.jigg.reading import read_sentences

    parsed = node("S[dcl]", "ba", np("John"), verb("runs"))
    success = ParseResult(parsed, -1, "success")
    failed = ParseResult(None, float("-inf"), "node_limit")
    sentences = read_sentences(jigg(((), success), (("Mary", "walks"), failed), ((), success)))
    assert [sentence.parse_failure for sentence in sentences] == [None, "node_limit", None]
    assert [len(sentence.derivations) for sentence in sentences] == [1, 0, 1]
    assert [token.base for token in sentences[1].tokens] == ["mary", "walk"]
