import types

import numpy as np
import pytest

from issen_ccg.application.parser import MAX_SENTENCE_TOKENS, ParseOptions, Parser
from issen_ccg.domain.result import ParseResult
from issen_ccg.domain.tree import CcgTree
from issen_ccg.ports.scorer import SentenceScores


class FakeScorer:
    categories = ("N",)

    def __init__(self):
        self.lengths = []

    def score(self, sentences):
        self.lengths.append([len(sentence) for sentence in sentences])
        return [
            SentenceScores(
                np.zeros((len(sentence), 1), dtype=np.float32),
                np.zeros((len(sentence), len(sentence) + 1), dtype=np.float32),
                ("NN",) * len(sentence),
            )
            for sentence in sentences
        ]


class FakeDecoder:
    grammar = types.SimpleNamespace(categories=("N",), lexical_categories=())

    def parse(self, tokens, lexical, heads, **options):
        return ParseResult(CcgTree("N", token=tokens[0], pos="NN"), -1, "success")


def parser(scorer):
    return Parser(scorer, FakeDecoder(), ParseOptions(max_categories=1))


def test_sentences_up_to_the_token_limit_are_parsed():
    scorer = FakeScorer()
    results = parser(scorer).parse([("w",) * MAX_SENTENCE_TOKENS])
    assert MAX_SENTENCE_TOKENS == 512
    assert [result.status for result in results] == ["success"]
    assert scorer.lengths == [[512]]


def test_too_long_sentences_fail_without_dropping_other_sentences():
    scorer = FakeScorer()
    sentences = [("first",), ("w",) * (MAX_SENTENCE_TOKENS + 1), ("last",)]
    results = parser(scorer).parse(sentences)
    assert [result.status for result in results] == ["success", "too_long", "success"]
    assert [result.tree.token for result in results if result.success] == ["first", "last"]
    assert not results[1].success
    assert scorer.lengths == [[1, 1]]


def test_only_too_long_sentences_do_not_reach_the_scorer():
    scorer = FakeScorer()
    results = parser(scorer).parse([("w",) * (MAX_SENTENCE_TOKENS + 1)])
    assert [result.status for result in results] == ["too_long"]
    assert scorer.lengths == []


def test_empty_sentences_are_rejected():
    with pytest.raises(ValueError):
        parser(FakeScorer()).parse([()])
