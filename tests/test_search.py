import numpy as np
import pytest

from issen_ccg.adapters.native_search import NativeSearch
from issen_ccg.domain.grammar import BinaryRule, Grammar, UnaryRule


def small_grammar():
    return Grammar(
        categories=("NP", "(S\\NP)/NP", "S\\NP", "S", "N"),
        unary=(UnaryRule(4, 0),),
        binary=(BinaryRule(1, 0, 2, "fa", 0), BinaryRule(0, 2, 3, "ba", 1)),
        roots=(3,),
    )


def test_search_scores_category_and_dependency_edges():
    decoder = NativeSearch(small_grammar())
    lexical = np.full((3, 5), -np.inf, dtype=np.float32)
    lexical[0, 4], lexical[1, 1], lexical[2, 0] = -0.2, -0.3, -0.4
    arcs = np.zeros((3, 3), dtype=np.float32)
    arcs[1, 0], arcs[2, 1] = -0.5, -0.6
    result = decoder.parse(("John", "likes", "Mary"), lexical, arcs)
    assert result.tree.tokens == ("John", "likes", "Mary")
    assert result.tree.category == "S"
    assert result.tree.leftmost_dependencies == (0, 0, 1)
    assert result.score == pytest.approx(-2.0)


def test_search_matches_exhaustive_best_derivation():
    grammar = Grammar(
        categories=("X",), unary=(), binary=(BinaryRule(0, 0, 0, "fa", 0),), roots=(0,)
    )
    decoder = NativeSearch(grammar)
    rng = np.random.default_rng(12)
    for length in range(2, 8):
        arcs = -rng.random((length, length), dtype=np.float32)
        lexical = -rng.random((length, 1), dtype=np.float32)
        scores = {}
        for size in range(1, length + 1):
            for begin in range(length - size + 1):
                end = begin + size
                scores[begin, end] = (
                    float(lexical[begin, 0])
                    if size == 1
                    else max(
                        scores[begin, middle] + scores[middle, end] + arcs[middle, begin]
                        for middle in range(begin + 1, end)
                    )
                )
        result = decoder.parse(tuple("x" for _ in range(length)), lexical, arcs)
        assert result.score == pytest.approx(scores[0, length], abs=1e-5)


def test_no_complete_parse_is_explicit():
    decoder = NativeSearch(small_grammar())
    lexical = np.full((1, 5), -np.inf, dtype=np.float32)
    lexical[0, 0] = 0
    result = decoder.parse(("John",), lexical, np.zeros((1, 1), dtype=np.float32))
    assert result.tree is None
    assert result.status == "no_parse"


def test_multiple_categories_match_full_dynamic_program():
    rules = (
        BinaryRule(0, 1, 0, "fa", 0),
        BinaryRule(1, 0, 1, "ba", 1),
        BinaryRule(0, 0, 1, "fa", 0),
        BinaryRule(1, 1, 0, "ba", 1),
    )
    decoder = NativeSearch(Grammar(("A", "B"), (), rules, (0,)))
    rng = np.random.default_rng(24)
    for length in range(2, 8):
        for _ in range(8):
            lexical = -rng.random((length, 2), dtype=np.float32)
            arcs = -rng.random((length, length), dtype=np.float32)
            chart = {(i, i + 1): lexical[i].copy() for i in range(length)}
            for size in range(2, length + 1):
                for begin in range(length - size + 1):
                    end = begin + size
                    values = np.full(2, -np.inf)
                    for middle in range(begin + 1, end):
                        for rule in rules:
                            value = (
                                chart[begin, middle][rule.left]
                                + chart[middle, end][rule.right]
                                + arcs[middle, begin]
                            )
                            values[rule.parent] = max(values[rule.parent], value)
                    chart[begin, end] = values
            result = decoder.parse(("x",) * length, lexical, arcs)
            assert result.score == pytest.approx(chart[0, length][0], abs=2e-5)


def test_node_limit_is_not_reported_as_a_success():
    decoder = NativeSearch(small_grammar())
    result = decoder.parse(
        ("x", "x"),
        np.zeros((2, 5), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        max_nodes=1,
    )
    assert result.status == "node_limit"
    assert result.tree is None
