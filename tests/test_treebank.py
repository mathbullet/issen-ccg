from issen_ccg.adapters.treebank import read_derivation


def test_auto_derivations_are_read():
    text = (
        r"(<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP John N>) ) "
        r"(<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBZ VBZ likes (S[dcl]\NP)/NP>) "
        r"(<L NP NNP NNP Mary NP>) ) )"
    )
    tree = read_derivation(text)
    assert tree.tokens == ("John", "likes", "Mary")
    assert tree.lexical_categories == ("N", "(S[dcl]\\NP)/NP", "NP")
    assert tree.leftmost_dependencies == (0, 0, 1)
    assert tree.pos_tags == ("NNP", "VBZ", "NNP")


def test_invalid_derivations_fail_explicitly():
    import pytest

    for text in ["", "(<T NP 0 3>)", "(<L NP NN NN x NP>) junk", "(<T NP 0 1>)"]:
        with pytest.raises(ValueError):
            read_derivation(text)
