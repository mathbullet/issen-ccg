import os
import sys
import types
from pathlib import Path

import pytest

from issen_ccg.adapters.morphodita import MorphoditaLemmatizer


class FakeTagger:
    def __init__(self, lemmas=None):
        self.lemmas = lemmas
        self.calls = 0

    def tag(self, forms, tagged):
        self.calls += 1
        lemmas = self.lemmas if self.lemmas is not None else [form.upper() for form in forms]
        tagged.extend(types.SimpleNamespace(lemma=lemma, tag="XX") for lemma in lemmas)


def install_fake_morphodita(monkeypatch, tagger):
    module = types.ModuleType("ufal.morphodita")
    module.Forms = list
    module.TaggedLemmas = list
    module.Tagger = types.SimpleNamespace(load=lambda path: tagger)
    package = types.ModuleType("ufal")
    package.morphodita = module
    monkeypatch.setitem(sys.modules, "ufal", package)
    monkeypatch.setitem(sys.modules, "ufal.morphodita", module)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "english.tagger"
    path.write_bytes(b"model")
    return path


def test_lemmas_follow_token_order_in_lowercase(monkeypatch, model):
    tagger = FakeTagger()
    install_fake_morphodita(monkeypatch, tagger)
    lemmatizer = MorphoditaLemmatizer(model)
    assert lemmatizer.lemmatize(["The", "women", "."]) == ("the", "women", ".")
    assert tagger.calls == 1


def test_empty_sentence_has_no_lemmas(monkeypatch, model):
    tagger = FakeTagger()
    install_fake_morphodita(monkeypatch, tagger)
    assert MorphoditaLemmatizer(model).lemmatize([]) == ()


@pytest.mark.parametrize(
    "tokens",
    ["The women", ["The", ""], ["The", "wo men"], ["The", 1]],
    ids=["sentence-string", "empty-token", "whitespace-token", "non-string-token"],
)
def test_invalid_tokens_are_rejected(monkeypatch, model, tokens):
    tagger = FakeTagger()
    install_fake_morphodita(monkeypatch, tagger)
    with pytest.raises(ValueError):
        MorphoditaLemmatizer(model).lemmatize(tokens)
    assert tagger.calls == 0


@pytest.mark.parametrize("lemmas", [["the"], ["the", ""]], ids=["count", "empty-lemma"])
def test_incompatible_morphodita_output_is_an_error(monkeypatch, model, lemmas):
    install_fake_morphodita(monkeypatch, FakeTagger(lemmas))
    with pytest.raises(RuntimeError):
        MorphoditaLemmatizer(model).lemmatize(["The", "women"])


def test_missing_model_reports_its_path(monkeypatch, tmp_path):
    install_fake_morphodita(monkeypatch, FakeTagger())
    missing = tmp_path / "missing.tagger"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        MorphoditaLemmatizer(missing)


def test_unreadable_model_is_an_error(monkeypatch, model):
    install_fake_morphodita(monkeypatch, None)
    with pytest.raises(RuntimeError, match=str(model)):
        MorphoditaLemmatizer(model)


def test_missing_dependency_explains_installation(monkeypatch, model):
    monkeypatch.setitem(sys.modules, "ufal", None)
    monkeypatch.setitem(sys.modules, "ufal.morphodita", None)
    with pytest.raises(RuntimeError, match=r"issen-ccg\[morphology\]"):
        MorphoditaLemmatizer(model)


@pytest.fixture(scope="module")
def real_lemmatizer():
    path = os.environ.get("ISSEN_MORPHODITA_MODEL")
    if not path:
        pytest.skip("ISSEN_MORPHODITA_MODEL is not set")
    return MorphoditaLemmatizer(Path(path))


def test_real_model_lemmatizes_an_english_sentence(real_lemmatizer):
    tokens = ("The", "women", "are", "sketching", "houses", ".")
    assert real_lemmatizer.lemmatize(tokens) == ("the", "woman", "be", "sketch", "house", ".")


def test_real_model_keeps_negative_prefixes(real_lemmatizer):
    assert real_lemmatizer.lemmatize(("She", "is", "unable", ".")) == ("she", "be", "unable", ".")
