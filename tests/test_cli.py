import sys
import xml.etree.ElementTree as ET

import pytest

from issen_ccg.cli import main as cli
from issen_ccg.domain.result import ParseResult
from issen_ccg.domain.tree import CcgTree


class FakeParser:
    def __init__(self):
        self.chunks = []

    def parse(self, sentences):
        self.chunks.append(len(sentences))
        return [
            ParseResult(None, float("-inf"), "timeout")
            if tokens[0] == "fail"
            else ParseResult(CcgTree("N", token=tokens[0], pos="NN"), -1, "success")
            for tokens in sentences
        ]


class FakeLemmatizer:
    models = []

    def __init__(self, model):
        self.models.append(model)

    def lemmatize(self, tokens):
        return tuple(token.upper() for token in tokens)


@pytest.fixture
def run(monkeypatch, tmp_path):
    parsers = []
    FakeLemmatizer.models = []

    def load_parser(*args, **kwargs):
        parsers.append(FakeParser())
        return parsers[-1]

    monkeypatch.setattr(cli, "load_parser", load_parser)
    monkeypatch.setattr(cli, "MorphoditaLemmatizer", FakeLemmatizer)

    def run(lines, *arguments):
        source = tmp_path / "input.txt"
        source.write_text("".join(f"{line}\n" for line in lines))
        monkeypatch.setattr(
            sys, "argv", ["issen-ccg", "--model", "model", "--input", str(source), *arguments]
        )
        try:
            return cli.main(), parsers
        except SystemExit as error:
            return error.code, parsers

    return run


def token_attributes(output):
    return [
        [
            (token.get("surf"), token.get("base"), token.get("pos"))
            for token in sentence.iter("token")
        ]
        for sentence in ET.fromstring(output).iter("sentence")
    ]


def test_jigg_writes_lemmas_from_the_morphodita_model(run, capsys):
    code, _ = run(["women"], "--format", "jigg", "--morphodita-model", "english.tagger")
    assert code == 0
    assert token_attributes(capsys.readouterr().out) == [[("women", "WOMEN", "NN")]]
    assert [str(model) for model in FakeLemmatizer.models] == ["english.tagger"]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--format", "jigg"),
        ("--format", "auto", "--morphodita-model", "english.tagger"),
        ("--format", "json", "--morphodita-model", "english.tagger"),
    ],
    ids=["jigg-without-model", "auto-with-model", "json-with-model"],
)
def test_incompatible_morphodita_arguments_fail_before_loading_models(run, arguments):
    code, parsers = run(["women"], *arguments)
    assert code == 2
    assert parsers == []
    assert FakeLemmatizer.models == []


@pytest.mark.parametrize("output_format", ["auto", "json"])
def test_auto_and_json_do_not_use_morphodita(run, monkeypatch, capsys, output_format):
    monkeypatch.setitem(sys.modules, "ufal", None)
    monkeypatch.setitem(sys.modules, "ufal.morphodita", None)
    code, _ = run(["women"], "--format", output_format)
    assert code == 0
    assert "women" in capsys.readouterr().out
    assert FakeLemmatizer.models == []


def test_lemmas_stay_aligned_across_failures_and_chunks(run, capsys):
    lines = [f"w{index}" if index % 7 else "fail" for index in range(130)]
    code, parsers = run(lines, "--format", "jigg", "--morphodita-model", "english.tagger")
    assert code == 3
    assert parsers[0].chunks == [128, 2]
    assert len(FakeLemmatizer.models) == 1
    expected = [[(line, line.upper(), "XX" if line == "fail" else "NN")] for line in lines]
    assert token_attributes(capsys.readouterr().out) == expected


@pytest.mark.parametrize("output_format", ["auto", "json", "jigg"])
def test_sentence_failures_exit_apart_from_argument_errors(run, capsys, output_format):
    model = ("--morphodita-model", "english.tagger") if output_format == "jigg" else ()
    code, _ = run(["women", "fail", "men"], "--format", output_format, *model)
    assert code == 3
    output = capsys.readouterr()
    assert "women" in output.out and "men" in output.out
    assert "1 sentence(s) could not be parsed" in output.err


def test_fully_parsed_input_exits_successfully(run):
    code, _ = run(["women", "men"], "--format", "jigg", "--morphodita-model", "english.tagger")
    assert code == 0
