from pathlib import Path

from issen_ccg.adapters.grammar_file import load_grammar
from issen_ccg.adapters.native_search import NativeSearch
from issen_ccg.adapters.onnx_scorer import OnnxScorer
from issen_ccg.application.parser import ParseOptions, Parser


def load_parser(model: str | Path, *, threads=1, batch_size=16, options=None) -> Parser:
    directory = Path(model)
    scorer = OnnxScorer(directory, threads=threads, batch_size=batch_size)
    decoder = NativeSearch(load_grammar(directory / "grammar.json"))
    return Parser(scorer, decoder, options or ParseOptions())
