import importlib.util

import issen_ccg
from issen_ccg.adapters import parser_loading


def test_public_load_parser_reads_models_through_the_adapter():
    assert issen_ccg.load_parser is parser_loading.load_parser
    assert importlib.util.find_spec("issen_ccg.bootstrap") is None
