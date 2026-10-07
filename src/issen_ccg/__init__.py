"""Independently learned combinatory categorial grammar parsing."""

from issen_ccg.adapters.parser_loading import load_parser
from issen_ccg.domain.result import ParseResult
from issen_ccg.domain.tree import CcgTree

__all__ = ["CcgTree", "ParseResult", "load_parser"]
