import ctypes as ct
import threading
import weakref
from pathlib import Path

import numpy as np

from issen_ccg.domain.grammar import Grammar
from issen_ccg.domain.result import ParseResult
from issen_ccg.domain.tree import CcgTree


class NativeSearch:
    """Owns an independent native chart and serializes access to its scratch storage."""

    def __init__(self, grammar: Grammar):
        path = Path(__file__).with_name("_search.so")
        if not path.is_file():
            candidates = sorted(Path(__file__).parent.glob("_search.*.so"))
            if candidates:
                path = candidates[0]
        if not path.is_file():
            raise RuntimeError("Native search is not built; reinstall the package to compile it")
        self.grammar = grammar
        self.lock = threading.Lock()
        self.library = ct.CDLL(str(path))
        ints, floats = ct.POINTER(ct.c_int), ct.POINTER(ct.c_float)
        self.library.issen_create.argtypes = [
            ct.c_int,
            ct.c_int,
            ints,
            ct.c_int,
            ints,
            ct.c_int,
            ints,
        ]
        self.library.issen_create.restype = ct.c_void_p
        self.library.issen_destroy.argtypes = [ct.c_void_p]
        self.library.issen_parse.argtypes = [
            ct.c_void_p,
            ct.c_int,
            floats,
            floats,
            ct.c_float,
            ct.c_int,
            ct.c_int,
            ct.c_float,
            ints,
            floats,
        ]
        self.library.issen_parse.restype = ct.c_int
        self.library.issen_nodes.argtypes = [ct.c_void_p]
        self.library.issen_nodes.restype = ints
        unary = np.asarray(
            [(r.child, r.parent, 2 if r.name == "tr" else 1) for r in grammar.unary], dtype=np.int32
        )
        kinds = {"fa": 1, "ba": 2, "conj": 4}
        binary = np.asarray(
            [
                (r.left, r.right, r.parent, r.head_child, kinds.get(r.name, 0))
                for r in grammar.binary
            ],
            dtype=np.int32,
        )
        roots = np.asarray(grammar.roots, dtype=np.int32)
        self.pointer = self.library.issen_create(
            len(grammar.categories),
            len(unary),
            unary.ctypes.data_as(ints),
            len(binary),
            binary.ctypes.data_as(ints),
            len(roots),
            roots.ctypes.data_as(ints),
        )
        if not self.pointer:
            raise MemoryError("native grammar allocation failed")
        self.finalizer = weakref.finalize(self, self.library.issen_destroy, self.pointer)

    def parse(
        self,
        tokens,
        lexical_scores,
        head_scores,
        *,
        pos=None,
        arc_weight=1.0,
        max_nodes=300_000,
        timeout_ms=10_000,
        unary_cost=0.0,
    ):
        tokens = tuple(tokens)
        size = len(tokens)
        if lexical_scores.shape != (size, len(self.grammar.categories)):
            raise ValueError("lexical score shape does not match tokens and grammar")
        if head_scores.shape != (size, size):
            raise ValueError("head score shape does not match tokens")
        if (
            np.isnan(lexical_scores).any()
            or np.isposinf(lexical_scores).any()
            or not np.isfinite(head_scores).all()
        ):
            raise ValueError("scores must be finite (except excluded lexical categories)")
        if (
            not np.isfinite(arc_weight)
            or arc_weight < 0
            or not np.isfinite(unary_cost)
            or unary_cost < 0
        ):
            raise ValueError("search weights must be finite and nonnegative")
        if max_nodes < 1 or size > 512:
            raise ValueError("invalid search limit")
        if pos is not None and len(pos) != size:
            raise ValueError("POS sequence length does not match tokens")
        tags = np.ascontiguousarray(lexical_scores, dtype=np.float32)
        arcs = np.ascontiguousarray(head_scores, dtype=np.float32)
        metadata = (ct.c_int * 4)()
        score = ct.c_float()
        floats = ct.POINTER(ct.c_float)
        with self.lock:
            status = self.library.issen_parse(
                self.pointer,
                size,
                tags.ctypes.data_as(floats),
                arcs.ctypes.data_as(floats),
                arc_weight,
                max_nodes,
                timeout_ms,
                unary_cost,
                metadata,
                ct.byref(score),
            )
            if status == 4:
                raise RuntimeError("native search failed")
            tree = None
            if status == 0:
                records = np.ctypeslib.as_array(
                    self.library.issen_nodes(self.pointer), shape=(metadata[1] * 7,)
                ).reshape(-1, 7)
                built = []
                rules = (*self.grammar.unary, *self.grammar.binary)
                for category, begin, _, left, right, rule, head in records:
                    if left < 0:
                        built.append(
                            CcgTree(
                                self.grammar.categories[category],
                                token=tokens[begin],
                                pos=pos[begin] if pos is not None else "XX",
                            )
                        )
                    else:
                        children = (built[left],) if right < 0 else (built[left], built[right])
                        built.append(
                            CcgTree(
                                self.grammar.categories[category],
                                children,
                                head_child=int(head),
                                combinator=rules[rule].name,
                            )
                        )
                tree = built[metadata[0]]
        names = ("success", "no_parse", "node_limit", "timeout")
        return ParseResult(tree, float(score.value), names[status], metadata[2], metadata[3])
