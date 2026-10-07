from dataclasses import dataclass

import numpy as np

from issen_ccg.domain.result import ParseResult
from issen_ccg.ports.decoder import Decoder
from issen_ccg.ports.scorer import Scorer, SentenceScores

MAX_SENTENCE_TOKENS = 512


@dataclass(frozen=True)
class ParseOptions:
    max_categories: int = 32
    category_margin: float = 12.0
    arc_weight: float = 1.0
    unary_cost: float = 0.1
    max_nodes: int = 300_000
    timeout_ms: int = 10_000

    def __post_init__(self):
        if not 1 <= self.max_categories or not 1 <= self.max_nodes:
            raise ValueError("candidate and node limits must be positive")
        if self.timeout_ms < 0:
            raise ValueError("timeout must be nonnegative")
        if any(
            not np.isfinite(value) or value < 0
            for value in (self.category_margin, self.arc_weight, self.unary_cost)
        ):
            raise ValueError("score weights and category margin must be finite and nonnegative")


class Parser:
    def __init__(self, scorer: Scorer, decoder: Decoder, options=None):
        options = options or ParseOptions()
        self.scorer, self.decoder, self.options = scorer, decoder, options
        if not 1 <= options.max_categories <= len(scorer.categories):
            raise ValueError("invalid category candidate limit")
        indices = {category: i for i, category in enumerate(decoder.grammar.categories)}
        self.mapping = np.asarray([indices[category] for category in scorer.categories])
        allowed = set(decoder.grammar.lexical_categories)
        self.allowed = np.asarray([not allowed or i in allowed for i in self.mapping])

    def decode(self, tokens, scores: SentenceScores):
        values = np.where(self.allowed[None, :], scores.categories, -np.inf)
        if values.shape != (len(tokens), len(self.mapping)):
            raise ValueError("scorer returned an incompatible category matrix")
        k = min(self.options.max_categories, values.shape[1])
        keep = np.argpartition(values, -k, axis=1)[:, -k:]
        rows = np.arange(len(tokens))[:, None]
        chosen = values[rows, keep]
        chosen = np.where(
            chosen >= values.max(-1, keepdims=True) - self.options.category_margin, chosen, -np.inf
        )
        lexical = np.full(
            (len(tokens), len(self.decoder.grammar.categories)), -np.inf, dtype=np.float32
        )
        lexical[rows, self.mapping[keep]] = chosen
        return self.decoder.parse(
            tuple(tokens),
            lexical,
            scores.heads,
            pos=scores.pos,
            arc_weight=self.options.arc_weight,
            unary_cost=self.options.unary_cost,
            max_nodes=self.options.max_nodes,
            timeout_ms=self.options.timeout_ms,
        )

    def parse(self, sentences):
        sentences = list(sentences)
        for sentence in sentences:
            if isinstance(sentence, str):
                raise ValueError("pass each sentence as a sequence of tokens, not a string")
            if any(
                not isinstance(token, str) or not token or any(c.isspace() for c in token)
                for token in sentence
            ):
                raise ValueError("tokens must be nonempty strings without whitespace")
            if not sentence:
                raise ValueError("sentences must contain at least one token")
        results = [ParseResult(None, float("-inf"), "too_long") for _ in sentences]
        parseable = [i for i, tokens in enumerate(sentences) if len(tokens) <= MAX_SENTENCE_TOKENS]
        if parseable:
            batch = [sentences[i] for i in parseable]
            for i, tokens, scores in zip(parseable, batch, self.scorer.score(batch), strict=True):
                results[i] = self.decode(tokens, scores)
        return results
