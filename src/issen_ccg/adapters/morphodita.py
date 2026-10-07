from collections.abc import Sequence
from pathlib import Path


class MorphoditaLemmatizer:
    def __init__(self, model: str | Path):
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(f"MorphoDiTa model not found: {path}")
        try:
            from ufal import morphodita
        except ImportError as error:
            raise RuntimeError(
                "MorphoDiTa is unavailable; install issen-ccg[morphology]"
            ) from error
        self._morphodita = morphodita
        self._tagger = morphodita.Tagger.load(str(path))
        if self._tagger is None:
            raise RuntimeError(f"Could not load MorphoDiTa model: {path}")

    def lemmatize(self, tokens: Sequence[str]) -> tuple[str, ...]:
        if isinstance(tokens, str):
            raise ValueError("pass a sequence of tokens, not a string")
        tokens = tuple(tokens)
        if any(
            not isinstance(token, str) or not token or any(c.isspace() for c in token)
            for token in tokens
        ):
            raise ValueError("tokens must be nonempty strings without whitespace")
        if not tokens:
            return ()
        tagged = self._morphodita.TaggedLemmas()
        self._tagger.tag(self._morphodita.Forms(tokens), tagged)
        lemmas = tuple(item.lemma.lower() for item in tagged)
        if len(lemmas) != len(tokens) or not all(lemmas):
            raise RuntimeError("MorphoDiTa returned lemmas incompatible with the tokens")
        return lemmas
