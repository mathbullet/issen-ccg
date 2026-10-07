import json
from pathlib import Path

import numpy as np

from issen_ccg.adapters.treebank import normalize_word
from issen_ccg.ports.scorer import SentenceScores


def log_softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class OnnxScorer:
    def __init__(self, directory: Path, threads=1, batch_size=16):
        import onnxruntime as ort

        ort.disable_telemetry_events()

        if threads < 1 or batch_size < 1:
            raise ValueError("threads and batch size must be positive")
        vocabulary = json.loads((directory / "vocabulary.json").read_text())
        self.words = {word: i for i, word in enumerate(vocabulary["words"])}
        self.characters = {char: i for i, char in enumerate(vocabulary["characters"])}
        self.categories = tuple(vocabulary["categories"])
        self.pos_tags = tuple(vocabulary["pos_tags"])
        self.batch_size = batch_size
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(directory / "scorer.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def score(self, sentences):
        if any(not sentence or len(sentence) > 512 for sentence in sentences):
            raise ValueError("sentences must contain between 1 and 512 tokens")
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        output = [None] * len(sentences)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            lengths = np.asarray([len(sentences[i]) for i in indices], dtype=np.int64)
            width = int(lengths.max())
            words = np.zeros((len(indices), width), dtype=np.int64)
            chars = np.zeros((len(indices), width, 32), dtype=np.int64)
            for row, index in enumerate(indices):
                for column, surface in enumerate(sentences[index]):
                    word = normalize_word(surface)
                    words[row, column] = self.words.get(word.lower(), 1)
                    for position, char in enumerate(word[:32]):
                        chars[row, column, position] = self.characters.get(char, 1)
            tags, heads, pos = self.session.run(
                None, {"words": words, "characters": chars, "lengths": lengths}
            )
            for row, index in enumerate(indices):
                size = int(lengths[row])
                output[index] = SentenceScores(
                    log_softmax(tags[row, :size]),
                    log_softmax(heads[row, :size, :size]),
                    tuple(self.pos_tags[i] for i in pos[row, :size].argmax(-1)),
                )
        return output
