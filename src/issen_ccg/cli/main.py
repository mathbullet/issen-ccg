import argparse
import sys
from pathlib import Path

from issen_ccg.adapters.morphodita import MorphoditaLemmatizer
from issen_ccg.adapters.parser_loading import load_parser
from issen_ccg.adapters.tree_output import to_auto, to_jigg, to_json
from issen_ccg.application.parser import ParseOptions
from issen_ccg.ports.lemmatizer import Lemmatizer

# argparse already exits with 2 on usage errors.
SENTENCE_FAILURE_EXIT = 3


def main():
    cli = argparse.ArgumentParser(description="IssenCCG: independently trained English CCG parser")
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument(
        "--input", type=Path, help="one space-tokenized sentence per line (default stdin)"
    )
    cli.add_argument("--format", choices=("auto", "json", "jigg"), default="auto")
    cli.add_argument("--threads", type=int, default=1)
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--max-nodes", type=int, default=300_000)
    cli.add_argument("--timeout-ms", type=int, default=10_000)
    cli.add_argument(
        "--morphodita-model", type=Path, help="MorphoDiTa tagger producing Jigg lemmas"
    )
    args = cli.parse_args()
    if args.format == "jigg" and args.morphodita_model is None:
        cli.error("--format jigg requires --morphodita-model")
    if args.format != "jigg" and args.morphodita_model is not None:
        cli.error("--morphodita-model is only used with --format jigg")
    try:
        lemmatizer: Lemmatizer | None = (
            MorphoditaLemmatizer(args.morphodita_model) if args.format == "jigg" else None
        )
        options = ParseOptions(max_nodes=args.max_nodes, timeout_ms=args.timeout_ms)
        parser = load_parser(
            args.model, threads=args.threads, batch_size=args.batch_size, options=options
        )
        with args.input.open() if args.input else sys.stdin as source:
            sentences = [tuple(line.split()) for line in source if line.strip()]
        # Bound scorer scratch tensors and score retention for large documents.
        results = []
        lemmas = []
        failures = 0
        for begin in range(0, len(sentences), 128):
            chunk = sentences[begin : begin + 128]
            parsed = parser.parse(chunk)
            if lemmatizer is not None:
                results.extend(parsed)
                lemmas.extend(lemmatizer.lemmatize(tokens) for tokens in chunk)
            else:
                for index, (tokens, result) in enumerate(
                    zip(chunk, parsed, strict=True), begin + 1
                ):
                    if args.format == "json":
                        print(to_json(str(index), tokens, result))
                    elif result.success:
                        print(f"ID={index} log_probability={result.score}")
                        print(to_auto(result.tree))
                    else:
                        print(f"ID={index} status={result.status}")
            failures += sum(not result.success for result in parsed)
        if lemmatizer is not None:
            print(to_jigg(sentences, results, lemmas_by_sentence=lemmas))
        if failures:
            print(f"{failures} sentence(s) could not be parsed", file=sys.stderr)
            return SENTENCE_FAILURE_EXIT
    except (ValueError, OSError, RuntimeError) as error:
        cli.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
