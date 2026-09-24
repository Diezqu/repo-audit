"""Validate a pinned corpus, run bounded evaluations, or summarize JSONL records."""

import argparse
import json
from pathlib import Path

from repo_audit.evaluation import load_corpus, run_evaluation, summarize_records, validate_corpus

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-corpus", help="check ten draft references, no API calls")
    validate.add_argument("--repo", type=Path, required=True, help="clean pinned source checkout")
    validate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    run = commands.add_parser("run", help="record one result per question and repeat")
    run.add_argument("--repo", type=Path, required=True, help="clean pinned source checkout")
    run.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run.add_argument("--output", type=Path, required=True, help="new JSONL file outside both repos")
    run.add_argument("--mode", choices=("demo", "real"), required=True)
    run.add_argument("--allow-api", action="store_true", help="acknowledge paid API calls in real mode")
    run.add_argument("--repeats", type=int, default=1, help="1–3; use 3 for latency range")
    run.add_argument("--id", action="append", dest="ids", help="select question id; repeat flag")
    run.add_argument("--max-questions", type=int, help="limit selected questions")
    run.add_argument("--max-concurrency", type=int, default=8)
    run.add_argument("--rates", type=Path, help="explicit JSON input/output rates by tier")

    summary = commands.add_parser("summarize", help="summarize records, without quality scoring")
    summary.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-corpus":
        result = validate_corpus(load_corpus(args.corpus), args.repo)
    elif args.command == "run":
        rates = json.loads(args.rates.read_text(encoding="utf-8")) if args.rates else None
        result = run_evaluation(
            args.corpus, args.repo, args.output, mode=args.mode, allow_api=args.allow_api,
            repeats=args.repeats, ids=set(args.ids) if args.ids else None,
            max_questions=args.max_questions, max_concurrency=args.max_concurrency, rates=rates,
        )
    else:
        result = summarize_records(args.records)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
