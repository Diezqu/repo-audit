"""Offline corpus validation and auditable, per-question evaluation records.

Draft reference answers are source-review aids, not human gold labels. This
module never scores answer correctness from strings or snippets.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from langchain_core.callbacks import BaseCallbackHandler

PINNED_COMMIT = "f4ae8bb0af04cb315eef262d38433af4b71d9c38"
PINNED_URL = "https://github.com/jlowin/fastmcp"
ENGINE_ROOT = Path(__file__).resolve().parents[2]
TIERS = ("cheap", "flagship")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect Git source ({args[0]})")
    return result.stdout.rstrip("\n")


def git_identity(root: Path) -> dict:
    """Capture the actual checkout commit and dirty state, including untracked files."""
    root = Path(root).resolve()
    return {
        "commit": _git(root, "rev-parse", "HEAD").strip(),
        "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=all")),
    }


def load_corpus(path: Path, *, expected_count: int = 10) -> list[dict]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid corpus JSON at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid corpus record at line {line_number}")
        records.append(record)
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} questions, found {len(records)}")
    ids = [record.get("id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Corpus ids must be unique nonempty strings")
    return records


def validate_corpus(
    records: list[dict], repo_root: Path, *, expected_commit: str = PINNED_COMMIT
) -> dict:
    """Check all draft references against the exact clean Git checkout."""
    identity = git_identity(repo_root)
    if identity["dirty"]:
        raise ValueError("Source checkout is dirty")
    if identity["commit"] != expected_commit:
        raise ValueError("Source commit does not match pinned commit")
    if len(expected_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_commit):
        raise ValueError("Expected commit must be a full lowercase SHA-1")
    reference_count = 0
    for record in records:
        if record.get("repo_commit") != expected_commit:
            raise ValueError(f"Corpus commit mismatch for {record.get('id')}")
        if expected_commit == PINNED_COMMIT and record.get("repo_url") != PINNED_URL:
            raise ValueError(f"Corpus repository URL mismatch for {record.get('id')}")
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError(f"Missing question for {record.get('id')}")
        if record.get("review_status") != "draft_requires_human_review":
            raise ValueError(f"Unexpected review status for {record.get('id')}")
        references = record.get("reference_citations")
        if not isinstance(references, list) or not references:
            raise ValueError(f"Missing draft references for {record.get('id')}")
        for citation in references:
            path = citation.get("file") if isinstance(citation, dict) else None
            if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or any(
                part in ("", ".", "..") for part in path.split("/")
            ):
                raise ValueError(f"Unsafe citation path for {record['id']}")
            start, end = citation.get("line_start"), citation.get("line_end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
                raise ValueError(f"Invalid citation line range for {record['id']}")
            source = _git(Path(repo_root), "show", f"{expected_commit}:{path}")
            lines = source.splitlines()
            if end > len(lines) or "\n".join(lines[start - 1:end]) != citation.get("snippet"):
                raise ValueError(f"Citation snippet mismatch for {record['id']} at {path}")
            reference_count += 1
    return {"question_count": len(records), "reference_count": reference_count,
            "repo_commit": expected_commit, "review_status": "draft_requires_human_review"}


class TierUsage(BaseCallbackHandler):
    """Thread-safe token accounting from returned model usage, with explicit gaps."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {}

    def on_llm_end(self, response, **kwargs):
        tags = kwargs.get("tags") or []
        tier = next((tag for tag in tags if tag in TIERS), "unknown")
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        if not usage:
            generations = getattr(response, "generations", [])
            first = next((item for group in generations for item in group), None)
            usage = getattr(getattr(first, "message", None), "usage_metadata", None) or {}
        input_count = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_count = usage.get("completion_tokens", usage.get("output_tokens"))
        complete = isinstance(input_count, int) and isinstance(output_count, int)
        with self._lock:
            stats = self._stats.setdefault(tier, {"calls": 0, "input_tokens": 0,
                                                  "output_tokens": 0,
                                                  "usage_unavailable_calls": 0})
            stats["calls"] += 1
            if complete:
                if stats["input_tokens"] is not None:
                    stats["input_tokens"] += input_count
                    stats["output_tokens"] += output_count
            else:
                stats["usage_unavailable_calls"] += 1
                stats["input_tokens"] = None
                stats["output_tokens"] = None

    def snapshot(self) -> dict:
        with self._lock:
            return {name: values.copy() for name, values in self._stats.items()}


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def measure_claims(claims: list) -> dict:
    claims = [_plain(claim) for claim in claims]
    valid = [claim for claim in claims if claim.get("citation_status") == "valid"
             and claim.get("citations")]
    return {
        "claim_count": len(claims),
        "raw_supported_count": sum(claim.get("status") == "supported" for claim in claims),
        "raw_citation_count": sum(len(claim.get("citations") or []) for claim in claims),
        "citation_valid_claim_count": len(valid),
        "citations_on_all_valid_claims": sum(len(claim["citations"]) for claim in valid),
        "semantic_verified_count": sum(claim.get("verdict") is not None for claim in claims),
    }


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("repo-audit", "langgraph", "langchain", "langchain-openai", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _models(mode: str) -> dict:
    if mode == "demo":
        return {tier: None for tier in TIERS}
    return {tier: os.environ[f"{tier.upper()}_MODEL"] for tier in TIERS}


def _check_real_settings() -> None:
    for tier in TIERS:
        prefix = tier.upper()
        missing = [f"{prefix}_{field}" for field in ("MODEL", "API_KEY", "BASE_URL")
                   if not os.getenv(f"{prefix}_{field}")]
        if missing:
            raise ValueError(f"Missing real-mode configuration: {', '.join(missing)}")


@contextmanager
def _run_environment(mode: str):
    names = ["REPO_AUDIT_MODE", "VERIFIER_ENABLED"]
    if mode == "demo":
        names.extend(f"{tier.upper()}_{field}" for tier in TIERS
                     for field in ("MODEL", "API_KEY", "BASE_URL"))
    old = {name: os.environ.get(name) for name in names}
    try:
        os.environ["REPO_AUDIT_MODE"] = mode
        os.environ["VERIFIER_ENABLED"] = "1"
        if mode == "demo":
            for name in names[2:]:
                os.environ[name] = ""  # load_dotenv() cannot refill an existing empty var
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _invoke_graph(question: str, root: Path, callbacks: list, max_concurrency: int):
    from repo_audit.graph import build_graph

    return build_graph().invoke(
        {"question": question, "repo_root": str(root)},
        config={"callbacks": callbacks, "max_concurrency": max_concurrency},
    )


def _cost(usage: dict, rates: dict | None) -> dict | None:
    if rates is None or not usage or any(
        item["input_tokens"] is None or item["output_tokens"] is None for item in usage.values()
    ):
        return None
    if any(tier not in rates["tiers"] for tier in usage):
        return None
    amount = sum(
        (item["input_tokens"] * rates["tiers"][tier]["input_per_million"]
         + item["output_tokens"] * rates["tiers"][tier]["output_per_million"]) / 1_000_000
        for tier, item in usage.items()
    )
    return {"amount": amount, "currency": rates["currency"],
            "rate_source": rates["source"], "rate_date": rates["date"]}


def validate_rates(rates: dict | None) -> dict | None:
    if rates is None:
        return None
    if not all(isinstance(rates.get(field), str) and rates[field] for field in
               ("currency", "source", "date")):
        raise ValueError("Rates require currency, source and date")
    tiers = rates.get("tiers")
    if not isinstance(tiers, dict) or any(tier not in tiers for tier in TIERS):
        raise ValueError("Rates require cheap and flagship tiers")
    for tier in TIERS:
        if not isinstance(tiers[tier], dict):
            raise ValueError(f"Invalid {tier} rates")
        for field in ("input_per_million", "output_per_million"):
            value = tiers[tier].get(field)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Invalid {tier} {field} rate")
    return rates


def run_evaluation(
    corpus_path: Path, repo_root: Path, output_path: Path, *, mode: str,
    allow_api: bool = False, repeats: int = 1, ids: set[str] | None = None,
    max_questions: int | None = None, max_concurrency: int = 8,
    rates: dict | None = None, expected_count: int = 10,
    expected_commit: str = PINNED_COMMIT, invoke=None,
) -> dict:
    """Write one JSONL record per attempted question/repeat; never overwrite results."""
    if mode not in ("demo", "real"):
        raise ValueError("mode must be demo or real")
    if mode == "real" and not allow_api:
        raise ValueError("Real mode requires allow_api acknowledgment")
    if mode == "real":
        _check_real_settings()
    if not 1 <= repeats <= 3 or max_concurrency < 1:
        raise ValueError("repeats must be 1–3 and max_concurrency positive")
    if max_questions is not None and max_questions < 1:
        raise ValueError("max_questions must be positive")
    rates = validate_rates(rates)
    corpus = load_corpus(corpus_path, expected_count=expected_count)
    repo_root = Path(repo_root).resolve()
    validate_corpus(corpus, repo_root, expected_commit=expected_commit)
    if ids is not None:
        unknown = ids - {item["id"] for item in corpus}
        if unknown:
            raise ValueError("Unknown question ids: " + ", ".join(sorted(unknown)))
        corpus = [item for item in corpus if item["id"] in ids]
    if max_questions is not None:
        corpus = corpus[:max_questions]
    if not corpus:
        raise ValueError("No questions selected")
    output_path = Path(output_path).resolve()
    if output_path.is_relative_to(ENGINE_ROOT) or output_path.is_relative_to(repo_root):
        raise ValueError("Results must be written outside the engine and source repositories")
    engine = git_identity(ENGINE_ROOT)
    corpus_sha256 = hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest()
    model_ids = _models(mode)
    versions = _versions()
    selected_count = len(corpus)
    try:
        from repo_audit.graph import MAX_QUERY_CALLS
    except ImportError:
        MAX_QUERY_CALLS = None
    budget = {"max_query_calls_per_worker": MAX_QUERY_CALLS,
              "max_concurrency": max_concurrency, "repeats": repeats}
    error_records = 0
    with output_path.open("x", encoding="utf-8") as output:
        for repeat in range(1, repeats + 1):
            for item in corpus:
                before = git_identity(repo_root)
                if before["dirty"] or before["commit"] != expected_commit:
                    raise ValueError("Source changed before next question; evaluation stopped")
                usage = TierUsage()
                started = time.perf_counter()
                result = None
                error_type = None
                try:
                    with _run_environment(mode):
                        result = (invoke(item["question"], repo_root, [usage]) if invoke else
                                  _invoke_graph(item["question"], repo_root, [usage], max_concurrency))
                    if not isinstance(result, dict):
                        raise TypeError("Graph returned a non-dict result")
                except Exception as exc:  # noqa: BLE001 - isolate one failed graph run
                    error_type = type(exc).__name__
                elapsed = time.perf_counter() - started
                after = git_identity(repo_root)
                stable = before == after and not after["dirty"]
                status = "source_changed" if not stable else ("error" if error_type else "ok")
                if status == "error":
                    error_records += 1
                raw_claims = _plain(result.get("claims", [])) if status == "ok" else []
                checked_claims = _plain(result.get("verified_claims", [])) if status == "ok" else []
                metrics = measure_claims(checked_claims) if status == "ok" else None
                subtasks = _plain(result.get("subtasks", [])) if status == "ok" else []
                tokens = usage.snapshot()
                record = {
                    "schema_version": 1, "question_id": item["id"], "question": item["question"],
                    "repeat": repeat, "status": status, "error_type": error_type,
                    "mode": mode, "measurement_eligible": mode == "real" and status == "ok",
                    "quality_metrics_status": "pending_human_labels",
                    "quality_metrics_eligible": False,
                    "human_gold_status": "pending", "repo_url": item["repo_url"],
                    "corpus_sha256": corpus_sha256,
                    "expected_repo_commit": expected_commit, "source_before": before,
                    "source_after": after, "engine": engine, "versions": versions,
                    "models": model_ids, "verifier": "deterministic_citation_check",
                    "budget": budget, "elapsed_seconds": elapsed,
                    "task_count": len(subtasks) if status == "ok" else None,
                    "configured_max_concurrency": max_concurrency,
                    "raw_claims": raw_claims, "claims": checked_claims,
                    "report": result.get("report") if status == "ok" else None,
                    "metrics": metrics, "usage_by_tier": tokens,
                    "cost": _cost(tokens, rates),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                if status == "source_changed":
                    return {"status": "source_changed", "records_written": (repeat - 1) *
                            selected_count + corpus.index(item) + 1, "output": str(output_path)}
    return {"status": "finished_with_errors" if error_records else "finished",
            "records_written": repeats * selected_count, "error_records": error_records,
            "output": str(output_path)}


def summarize_records(path: Path) -> dict:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    successful = [item for item in records if item.get("status") == "ok"]
    real = [item for item in successful if item.get("mode") == "real"]
    return {"record_count": len(records), "success_count": len(successful),
            "error_count": sum(item.get("status") == "error" for item in records),
            "source_changed_count": sum(item.get("status") == "source_changed" for item in records),
            "real_record_count": len(real), "demo_record_count": sum(
                item.get("mode") == "demo" for item in records),
            "quality_metrics_status": "pending_human_labels",
            "latency_seconds_by_question": {
                question_id: sorted(item["elapsed_seconds"] for item in real
                                    if item["question_id"] == question_id)
                for question_id in sorted({item["question_id"] for item in real})
            }}
