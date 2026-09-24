import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from repo_audit.evaluation import (
    TierUsage,
    _cost,
    load_corpus,
    measure_claims,
    run_evaluation,
    summarize_records,
    validate_corpus,
    validate_rates,
)

COMMIT = "f4ae8bb0af04cb315eef262d38433af4b71d9c38"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "a.py").write_text("one\ntwo\n")
    subprocess.run(["git", "-C", str(root), "add", "a.py"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=t@example.test",
         "commit", "-qm", "source"], check=True,
    )
    return root


def corpus_file(tmp_path, commit):
    path = tmp_path / "corpus.jsonl"
    record = {"id": "Q1", "question": "Where?", "repo_url": "https://example.test/public",
              "repo_commit": commit, "expected_answer": "one", "review_status":
              "draft_requires_human_review", "reference_citations": [
                  {"file": "a.py", "line_start": 1, "line_end": 2,
                   "snippet": "one\ntwo"}]}
    path.write_text(json.dumps(record) + "\n")
    return path


def test_corpus_requires_pinned_clean_matching_source(source, tmp_path):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    assert validate_corpus(load_corpus(corpus, expected_count=1), source, expected_commit=git(source, "rev-parse", "HEAD"))["reference_count"] == 1
    (source / "untracked").write_text("dirty")
    with pytest.raises(ValueError, match="dirty"):
        validate_corpus(load_corpus(corpus, expected_count=1), source, expected_commit=git(source, "rev-parse", "HEAD"))
    (source / "untracked").unlink()
    wrong = corpus_file(tmp_path, "0" * 40)
    with pytest.raises(ValueError, match="commit"):
        validate_corpus(load_corpus(wrong, expected_count=1), source, expected_commit=git(source, "rev-parse", "HEAD"))


def test_corpus_rejects_false_snippet_and_unsafe_path(source, tmp_path):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    record = json.loads(corpus.read_text())
    record["reference_citations"][0]["snippet"] = "wrong"
    corpus.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="snippet"):
        validate_corpus(load_corpus(corpus, expected_count=1), source, expected_commit=git(source, "rev-parse", "HEAD"))
    record["reference_citations"][0]["file"] = "../outside.py"
    corpus.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="path"):
        validate_corpus(load_corpus(corpus, expected_count=1), source, expected_commit=git(source, "rev-parse", "HEAD"))


def test_corpus_rejects_repo_subdirectory_even_when_reference_resolves_at_root(source, tmp_path):
    (source / "sub").mkdir()
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    with pytest.raises(ValueError, match="top-level"):
        validate_corpus(load_corpus(corpus, expected_count=1), source / "sub",
                        expected_commit=git(source, "rev-parse", "HEAD"))


def test_usage_missing_is_unavailable_and_parallel_counts_are_safe():
    usage = TierUsage()
    response = SimpleNamespace(llm_output={}, generations=[])
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda _: usage.on_llm_end(response, tags=["cheap"]), range(100)))
    assert usage.snapshot()["cheap"] == {
        "calls": 100, "input_tokens": None, "output_tokens": None,
        "usage_unavailable_calls": 100, "error_calls": 0,
    }
    usage.on_llm_end(SimpleNamespace(llm_output={"token_usage": {
        "prompt_tokens": 3, "completion_tokens": 5}}, generations=[]), tags=["cheap"])
    assert usage.snapshot()["cheap"]["input_tokens"] is None


def test_usage_failed_call_is_counted_and_makes_cost_unknown():
    usage = TierUsage()
    usage.on_llm_end(SimpleNamespace(llm_output={"token_usage": {
        "prompt_tokens": 10, "completion_tokens": 5}}, generations=[]), tags=["flagship"])
    usage.on_llm_error(TimeoutError("private credential"), tags=["cheap"], run_id=uuid.uuid4())
    stats = usage.snapshot()
    assert stats["flagship"]["calls"] == 1
    assert stats["cheap"] == {"calls": 1, "error_calls": 1, "input_tokens": None,
                              "output_tokens": None, "usage_unavailable_calls": 1}
    rates = {"currency": "USD", "source": "test", "date": "2026-09-24", "tiers": {
        tier: {"input_per_million": 1, "output_per_million": 1}
        for tier in ("cheap", "flagship")}}
    assert _cost(stats, rates) is None
    assert "private credential" not in json.dumps(stats)


def test_cost_never_returns_nonfinite_amount():
    rates = {"currency": "USD", "source": "test", "date": "2026-09-24", "tiers": {
        "cheap": {"input_per_million": 1e308, "output_per_million": 0},
        "flagship": {"input_per_million": 0, "output_per_million": 0}}}
    assert _cost({"cheap": {"input_tokens": 1000, "output_tokens": 0}}, rates) is None
    assert _cost({"cheap": {"input_tokens": 10 ** 1000, "output_tokens": 0}}, rates) is None


def test_evaluator_marks_worker_failures_without_misclassifying_insufficiency(source, tmp_path):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    commit = git(source, "rev-parse", "HEAD")
    def claim(worker_id, error=None):
        value = {"worker_id": worker_id, "statement": "No evidence", "status": "insufficient",
                 "citations": [], "citation_status": "missing"}
        if error:
            value["worker_error"] = error
        return value
    for name, claims, expected in (
        ("partial", [claim("w1", "ReadTimeout"), claim("w2")], "partial_failure"),
        ("all", [claim("w1", "ReadTimeout")], "error"),
        ("ordinary", [claim("w1")], "ok"),
    ):
        output = tmp_path / f"{name}.jsonl"
        result = run_evaluation(corpus, source, output, mode="demo", expected_count=1,
                                expected_commit=commit, invoke=lambda *_args: {
                                    "subtasks": ["a"], "claims": claims,
                                    "verified_claims": claims, "report": "partial report"})
        record = json.loads(output.read_text())
        assert record["status"] == expected
        assert record["failed_worker_count"] == (0 if name == "ordinary" else 1)
        assert record["failed_worker_ids"] == ([] if name == "ordinary" else ["w1"])
        assert record["report"] == "partial report"
        assert record["claims"] == claims
        assert record["measurement_eligible"] is False
        assert result["error_records"] == (0 if name == "ordinary" else 1)
        summary = summarize_records(output)
        assert summary["partial_failure_count"] == (1 if name == "partial" else 0)
        assert summary["error_count"] == (1 if name == "all" else 0)


def test_real_failed_callback_does_not_report_cost_or_success_latency(source, tmp_path, monkeypatch):
    commit = git(source, "rev-parse", "HEAD")
    corpus = corpus_file(tmp_path, commit)
    output = tmp_path / "partial-real.jsonl"
    for tier in ("CHEAP", "FLAGSHIP"):
        for field in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.setenv(f"{tier}_{field}", "test-only")

    def partial_graph(_question, _root, callbacks):
        usage = callbacks[0]
        usage.on_llm_end(SimpleNamespace(llm_output={"token_usage": {
            "prompt_tokens": 10, "completion_tokens": 5}}, generations=[]), tags=["flagship"])
        usage.on_llm_error(TimeoutError("private credential"), tags=["cheap"],
                           run_id=uuid.uuid4())
        claims = [
            {"worker_id": "w1", "status": "insufficient", "worker_error": "TimeoutError"},
            {"worker_id": "w2", "status": "insufficient"},
        ]
        return {"subtasks": ["a", "b"], "claims": claims, "verified_claims": claims,
                "report": "surviving report"}

    rates = {"currency": "USD", "source": "test", "date": "2026-09-24", "tiers": {
        tier: {"input_per_million": 1, "output_per_million": 1}
        for tier in ("cheap", "flagship")}}
    run_evaluation(corpus, source, output, mode="real", allow_api=True, expected_count=1,
                   expected_commit=commit, invoke=partial_graph, rates=rates)
    record = json.loads(output.read_text())
    assert record["status"] == "partial_failure"
    assert record["measurement_eligible"] is False
    assert record["usage_by_tier"]["cheap"]["error_calls"] == 1
    assert record["cost"] is None
    assert record["report"] == "surviving report"
    assert "private credential" not in output.read_text()
    assert summarize_records(output)["latency_seconds_by_question"] == {}


def test_claim_metrics_do_not_infer_semantic_accuracy():
    citation = {"file": "a.py", "line_start": 1, "line_end": 1, "snippet": "one"}
    claims = [
        {"status": "supported", "citation_status": "valid", "citations": [citation]},
        {"status": "supported", "citation_status": "invalid", "citations": [citation]},
        {"status": "insufficient", "citation_status": "missing", "citations": []},
    ]
    assert measure_claims(claims)["raw_supported_count"] == 2
    assert measure_claims(claims)["citation_valid_claim_count"] == 1
    assert measure_claims(claims)["raw_citation_count"] == 2
    assert measure_claims(claims)["citations_on_all_valid_claims"] == 1
    assert measure_claims(claims)["semantic_verified_count"] == 0


def test_run_records_error_and_source_change_without_success(source, tmp_path, monkeypatch):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    output = tmp_path / "results.jsonl"
    monkeypatch.delenv("CHEAP_API_KEY", raising=False)
    monkeypatch.delenv("FLAGSHIP_API_KEY", raising=False)

    def changing_graph(_question, _root, _callbacks):
        (source / "a.py").write_text("changed\n")
        return {"subtasks": [], "claims": [], "verified_claims": [], "report": "fake"}

    run_evaluation(corpus, source, output, mode="demo", repeats=1, expected_count=1,
                   expected_commit=git(source, "rev-parse", "HEAD"), invoke=changing_graph)
    record = json.loads(output.read_text())
    assert record["status"] == "source_changed"
    assert record["report"] is None
    (source / "a.py").write_text("one\ntwo\n")
    with pytest.raises(FileExistsError):
        run_evaluation(corpus, source, output, mode="demo", repeats=1, expected_count=1,
                       expected_commit=git(source, "rev-parse", "HEAD"), invoke=changing_graph)


def test_run_exception_is_sanitized_and_not_success(source, tmp_path, monkeypatch):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    output = tmp_path / "results.jsonl"
    monkeypatch.delenv("CHEAP_API_KEY", raising=False)
    monkeypatch.delenv("FLAGSHIP_API_KEY", raising=False)

    def failing_graph(*_args):
        raise RuntimeError("private secret token")

    run_evaluation(corpus, source, output, mode="demo", repeats=1, expected_count=1,
                   expected_commit=git(source, "rev-parse", "HEAD"), invoke=failing_graph)
    record = json.loads(output.read_text())
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert "private secret token" not in output.read_text()
    assert record["usage_by_tier"] == {}
    assert record["mode"] == "demo"
    assert record["quality_metrics_eligible"] is False


def test_run_non_dict_graph_result_is_sanitized(source, tmp_path):
    commit = git(source, "rev-parse", "HEAD")
    output = tmp_path / "non-dict.jsonl"
    run_evaluation(corpus_file(tmp_path, commit), source, output, mode="demo",
                   expected_count=1, expected_commit=commit, invoke=lambda *_: "bad result")
    record = json.loads(output.read_text())
    assert record["status"] == "error"
    assert record["error_type"] == "TypeError"
    assert record["claims"] == []


def test_real_requires_explicit_ack_and_both_tiers(source, tmp_path, monkeypatch):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    for name in ("CHEAP", "FLAGSHIP"):
        for key in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{name}_{key}", raising=False)
    with pytest.raises(ValueError, match="allow_api"):
        run_evaluation(corpus, source, tmp_path / "out.jsonl", mode="real", repeats=1,
                       expected_count=1, expected_commit=git(source, "rev-parse", "HEAD"), invoke=lambda *_: {})
    with pytest.raises(ValueError, match="CHEAP"):
        run_evaluation(corpus, source, tmp_path / "out.jsonl", mode="real", allow_api=True,
                       repeats=1, expected_count=1, expected_commit=git(source, "rev-parse", "HEAD"), invoke=lambda *_: {})
    assert not (tmp_path / "out.jsonl").exists()


def test_demo_clears_all_model_settings_and_does_not_call_api(source, tmp_path, monkeypatch):
    corpus = corpus_file(tmp_path, git(source, "rev-parse", "HEAD"))
    for name in ("CHEAP", "FLAGSHIP"):
        for key in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.setenv(f"{name}_{key}", "private-value")

    def offline_graph(_question, _root, _callbacks):
        import os

        assert os.environ["REPO_AUDIT_MODE"] == "demo"
        assert all(os.environ[f"{name}_{key}"] == "" for name in ("CHEAP", "FLAGSHIP")
                   for key in ("MODEL", "API_KEY", "BASE_URL"))
        return {"subtasks": [], "claims": [], "verified_claims": [], "report": "demo"}

    output = tmp_path / "demo.jsonl"
    run_evaluation(corpus, source, output, mode="demo", expected_count=1,
                   expected_commit=git(source, "rev-parse", "HEAD"), invoke=offline_graph)
    record = json.loads(output.read_text())
    assert record["status"] == "ok"
    assert record["quality_metrics_eligible"] is False
    assert record["models"] == {"cheap": None, "flagship": None}
    assert "private-value" not in output.read_text()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, False])
@pytest.mark.parametrize("field", ["input_per_million", "output_per_million"])
def test_rates_reject_invalid_prices(value, field):
    rates = {"currency": "USD", "source": "manual", "date": "2026-09-24", "tiers": {
        tier: {"input_per_million": 0, "output_per_million": 2.5}
        for tier in ("cheap", "flagship")
    }}
    rates["tiers"]["cheap"][field] = value
    with pytest.raises(ValueError, match="Invalid cheap"):
        validate_rates(rates)


def test_rates_accept_zero_integer_and_float_prices():
    rates = {"currency": "USD", "source": "manual", "date": "2026-09-24", "tiers": {
        "cheap": {"input_per_million": 0, "output_per_million": 2.5},
        "flagship": {"input_per_million": 3, "output_per_million": 4.5},
    }}
    assert validate_rates(rates) == rates


@pytest.fixture
def ignored_source(source):
    (source / ".gitignore").write_text("scratch.py\n")
    git(source, "add", ".gitignore")
    git(source, "-c", "user.name=test", "-c", "user.email=t@example.test",
        "commit", "-qm", "ignore scratch")
    return source


def test_run_rejects_ignored_source_files_before_invocation(ignored_source, tmp_path):
    source = ignored_source
    commit = git(source, "rev-parse", "HEAD")
    corpus = corpus_file(tmp_path, commit)
    (source / "scratch.py").write_text("uncommitted source\n")
    assert git(source, "status", "--porcelain", "--untracked-files=all") == ""
    output = tmp_path / "ignored.jsonl"

    def must_not_run(*_args):
        pytest.fail("Graph must not run against ignored source files")

    with pytest.raises(ValueError, match="ignored"):
        run_evaluation(corpus, source, output, mode="demo", expected_count=1,
                       expected_commit=commit, invoke=must_not_run)
    assert not output.exists()


def test_run_invalidates_result_when_ignored_file_appears(ignored_source, tmp_path):
    source = ignored_source
    commit = git(source, "rev-parse", "HEAD")
    corpus = corpus_file(tmp_path, commit)
    output = tmp_path / "changed.jsonl"

    def changing_graph(*_args):
        (source / "scratch.py").write_text("uncommitted source\n")
        return {"subtasks": [], "claims": [], "verified_claims": [], "report": "fake"}

    result = run_evaluation(corpus, source, output, mode="demo", repeats=3,
                            expected_count=1, expected_commit=commit, invoke=changing_graph)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["status"] == "source_changed"
    assert result["records_written"] == len(records) == 1
    assert records[0]["source_before"]["ignored"] is False
    assert records[0]["source_after"]["ignored"] is True
    assert records[0]["status"] == "source_changed"
    assert records[0]["measurement_eligible"] is False
    assert records[0]["report"] is None
