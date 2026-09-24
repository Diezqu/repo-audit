import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from repo_audit.evaluation import (
    TierUsage,
    load_corpus,
    measure_claims,
    run_evaluation,
    validate_corpus,
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


def test_usage_missing_is_unavailable_and_parallel_counts_are_safe():
    usage = TierUsage()
    response = SimpleNamespace(llm_output={}, generations=[])
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda _: usage.on_llm_end(response, tags=["cheap"]), range(100)))
    assert usage.snapshot()["cheap"] == {
        "calls": 100, "input_tokens": None, "output_tokens": None,
        "usage_unavailable_calls": 100,
    }
    usage.on_llm_end(SimpleNamespace(llm_output={"token_usage": {
        "prompt_tokens": 3, "completion_tokens": 5}}, generations=[]), tags=["cheap"])
    assert usage.snapshot()["cheap"]["input_tokens"] is None


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
