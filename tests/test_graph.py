"""Graph tests for worker output, deterministic citation checks, and reports."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from repo_audit import graph
from repo_audit.graph import (
    Citation,
    Claim,
    SubTask,
    _render_claims,
    build_graph,
    fan_out,
    planner,
    verifier,
    worker,
)


def _clear_keys(monkeypatch):
    monkeypatch.setenv("REPO_AUDIT_MODE", "demo")
    for prefix in ("CHEAP", "FLAGSHIP"):
        for k in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{prefix}_{k}", raising=False)


def test_planner_produces_subtask_structure(monkeypatch):
    _clear_keys(monkeypatch)
    result = planner({"question": "q", "repo_root": "."})
    subtasks = result["subtasks"]
    assert len(subtasks) >= 2
    for st in subtasks:
        assert isinstance(st, SubTask)
        assert st.target_module
        assert len(st.questions) >= 1


def test_real_planner_prompt_contains_users_question(monkeypatch, tmp_path):
    class CapturingLLM:
        def with_structured_output(self, schema, *, method):
            assert schema is graph.Plan
            assert method == "function_calling"
            return self

        def invoke(self, prompt):
            assert "How does the graph enforce tool budgets?" in prompt
            return {"subtasks": [{"target_module": "src/", "questions": ["How?"]}]}

    monkeypatch.setattr(
        graph.config, "try_flagship_tier",
        lambda: SimpleNamespace(client=lambda **kwargs: CapturingLLM()),
    )
    result = planner({"question": "How does the graph enforce tool budgets?", "repo_root": str(tmp_path)})
    assert result["subtasks"] == [SubTask(target_module="src/", questions=["How?"])]


@pytest.mark.parametrize("question", ["", "  \n  "])
def test_planner_rejects_blank_question(monkeypatch, question):
    _clear_keys(monkeypatch)
    with pytest.raises(ValueError, match="question"):
        planner({"question": question, "repo_root": "."})


@pytest.mark.parametrize(
    "subtasks",
    [
        [],
        [{"target_module": "src/", "questions": ["q"]}] * 9,
        [{"target_module": "   ", "questions": ["q"]}],
        [{"target_module": "src/", "questions": []}],
        [{"target_module": "src/", "questions": ["q"] * 4}],
        [{"target_module": "src/", "questions": ["  "]}],
    ],
)
def test_plan_rejects_invalid_subtasks(subtasks):
    with pytest.raises(ValidationError):
        graph.Plan.model_validate({"subtasks": subtasks})


def test_real_planner_validates_dict_returned_by_client(monkeypatch, tmp_path):
    class InvalidLLM:
        def with_structured_output(self, schema, *, method):
            return self

        def invoke(self, prompt):
            return {"subtasks": []}

    monkeypatch.setattr(
        graph.config, "try_flagship_tier",
        lambda: SimpleNamespace(client=lambda **kwargs: InvalidLLM()),
    )
    with pytest.raises(ValidationError):
        planner({"question": "What does this do?", "repo_root": str(tmp_path)})


def test_worker_produces_claim_structure(monkeypatch):
    _clear_keys(monkeypatch)
    subtask = SubTask(target_module="src/", questions=["q1", "q2"])
    result = worker({"subtask": subtask, "worker_id": "worker_1", "repo_root": "."})
    claims = result["claims"]
    assert len(claims) == 2
    for c in claims:
        assert isinstance(c, Claim)
        assert c.worker_id == "worker_1"
        assert c.target_module == "src/"
        assert c.status in ("supported", "insufficient")
    # 假数据模式刻意让第一个问题 supported+citation、其余 insufficient——
    # 这样一次 worker() 调用就能覆盖 synthesizer 需要用到的两种真实取值。
    assert claims[0].status == "supported"
    assert claims[0].citations
    assert claims[1].status == "insufficient"
    assert claims[1].citations == []


def test_query_batch_stops_at_eight_and_replies_to_every_call(tmp_path):
    (tmp_path / "a.py").write_text("only source line\n")
    tool_map = {tool.name: tool for tool in graph._make_tools(tmp_path)}
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.py"}, "id": call_id}
        for call_id in ("one", "two", "three")
    ])
    messages = []

    output, used = graph._handle_tool_calls(ai_msg, tool_map, messages, 7)

    assert output is None
    assert used == 8
    assert [msg.tool_call_id for msg in messages] == ["one", "two", "three"]
    assert "only source line" in messages[0].content
    assert all("预算" in msg.content for msg in messages[1:])


def test_malformed_query_consumes_budget_and_valid_call_can_recover(tmp_path):
    (tmp_path / "a.py").write_text("actual source\n")
    tool_map = {tool.name: tool for tool in graph._make_tools(tmp_path)}
    ai_msg = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "read_file", "args": "{bad", "id": "bad", "error": "invalid JSON"}
        ],
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "good"}],
    )
    messages = []

    _, used = graph._handle_tool_calls(ai_msg, tool_map, messages, 6)

    assert used == 8
    assert [msg.tool_call_id for msg in messages] == ["bad", "good"]
    assert "解析失败" in messages[0].content
    assert "actual source" in messages[1].content


def test_malformed_query_at_budget_limit_blocks_next_valid_call(tmp_path):
    (tmp_path / "a.py").write_text("must not be read\n")
    tool_map = {tool.name: tool for tool in graph._make_tools(tmp_path)}
    ai_msg = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "read_file", "args": "{bad", "id": "bad", "error": "invalid JSON"}
        ],
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "later"}],
    )
    messages = []

    _, used = graph._handle_tool_calls(ai_msg, tool_map, messages, 7)

    assert used == 8
    assert [msg.tool_call_id for msg in messages] == ["bad", "later"]
    assert "预算" in messages[1].content
    assert "must not be read" not in messages[1].content


def test_unknown_tool_returns_error_for_its_call_id(tmp_path):
    tool_map = {tool.name: tool for tool in graph._make_tools(tmp_path)}
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "unknown_source", "args": {}, "id": "unknown"}
    ])
    messages = []

    output, used = graph._handle_tool_calls(ai_msg, tool_map, messages, 0)

    assert output is None
    assert used == 1
    assert len(messages) == 1
    assert messages[0].tool_call_id == "unknown"
    assert "unknown_source" in messages[0].content


def test_file_read_os_error_returns_tool_message(tmp_path):
    class UnreadableTool:
        def invoke(self, args):
            raise OSError("file disappeared")

    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "vanished.py"}, "id": "read"}
    ])
    messages = []
    output, used = graph._handle_tool_calls(
        ai_msg, {"read_file": UnreadableTool()}, messages, 0
    )
    assert output is None
    assert used == 1
    assert messages[0].tool_call_id == "read"
    assert "file disappeared" in messages[0].content


def test_later_invalid_submit_does_not_erase_earlier_valid_submit(tmp_path):
    tool_map = {tool.name: tool for tool in graph._make_tools(tmp_path)}
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "submit_claims", "args": {"claims": [
            {"statement": "Evidence found", "status": "insufficient", "citations": []}
        ]}, "id": "valid"},
        {"name": "submit_claims", "args": {"claims": "invalid"}, "id": "invalid"},
    ])
    messages = []

    output, _ = graph._handle_tool_calls(ai_msg, tool_map, messages, 0)

    assert output.claims[0].statement == "Evidence found"
    assert [msg.tool_call_id for msg in messages] == ["valid", "invalid"]
    assert "提交失败" in messages[1].content


def test_closing_stage_refuses_source_queries_even_with_budget_left(tmp_path):
    (tmp_path / "a.py").write_text("unavailable in closing stage\n")
    tool_map = {tool.name: tool for tool in graph._make_tools(tmp_path)}
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.py"}, "id": "closing-query"}
    ])
    messages = []

    output, used = graph._handle_tool_calls(
        ai_msg, tool_map, messages, 0, allow_queries=False
    )

    assert output is None
    assert used == 1
    assert len(messages) == 1
    assert messages[0].tool_call_id == "closing-query"
    assert "unavailable in closing stage" not in messages[0].content
    assert "不可" in messages[0].content


def test_worker_closing_stage_does_not_run_source_query(monkeypatch, tmp_path):
    (tmp_path / "a.py").write_text("private source content\n")

    class ScriptedLLM:
        def __init__(self):
            self.responses = iter([
                AIMessage(content="No tools"),
                AIMessage(content="", tool_calls=[
                    {"name": "read_file", "args": {"path": "a.py"}, "id": "late-query"}
                ]),
                AIMessage(content="", tool_calls=[
                    {"name": "submit_claims", "args": {"claims": [
                        {"statement": "Not verified", "status": "insufficient", "citations": []}
                    ]}, "id": "submit"}
                ]),
            ])
            self.last_messages = []

        def bind_tools(self, tools, tool_choice=None):
            return self

        def invoke(self, messages):
            self.last_messages = list(messages)
            return next(self.responses)

    llm = ScriptedLLM()
    monkeypatch.setattr(
        graph.config, "try_cheap_tier",
        lambda: SimpleNamespace(client=lambda **kwargs: llm),
    )

    result = worker({
        "subtask": SubTask(target_module="a.py", questions=["What does it do?"]),
        "worker_id": "worker_1",
        "repo_root": str(tmp_path),
    })

    assert result["claims"][0].statement == "Not verified"
    late_reply = next(
        msg for msg in llm.last_messages
        if isinstance(msg, ToolMessage) and msg.tool_call_id == "late-query"
    )
    assert "不可" in late_reply.content
    assert "private source content" not in late_reply.content


def test_fanout_scales_with_subtasks(monkeypatch):
    """Send 扇出数量 = 子任务数量（而非写死的节点数）。"""
    _clear_keys(monkeypatch)
    subtasks = [SubTask(target_module=f"m{i}", questions=["q"]) for i in range(4)]
    state = {"question": "q", "repo_root": ".", "subtasks": subtasks, "claims": [], "report": ""}
    sends = fan_out(state)
    assert len(sends) == 4
    assert sends[2].arg["worker_id"] == "worker_3"
    assert sends[2].arg["repo_root"] == "."
    assert sends[2].arg["subtask"] is subtasks[2]



def _claim(*, citations=None, status="supported", verdict=None, citation_status="unchecked"):
    if citations is None:
        citations = [Citation(file="a.py", line_start=1, line_end=1, snippet="line1")]
    return Claim(statement="s", status=status, citations=citations,
                 worker_id="worker_1", target_module="m", verdict=verdict,
                 citation_status=citation_status)


def _verify(tmp_path, monkeypatch, claims, *, enabled=True):
    monkeypatch.setenv("VERIFIER_ENABLED", "true" if enabled else "false")
    return verifier({"repo_root": str(tmp_path), "claims": claims})["verified_claims"]


def test_worker_rejects_unsupported_submission(tmp_path):
    tool_map = {t.name: t for t in graph._make_tools(tmp_path)}
    msg = AIMessage(content="", tool_calls=[{"name": "submit_claims", "args": {
        "claims": [{"statement": "s", "status": "supported", "citations": []}]}, "id": "s"}])
    messages = []
    output, _ = graph._handle_tool_calls(msg, tool_map, messages, 0)
    assert output is None
    assert "提交失败" in messages[0].content


@pytest.mark.parametrize("statement,status", [("  ", "supported"), ("s", "refuted"), ("s", "unknown")])
def test_claim_draft_validates_worker_fields(statement, status):
    with pytest.raises(ValidationError):
        graph.ClaimDraft(statement=statement, status=status, citations=[])


def test_worker_cannot_set_verification_fields():
    with pytest.raises(ValidationError):
        graph.ClaimDraft.model_validate({"statement": "s", "status": "insufficient",
            "citations": [], "citation_status": "valid", "verdict": "supported"})


def test_verifier_validates_exact_source_without_semantic_verdict(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    c = _claim(citations=[Citation(file="a.py", line_start=1, line_end=1,
                                   snippet="     1\tline1")])
    result = _verify(tmp_path, monkeypatch, [c])[0]
    assert result.citation_status == "valid"
    assert result.verdict is None
    assert c.citation_status == "unchecked"


@pytest.mark.parametrize("snippet", [
    "999\talpha",
    "   999\talpha",
    "1\talpha",
    "     1\talpha",
    "     2\talpha\n     4\tbeta",
    "     2\talpha\n     2\tbeta",
    "     2\talpha\nbeta",
])
def test_verifier_rejects_forged_numbered_snippets(tmp_path, monkeypatch, snippet):
    (tmp_path / "a.py").write_text("123\talpha\nalpha\nbeta\n")
    claim = _claim(citations=[Citation(file="a.py", line_start=1, line_end=3,
                                       snippet=snippet)])
    result = _verify(tmp_path, monkeypatch, [claim])[0]
    assert result.citation_status == "invalid"


@pytest.mark.parametrize("snippet", [
    "123\talpha\nalpha",
    "     2\talpha\n     3\tbeta",
])
def test_verifier_preserves_literal_and_correctly_numbered_snippets(tmp_path, monkeypatch, snippet):
    (tmp_path / "a.py").write_text("123\talpha\nalpha\nbeta\n")
    claim = _claim(citations=[Citation(file="a.py", line_start=1, line_end=3,
                                       snippet=snippet)])
    assert _verify(tmp_path, monkeypatch, [claim])[0].citation_status == "valid"


def test_verifier_numbered_snippet_must_stay_within_citation_range(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("alpha\nalpha\n")
    claim = _claim(citations=[Citation(file="a.py", line_start=1, line_end=1,
                                       snippet="     2\talpha")])
    assert _verify(tmp_path, monkeypatch, [claim])[0].citation_status == "invalid"


@pytest.mark.parametrize("citation", [
    Citation(file="missing.py", line_start=1, line_end=1, snippet="line1"),
    Citation(file="../outside.py", line_start=1, line_end=1, snippet="line1"),
    Citation(file="a.py", line_start=1, line_end=9, snippet="line1"),
    Citation(file="a.py", line_start=2, line_end=1, snippet="line1"),
    Citation(file="a.py", line_start=1, line_end=1, snippet="forged"),
    Citation(file="a.py", line_start=1, line_end=1, snippet="  "),
])
def test_verifier_marks_bad_citation_invalid(tmp_path, monkeypatch, citation):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    result = _verify(tmp_path, monkeypatch, [_claim(citations=[citation])])[0]
    assert result.citation_status == "invalid"
    assert result.citation_reason
    assert result.verdict is None


def test_verifier_all_or_nothing(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("line1\n")
    cites = [Citation(file="a.py", line_start=1, line_end=1, snippet=s)
             for s in ("line1", "forged")]
    result = _verify(tmp_path, monkeypatch, [_claim(citations=cites)])[0]
    assert result.citation_status == "invalid"
    assert _render_claims([result])[1] == ""


def test_verifier_missing_citations(tmp_path, monkeypatch):
    result = _verify(tmp_path, monkeypatch, [_claim(citations=[], status="insufficient")])[0]
    assert result.citation_status == "missing"


def test_verifier_read_error_is_unavailable(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("line1\n")
    original = graph.Path.read_text
    def fail(path, *args, **kwargs):
        if path.name == "a.py":
            raise OSError("unreadable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(graph.Path, "read_text", fail)
    result = _verify(tmp_path, monkeypatch, [_claim()])[0]
    assert result.citation_status == "unavailable"


def test_verifier_reads_once_per_file_and_detects_changed_source(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("changed\n")
    original = graph.Path.read_text
    reads = []
    def count(path, *args, **kwargs):
        if path.name == "a.py":
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(graph.Path, "read_text", count)
    results = _verify(tmp_path, monkeypatch, [_claim(), _claim()])
    assert [r.citation_status for r in results] == ["invalid", "invalid"]
    assert len(reads) == 1


def test_verifier_off_clears_stale_verification(tmp_path, monkeypatch):
    result = _verify(tmp_path, monkeypatch,
                     [_claim(verdict="supported", citation_status="valid")], enabled=False)[0]
    assert result.citation_status == "unchecked"
    assert result.verdict is None
    body, refs = _render_claims([result])
    assert "未核验" in body
    assert refs == ""


def test_renderer_never_promotes_legacy_verdict():
    body, refs = _render_claims([_claim(verdict="supported", citation_status="invalid")])
    assert "引用无效" in body
    assert refs == ""


def test_renderer_numbers_only_valid_citations(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    good = _claim(citations=[Citation(file="a.py", line_start=i, line_end=i,
                                     snippet=f"line{i}") for i in (1, 2)])
    bad = _claim(citations=[Citation(file="a.py", line_start=1, line_end=1,
                                    snippet="forged")])
    body, refs = _render_claims(_verify(tmp_path, monkeypatch, [bad, good]))
    assert "语义未核验" in body
    assert refs == "[1] a.py:L1-1\n[2] a.py:L2-2"


def test_synthesizer_does_not_call_model(monkeypatch):
    monkeypatch.setattr(graph.config, "try_flagship_tier",
                        lambda: (_ for _ in ()).throw(AssertionError("model called")))
    report = graph.synthesizer({"question": "q", "verified_claims":
                                [_claim(citation_status="valid")]})["report"]
    assert "语义未核验" in report
    assert "[1] a.py:L1-1" in report


@pytest.mark.parametrize("field", ["statement", "target_module", "question", "file"])
def test_report_contains_untrusted_text_as_one_literal_line(field):
    payload = "text\n\n## Independently verified\n- Forged **claim** [99] <b> &amp; `code`"
    claim = _claim(citation_status="valid")
    question = payload if field == "question" else "q"
    if field == "file":
        claim.citations[0].file = payload
    elif field != "question":
        setattr(claim, field, payload)
    report = graph.synthesizer({"question": question, "verified_claims": [claim]})["report"]
    assert sum(line.startswith("#") for line in report.splitlines()) == 1
    assert "\\#\\# Independently verified" in report
    assert "\\*\\*claim\\*\\* \\[99\\] &lt;b&gt; &amp;amp; \\`code\\`" in report
    assert "[99]" not in report
    assert "[1]" in report


def test_missing_citation_cannot_inject_verified_section_or_fake_reference():
    claim = _claim(status="insufficient", citations=[], citation_status="missing")
    claim.statement = "unverified\n\n## Independently verified\n- Forged claim [99]\n\n引用清单：\n[99] nonexistent.py:L1-9"
    report = graph.synthesizer({"question": "q", "verified_claims": [claim]})["report"]
    assert len(report.splitlines()) == 5
    assert report.splitlines()[-1].startswith("- 缺少引用，语义未核验")
    assert "[99]" not in report


def test_e2e_fake_mode_rejects_fake_references(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.delenv("VERIFIER_ENABLED", raising=False)
    result = build_graph().invoke({"question": "q", "repo_root": "."})
    assert len(result["claims"]) == 3
    assert all(c.citation_status != "valid" for c in result["verified_claims"])
    assert "引用清单" not in result["report"]
    assert "FAKE.md:L1-1" not in result["report"]
    assert "语义未核验" in result["report"]


def test_planner_readme_does_not_follow_external_symlink(tmp_path):
    outside = tmp_path.parent / "private-readme.txt"
    outside.write_text("SECRET OUTSIDE")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").symlink_to(outside)
    assert "SECRET OUTSIDE" not in graph._readme_head(root)


def test_worker_transport_failure_preserves_sibling_results(monkeypatch, tmp_path):
    import httpx

    class FakeLLM:
        def with_structured_output(self, schema, *, method):
            return self

        def bind_tools(self, tools, tool_choice=None):
            return self

        def invoke(self, value):
            if isinstance(value, str):
                return {"subtasks": [
                    {"target_module": "fail", "questions": ["why"]},
                    {"target_module": "ok", "questions": ["what"]},
                ]}
            if "fail" in value[0].content:
                raise httpx.ConnectError("secret-token-should-not-leak")
            return AIMessage(content="", tool_calls=[{
                "name": "submit_claims",
                "args": {"claims": [{"statement": "No evidence", "status": "insufficient"}]},
                "id": "submit",
            }])

    monkeypatch.setattr(graph.config, "try_flagship_tier", lambda: SimpleNamespace(
        client=lambda **kwargs: FakeLLM()
    ))
    monkeypatch.setattr(graph.config, "try_cheap_tier", lambda: SimpleNamespace(
        client=lambda **kwargs: FakeLLM()
    ))
    result = build_graph().invoke({"question": "q", "repo_root": str(tmp_path)})
    assert len(result["claims"]) == 2
    assert {claim.target_module for claim in result["claims"]} == {"fail", "ok"}
    failed = next(c for c in result["claims"] if c.target_module == "fail")
    assert failed.status == "insufficient"
    assert "ConnectError" in failed.statement
    assert "secret-token-should-not-leak" not in result["report"]


def test_worker_programming_error_still_propagates(monkeypatch, tmp_path):
    class BrokenLLM:
        def bind_tools(self, tools, tool_choice=None):
            return self

        def invoke(self, messages):
            raise TypeError("bug in worker integration")

    monkeypatch.setattr(graph.config, "try_cheap_tier", lambda: SimpleNamespace(
        client=lambda **kwargs: BrokenLLM()
    ))
    task = {"subtask": SubTask(target_module="src", questions=["q"]),
            "worker_id": "w1", "repo_root": str(tmp_path)}
    with pytest.raises(TypeError, match="bug in worker integration"):
        worker(task)


def test_cli_missing_config_fails_before_graph_invocation(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("REPO_AUDIT_MODE", "real")
    for prefix in ("CHEAP", "FLAGSHIP"):
        for suffix in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    monkeypatch.setattr(graph, "build_graph", lambda: pytest.fail("graph must not start"))
    with pytest.raises(SystemExit) as raised:
        graph.main([str(tmp_path), "question"])
    assert raised.value.code != 0
    assert "Missing env vars" in capsys.readouterr().err


def test_cli_demo_runs_with_keys_and_disables_langfuse(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("REPO_AUDIT_MODE", "real")
    for prefix in ("CHEAP", "FLAGSHIP"):
        for suffix in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.setenv(f"{prefix}_{suffix}", "dummy")
    for suffix in ("PUBLIC_KEY", "SECRET_KEY", "HOST"):
        monkeypatch.setenv(f"LANGFUSE_{suffix}", "dummy")
    monkeypatch.setattr(graph.config, "CallbackHandler", lambda: pytest.fail("Langfuse started"))
    assert graph.main(["--demo", str(tmp_path), "question"]) == 0
    assert "假数据" in capsys.readouterr().out


def test_cli_rejects_blank_question(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        graph.main(["--demo", str(tmp_path), "   "])
    assert raised.value.code != 0
    assert "nonblank" in capsys.readouterr().err


def test_cli_rejects_missing_root(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        graph.main(["--demo", str(tmp_path / "missing"), "q"])
    assert raised.value.code != 0
    assert "not a directory" in capsys.readouterr().err


def test_worktree_git_metadata_file_is_not_evidence(tmp_path):
    from repo_audit.repo_tools import grep_repo, repo_stats, repo_tree

    (tmp_path / ".git").write_text("gitdir: SECRET METADATA\n")
    (tmp_path / "source.py").write_text("print('visible')\n")
    assert ".git" not in repo_tree(tmp_path)
    assert repo_stats(tmp_path).total_files == 1
    assert "SECRET METADATA" not in grep_repo(tmp_path, "SECRET")
