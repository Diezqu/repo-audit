"""端到端骨架测试：假数据模式，无网络、无密钥。

覆盖点对应 T7 交付清单里点名要断言的三件事：
  - planner 产出 SubTask 结构（target_module + questions，不是消费版的字符串列表）；
  - worker 产出 Claim 结构（带 worker_id/target_module 的运行时装配版）；
  - 端到端 invoke 出 report，且引用清单格式正确（"[n] file:L起-L止"）。

D3 新增一块：Verifier 管道层（判定规则本体不在这里测——那部分今晚还没
实现，测的是"开关关=行为不变""开关开+假引用=refuted""存在性检查通过=
抛 NotImplementedError 且节点兜底不崩"这三件管道层该保证的事）。
"""

import pytest

from repo_audit.graph import (
    Citation,
    Claim,
    SubTask,
    _judge_claim,
    _render_claims,
    build_graph,
    fan_out,
    planner,
    verifier,
    worker,
)


def _clear_keys(monkeypatch):
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


def test_e2e_fake_mode(monkeypatch):
    _clear_keys(monkeypatch)
    result = build_graph().invoke({"question": "这个仓库是干什么的？", "repo_root": "."})

    # 假 Planner 拆 2 个子任务（2 问 + 1 问）→ 2 个 Worker → 3 条 claim
    assert {c.worker_id for c in result["claims"]} == {"worker_1", "worker_2"}
    assert len(result["claims"]) == 3

    report = result["report"]
    assert "这个仓库是干什么的？" in report
    # 末尾引用清单格式："[n] file:L起-L止"
    assert "引用清单" in report
    assert "[1] FAKE.md:L1-1" in report
    # insufficient 的结论显式标"存疑：证据不足"
    assert "存疑：证据不足" in report


def _make_claim(
    *, file="a.py", line_start=1, line_end=1,
    status="supported", citations=None, verdict=None,
):
    """测试用最小 Claim 工厂——D3 新增的测试大多只关心 citations/verdict，
    其余字段给个能用的默认值，减少每个用例里的重复样板。"""
    if citations is None:
        citations = [Citation(file=file, line_start=line_start, line_end=line_end, snippet="x")]
    return Claim(
        statement="s", status=status, citations=citations,
        worker_id="worker_1", target_module="m", verdict=verdict,
    )


# ──────────────────────────────────────────────────────────────
# D3：Verifier 管道层（判定规则本体不测——_judge_claim 里那段还没实现，
# 测的是"开关关=行为不变""开关开+假引用=refuted""存在性检查通过=抛
# NotImplementedError 且节点兜底不崩"这三件管道层该保证的事）
# ──────────────────────────────────────────────────────────────

def test_verifier_disabled_is_passthrough(monkeypatch):
    """开关默认关闭：verified_claims 与 claims 逐字一致，verdict 全部
    None——这是验收标准，不是顺带的优化：今天的行为必须和还没有 Verifier
    这一层完全一样。"""
    monkeypatch.delenv("VERIFIER_ENABLED", raising=False)
    claims = [_make_claim()]
    state = {"question": "q", "repo_root": ".", "subtasks": [], "claims": claims, "report": ""}
    result = verifier(state)
    assert result["verified_claims"] == claims
    assert all(c.verdict is None for c in result["verified_claims"])


def test_verifier_enabled_refutes_nonexistent_file(monkeypatch, tmp_path):
    """开关打开 + 假 claim 引用一个真实不存在的文件 → verdict=refuted。"""
    monkeypatch.setenv("VERIFIER_ENABLED", "true")
    claim = _make_claim(file="does_not_exist.py")
    state = {
        "question": "q", "repo_root": str(tmp_path), "subtasks": [],
        "claims": [claim], "report": "",
    }
    result = verifier(state)
    assert result["verified_claims"][0].verdict == "refuted"
    # 原 claim 对象不应被就地改动——verifier 返回的是新对象（model_copy）
    assert claim.verdict is None


def test_verifier_enabled_falls_back_to_none_on_notimplemented(monkeypatch, tmp_path):
    """存在性检查通过的合法引用 → _judge_claim 抛 NotImplementedError →
    节点兜底捕获、verdict=None，整个 run 不崩。"""
    monkeypatch.setenv("VERIFIER_ENABLED", "true")
    (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
    claim = _make_claim(file="a.py", line_start=1, line_end=2)
    state = {
        "question": "q", "repo_root": str(tmp_path), "subtasks": [],
        "claims": [claim], "report": "",
    }
    result = verifier(state)  # 不应抛异常
    assert result["verified_claims"][0].verdict is None


def test_verifier_enabled_needs_no_llm_tier(monkeypatch, tmp_path):
    """开关只读环境变量，不碰 config.try_*_tier——即便两档模型 key 都缺失
    （假数据模式），Verifier 打开时依然能正常跑存在性检查，因为它今晚
    根本不调用任何模型。"""
    for prefix in ("CHEAP", "FLAGSHIP"):
        for k in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{prefix}_{k}", raising=False)
    monkeypatch.setenv("VERIFIER_ENABLED", "true")
    claim = _make_claim(file="does_not_exist.py")
    state = {
        "question": "q", "repo_root": str(tmp_path), "subtasks": [],
        "claims": [claim], "report": "",
    }
    result = verifier(state)
    assert result["verified_claims"][0].verdict == "refuted"


def test_judge_claim_raises_notimplemented_when_citation_exists(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
    claim = _make_claim(file="a.py", line_start=1, line_end=2)
    with pytest.raises(NotImplementedError):
        _judge_claim(tmp_path, claim)


def test_judge_claim_nonexistent_file_is_refuted(tmp_path):
    claim = _make_claim(file="nope.py")
    assert _judge_claim(tmp_path, claim) == "refuted"


def test_judge_claim_path_escape_is_refuted(tmp_path):
    claim = _make_claim(file="../../../etc/passwd")
    assert _judge_claim(tmp_path, claim) == "refuted"


def test_judge_claim_out_of_range_lines_is_refuted(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    claim = _make_claim(file="a.py", line_start=1, line_end=99)
    assert _judge_claim(tmp_path, claim) == "refuted"


def test_judge_claim_reversed_range_is_refuted(tmp_path):
    """line_start > line_end——首尾颠倒的区间同样不可能是真实引用。"""
    (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
    claim = _make_claim(file="a.py", line_start=3, line_end=1)
    assert _judge_claim(tmp_path, claim) == "refuted"


def test_judge_claim_no_citations_raises_notimplemented(tmp_path):
    """insufficient 的结论没有 citations——存在性检查循环零次、无失败，
    照样落到同一个 NotImplementedError，不是这里替它判"不需要核验"。"""
    claim = _make_claim(status="insufficient", citations=[])
    with pytest.raises(NotImplementedError):
        _judge_claim(tmp_path, claim)


# ──────────────────────────────────────────────────────────────
# D3：_render_claims 的 verdict 优先渲染
# ──────────────────────────────────────────────────────────────

def test_render_claims_refuted_gets_rejected_style_and_no_citation():
    claim = _make_claim(verdict="refuted")
    body, citations_block = _render_claims([claim])
    assert "已核验驳回" in body
    assert claim.statement in body
    assert citations_block == ""  # 驳回的引用不进入可信引用清单，不给编号


def test_render_claims_verdict_supported_renders_as_supported():
    claim = _make_claim(verdict="supported")
    body, citations_block = _render_claims([claim])
    assert "存疑" not in body
    assert "[1]" in body
    assert citations_block == "[1] a.py:L1-1"


def test_render_claims_verdict_none_falls_back_to_status():
    """verdict=None 时沿用现状：完全按 status/citations 判断，行为与升级
    前的 _render_claims 逐字一致。"""
    supported = _make_claim(verdict=None, status="supported")
    insufficient = _make_claim(verdict=None, status="insufficient", citations=[])
    body, citations_block = _render_claims([supported, insufficient])
    assert "[1]" in body
    assert "存疑：证据不足" in body
    assert citations_block == "[1] a.py:L1-1"


def test_e2e_verifier_enabled_fake_mode(monkeypatch):
    """完整 e2e、Verifier 开着跑一次：Planner/Worker 仍是假数据模式（无
    密钥），假数据的第一条 claim 引用 "FAKE.md"——在真实 repo_root="."
    （本仓库）下并不存在，存在性检查应判它 refuted，报告要体现"已核验
    驳回"而不是原来的支持渲染。证明 verifier 作为静态边接入 build_graph
    后，三个开关外的节点（Planner/Worker/Synthesizer）不受影响、整条链路
    接得通。"""
    _clear_keys(monkeypatch)
    monkeypatch.setenv("VERIFIER_ENABLED", "true")
    result = build_graph().invoke({"question": "这个仓库是干什么的？", "repo_root": "."})
    report = result["report"]
    assert "已核验驳回" in report
    # claims（Worker 原始产出）不受 verifier 影响，仍是升级前的样子
    assert {c.worker_id for c in result["claims"]} == {"worker_1", "worker_2"}
    assert all(c.verdict is None for c in result["claims"])
