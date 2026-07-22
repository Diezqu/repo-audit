"""端到端骨架测试：假数据模式，无网络、无密钥。

覆盖点对应 T7 交付清单里点名要断言的三件事：
  - planner 产出 SubTask 结构（target_module + questions，不是消费版的字符串列表）；
  - worker 产出 Claim 结构（带 worker_id/target_module 的运行时装配版）；
  - 端到端 invoke 出 report，且引用清单格式正确（"[n] file:L起-L止"）。
"""

from repo_audit.graph import Claim, SubTask, build_graph, fan_out, planner, worker


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
