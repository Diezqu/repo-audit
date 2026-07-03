"""端到端骨架测试：假数据模式，无网络、无密钥。"""

from decision_engine.graph import build_graph


def _clear_keys(monkeypatch):
    for prefix in ("CHEAP", "FLAGSHIP"):
        for k in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{prefix}_{k}", raising=False)


def test_e2e_fake_mode(monkeypatch):
    _clear_keys(monkeypatch)
    result = build_graph().invoke({"question": "测试问题"})

    # 动态扇出：假 Planner 拆 2 个子问题 → 应派出 2 个 Worker
    assert {ev.worker_id for ev in result["evidence"]} == {"worker_1", "worker_2"}
    # 并行追加没有互相覆盖
    assert len(result["evidence"]) == 2
    # 报告成型且含原问题
    assert "测试问题" in result["report"]


def test_fanout_scales_with_subtasks(monkeypatch):
    """Send 扇出数量 = 子问题数量（而非写死的节点数）。"""
    _clear_keys(monkeypatch)
    from decision_engine.graph import fan_out

    sends = fan_out({"question": "q", "subtasks": ["a", "b", "c", "d"], "evidence": [], "report": ""})
    assert len(sends) == 4
    assert sends[2].arg["worker_id"] == "worker_3"
