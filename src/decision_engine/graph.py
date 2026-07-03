"""决策引擎 v1：Planner → 动态并行 Worker 池 → 合成器。

相比 v0 的两点升级：
1. 真模型接入：Planner/合成器走旗舰档、Worker 走便宜档（分级路由见 config.py）。
   没配 key 时自动退回假数据模式——骨架照样跑，CI 不需要密钥。
2. 动态扇出：v0 写死两个 Worker；现在 Planner 拆出几个子问题，
   运行时就现场派几个 Worker（LangGraph 的 Send 机制）。

诚实标注：本版 Worker 还没接搜索工具（M2 的活），只凭模型自身知识作答，
证据来源统一标 model://<模型名>，明示"未经外部检索核验"。
"""

import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from decision_engine import config


# ──────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────

class Evidence(BaseModel):
    """一条证据。结构化而非纯文本：Verifier 要按来源计票、报告要带引用。"""

    claim: str        # 这条证据支持的结论
    source_url: str   # 来源（M2 前为 model://，之后为真实 URL）
    excerpt: str      # 原文摘录（防转述失真；模型知识作答时为空）
    worker_id: str    # 哪个 Worker 带回来的


class State(TypedDict):
    """共享状态。evidence 声明为"并行追加"（operator.add）；
    其余字段都只有一个写入者，用默认的覆盖语义。"""

    question: str
    subtasks: list[str]
    evidence: Annotated[list[Evidence], operator.add]
    report: str


class WorkerInput(TypedDict):
    """派给单个 Worker 的任务书——Send 的载荷，不是全局 State。"""

    subtask: str
    worker_id: str


# ──────────────────────────────────────────────────────────────
# 节点
# ──────────────────────────────────────────────────────────────

class _Plan(BaseModel):
    subtasks: list[str] = Field(description="3-5 个可独立并行调研的子问题")


PLANNER_PROMPT = """你是决策调研的规划者。用户面临一个消费/选品决策，\
把它拆解成 3-5 个可以独立并行调研的子问题。要求：
- 覆盖面：产品本身的口碑与缺陷、替代品对比、与用户需求的匹配度、价格与渠道
- 每个子问题自包含：调研者看不到原问题也能独立去查

用户的决策问题：{question}"""


def planner(state: State) -> dict:
    """把大问题拆成可并行调研的子问题（旗舰档：拆解质量决定全局）。"""
    tier = config.try_flagship_tier()
    if tier is None:  # 假数据模式
        q = state["question"]
        return {"subtasks": [f"「{q}」的口碑与缺陷？", f"「{q}」有哪些替代品？"]}
    # method="function_calling"：DeepSeek 不支持 OpenAI 新的 json_schema
    # response_format（400: This response_format type is unavailable now），
    # 用工具调用协议拿结构化输出，两家都兼容。
    llm = tier.client(temperature=0).with_structured_output(_Plan, method="function_calling")
    plan = llm.invoke(PLANNER_PROMPT.format(question=state["question"]))
    return {"subtasks": plan.subtasks}


def fan_out(state: State) -> list[Send]:
    """Planner 完成后，按子问题数量现场派发 Worker。

    子问题有几个是运行时才知道的，所以不能用静态边，
    要用 Send：每个 Send = 一个并行分支 + 它专属的任务书。
    """
    return [
        Send("worker", WorkerInput(subtask=t, worker_id=f"worker_{i + 1}"))
        for i, t in enumerate(state["subtasks"])
    ]


class _Findings(BaseModel):
    findings: list[str] = Field(description="2-4 条独立、具体的调研结论，每条一句话")


WORKER_PROMPT = """你是调研员。回答下面这个子问题，给出 2-4 条独立、具体的结论。\
只说你有把握的事实；没把握的就明确说"不确定"，不要编造。

子问题：{subtask}"""


def worker(task: WorkerInput) -> dict:
    """领一个子问题，带证据回来（便宜档：跑量的活）。"""
    tier = config.try_cheap_tier()
    if tier is None:  # 假数据模式
        return {
            "evidence": [
                Evidence(
                    claim=f"关于[{task['subtask']}]的假证据结论",
                    source_url="fake://placeholder",
                    excerpt="",
                    worker_id=task["worker_id"],
                )
            ]
        }
    llm = tier.client(temperature=0).with_structured_output(_Findings, method="function_calling")
    found = llm.invoke(WORKER_PROMPT.format(subtask=task["subtask"]))
    source = f"model://{tier.model}"  # M2 接入真实搜索前的诚实标注
    return {
        "evidence": [
            Evidence(claim=c, source_url=source, excerpt="", worker_id=task["worker_id"])
            for c in found.findings
        ]
    }


SYNTH_PROMPT = """你是决策报告撰写者。基于下列证据，对用户的问题给出结论明确的决策建议。
要求：
- 结论先行（买 / 不买 / 买哪个），再给理由
- 每条论据后用 [n] 标注对应的证据编号
- 证据不足或互相矛盾的地方明说，不要编造

用户问题：{question}

证据清单：
{evidence}"""


def synthesizer(state: State) -> dict:
    """汇总全部证据 → 带引用的决策报告（旗舰档：最终质量门面）。"""
    numbered = "\n".join(
        f"[{i}] {ev.claim}（来源: {ev.source_url}，{ev.worker_id}）"
        for i, ev in enumerate(state["evidence"], 1)
    )
    tier = config.try_flagship_tier()
    if tier is None:  # 假数据模式
        return {"report": f"# 决策报告：{state['question']}\n\n{numbered}"}
    msg = tier.client(temperature=0).invoke(
        SYNTH_PROMPT.format(question=state["question"], evidence=numbered)
    )
    return {"report": f"{msg.content}\n\n---\n引用来源：\n{numbered}"}


# ──────────────────────────────────────────────────────────────
# 组装
# ──────────────────────────────────────────────────────────────

def build_graph():
    """START → planner ═(Send×N 动态并行)═> worker → synthesizer → END"""
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("worker", worker)
    g.add_node("synthesizer", synthesizer)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["worker"])
    g.add_edge("worker", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


if __name__ == "__main__":
    # 用法：python -m decision_engine.graph "该不该买 XXX"
    question = " ".join(sys.argv[1:]) or "该不该买 Kragg 软质一脚蹬"
    result = build_graph().invoke({"question": question})
    print(result["report"])
