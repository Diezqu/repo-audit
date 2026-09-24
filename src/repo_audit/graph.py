"""Plan repository questions, collect worker claims, and verify citations.

Citation checks compare source coordinates and exact snippets. Semantic
support is not judged in this release; reports label that limit explicitly.
"""

import argparse
import html
import operator
import re
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from httpx import TransportError
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langfuse import get_client
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import APIError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from repo_audit import config
from repo_audit.repo_tools import (
    PathEscapeError,
    _resolve_within,
    repo_stats,
)
from repo_audit.repo_tools import (
    grep_repo as _grep_repo,
)
from repo_audit.repo_tools import (
    read_file as _read_file,
)
from repo_audit.repo_tools import (
    repo_tree as _repo_tree,
)

# ──────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────

_NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SubTask(BaseModel):
    """Planner 拆出的一个可并行、自包含的子任务（草案 §1.2）。

    对照消费版 `_Plan(subtasks: list[str])`：从"一串字符串"升成"带目标模块
    的对象"，因为 Worker 现在要知道去哪块代码查，而不是拿一句话满仓库乱翻。
    """

    target_module: _NonBlank  # 该子任务聚焦的目录/模块/文件（从目录树里挑）
    questions: list[_NonBlank] = Field(min_length=1, max_length=3)
    # v1 先不加 question_type 字段——本周没有消费方（Verifier 不看它、报告
    # 不按它分组），没有消费方的字段就是负债，见草案 §1.4 末段。


class Plan(BaseModel):
    subtasks: list[SubTask] = Field(
        min_length=1, max_length=8,
        description="1~8 个可并行、独立调研的子任务；仓库过小可酌减，不许注水凑数",
    )


class Citation(BaseModel):
    """Source coordinates and the exact excerpt supplied by a worker."""

    file: str
    line_start: int
    line_end: int
    snippet: str


class ClaimDraft(BaseModel):
    """Worker supplied data. Verification fields are never accepted here."""

    model_config = ConfigDict(extra="forbid")
    statement: _NonBlank
    status: Literal["supported", "insufficient"]
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def supported_needs_citation(self):
        if self.status == "supported" and not self.citations:
            raise ValueError("supported claim requires at least one citation")
        return self


class WorkerOutput(BaseModel):
    claims: list[ClaimDraft] = Field(
        description="每个子问题至少对应一条结论；查不实就出 insufficient，不许省略"
    )


class Claim(ClaimDraft):
    """Worker claim with runtime provenance and independent citation state.

    status is a worker self-report. citation_status checks source coordinates and
    exact text only. verdict is reserved for a future semantic judge and remains
    None in this deterministic release.
    """

    worker_id: str
    target_module: str
    worker_error: str | None = None
    citation_status: Literal["unchecked", "valid", "invalid", "missing", "unavailable"] = "unchecked"
    citation_reason: str | None = None
    verdict: str | None = None


class State(TypedDict):
    """共享状态。claims 声明为"并行追加"（operator.add）——Worker 的 Send
    扇入需要这个语义，多个并行 Worker 各自回填一段 claims，靠它拼成完整
    列表。

    verified_claims 故意是普通覆盖字段（无 reducer，最后一次写入生效），
    不是"claims 的另一份拷贝"这么简单——它是 verifier() 存在的直接原因：
    claims 的 reducer 是 operator.add，绑在这个 channel 上，对它的每一次
    写入都会触发（不分是不是来自 Send）；如果 verifier 也写 claims，
    LangGraph 会把它的产出与已有 claims 相加（列表相加=拼接），变成"一份
    没 verdict 的 + 一份有 verdict 的"两份重复结论，而不是替换。
    verified_claims 用默认覆盖语义，一次性整体替换，才是 verifier 真正
    需要的语义——claims 保留不动，含义收窄为"Worker 的原始产出，只服务
    Send 扇入"，synthesizer 改读 verified_claims（见 verifier() docstring
    的完整论证）。

    其余字段只有一个写入者，同样用默认的覆盖语义。"""

    question: str
    repo_root: str
    subtasks: list[SubTask]
    claims: Annotated[list[Claim], operator.add]
    verified_claims: list[Claim]
    report: str


class WorkerInput(TypedDict):
    """派给单个 Worker 的任务书——Send 的载荷，不是全局 State。repo_root
    显式放进载荷（而不是指望 Worker 从别处读全局 State）：Worker 是并行
    分支，只应该看到它自己需要的那一份上下文。"""

    subtask: SubTask
    worker_id: str
    repo_root: str


# ──────────────────────────────────────────────────────────────
# 节点：Planner
# ──────────────────────────────────────────────────────────────

def _readme_head(root: Path, *, max_lines: int = 60, max_chars: int = 2000) -> str:
    """Planner 地图三件套之一：README 开头（草案 §1.1）。README.md 优先于
    README.rst；两者都没有时显式写"（无 README）"而不是空字符串——空字符串
    塞进提示词模板会让这一段看起来像"没读到"，和"确实没有"是两回事，前者是
    该修的 bug、后者是这个仓库的真实状态，提示词读者（Planner 和读代码的人）
    都需要分得清。
    """
    for name in ("README.md", "README.rst"):
        path = root / name
        if not path.is_symlink() and path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[:max_lines])[:max_chars]
    return "（无 README）"


PLANNER_PROMPT = """你是代码库调研的规划者。你面前只有一个陌生仓库的「地图」——统计信息、浅层目录树、README 开头——你的职责是围绕用户问题拆出可以并行、独立调研的子任务，交给下游的 Worker 逐个去代码里取证。

用户问题：{question}

严格约束：
- 你只做拆解，不做回答。你看不到文件正文，任何结论都不该由你给出——读代码下结论是 Worker 的活。
- 每个子任务必须自包含：Worker 看不到这个原始全局目标，只拿到你写的这一条，也要能独立照着去查。
- 子任务之间尽量不重叠：它们会被并行分派，重叠等于花两份钱查同一件事。

拆分原则（两条一起用）：
1. 按模块边界拆——优先沿目录树里看得见的模块 / 子包 / 顶层目录切，一个子任务聚焦一块（例：checkpoint 持久化层、graph 编排层、CLI 入口层各一条）。
2. 按问题类型拆——覆盖下面四类，别全堆在"这是什么"上：
   - 功能定位：某模块/文件是干什么的、对外暴露什么接口
   - 机制原理：某能力具体怎么实现（关键数据结构、控制流、核心算法）
   - 边界条件：错误处理、并发、持久化、失败恢复这些容易踩的边界
   - 设计取舍：为什么这么设计、有没有可插拔/开关/降级的点

每个子任务给出两样：
- target_module：该子任务落在哪个目录/模块/文件（从上面的目录树里挑；拿不准就填最相关的顶层目录，别填一个树里没有的路径）
- questions：1~3 个"读这块代码就能回答"的具体问题。要能被 file:行号 回答，别问"介绍一下 X"这种泛问。

数量：1~8 个。按用户问题所需范围拆分，仓库很小就少拆几个（宁少毋凑）；很大也别超 8——把同一模块的问题合并成一条。

──────── 仓库地图 ────────
[统计]
{repo_stats}

[目录树 · 深度≤2]
{tree}

[README 开头]（可能为空）
{readme_head}"""


def planner(state: State) -> dict:
    """围绕用户问题拆成 1~8 个可并行调研的子任务（旗舰档：拆解质量
    决定全局，草案 §1.4）。"""
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonblank string")
    tier = config.try_flagship_tier()
    if tier is None:  # Explicit demo mode only.
        return {
            "subtasks": [
                SubTask(
                    target_module="src/",
                    questions=["这个目录对外暴露什么接口？", "有没有明显的边界条件处理？"],
                ),
                SubTask(target_module="tests/", questions=["测试覆盖了哪些行为？"]),
            ]
        }
    root = Path(state["repo_root"])
    prompt = PLANNER_PROMPT.format(
        question=question,
        repo_stats=repo_stats(root).render(),
        tree=_repo_tree(root, 2),
        readme_head=_readme_head(root),
    )
    # method="function_calling"：DeepSeek 不支持 OpenAI 新的 json_schema
    # response_format（400: This response_format type is unavailable now，
    # BATTLE_LOG 2026-07-03），用两家都兼容的工具调用协议取结构化输出。
    # 这条不许改——踩过一次真故障。
    llm = tier.client(temperature=0).with_structured_output(Plan, method="function_calling")
    plan = Plan.model_validate(llm.invoke(prompt))
    return {"subtasks": plan.subtasks}


def fan_out(state: State) -> list[Send]:
    """Planner 完成后，按子任务数量现场派发 Worker（Send 动态并行）。

    子任务有几个是运行时才知道的，所以不能用静态边，要用 Send：
    每个 Send = 一个并行分支 + 它专属的任务书。
    """
    return [
        Send(
            "worker",
            WorkerInput(subtask=task, worker_id=f"worker_{i + 1}", repo_root=state["repo_root"]),
        )
        for i, task in enumerate(state["subtasks"])
    ]


# ──────────────────────────────────────────────────────────────
# 节点：Worker
# ──────────────────────────────────────────────────────────────

# 查证类工具调用累计上限：grep 定位(1~2) + read 精读(2~3) + 追一层引用(2~3)
# ≈ 8（草案 §2.4）。真正的护栏不是"8"这个数字，是"到点即降级"这条规则——
# 数字是可调参数，评测阶段可回调；规则不变，诚实性不受影响。
MAX_QUERY_CALLS = 8

# 只有这三个是"查证"类工具，计入上限；submit_claims 是交卷动作，不计数。
_QUERY_TOOL_NAMES = {"repo_tree", "read_file", "grep_repo"}


class _TreeArgs(BaseModel):
    """repo_tree 的模型可见参数。字段名与 WORKER_PROMPT 里的描述文字逐字
    对齐（path/depth）——2026-07-22 真实调用验证过：提示词写 repo_tree(path,
    depth) 而 schema 只有 max_depth 时，DeepSeek 会照着提示词的名字传参，
    pydantic 默认 extra=ignore 把不认识的参数**静默扔掉**，模型以为在看
    src/fastmcp/tools 的子树、实际拿到的是整棵根目录树，还据此误判"目录
    不存在"。所以这里两条一起上：字段名对齐提示词 + extra="forbid"（真有
    对不上的参数就报错回给模型自己纠正，绝不静默吞掉——静默的错误比响亮
    的错误危险得多，这一课与 BATTLE_LOG 7/03"不能吃库默认值"同源）。"""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(".", description="要看的子目录（仓库相对路径），默认仓库根")
    depth: int = Field(2, description="从该目录往下展示的层数")


class _GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(..., description="正则表达式（Python 语义）")
    path_glob: str | None = Field(None, description="限定搜索范围的文件 glob，如 src/**/*.py")


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="仓库相对路径")
    start: int | None = Field(None, description="起始行号（1-based，闭区间）")
    end: int | None = Field(None, description="结束行号（闭区间）")


def _make_tools(root: Path) -> list:
    """把 T3 工具四件套用闭包绑定死 root，只把模型需要看见的参数暴露成
    工具签名（args_schema 见上面三个 _*Args——字段名与提示词描述逐字对齐，
    extra="forbid" 拒绝静默吞参）。

    root 完全不出现在暴露给模型的任何参数里——闭包绑定，而不是让模型
    自己填一个 root 参数。Worker 一次只服务一个 repo_root，模型的工具
    调用参数里不该有"填个别的目录试试"这个选项，哪怕 T3 的路径护栏
    （_resolve_within）最终仍会拦下越界访问，闭包绑定是更早、更干净的
    一层——两层防线不冲突，闭包挡意图，路径护栏挡万一。

    repo_tree 的子目录下钻：repo_tools.repo_tree 本身只会从传入的 root
    整棵渲染，这里用 _resolve_within 先把模型给的 path 安全解析到 root
    内部，再以该子目录为树根渲染——复用的是 T3 已有的路径护栏，不是新
    发明一层校验（2026-07-22 真实调用里模型第一轮就想看子树，这不是
    可有可无的便利，是它的实际工作方式）。"""

    @tool(args_schema=_TreeArgs)
    def repo_tree(path: str = ".", depth: int = 2) -> str:
        """看某个子目录的结构（默认仓库根），找该读哪些文件。"""
        target = _resolve_within(root, path)
        if not target.is_dir():
            raise FileNotFoundError(f"不是目录：{path}")
        return _repo_tree(target, depth)

    @tool(args_schema=_GrepArgs)
    def grep_repo(pattern: str, path_glob: str | None = None) -> str:
        """全库正则搜索，返回 file:行号 + 命中行——用它定位符号/关键字。"""
        return _grep_repo(root, pattern, path_glob)

    @tool(args_schema=_ReadArgs)
    def read_file(path: str, start: int | None = None, end: int | None = None) -> str:
        """读带行号的文件内容——用它精读命中处的上下文、拿到确切行号。"""
        return _read_file(root, path, start, end)

    @tool
    def submit_claims(claims: list[ClaimDraft]) -> str:
        """调查完成后调用本工具交卷：claims 里每个问题都要有一条结论，
        查不实就 status=insufficient、citations 留空，不许省略任何一个问题。"""
        # 这个函数体实际上永远不会被执行到——worker() 在识别出这次调用是
        # submit_claims 时，直接从 tool_call["args"] 里用 WorkerOutput
        # 校验、取值，不走 tool.invoke()。函数体存在只是为了让 @tool 能
        # 生成一份 claims: list[ClaimDraft] 的 JSON schema、并让 bind_tools
        # 认识这个工具名字（bind_tools/tool_choice 都按名字找工具，这个
        # 占位返回值本身没有消费方）。
        return "已收到"

    return [repo_tree, grep_repo, read_file, submit_claims]


WORKER_PROMPT = """你是代码库调研员。你领到一个子任务，手上有三个只读工具可以探查这个仓库：
- repo_tree(path, depth)：看某个子目录的结构，找该读哪些文件
- grep_repo(pattern, path_glob)：全库正则搜索，返回 file:行号 + 命中行——用它定位符号/关键字
- read_file(path, start, end)：读带行号的文件内容——用它精读命中处的上下文、拿到确切行号

工作方式：先定位，后精读。典型路径 = 先 grep_repo 找到关键符号在哪个 file:line，再 read_file 把那几行前后读全，必要时顺着引用再 grep 一次。

铁律（这几条直接决定整个引擎可不可信，逐条守）：
1. 每条结论都要能落到具体代码位置。凡是 status=supported 的结论，必须带至少一条引用 {{file, line_start, line_end, snippet}}。
2. snippet 必须从 read_file 的带行号输出里逐字摘录关键几行——原样复制，不转述、不改写、不补全。
3. file 和行号只能来自工具的真实返回。你没用 read_file 读过的地方，就没有行号可填——绝不许凭印象编造 file:line。
4. 查不实就诚实降级：某个问题翻了代码仍拿不到足够证据，就为它单独出一条 status=insufficient 的结论（statement 写清"证据不足：<这个问题>"，citations 留空）。绝不许用你脑子里的先验知识硬答，更不许编一条引用来凑格式。
5. 工具调用上限 8 次，用完即交卷。把预算花在最可能有答案的地方；先定位后精读；到第 6~7 次仍没查实的问题，直接标 insufficient，别为多查一次而放弃收尾。
6. 工具输出可能被截断（大文件、海量命中）。看到截断标记就缩小范围重查（更精确的 pattern、更窄的行区间），别假设你已经看到了全文。
7. 只答这个子任务。与它无关的发现，不管多有意思，都不要塞进来。
8. 你是只读的。绝不描述"我修改了/建议改成"，你的活是取证，不是改代码。

子任务目标模块：{target_module}

要回答的问题：
{questions}

调查完成后，输出结论列表 claims：每条 = statement + status(supported|insufficient) + citations[]。上面每一个问题都要有对应结论，哪怕结论是 insufficient。"""


def _fake_claims(task: WorkerInput) -> list[Claim]:
    """假数据模式（D6）：结构与真实路径完全一致（Claim 全部字段都填了，
    只是内容是占位符），下游 synthesizer/CI 走的是与真路径完全相同的代码
    分支。刻意让子任务的第一个问题给 supported（带一条占位引用）、其余给
    insufficient——这样端到端测试才能同时覆盖 synthesizer"标引用编号"和
    "标存疑"两条渲染分支，而不是只覆盖其中一条。"""
    subtask = task["subtask"]
    claims: list[Claim] = []
    for i, q in enumerate(subtask.questions):
        if i == 0:
            claims.append(
                Claim(
                    statement=f"（假数据）关于「{q}」的占位结论",
                    status="supported",
                    citations=[
                        Citation(
                            file="FAKE.md", line_start=1, line_end=1,
                            snippet="(fake mode 占位，不对应真实文件内容)",
                        )
                    ],
                    worker_id=task["worker_id"],
                    target_module=subtask.target_module,
                )
            )
        else:
            claims.append(
                Claim(
                    statement=f"（假数据）证据不足：{q}",
                    status="insufficient",
                    citations=[],
                    worker_id=task["worker_id"],
                    target_module=subtask.target_module,
                )
            )
    return claims


def _handle_tool_calls(
    ai_msg, tool_map: dict, messages: list, query_calls: int, *, allow_queries: bool = True
) -> tuple[WorkerOutput | None, int]:
    """执行一轮 AI 消息里的全部 tool_calls，把结果回填成 ToolMessage。

    一轮里可能同时出现多个 tool_calls（模型并行发起）——OpenAI 协议要求
    这一轮里每一个 tool_call_id 都必须有对应的 ToolMessage 回复，所以这里
    不能"找到合法 submit_claims 就提前返回"，必须把该轮全部 tool_calls
    处理完（该调用的调用、该报错的报错），只是在发现合法 submit_claims 时
    记下 output，跟其余 ToolMessage 一起处理完再返回。

    这里还必须处理 ai_msg.invalid_tool_calls（真实调用中踩到的故障，写
    在这里免得下一个接线的人重踩）：模型吐出的参数不是合法 JSON 时（
    submit_claims 的嵌套 schema 比三个查证工具复杂得多，最容易在这里
    出错），LangChain 把这次调用放进 invalid_tool_calls 而不是
    tool_calls，乍看好像可以不管——但 langchain_openai 把 AIMessage
    重新序列化回下一轮 API 请求时，会把 tool_calls 和 invalid_tool_calls
    合并进同一个 "tool_calls" 字段（见 langchain_openai.chat_models.base.
    _convert_message_to_dict），也就是说 API 侧仍然认为这个 tool_call_id
    需要一条回复。漏掉不回复，下一次 invoke() 就会 400："An assistant
    message with 'tool_calls' must be followed by tool messages responding
    to each 'tool_call_id'."——这正是本次接线唯一一处真实踩到的故障，是
    直接拿这仓库自身当测试目标、跑真实 API 才暴露出来的（假数据模式的
    单测不会调用真模型，测不到这条路径）。
    """
    output: WorkerOutput | None = None

    for invalid in ai_msg.invalid_tool_calls:
        if invalid.get("id") is None:
            continue  # 极端情况下连 id 都没有，没法定向回复给哪条调用，只能跳过
        if invalid.get("name") != "submit_claims":
            query_calls = min(MAX_QUERY_CALLS, query_calls + 1)
        messages.append(
            ToolMessage(
                content=f"调用解析失败（参数不是合法 JSON）：{invalid.get('error')}；请重新调用，"
                "确保参数是合法 JSON。",
                tool_call_id=invalid["id"],
            )
        )

    for tool_call in ai_msg.tool_calls:
        name = tool_call["name"]
        if name == "submit_claims":
            try:
                submitted = WorkerOutput.model_validate(tool_call["args"])
            except Exception as exc:  # noqa: BLE001 — 参数形状不对，回错让模型自己改，不崩
                messages.append(
                    ToolMessage(content=f"提交失败，参数不合法：{exc}", tool_call_id=tool_call["id"])
                )
                continue
            if output is None:
                output = submitted
            messages.append(ToolMessage(content="已收到", tool_call_id=tool_call["id"]))
            continue

        budget_available = query_calls < MAX_QUERY_CALLS
        query_calls = min(MAX_QUERY_CALLS, query_calls + 1)
        if name not in _QUERY_TOOL_NAMES or name not in tool_map:
            messages.append(
                ToolMessage(content=f"未知工具：{name}", tool_call_id=tool_call["id"])
            )
            continue
        if not allow_queries:
            messages.append(
                ToolMessage(content="交卷阶段不可调用查证工具", tool_call_id=tool_call["id"])
            )
            continue
        if not budget_available:
            messages.append(
                ToolMessage(content="查证工具预算已用尽", tool_call_id=tool_call["id"])
            )
            continue
        try:
            result = tool_map[name].invoke(tool_call["args"])
        except (PathEscapeError, ValueError, OSError) as exc:
            # 铁律 3/6 的下游消费方式：工具异常不是致命错误，是模型给错了
            # 参数（越界路径/非法正则/文件不存在），把错误文本原样回给
            # 模型自己调整，绝不让整个 Worker 崩掉。
            result = f"工具出错：{exc}"
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    return output, query_calls


def _provider_failure_claims(task: WorkerInput, exc: APIError | TransportError) -> list[Claim]:
    """Keep a failed worker in the report without exposing provider error text."""
    subtask = task["subtask"]
    return [
        Claim(
            statement=f"证据不足：{question}（模型调用失败：{type(exc).__name__}）",
            status="insufficient",
            citations=[],
            worker_id=task["worker_id"],
            target_module=subtask.target_module,
            worker_error=type(exc).__name__,
        )
        for question in subtask.questions
    ]


def worker(task: WorkerInput) -> dict:
    """领一个 SubTask：手写工具循环反复查证，最终收敛到 submit_claims
    （便宜档：跑量的活）。

    为什么手写循环、不用 create_react_agent：拍板已定——手写循环的每一步
    （数了几次查证工具调用、超限后怎么强制收尾、工具异常怎么喂回去)都是
    显式代码，面试时能逐行讲清楚；create_react_agent 把这整个循环封装成
    黑盒，出问题或要讲设计取舍时反而讲不清楚。这与 D2 选 LangGraph 而非
    CrewAI 的理由同源：显式状态机 > 封装好的黑盒。
    """
    subtask = task["subtask"]
    tier = config.try_cheap_tier()
    if tier is None:  # Explicit demo mode only.
        return {"claims": _fake_claims(task)}

    root = Path(task["repo_root"])
    tools = _make_tools(root)
    tool_map = {t.name: t for t in tools}
    base_llm = tier.client(temperature=0)
    llm = base_llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        HumanMessage(
            content=WORKER_PROMPT.format(
                target_module=subtask.target_module,
                questions="\n".join(f"- {q}" for q in subtask.questions),
            )
        )
    ]

    query_calls = 0
    output: WorkerOutput | None = None
    rounds = 0
    budget_warned = False
    # 安全帽：正常情况下靠 query_calls 计数退出；但如果模型完全不碰查证
    # 工具、只反复交出参数不合法的 submit_claims，query_calls 永远不会
    # 涨到上限，那条退出条件就失效了——rounds 是独立于"模型行为是否配合"
    # 的硬顶，保证无论模型怎么不听话，这里都不会真的死循环。这是"8 次
    # 查证上限"这条护栏之外，另加的一层工程纪律，草案没写但同一个精神：
    # 单人维护需要处处有硬闸（草案 §2.4"问 2"答题要点）。
    max_rounds = MAX_QUERY_CALLS + 4

    while output is None and query_calls < MAX_QUERY_CALLS and rounds < max_rounds:
        rounds += 1
        try:
            ai_msg = llm.invoke(messages)
        except (APIError, TransportError) as exc:
            return {"claims": _provider_failure_claims(task, exc)}
        messages.append(ai_msg)
        if not ai_msg.tool_calls and not ai_msg.invalid_tool_calls:
            break  # 模型放弃了工具协议、直接吐了段文字——没什么好等的，进强制交卷
        # 注意：只要 tool_calls/invalid_tool_calls 任一非空就必须先调用
        # _handle_tool_calls 把这一轮的每个 tool_call_id 都回复掉，不能因为
        # tool_calls 为空就跳过（invalid_tool_calls 非空时同样欠着 API 的
        # 回复，见 _handle_tool_calls 顶部的故障说明）——这也是为什么上面
        # 的 break 条件要两个都判空，才能安全地"什么都不用回复"。
        output, query_calls = _handle_tool_calls(ai_msg, tool_map, messages, query_calls)
        # 预算收敛提醒：2026-07-22 对 FastMCP 真跑发现，大仓库里模型会把
        # 8 次预算全花在"再查一点"上、从不主动交卷（小仓库 2~3 次就查完，
        # 暴露不了这个问题）。提示词里的铁律 5 写了"到点即交卷"，但模型
        # 对"已经用到第几次"没有稳定的自我感知——运行时在只剩 2 次时补一条
        # 显式提醒，把计数这件事从模型的记忆里挪到确定性代码里。只提醒
        # 一次：反复催会把上下文塞满同一句话。
        if output is None and not budget_warned and query_calls >= MAX_QUERY_CALLS - 2:
            budget_warned = True
            messages.append(
                HumanMessage(
                    content=f"提醒：查证调用已用 {query_calls}/{MAX_QUERY_CALLS} 次，只剩 "
                    f"{MAX_QUERY_CALLS - query_calls} 次。请立即收敛：已查实的结论整理成 "
                    "supported+引用，仍未查实的问题按铁律 4 标 insufficient，调用 "
                    "submit_claims 交卷。"
                )
            )

    if output is None:
        # 超限（或模型提前放弃协议）后的强制交卷。两个要点，都是 2026-07-22
        # 对 FastMCP 真跑时踩出来的（BATTLE_LOG 当日条目）：
        # 1. 只绑 submit_claims 这一个工具——第一版把四个工具全绑上再加
        #    tool_choice="submit_claims"，DeepSeek 对"指名工具"的 tool_choice
        #    根本不遵守，照样去调 read_file，强制形同虚设。把查证工具从
        #    绑定列表里拿掉才是结构性保证：模型想不配合也没有别的工具可调。
        #    （又一例"OpenAI 兼容是程度词"，同 BATTLE_LOG 7/03。）
        # 2. 附一条显式的交卷指令——tool_choice 靠不住，就让自然语言和
        #    工具列表一起把路收窄。
        # 换挡重试：先试指名 tool_choice（规范供应商一次就中），供应商不认
        # 这个形态（400 或被无视）就退到 "auto"——此时列表里也只有
        # submit_claims 一个选项。两轮都失败才落到最终的 insufficient 降级。
        messages.append(
            HumanMessage(
                content="查证预算已用尽，现在只剩 submit_claims 一个工具。立即交卷："
                "已查实的结论给 supported+引用（引用必须来自你前面真实读到的 file:行号），"
                "未查实的问题标 insufficient。不要输出其他内容。"
            )
        )
        submit_only = [tool_map["submit_claims"]]
        last_provider_error = None
        for choice in ("submit_claims", "auto"):
            forced_llm = base_llm.bind_tools(submit_only, tool_choice=choice)
            try:
                ai_msg = forced_llm.invoke(messages)
            except (APIError, TransportError) as exc:
                last_provider_error = exc
                continue
            last_provider_error = None
            messages.append(ai_msg)
            output, _ = _handle_tool_calls(
                ai_msg, tool_map, messages, query_calls, allow_queries=False
            )
            if output is not None:
                break
        if output is None and last_provider_error is not None:
            return {"claims": _provider_failure_claims(task, last_provider_error)}

    if output is None:
        # 强制那一轮仍没拿到合法结构化输出（模型没配合协议/参数校验失败）
        # ——诚实降级：每个问题单独出一条 insufficient，绝不让整个 Worker
        # 挂掉、也绝不编一条引用来凑数（铁律 4 的运行时兜底版本）。
        output = WorkerOutput(
            claims=[
                ClaimDraft(statement=f"证据不足：{q}", status="insufficient", citations=[])
                for q in subtask.questions
            ]
        )

    claims = [
        Claim(
            statement=c.statement,
            status=c.status,
            citations=c.citations,
            worker_id=task["worker_id"],
            target_module=subtask.target_module,
        )
        for c in output.claims
    ]
    return {"claims": claims}


# ──────────────────────────────────────────────────────────────
# 节点：Verifier（确定性引用检查）
# ──────────────────────────────────────────────────────────────

def _snippet_matches(citation: Citation, lines: list[str]) -> bool:
    """Accept literal source, or complete read_file rows at their stated lines."""
    if not citation.snippet.strip():
        return False
    source = "\n".join(lines[citation.line_start - 1:citation.line_end])
    if citation.snippet in source:
        return True

    previous = None
    excerpts = []
    for row in citation.snippet.split("\n"):
        prefix, separator, excerpt = row.partition("\t")
        try:
            number = int(prefix)
        except ValueError:
            return False
        if (
            not separator
            or prefix != f"{number:>6}"
            or not citation.line_start <= number <= citation.line_end
            or (previous is not None and number != previous + 1)
            or excerpt != lines[number - 1]
        ):
            return False
        previous = number
        excerpts.append(excerpt)
    return bool("\n".join(excerpts).strip())


def _judge_claim(
    root: Path, claim: Claim, cache: dict[Path, tuple[list[str] | None, str | None]] | None = None,
) -> tuple[Literal["valid", "invalid", "missing", "unavailable"], str | None]:
    """Check every citation against one source snapshot per file.

    Aggregation is all-or-nothing: any invalid citation makes the claim's
    citation state invalid. Read errors are unavailable, never semantic refutation.
    """
    if not claim.citations:
        return "missing", "no citations supplied"
    if cache is None:
        cache = {}
    failures: list[tuple[str, str]] = []
    for c in claim.citations:
        try:
            path = _resolve_within(root, c.file)
        except (PathEscapeError, OSError, ValueError):
            failures.append(("invalid", f"path outside repository: {c.file}"))
            continue
        if path not in cache:
            try:
                if not path.is_file():
                    cache[path] = (None, "file does not exist")
                else:
                    cache[path] = (path.read_text(encoding="utf-8", errors="replace").splitlines(), None)
            except OSError as exc:
                cache[path] = (None, f"source unreadable: {exc}")
        lines, error = cache[path]
        if error:
            kind = "unavailable" if error.startswith("source unreadable") else "invalid"
            failures.append((kind, f"{c.file}: {error}"))
            continue
        assert lines is not None
        if not 1 <= c.line_start <= c.line_end <= len(lines):
            failures.append(("invalid", f"{c.file}: invalid line range"))
            continue
        if not _snippet_matches(c, lines):
            failures.append(("invalid", f"{c.file}: snippet differs from source"))
    if failures:
        return next(((kind, reason) for kind, reason in failures if kind == "invalid"), failures[0])
    return "valid", None


def verifier(state: State) -> dict:
    """Populate citation evidence independently of worker status and semantics."""
    if not config.verifier_enabled():
        return {"verified_claims": [c.model_copy(update={
            "citation_status": "unchecked", "citation_reason": "verifier disabled", "verdict": None,
        }) for c in state["claims"]]}
    root = Path(state["repo_root"])
    cache: dict[Path, tuple[list[str] | None, str | None]] = {}
    verified = []
    for claim in state["claims"]:
        citation_status, citation_reason = _judge_claim(root, claim, cache)
        verified.append(claim.model_copy(update={
            "citation_status": citation_status, "citation_reason": citation_reason, "verdict": None,
        }))
    return {"verified_claims": verified}


# ──────────────────────────────────────────────────────────────
# 节点：Synthesizer
# ──────────────────────────────────────────────────────────────

def _report_text(value: str) -> str:
    """Keep external text on one line, with HTML and inline Markdown inert."""
    text = " ".join(value.split())
    return html.escape(re.sub(r"([\\`*_{}\[\]()#+!|~])", r"\\\1", text))


def _render_claims(claims: list[Claim]) -> tuple[str, str]:
    """Deterministically label claims; number only citation-valid references."""
    claim_lines: list[str] = []
    citation_lines: list[str] = []
    labels = {
        "valid": "引用有效，语义未核验",
        "invalid": "引用无效，语义未核验",
        "missing": "缺少引用，语义未核验",
        "unavailable": "引用无法读取，语义未核验",
        "unchecked": "引用未核验，语义未核验",
    }
    for claim in claims:
        label = labels[claim.citation_status]
        if claim.status == "insufficient" or claim.verdict == "refuted":
            label += "；存疑：证据不足"
        marks = []
        if claim.citation_status == "valid":
            for c in claim.citations:
                n = len(citation_lines) + 1
                marks.append(f"[{n}]")
                citation_lines.append(f"[{n}] {_report_text(c.file)}:L{c.line_start}-{c.line_end}")
        claim_lines.append(
            f"- {label} —— {_report_text(claim.statement)} {''.join(marks)}"
            f"（{_report_text(claim.target_module)}）"
        )
    return "\n".join(claim_lines) if claim_lines else "（无结论）", "\n".join(citation_lines)


def synthesizer(state: State) -> dict:
    """Compose a report from verified_claims without a model rewrite."""
    body, citations = _render_claims(state["verified_claims"])
    report = (
        f"# 代码库调研报告：{_report_text(state['question'])}\n\n"
        f"语义未核验；以下仅检查引用与源码是否一致。\n\n{body}"
    )
    if citations:
        report += f"\n\n---\n引用清单：\n{citations}"
    return {"report": report}


# ──────────────────────────────────────────────────────────────
# 组装
# ──────────────────────────────────────────────────────────────

def build_graph():
    """START → planner ═(Send×N 动态并行)═> worker → verifier → synthesizer → END"""
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("worker", worker)
    g.add_node("verifier", verifier)
    g.add_node("synthesizer", synthesizer)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["worker"])
    g.add_edge("worker", "verifier")
    g.add_edge("verifier", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a repository with cited evidence")
    parser.add_argument("--demo", action="store_true", help="run with fabricated sample claims")
    parser.add_argument("repo", nargs="?", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("question", nargs="*", help="natural-language repository question")
    args = parser.parse_args(argv)
    if args.demo:
        import os

        os.environ["REPO_AUDIT_MODE"] = "demo"
    root = Path(args.repo).resolve()
    if not root.is_dir():
        parser.error(f"repository path is not a directory: {root}")
    question = " ".join(args.question) if args.question else "这个仓库的整体架构是怎样的？"
    if not question.strip():
        parser.error("question must be nonblank")
    try:
        config.try_flagship_tier()  # validates both real tiers before graph/model calls
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    handler = config.langfuse_handler()
    invoke_kwargs = {"config": {"callbacks": [handler]}} if handler is not None else {}
    try:
        result = build_graph().invoke(
            {"question": question, "repo_root": str(root)}, **invoke_kwargs
        )
        print(result["report"])
        return 0
    finally:
        if handler is not None:
            get_client().shutdown()


if __name__ == "__main__":
    sys.exit(main())
