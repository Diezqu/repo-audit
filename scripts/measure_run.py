"""计量跑批：跑一次完整 ask，输出可上简历的硬数字（耗时/token/成本/并发/结论分布）。

为什么单独写这个脚本、而不是等 Langfuse：Langfuse 埋点已接（T8），但要
真看到 trace 得先注册账号填 key；在那之前，"一次 ask 到底花多少钱、并发
几个 Worker"这类数字不该是估的。这里用 LangChain 原生的回调把每次 LLM
调用的 token 用量按档累加——数据来源是各家 API 自己回的 usage 字段，不是
本地估算的 tokenizer 近似值。

用法：
    .venv/bin/python scripts/measure_run.py <目标仓库> "<问题>"
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

ENGINE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ENGINE_ROOT / ".env")

from repo_audit import config  # noqa: E402
from repo_audit.graph import build_graph  # noqa: E402

# 单价（元/百万 token）：DeepSeek 官网价，便宜档实际用的就是 deepseek-chat。
# 旗舰档按同一张表算——两档若配成同一家同一模型，成本差异体现在调用次数
# 与上下文长度上，而不是单价上；换供应商时改这里一处即可。
PRICE = {
    "cheap": {"in": 2.0, "out": 8.0},
    "flagship": {"in": 2.0, "out": 8.0},
}


class TierUsage(BaseCallbackHandler):
    """按模型档累加 token。tier 从 ChatOpenAI 构造时绑定的 tags 里读——
    与 T8 给 Langfuse 用的是同一份归因标签（config.ModelTier.client），
    不另建一套映射表。"""

    def __init__(self):
        self.stats: dict[str, dict] = {}

    def on_llm_end(self, response, **kwargs):
        tags = kwargs.get("tags") or []
        tier = next((t for t in tags if t in ("cheap", "flagship")), "unknown")
        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage:
            gens = [g for gl in response.generations for g in gl]
            meta = getattr(gens[0].message, "usage_metadata", None) if gens else None
            if meta:
                usage = {
                    "prompt_tokens": meta.get("input_tokens", 0),
                    "completion_tokens": meta.get("output_tokens", 0),
                }
        s = self.stats.setdefault(tier, {"calls": 0, "in": 0, "out": 0})
        s["calls"] += 1
        s["in"] += usage.get("prompt_tokens", 0)
        s["out"] += usage.get("completion_tokens", 0)


if len(sys.argv) < 3:
    print(f'用法：{sys.argv[0]} <目标仓库> "<问题>"')
    sys.exit(1)

target, question = Path(sys.argv[1]).resolve(), sys.argv[2]
usage_cb = TierUsage()
callbacks = [usage_cb]
if (lf := config.langfuse_handler()) is not None:
    callbacks.append(lf)

t0 = time.perf_counter()
result = build_graph().invoke(
    {"question": question, "repo_root": str(target)}, config={"callbacks": callbacks}
)
elapsed = time.perf_counter() - t0

claims = result["verified_claims"]
workers = {c.worker_id for c in claims}
supported = [c for c in claims if c.status == "supported"]
cites = sum(len(c.citations) for c in supported)
cost = sum(
    s["in"] / 1e6 * PRICE.get(t, PRICE["cheap"])["in"]
    + s["out"] / 1e6 * PRICE.get(t, PRICE["cheap"])["out"]
    for t, s in usage_cb.stats.items()
)

print(f"\n{'=' * 58}\n计量结果：{target.name}    问题：{question[:40]}…\n{'=' * 58}")
print(f"耗时           : {elapsed:.1f}s")
print(f"并发 Worker    : {len(workers)}（Planner 拆 {len(result['subtasks'])} 个子任务）")
print(f"结论           : {len(claims)} 条 = supported {len(supported)} / "
      f"insufficient {len(claims) - len(supported)}")
print(f"引用           : {cites} 条 file:line")
for tier, s in sorted(usage_cb.stats.items()):
    print(f"  {tier:9s}: {s['calls']:2d} 次调用, 输入 {s['in']:>7,} tok, 输出 {s['out']:>6,} tok")
total_in = sum(s["in"] for s in usage_cb.stats.values())
total_out = sum(s["out"] for s in usage_cb.stats.values())
print(f"  合计     : 输入 {total_in:,} tok, 输出 {total_out:,} tok")
print(f"成本           : ¥{cost:.4f}（按 DeepSeek 现价，见脚本 PRICE 表）")
