"""Measure one graph run: elapsed time, token usage, and distinct evidence states.

No observed concurrency or cost estimate is inferred from task count or
unconfigured provider pricing. A later evaluation script will add reproducible metrics.
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
worker_tasks = len(result["subtasks"])
self_reported = sum(c.status == "supported" for c in claims)
valid_claims = sum(c.citation_status == "valid" for c in claims)
valid_citations = sum(len(c.citations) for c in claims if c.citation_status == "valid")
semantic_verified = sum(c.verdict == "supported" and c.citation_status == "valid" for c in claims)

print(f"\n{'=' * 58}\n计量结果：{target.name}    问题：{question[:40]}…\n{'=' * 58}")
print(f"耗时           : {elapsed:.1f}s")
print(f"Worker 任务数  : {worker_tasks} 个任务（不代表实际并发数）")
print(f"结论总数       : {len(claims)} 条结论")
print(f"Worker 自述支持: {self_reported} 条结论")
print(f"引用有效的结论 : {valid_claims} 条结论")
print(f"有效引用总数   : {valid_citations} 条引用")
print(f"语义已核验支持 : {semantic_verified} 条结论")
for tier, stats in sorted(usage_cb.stats.items()):
    print(f"  {tier:9s}: {stats['calls']:2d} 次调用, 输入 {stats['in']:>7,} tok, 输出 {stats['out']:>6,} tok")
total_in = sum(stats["in"] for stats in usage_cb.stats.values())
total_out = sum(stats["out"] for stats in usage_cb.stats.values())
print(f"  合计     : 输入 {total_in:,} tok, 输出 {total_out:,} tok")
