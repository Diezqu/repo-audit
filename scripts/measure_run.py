"""Measure one graph run: elapsed time, token usage, and distinct evidence states.

No observed concurrency or cost estimate is inferred from task count or
unconfigured provider pricing. A later evaluation script will add reproducible metrics.
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ENGINE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ENGINE_ROOT / ".env")

from repo_audit import config  # noqa: E402
from repo_audit.evaluation import TierUsage  # noqa: E402
from repo_audit.graph import build_graph  # noqa: E402

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
print("Token 用量为回调诊断值；失败或缺失用量时显示未知，不推算费用。")
usage = usage_cb.snapshot()
for tier, stats in sorted(usage.items()):
    input_text = f"{stats['input_tokens']:,}" if stats["input_tokens"] is not None else "未知"
    output_text = f"{stats['output_tokens']:,}" if stats["output_tokens"] is not None else "未知"
    print(f"  {tier:9s}: {stats['calls']:2d} 次调用 ({stats['error_calls']} 次失败), "
          f"输入 {input_text} tok, 输出 {output_text} tok")
total_in = (sum(stats["input_tokens"] for stats in usage.values())
            if all(stats["input_tokens"] is not None for stats in usage.values()) else None)
total_out = (sum(stats["output_tokens"] for stats in usage.values())
             if all(stats["output_tokens"] is not None for stats in usage.values()) else None)
print(f"  合计     : 输入 {f'{total_in:,}' if total_in is not None else '未知'} tok, "
      f"输出 {f'{total_out:,}' if total_out is not None else '未知'} tok")
