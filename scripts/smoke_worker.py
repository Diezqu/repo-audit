"""D1 烟测：单个 Worker 对一个真实仓库答出带 file:line 引用的子问题。

用法：
    .venv/bin/python scripts/smoke_worker.py <目标仓库路径> <关键字式子任务>
    例： .venv/bin/python scripts/smoke_worker.py /tmp/fastmcp add_tool

两个已知的过渡态（都是 D2 接 prompts-v2 工具循环时消掉的，不是 bug）：
1. RepoSource 目前把整句子任务当 grep pattern 用——所以子任务必须是
   「代码里真实出现的关键字」，中文自然语言问题会零命中、落到降级路径。
2. RepoSource 的 root 走 available_sources() 的 cwd 默认值——所以本脚本
   先加载引擎自己的 .env（否则 chdir 后 load_dotenv 找不到密钥），再
   chdir 进目标仓库。

判定标准（对应方案 E 冲刺日历 D1 验收线）：
  至少一条 Evidence 的 source_url 形如 "path:行号" —— 即结论真的落到了
  目标仓库的具体代码位置，而不是模型凭知识作答（那种来源是 model://）。
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ENGINE_ROOT = Path(__file__).resolve().parent.parent

# 顺序要求：先锁引擎的 .env，再 chdir，最后才 import graph（config 在 import
# 时也会 load_dotenv，但 dotenv 不覆盖已存在的变量，所以这里先到先得是安全的）。
load_dotenv(ENGINE_ROOT / ".env")

target = Path(sys.argv[1]).resolve()
subtask = sys.argv[2]
os.chdir(target)

from repo_audit.graph import worker  # noqa: E402（依赖上面的 env/chdir 顺序）

result = worker({"subtask": subtask, "worker_id": "smoke_1"})
evidence = result["evidence"]

cited = [e for e in evidence if re.search(r".+:\d+$", e.source_url)]
print(f"目标仓库: {target.name}    子任务: {subtask}")
print(f"证据 {len(evidence)} 条，其中带 file:line 引用 {len(cited)} 条\n")
for e in evidence:
    mark = "✅" if re.search(r".+:\d+$", e.source_url) else "▫️"
    print(f"{mark} {e.claim}")
    print(f"   来源: {e.source_url}")
    if e.excerpt:
        print(f"   摘录: {e.excerpt[:100]}")

if not cited:
    print("\n烟测失败：没有任何一条结论落到 file:line。")
    sys.exit(1)
print("\n烟测通过：D1 验收线达成。")
