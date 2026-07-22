"""D2 烟测：端到端跑通整图（Planner → Worker×N → Synthesizer），对一个真实
仓库提出自然语言问题，验证至少一条 supported 结论的引用真的落在目标仓库里。

相比 D1 版本（scripts/smoke_worker.py 的上一版，见 git 历史）的两点变化：
1. repo_root 现在是 State 的显式字段（本次 T7 接线的一部分），不再需要
   os.chdir 把"当前目录"偷换成目标仓库——旧版靠 RepoSource 默认读 cwd 才
   需要这个 hack；新版 Worker 的工具循环直接从 WorkerInput.repo_root 拿
   路径，本脚本因此也不用再摆弄 cwd。
2. 烟测对象从"单个 worker() 函数"升级成"整张图"：D1 只需要验证 Worker
   能不能查到 file:line；D2 的 Worker 是手写工具循环 + Planner 拆子任务
   + Synthesizer 合成报告，三层任何一层接线接错都可能只在端到端层面才
   炸出来，所以这里跑 build_graph().invoke()，而不是只调 worker() 这一
   个节点。

用法：
    .venv/bin/python scripts/smoke_worker.py <目标仓库路径> "<自然语言问题>"
    例： .venv/bin/python scripts/smoke_worker.py . "graph.py 里 Worker 是怎么做工具循环的？"

判定标准：报告对应的 claims 里至少一条 status=supported 且带 citation；
用 T3 的 read_file 独立回读该 citation 的 file:line_start-line_end，确认
读出来非空——这一步不是复述 Worker 已经说过的话，是重新打开文件再核验一遍
"这个引用指向的位置真的存在内容"，某种程度上是明天才落地的 Verifier 的
最小可行版本（这里只核验"存在"，不核验"内容是否真支持结论"——那是 Verifier
要做的更深一层核验）。
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ENGINE_ROOT = Path(__file__).resolve().parent.parent

# 先加载引擎自己的 .env 再 import graph（config.py 在 import 时也会
# load_dotenv，但 dotenv 不覆盖已存在的变量，所以这里先到先得是安全的）。
load_dotenv(ENGINE_ROOT / ".env")

if len(sys.argv) < 3:
    print(f'用法：{sys.argv[0]} <目标仓库路径> "<自然语言问题>"')
    sys.exit(1)

target = Path(sys.argv[1]).resolve()
question = sys.argv[2]

if not target.is_dir():
    # 提前给一个可读的错误，而不是让 planner 里的 repo_tree/repo_stats 因为
    # os.walk/iterdir 撞上不存在的路径而甩出一截 Pregel 调用栈的 traceback——
    # 两者最终都 exit 1，但这里的信息量对"人到底填错了什么"更直接。
    print(f"目标路径不存在或不是目录：{target}")
    sys.exit(1)

from langfuse import get_client  # noqa: E402
from repo_audit import config  # noqa: E402
from repo_audit.graph import build_graph  # noqa: E402 （依赖上面 load_dotenv 先执行）
from repo_audit.repo_tools import PathEscapeError, read_file  # noqa: E402

# T8：Langfuse 观测埋点。无密钥时返回 None、不传 callbacks，行为与接入前
# 完全一样（设计说明见 config.langfuse_handler）。
handler = config.langfuse_handler()
invoke_kwargs = {"config": {"callbacks": [handler]}} if handler is not None else {}

try:
    result = build_graph().invoke(
        {"question": question, "repo_root": str(target)}, **invoke_kwargs
    )
    claims = result["claims"]
    report = result["report"]

    print(f"目标仓库: {target}    问题: {question}")
    print(f"共 {len(claims)} 条 claim\n")
    print(report)

    verified = False
    for claim in claims:
        if claim.status != "supported" or not claim.citations:
            continue
        for cite in claim.citations:
            try:
                text = read_file(target, cite.file, cite.line_start, cite.line_end)
            except (FileNotFoundError, PathEscapeError, ValueError) as exc:
                print(f"\n（引用回读失败，跳过：{cite.file}:{cite.line_start}-{cite.line_end} —— {exc}）")
                continue
            if text.strip() and "空范围" not in text:
                verified = True
                print(
                    f"\n独立复核通过：{cite.file}:L{cite.line_start}-{cite.line_end} 非空"
                    f"\n对应结论：{claim.statement}"
                )
                break
        if verified:
            break

    if not verified:
        print("\n烟测失败：没有任何一条 supported 结论的引用能被独立回读验证。")
        sys.exit(1)
    print("\n烟测通过：D2 验收线达成。")
finally:
    # 短脚本进程退出前必须显式收尾，理由与 graph.py __main__ 那份注释相同：
    # Langfuse 4.x 批量异步上报，不 flush 直接退出会丢 trace；放 finally
    # 而不是紧跟在 invoke() 后面，是为了上面任何一步抛异常（包括"烟测失败"
    # 那条 sys.exit(1)）时也能先把已产生的 trace 发出去再真正退出——失败
    # 的这次调用恰恰是最需要看观测数据排查的一次。
    if handler is not None:
        get_client().shutdown()
