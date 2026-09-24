"""Run a full graph smoke check against a repository.

Success means at least one worker-supported claim has citations that match
source text. Citation validity does not establish semantic support.
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

# T8：Langfuse 观测埋点。无密钥时返回 None、不传 callbacks，行为与接入前
# 完全一样（设计说明见 config.langfuse_handler）。
handler = config.langfuse_handler()
invoke_kwargs = {"config": {"callbacks": [handler]}} if handler is not None else {}

try:
    result = build_graph().invoke(
        {"question": question, "repo_root": str(target)}, **invoke_kwargs
    )
    claims = result["verified_claims"]
    report = result["report"]

    print(f"目标仓库: {target}    问题: {question}")
    print(f"共 {len(claims)} 条 claim\n")
    print(report)

    valid_claims = [
        claim for claim in claims
        if claim.status == "supported" and claim.citation_status == "valid"
    ]
    if not valid_claims:
        print("\n烟测失败：没有引用有效的 Worker supported 结论。")
        sys.exit(1)
    print(f"\n烟测通过：{len(valid_claims)} 条 Worker supported 结论的引用有效；语义未核验。")
finally:
    # 短脚本进程退出前必须显式收尾，理由与 graph.py __main__ 那份注释相同：
    # Langfuse 4.x 批量异步上报，不 flush 直接退出会丢 trace；放 finally
    # 而不是紧跟在 invoke() 后面，是为了上面任何一步抛异常（包括"烟测失败"
    # 那条 sys.exit(1)）时也能先把已产生的 trace 发出去再真正退出——失败
    # 的这次调用恰恰是最需要看观测数据排查的一次。
    if handler is not None:
        get_client().shutdown()
