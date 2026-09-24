# repo-audit · 代码库调研与引用检查

输入一个本地仓库和问题，程序读取仓库地图，拆出有界子任务，并行检索源码，再输出逐条列明引用状态的报告。**引用有效只表示所指文件、行号和原文片段能被回读；它不证明结论在语义上成立。** 报告保留这一边界，仍需人工判断代码是否支持结论。

## 快速开始

需要 Python 3.12 及以上；CI 覆盖 3.12 和 3.13。以下命令在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m repo_audit.graph --demo . "这个仓库如何组织代码？"
```

`--demo` 显式运行不调用模型的流程示例。其结论是占位内容，**不能用于判断目标仓库**；即使已配置模型密钥，`--demo` 也只运行示例模式。路径必须指向存在的目录。若省略问题，CLI 使用默认的架构问题；显式给出的空白问题会报错。

上面命令的实际输出（**虚构数据**；占位引用 `FAKE.md` 并非真实证据）：

```text
# 代码库调研报告：这个仓库如何组织代码？

语义未核验；以下仅检查引用与源码是否一致。

- 引用无效，语义未核验 —— （假数据）关于「这个目录对外暴露什么接口？」的占位结论 （src/）
- 缺少引用，语义未核验；存疑：证据不足 —— （假数据）证据不足：有没有明显的边界条件处理？ （src/）
- 引用无效，语义未核验 —— （假数据）关于「测试覆盖了哪些行为？」的占位结论 （tests/）
```

真实调研默认运行真实模式。先复制 `.env.example` 为 `.env`，配置 `CHEAP_MODEL`、`CHEAP_API_KEY`、`CHEAP_BASE_URL` 和 `FLAGSHIP_MODEL`、`FLAGSHIP_API_KEY`、`FLAGSHIP_BASE_URL`，再执行：

```bash
cp .env.example .env
# 编辑 .env，填入六项模型配置
python -m repo_audit.graph /path/to/repository "这个仓库如何处理失败的工具调用？"
```

真实模式在调用模型前要求两档配置齐全；配置缺失会报错退出，不会混用真实与占位结果。两档是 Planner 与 Worker 的配置角色，不要求必须使用不同的模型或供应商。模型请求可能产生费用；本项目没有发布可靠的每次运行成本估计。不要在公开仓库提交 `.env`、密钥或私有仓库内容。可选的 Langfuse 配置见 `.env.example`；不配置时不启用追踪。

本地验证：

```bash
ruff check .
pytest -q
```

## 工作流程

```text
问题 + repo_stats + 浅层目录树 + README 开头
                  │
                  ▼
         Planner：1–8 个子任务
                  │ LangGraph Send
                  ▼
      Worker：tree / grep / read 取证
                  │ 每个 Worker 最多尝试 8 次查询工具调用
                  ▼
         Verifier：确定性引用检查
                  ▼
       确定性报告：引用状态 + 语义未核验
```

子任务数是 **1–8**，表示最多派发的 Worker 任务数，**不是实测同时运行数**。每个子任务有 1–3 个具体问题。查询预算按尝试计数，预算耗尽后不得继续读仓库；`submit_claims` 是交卷动作。Worker 自报的 `supported` 只表示它认为自己找到依据。Verifier 默认开启，检查引用路径、行号范围和非空原文片段；任何一条引用无效，整条结论的引用状态为无效。无引用、读取失败、显式关闭检查和有效引用各有不同状态。它不会把结论判成“语义支持”或“语义反驳”。

报告由确定性模板生成，目前没有自由生成的 LLM 合成器。真实模式若个别 Worker 遇到供应商或传输错误，该任务会降级为证据不足并保留其他任务结果；配置错误会在规划前失败。仓库内容可能在工具读取与引用检查之间变化，因此引用检查对应**检查时**的文件内容，不是仓库快照。

## 边界与评测

- 工具入口限制在给定仓库根目录内；遍历跳过符号链接，直接读取只允许解析后仍在根目录内的文件。这是应用层路径约束，**不是操作系统沙箱**，也不保证抵御并发修改文件系统造成的竞态。
- `grep_repo` 优先使用 `rg`，没有时使用 Python 回退。两条路径的正则和忽略规则可能不同；命中与否都不能直接视作完整性证明。
- 当前没有 checkpoint 持久化、向量数据库、MCP 服务、独立语义判定或自动生成架构文档命令。LangGraph 提供相关能力，不代表本项目已接线。
- [评测草案](eval/README.md) 含固定提交上的 10 道待人工复核问题；它还不是人工金标，也没有已发布的模型质量分数。评测应分别统计 Worker 自报、引用有效性和人工语义判断，保留模型配置、语料提交与运行记录。

评测脚本支持离线校验、限定题目运行和汇总。先准备评测 README 指定提交的**干净专用检出**，包括不能存在未跟踪或被忽略的文件；输出文件必须新建在引擎与目标仓库之外：

```bash
PYTHONPATH=src python scripts/evaluate.py validate-corpus --repo /path/to/fastmcp
PYTHONPATH=src python scripts/evaluate.py run --repo /path/to/fastmcp \
  --mode demo --id FMCP-01 --output /tmp/repo-audit-demo.jsonl
PYTHONPATH=src python scripts/evaluate.py summarize \
  --records /tmp/repo-audit-demo.jsonl
```

这些演示记录同样是合成数据，不能用于回答质量统计。真实评测还需显式 `--mode real --allow-api`、六项模型环境变量和新的输出路径；详见 [评测说明](eval/README.md)。当前结果汇总只报告运行记录和时间，不提供语义质量分数；费用估计须单独给出可核对的费率。

设计与现状见 [架构说明](docs/current-architecture.md) 和 [决策记录](DECISIONS.md)；故障复盘见 [BATTLE_LOG.md](BATTLE_LOG.md)。D1–D16 是历史决策，当前行为以本文、架构说明和代码为准。
