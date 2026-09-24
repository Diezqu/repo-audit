# 当前架构与行为边界

本文描述当前 `python -m repo_audit.graph` 路径。历史设计推导见 [DECISIONS.md](../DECISIONS.md)，安装与命令见 [README](../README.md)。

```text
CLI：仓库目录 + 非空问题 + 可选 --demo
  └─ Planner：repo_stats、浅层 repo_tree、README 开头 → 1–8 个 SubTask
       └─ LangGraph Send → Worker × 子任务数
            ├─ repo_tree / grep_repo / read_file，最多尝试 8 次查询
            └─ submit_claims → Claim（自报状态 + Citation）
                 └─ reducer 合并 → Verifier（默认开启）
                      └─ citation_status + citation_reason；语义 verdict 留空
                           └─ 确定性报告
```

## 状态与节点

- `State` 保存问题、仓库根目录、计划、Worker 原始 `claims`、Verifier 处理后的 `verified_claims` 和报告。`claims` 通过 reducer 合并并行分支；Verifier 写单独字段，避免把处理后的结果再次追加。
- Planner 的结构化计划限制 1–8 个子任务；每项含目标模块及 1–3 个非空问题。数量是任务数，不能解释成真实并发量。
- Worker 用便宜档模型自主决定何时查目录、搜索和精读。查询预算在调用前执行，非法尝试也计入；交卷不计查询次数。Worker 自报 `supported` 或 `insufficient`，没有引用的“支持”不能当作已证实。
- Verifier 回读引用文件，检查仓库相对路径、闭区间行号和非空原文片段。同一批核验中每个文件只读取一次。结果区分 `valid`、`invalid`、`missing`、`unavailable`、`unchecked`。一个 Claim 有多条引用时，引用有效要求全部通过。源码在 Worker 读取与 Verifier 读取间变化会导致检查失败，但不代表命题为假。
- 报告只列出其引用状态及 Worker 自报状态；有效引用仍标“语义未核验”。目前不调用自由生成的合成模型，也没有独立语义 Judge。

## 运行模式与失败

默认真实模式要求旗舰档 Planner 与便宜档 Worker 的模型名、密钥和端点均配置完整。`--demo` 或 `REPO_AUDIT_MODE=demo` 才使用占位计划和结论；演示报告不代表对目标仓库的调研。配置缺失在规划前失败。模型客户端默认超时 60 秒、最多重试 1 次。真实模式中，一个 Worker 的供应商或传输异常会形成含异常类型、无原始错误文本的证据不足项，并在运行时 `Claim.worker_error` 标记；其余 Worker 继续汇总。评测将部分 Worker 失败标为 `partial_failure`，全部失败标为 `error`，均不纳入成功的真实运行延迟。普通证据不足没有 `worker_error`。程序错误与配置错误不作为“证据不足”吞掉。可选 Langfuse 仅在配置齐全且非演示模式时启用。

## 文件系统与检索

模型只获得绑定到指定 `repo_root` 的工具。显式路径读取先解析并检查是否仍在根目录；`read_file` 和 Verifier 同时拒绝根目录 `.git` 元数据（含大小写变体及解析到该处的符号链接）。遍历、统计和搜索跳过符号链接及 `.git` 元数据。`rg` 使用 `--no-config --no-follow --hidden --no-ignore` 和显式 `.` 搜索根，缺席时使用 Python 回退。工具仍跳过项目定义的噪音目录。返回有行数与字符上限，并提示截断；两种 grep 引擎的正则、glob、二进制文件处理可能不同。Planner 的 README/入口文件读取同样遵守根目录边界。

这是应用层约束，**不是操作系统沙箱**。仓库内容可能包含提示注入，或被外部进程在读取期间改动；本实现不承诺对抗并发符号链接替换和快照一致性。处理不可信仓库时，应在适当的外部隔离环境运行，并人工核对结论。

## 尚未实现

没有持久化 checkpoint、向量数据库、MCP 服务、`ask`/`onboard` 子命令、自动生成架构文档或语义裁决。固定提交的 [十题评测草案](../eval/README.md) 尚待人工标注；引用检查结果不能替代语义质量测量或成本测量。

`scripts/evaluate.py` 已实现 `validate-corpus`、`run`、`summarize`。离线校验要求 `--repo` 为固定提交的 Git 顶层目录且工作树干净，包括没有未跟踪和被忽略的文件。运行记录须写到引擎与目标仓库外的新文件；真实运行需六项模型配置及显式 `--mode real --allow-api`。记录中的任务数、配置并发上限、Token 用量、引用有效性和延迟各有独立字段；没有人工语义标签时不计算语义正确率。评测步骤与字段见 [评测说明](../eval/README.md)。
