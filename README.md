# repo-audit · 多智能体代码库调研核验引擎

> 原名 decision-engine（消费决策调研），2026-07-21 经 D15 决策换域改名，git 历史完整保留——决策过程本身见 [DECISIONS.md](DECISIONS.md)。

给它一个本地代码仓库和问题，Planner 根据仓库地图拆解任务，并行 Worker 用只读工具检索源码，合成器生成带 `file:行号` 引用的回答。目前 Verifier **默认关闭**；开启后会独立回读源码，驳回越界、无效行号及与源码不符的引用片段。

## 当前能力与边界

- 已实现 LangGraph Planner → 动态并行 Worker → Verifier → Synthesizer；Worker 使用 `repo_tree`、`read_file`、`grep_repo`，有路径限制、输出截断和工具调用预算。
- Verifier 当前只做**确定性的引用检查**：文件路径、行号范围和片段内容。引用存在不代表结论的语义得到证明；语义判定尚未实现，不能把 Worker 的 `supported` 当成独立核验结果。
- 无模型密钥时会运行假数据模式，用于检查流程和测试；输出不代表对目标仓库完成了真实调研。
- 评测集、Verifier 开关对照实验和 FastMCP 服务尚未实现；目前没有可复现的引用错误率数字。

## 目标形态

```bash
repo-audit ask ./langgraph/ "checkpoint 能否跨进程恢复？"
# ✅ 支持，需配置持久化 checkpointer
#    依据: libs/checkpoint/sqlite.py L41-88 [已核验]
repo-audit onboard ./langgraph/   # 架构全景 / 代码意图 / 特例规则 三份文档（v1.1）
```

## 架构

```
输入：仓库 + 问题
   │
   ▼
Planner（旗舰档）：读地图（repo_stats + 目录树 + README 头部）
   → 拆 3~8 个子任务，写入 LangGraph 状态
   │
   ▼ Send 并行
Worker 池（便宜档 × N）：tree / read_file / grep 取证
   → 结构化 Claim {结论, [file, L起-L止, snippet]}
   │  （每 Worker 工具调用上限 8 次——成本护栏）
   ▼
Verifier（默认关闭，可开关）：回读文件、行号和片段
   → 无效引用标 refuted；有效引用仍待语义判定
   │
   ▼
合成器（旗舰档）→ 带引用回答，证据不足显式标注
```

## 关键选型（完整推导见 [DECISIONS.md](DECISIONS.md)）

| 选型 | 定案 | 一句话理由 |
|---|---|---|
| 编排 | LangGraph 显式状态机 | 断点续跑、逐节点测试、Send 并行原生语义（D2） |
| 检索 | 结构化导航（tree/read/grep），**无向量库** | 代码=精确标识符世界，grep 零误差零基建；embedding 切碎代码结构（D15） |
| 模型 | DeepSeek/Qwen 便宜档 + 旗舰档双档路由 | 翻文件是体力活，规划合成是脑力活；成本可归因 |
| 核验 | 独立 Verifier 节点 + 开关 flag | 先做可确定的引用检查；语义判定仍待实现 |
| AST/调用图 | v1 不做 | 时间盒守恒；grep 覆盖八成需求 |

## 现状（2026-07-21 起七天冲刺）

- [x] LangGraph 骨架：Planner → 动态并行 Worker（Send）→ 合成器
- [x] 双档模型路由 / 结构化证据 / pytest + CI
- [x] 15 条架构决策记录 + 战伤日志（含一次完整止损：D10–D15 消费数据行业级死题 → 保引擎换领域）
- [x] 仓库工具层（tree / read_file / grep / repo_stats，路径白名单 + 输出截断，见 D16）
- [x] 本地仓库只读取证工具替换旧网搜路径（见 D16）
- [x] Verifier 开关与文件、行号、片段检查
- [ ] 结论与引用之间的独立语义判定
- [ ] 30 题评测集（LangGraph + FastMCP，pin commit）+ 10 题人工金标 + judge + 回归 CI
- [ ] Verifier 开/关 ablation 数字
- [ ] FastMCP server 化（`ask_repo` 工具，Claude Code / Cursor 可直接调用）

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入两档模型的 API key
pytest                 # 冒烟测试
python -m repo_audit.graph /path/to/repo "这个仓库的整体架构是怎样的？"
```

不配置模型密钥时，上面的命令会输出标明“假数据”的流程示例。真实调研需要在 `.env` 中配置模型；`VERIFIER_ENABLED=true` 可开启引用检查。当前没有 `repo-audit ask` 或 `onboard` 命令，前面的“目标形态”仅是规划。

## 决策与战伤

- [DECISIONS.md](DECISIONS.md) — 15 条技术决策：每条含备选、取舍与「面试一句话」
- [BATTLE_LOG.md](BATTLE_LOG.md) — 真实故障的排查与修复记录
