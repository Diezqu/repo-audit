# repo-audit · 多智能体代码库调研核验引擎

> 原名 decision-engine（消费决策调研），2026-07-21 经 D15 决策换域改名，git 历史完整保留——决策过程本身见 [DECISIONS.md](DECISIONS.md)。

丢给它一个代码仓库，得到**带 `file:行号` 引用、且经独立核验**的架构文档与问答。Planner 读仓库地图拆解 → 并行 Worker 用本地工具取证 → Verifier 回读源码逐条核验 → 合成器只组装核验通过的结论，没通过的显式标「存疑」，绝不静默混入。

**它不是又一个代码问答 bot**，区别在「核验 + 数字」：

- 每条关键结论由独立 Verifier 回到引用位置查证「这几行代码真的支持这个说法吗」；
- Verifier 可开关——同一评测集开/关各跑一遍，产出**引用错误率 ablation** 这一硬指标；
- 评测集 pin 死 commit，人工金标 + LLM judge + 回归 CI，数字可复现。

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
Verifier（中档，可开关）：重新打开每条引用位置逐条判定
   → supported / refuted / insufficient；非 supported 打回或降级
   │
   ▼
合成器（旗舰档）→ 带引用回答 / 三文档，存疑显式标注
```

## 关键选型（完整推导见 [DECISIONS.md](DECISIONS.md)）

| 选型 | 定案 | 一句话理由 |
|---|---|---|
| 编排 | LangGraph 显式状态机 | 断点续跑、逐节点测试、Send 并行原生语义（D2） |
| 检索 | 结构化导航（tree/read/grep），**无向量库** | 代码=精确标识符世界，grep 零误差零基建；embedding 切碎代码结构（D15） |
| 模型 | DeepSeek/Qwen 便宜档 + 旗舰档双档路由 | 翻文件是体力活，规划合成是脑力活；成本可归因 |
| 核验 | 独立 Verifier 节点 + 开关 flag | 写结论的模型自查会偏袒；开关预埋 ablation |
| AST/调用图 | v1 不做 | 时间盒守恒；grep 覆盖八成需求 |

## 现状（2026-07-21 起七天冲刺）

- [x] LangGraph 骨架：Planner → 动态并行 Worker（Send）→ 合成器
- [x] 双档模型路由 / 结构化证据 / pytest + CI
- [x] 15 条架构决策记录 + 战伤日志（含一次完整止损：D10–D15 消费数据行业级死题 → 保引擎换领域）
- [x] 仓库工具层（tree / read_file / grep / repo_stats，路径白名单 + 输出截断，见 D16）
- [x] 证据源改造：RepoSource 接入本地仓库取证，替换博查/DDG 网搜实现（见 D16）
- [ ] Verifier v1 + 开关
- [ ] 30 题评测集（LangGraph + FastMCP，pin commit）+ 10 题人工金标 + judge + 回归 CI
- [ ] Verifier 开/关 ablation 数字
- [ ] FastMCP server 化（`ask_repo` 工具，Claude Code / Cursor 可直接调用）

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入两档模型的 API key
pytest                 # 冒烟测试
```

## 决策与战伤

- [DECISIONS.md](DECISIONS.md) — 15 条技术决策：每条含备选、取舍与「面试一句话」
- [BATTLE_LOG.md](BATTLE_LOG.md) — 真实故障的排查与修复记录
