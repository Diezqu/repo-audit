# decision-engine · 多智能体决策调研核验引擎

自建多智能体决策调研核验引擎：Planner 拆解 → 并行 Worker 检索 → Verifier 对抗核实 → 带引用的决策报告。私有 Obsidian 知识库经自建 MCP 连接器接入，作为长期偏好记忆源（选品护栏、身体数据、历史决策）；全链路 Langfuse 观测 + 人工金标评测集 + 对抗核实 ablation 量化。

> 🚧 进行中。引擎代码与合成 fixture 开源；真实个人库永不入库。

## 架构

```
用户问题（该不该买 X / 选 A 还是 B）
   │
   ▼
Planner（旗舰档模型）
   ├─ 读 MCP 库连接器：选品护栏 / 身体数据 / 现有单品 / 历史决策
   └─ 拆解为 3-5 个子任务 → 写入 LangGraph 状态
   │
   ▼ 并行
Worker 池（便宜档模型 × N）
   └─ 每个 Worker 查一个子问题，带回带来源的证据
   │
   ▼
Verifier（中档，每条关键结论 3 票对抗核实）
   │
   ▼
合成器（旗舰档）→ 带引用决策报告
```

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入两档模型的 API key
pytest                 # 冒烟测试
```

## 战伤日志

真实故障的排查与修复记录见 [BATTLE_LOG.md](BATTLE_LOG.md)。
