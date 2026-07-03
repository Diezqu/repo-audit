"""LangGraph 状态机 —— 【Ziyang 亲手写，AI 不代写】

这是整个项目的面试核心。动手前先在纸上回答（答案会被当场拷问）：

1. State 里放什么字段？
   - 用户问题、子任务列表、每个 Worker 带回的证据（含来源）、最终报告……
   - 并行 Worker 同时往一个字段写，会不会互相覆盖？LangGraph 怎么声明
     「这个字段是累加的」？（提示：查 Annotated + reducer）

2. Planner → N 个并行 Worker 的扇出怎么表达？
   - 子任务数量是运行时才知道的（3-5 个），静态加边行不通。
     LangGraph 里动态扇出的机制叫什么？

3. 一个 Worker 挂了（超时/API 报错），整个 run 就废吗？
   - 「失败接管」在状态机里长什么样：重试？降级标记？还是跳过并在
     合成时声明证据不全？——选一个，说清为什么。

4. 为什么是 LangGraph 而不是 CrewAI / AutoGen / 裸写 asyncio？
   - 答案必须落在「显式状态机带来什么」：断点续跑、可观测、可测试。

骨架顺序建议：先让 Planner → 2 个写死的 Worker → 合成器跑通直线流，
再改成动态扇出。第一版允许丑，不允许不是你写的。
"""

# from langgraph.graph import StateGraph, START, END
# from decision_engine.config import cheap_tier, flagship_tier
