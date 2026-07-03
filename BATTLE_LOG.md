# BATTLE_LOG · 战伤日志

> 只记真实发生的故障。格式：`日期 | 组件 | 现象 → 排查过程 → 根因 → 修复 → 前后数字`。
> 规矩：遇到 bug 本人先排查（AI 只给方向提示），修完当天记录。宁缺毋滥。

<!-- 条目从这里往下追加 -->

2026-07-03 | Planner（结构化输出） | 首次真模型调用即崩：DeepSeek 返回 400 "This response_format type is unavailable now" → 读栈定位到 `with_structured_output()`，其默认走 OpenAI 新版 `json_schema` response_format 协议 → 根因：DeepSeek 的 OpenAI 兼容层未实现该协议 → 修复：显式指定 `method="function_calling"`，改走两家都支持的工具调用协议取结构化输出 → 修复后端到端跑通：4 Worker 并行、11 条证据、单报告延迟 15.4s。教训：**"OpenAI 兼容"是个程度词，不是布尔值**——跨供应商时结构化输出的协议路径必须显式钉死，不能吃库的默认值。
