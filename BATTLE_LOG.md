# BATTLE_LOG · 战伤日志

> 只记真实发生的故障。格式：`日期 | 组件 | 现象 → 排查过程 → 根因 → 修复 → 前后数字`。
> 规矩：遇到 bug 本人先排查（AI 只给方向提示），修完当天记录。宁缺毋滥。

<!-- 条目从这里往下追加 -->

2026-07-03 | Planner（结构化输出） | 首次真模型调用即崩：DeepSeek 返回 400 "This response_format type is unavailable now" → 读栈定位到 `with_structured_output()`，其默认走 OpenAI 新版 `json_schema` response_format 协议 → 根因：DeepSeek 的 OpenAI 兼容层未实现该协议 → 修复：显式指定 `method="function_calling"`，改走两家都支持的工具调用协议取结构化输出 → 修复后端到端跑通：4 Worker 并行、11 条证据、单报告延迟 15.4s。教训：**"OpenAI 兼容"是个程度词，不是布尔值**——跨供应商时结构化输出的协议路径必须显式钉死，不能吃库的默认值。

2026-07-21 | RepoSource（D1 过渡检索层） | D1 烟测发现，用中文自然语言子任务（如「这个库怎么注册工具」）跑 Worker 时 grep 零命中，证据全部落到 model:// 降级路径；只有直接用代码里真实出现的英文关键字（如 add_tool）当子任务才能命中 → RepoSource.search 把 Planner 拆出的整句自然语言子问题原样当 grep pattern 丢进仓库搜索；不是合法正则时还会整串转义按字面量搜——一句中文问句在英文代码库里做字面量匹配，命中率必然为零 → 根因：过渡设计把「检索词生成」和「检索执行」耦死在一层，它假设子任务本身就是可 grep 的字符串，但 Planner 的产出是给人读的自然语言 → 修复：D2 晚用 Worker 自主工具循环取代（模型自己决定 grep 哪个英文符号、read 哪个文件、用 submit_claims 交卷），RepoSource 退役删除 → 中文子任务命中 0 条→降级；同关键字英文子任务 3 条带 file:line 证据（D1 烟测原始记录）。教训：**检索词生成是模型的活、检索执行才是工具的活，两者不能在接口上混为一谈**。

2026-07-22 | Worker（工具循环 × LangChain 消息协议） | T7 接线后拿本仓库真跑，Worker 第二轮 invoke 直接 400 "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'" → 逐条比对上一轮 tool_call_id 与已回复的 ToolMessage，缺回复的那条不在 ai_msg.tool_calls 里，而在 invalid_tool_calls——submit_claims 的嵌套 claims 参数不是合法 JSON 时，LangChain 把这次调用归到 invalid 列表 → 根因：第一版只回复 tool_calls；但 langchain_openai 把 AIMessage 重新序列化回下一轮请求时，会把 tool_calls 与 invalid_tool_calls 合并进同一个 "tool_calls" 字段，API 侧仍认为那条调用欠一条回复 → 修复：_handle_tool_calls 对 invalid_tool_calls 也逐条回 ToolMessage（"参数不是合法 JSON，请重发"），让模型自己纠正 → 修复后同类子任务真跑通过，Worker 对 graph.py 自身产出带准确 file:line 的 supported 结论（假数据模式的单测永远测不到这条路径——它不调真模型）。教训：**协议债不会因为解析失败而消失**——模型发出的每条 tool_call 无论参数合不合法都欠 API 一条回复，只处理"合法的那部分"等于默默欠债，下一轮才爆。

2026-07-22 | Worker（强制交卷 × 工具 schema） | 自查本仓库通过，首次对外部仓库（FastMCP）端到端却全灭：6 个 Worker 全部交出"证据不足"兜底、36 秒白卷收场 → 写单 Worker 逐轮调试脚本重放，一次抓到两个静默失效：① 模型第一轮就按提示词描述传 repo_tree(path='src/fastmcp/tools', depth=3)，但 schema 里只有 max_depth——pydantic 默认 extra=ignore 把不认识的参数静默扔掉，模型以为在看子树、实际拿到整棵根目录树，还据此误判"目录不存在"；② 到 8 次上限进强制交卷，tool_choice="submit_claims" 被 DeepSeek 无视，模型照样调 read_file，output 永远为 None → 根因：提示词与 schema 字段名两套账 + 依赖供应商根本没实现的指名 tool_choice，两者都不报任何错 → 修复：① 工具 schema 字段名与提示词逐字对齐、extra="forbid" 让错参数响亮报错回给模型自己改、repo_tree 复用 _resolve_within 真正支持子目录下钻；② 强制轮只绑 submit_claims 一个工具 + 显式交卷指令，tool_choice 按 submit_claims→auto 换挡重试；③ 预算只剩 2 次时注入收敛提醒 → 修复前 FastMCP 全问题 0 条 supported；修复后同一问题 46 条 file:line 引用、独立复核通过（46 秒）。教训：**"强制"必须是结构性的，不能是参数性的**——把别的工具从绑定列表里拿掉才叫强制，tool_choice 只是供应商可以不理的请求；静默吞参数比响亮报错危险得多，extra="forbid" 是给模型的纠错反馈通道，不是代码洁癖。
