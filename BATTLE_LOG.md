# BATTLE_LOG · 战伤日志

> 记录复现过的故障及修复证据。旧条目是当时的运行观察和排障叙述，不代表当前性能或质量基线；其中的 Worker 数是派发任务数，不能直接解释成实测并发。当前实现边界见 [README](README.md) 与 [当前架构](docs/current-architecture.md)。近期修改由 AI 辅助完成，本日志不将其归为个人手写成果。

<!-- 条目从这里往下追加 -->

2026-07-03 | Planner（结构化输出） | 首次真模型调用即崩：DeepSeek 返回 400 "This response_format type is unavailable now" → 读栈定位到 `with_structured_output()`，其默认走 OpenAI 新版 `json_schema` response_format 协议 → 根因：DeepSeek 的 OpenAI 兼容层未实现该协议 → 修复：显式指定 `method="function_calling"`，改走两家都支持的工具调用协议取结构化输出 → 修复后端到端跑通：4 Worker 并行、11 条证据、单报告延迟 15.4s。教训：**"OpenAI 兼容"是个程度词，不是布尔值**——跨供应商时结构化输出的协议路径必须显式钉死，不能吃库的默认值。

2026-07-21 | RepoSource（D1 过渡检索层） | D1 烟测发现，用中文自然语言子任务（如「这个库怎么注册工具」）跑 Worker 时 grep 零命中，证据全部落到 model:// 降级路径；只有直接用代码里真实出现的英文关键字（如 add_tool）当子任务才能命中 → RepoSource.search 把 Planner 拆出的整句自然语言子问题原样当 grep pattern 丢进仓库搜索；不是合法正则时还会整串转义按字面量搜——一句中文问句在英文代码库里做字面量匹配，命中率必然为零 → 根因：过渡设计把「检索词生成」和「检索执行」耦死在一层，它假设子任务本身就是可 grep 的字符串，但 Planner 的产出是给人读的自然语言 → 修复：D2 晚用 Worker 自主工具循环取代（模型自己决定 grep 哪个英文符号、read 哪个文件、用 submit_claims 交卷），RepoSource 退役删除 → 中文子任务命中 0 条→降级；同关键字英文子任务 3 条带 file:line 证据（D1 烟测原始记录）。教训：**检索词生成是模型的活、检索执行才是工具的活，两者不能在接口上混为一谈**。

2026-07-22 | Worker（工具循环 × LangChain 消息协议） | T7 接线后拿本仓库真跑，Worker 第二轮 invoke 直接 400 "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'" → 逐条比对上一轮 tool_call_id 与已回复的 ToolMessage，缺回复的那条不在 ai_msg.tool_calls 里，而在 invalid_tool_calls——submit_claims 的嵌套 claims 参数不是合法 JSON 时，LangChain 把这次调用归到 invalid 列表 → 根因：第一版只回复 tool_calls；但 langchain_openai 把 AIMessage 重新序列化回下一轮请求时，会把 tool_calls 与 invalid_tool_calls 合并进同一个 "tool_calls" 字段，API 侧仍认为那条调用欠一条回复 → 修复：_handle_tool_calls 对 invalid_tool_calls 也逐条回 ToolMessage（"参数不是合法 JSON，请重发"），让模型自己纠正 → 修复后同类子任务真跑通过，Worker 对 graph.py 自身产出带准确 file:line 的 supported 结论（假数据模式的单测永远测不到这条路径——它不调真模型）。教训：**协议债不会因为解析失败而消失**——模型发出的每条 tool_call 无论参数合不合法都欠 API 一条回复，只处理"合法的那部分"等于默默欠债，下一轮才爆。

2026-07-22 | Worker（强制交卷 × 工具 schema） | 自查本仓库通过，首次对外部仓库（FastMCP）端到端却全灭：6 个 Worker 全部交出"证据不足"兜底、36 秒白卷收场 → 写单 Worker 逐轮调试脚本重放，一次抓到两个静默失效：① 模型第一轮就按提示词描述传 repo_tree(path='src/fastmcp/tools', depth=3)，但 schema 里只有 max_depth——pydantic 默认 extra=ignore 把不认识的参数静默扔掉，模型以为在看子树、实际拿到整棵根目录树，还据此误判"目录不存在"；② 到 8 次上限进强制交卷，tool_choice="submit_claims" 被 DeepSeek 无视，模型照样调 read_file，output 永远为 None → 根因：提示词与 schema 字段名两套账 + 依赖供应商根本没实现的指名 tool_choice，两者都不报任何错 → 修复：① 工具 schema 字段名与提示词逐字对齐、extra="forbid" 让错参数响亮报错回给模型自己改、repo_tree 复用 _resolve_within 真正支持子目录下钻；② 强制轮只绑 submit_claims 一个工具 + 显式交卷指令，tool_choice 按 submit_claims→auto 换挡重试；③ 预算只剩 2 次时注入收敛提醒 → 修复前 FastMCP 全问题 0 条 supported；修复后同一问题 46 条 file:line 引用、独立复核通过（46 秒）。教训：**"强制"必须是结构性的，不能是参数性的**——把别的工具从绑定列表里拿掉才叫强制，tool_choice 只是供应商可以不理的请求；静默吞参数比响亮报错危险得多，extra="forbid" 是给模型的纠错反馈通道，不是代码洁癖。

2026-07-22 | 评测素材（临时目录克隆） | 计量跑批突然全灭：同一问题两小时前 46 条引用，重跑 0 条 supported、24 条全 insufficient，第一反应是当晚脚手架改坏了 Worker → 先跑单 Worker 复现，报错文本写着"grep_repo 搜索 def tool 未找到任何匹配"——引擎说的是仓库里没有，不是工具坏了 → 临时目录中源码文件已被系统回收 → 根因：把评测仓库克隆到可能被系统回收的路径 → 修复：移到固定目录并记录 commit（f4ae8bb0） → 素材恢复后的单次观察：派发 8 个 Worker 任务、34 条结论（28 条 Worker 自报 supported、6 条 insufficient）、99 条 file:line 引用、56.5s、¥0.87。这些是历史单次输出，未证明引用的语义正确性，也不是可复现的成本或质量基线。教训：评测素材的位置与提交版本都是实验条件。

2026-09-24 | CI / Ruff | 远端 CI 因未固定的 Ruff 规则漂移出现 10 条 lint 错误，测试阶段未运行 → 将开发依赖固定为 `ruff==0.15.20`，明确项目 lint 规则并修正触发项 → 在 Python 3.12、3.13 环境分别通过 56 个测试与 lint；随后同一 PR 的远端 3.12、3.13 CI 作业均成功。该结果是当时基线，不代表后续功能变更已通过。

2026-09-24 | 仓库根目录边界 | 用临时仓库构造指向根目录外的文件链接和目录链接，修复前 `repo_tree` 会列出外部文件，Python grep 会命中外部哨兵文本 → 原有显式 `read_file` 的解析检查没有覆盖目录枚举、统计和 grep 回退；`rg` 路径还可能受用户配置影响 → 修复：枚举统一跳过符号链接，入口文件提示不读取链接，`rg` 加 `--no-config --no-follow` 并显式指定搜索根；保留直接 `read_file` 对“解析后仍在根目录内”链接的支持 → 回归测试覆盖 tree、stats、Python/rg grep、外部与内部链接。此修复是应用层路径限制，不提供并发文件系统变更下的沙箱保证。
