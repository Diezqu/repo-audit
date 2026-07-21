"""证据源层：EvidenceSource 接口 + 可插拔后端。

D15 换域后的证据源只有一层——本地仓库：

  RepoSource   用 T3 的 grep_repo 在给定仓库里定位候选行，替代原来的
               博查/DDG 网络搜索（git 历史留痕，见 DECISIONS.md D13/D15）。

引擎只依赖 search_with_fallback()，换后端零改动——这层抽象曾经是「国内
部署换国产 API」承诺的落点（D13），现在是「证据从脏的开放网页换成干净
本地仓库」（D15）落地时唯一需要改的文件：graph.py 里的 worker() 调用
sources.search_with_fallback(subtask) 这一行代码本身完全不用动。
"""

import re
from pathlib import Path

from pydantic import BaseModel

from repo_audit.repo_tools import grep_repo


class SearchResult(BaseModel):
    """一条检索结果：Worker 提取结论时的原始材料。

    字段名沿用旧版网络搜索时代的命名（title/url/snippet），换成仓库场景后
    语义随之转译：title = 命中所在的相对路径，url = "file:line" 引用（不再
    是网页地址，但恰好是 Evidence.source_url 需要的确切格式），snippet =
    命中那一行的原文。字段形状不变是刻意的——下游 Worker 的提取逻辑
    （numbered 列表拼 prompt、按 source_index 挂引用）不需要跟着改一行。
    """

    title: str
    url: str
    snippet: str


class RepoSource:
    """本地代码仓库证据源（D15 落地）：不再打网络请求，改用 grep_repo 在
    root 指向的仓库里找可能相关的代码行。

    query 通常是 Planner 拆出的自然语言子问题，不保证是合法正则——
    Chinese 标点、括号这些在自然语言里很常见但在正则里有特殊含义，
    所以编译失败时退化成整串转义后按字面量搜索，而不是直接报错让整个
    Worker 挂掉（与 search_with_fallback 一贯的"单源故障不致命"原则一致）。
    """

    name = "repo"

    def __init__(self, root: str | Path):
        self._root = Path(root)

    def search(self, query: str, max_results: int = 6) -> list[SearchResult]:
        try:
            re.compile(query)
            pattern = query
        except re.error:
            pattern = re.escape(query)

        try:
            raw = grep_repo(self._root, pattern, max_results=max_results)
        except Exception:
            return []  # 仓库不存在/不可读等——按证据源惯例，单源故障不致命

        results = []
        for line in raw.splitlines():
            # grep_repo 的正常命中行是严格的 "path:行号:内容" 三段；
            # "(无匹配)" 和截断提示行都凑不出这个形状，天然被下面的校验滤掉，
            # 不需要单独识别这两种哨兵值。
            parts = line.split(":", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            file_part, lineno, content = parts
            results.append(
                SearchResult(
                    title=file_part,
                    url=f"{file_part}:{lineno}",
                    snippet=content.strip()[:300],
                )
            )
        return results[:max_results]


def available_sources(root: str | Path | None = None) -> list:
    """给出当前可用的证据源。目前只有 RepoSource 一个。

    root 留成可选参数并默认 Path.cwd()，是给 graph.py 现状的一个妥协：
    State/Worker 目前还没有 repo_root 字段（Planner/Worker 提示词仍是
    消费决策版，切换到仓库场景是另一条线的任务），先保证 sources 这一层
    接口完整、可独立测试；等提示词切换后，graph.py 需要把真正的仓库路径
    经 State 传进来，届时这里改成必填参数即可，调用方式不受影响。
    """
    target = Path(root) if root is not None else Path.cwd()
    return [RepoSource(target)]


def search_with_fallback(
    query: str, max_results: int = 6, root: str | Path | None = None
) -> tuple[str, list[SearchResult]]:
    """依次尝试各源，返回 (源名, 结果)。全部失败返回 ("none", [])——
    调用方据此降级（模型知识作答并如实标注），而不是崩掉整个 run。"""
    for src in available_sources(root):
        try:
            rows = src.search(query, max_results)
            if rows:
                return src.name, rows
        except Exception:
            continue  # 单源故障不致命，换下一个源
    return "none", []
