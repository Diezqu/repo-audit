"""证据源层：EvidenceSource 接口 + 可插拔后端（D13 中国优先架构）。

分层（按优先级）：
  1. BochaSource   国产搜索 API，国内直连，默认主干（有 BOCHA_API_KEY 时启用）
  2. DDGSource     DuckDuckGo，海外源增强（网络可达时可用，无需 key）
  后续：MediaCrawler 适配器（封闭平台）、用户补充证据通道。

引擎只依赖 search_with_fallback()，换后端零改动——
这层抽象同时也是「国内部署换国产 API」承诺的落点。
"""

import os

import httpx
from pydantic import BaseModel


class SearchResult(BaseModel):
    """一条搜索结果：Worker 提取结论时的原始材料。"""

    title: str
    url: str
    snippet: str


class BochaSource:
    """博查 Web Search API（open.bochaai.com）：对标 Bing、LLM 优化、国内直连。"""

    name = "bocha"

    def __init__(self, api_key: str):
        self._key = api_key

    def search(self, query: str, max_results: int = 6) -> list[SearchResult]:
        resp = httpx.post(
            "https://api.bochaai.com/v1/web-search",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"query": query, "count": max_results, "summary": True},
            timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json()["data"]["webPages"]["value"]
        return [
            SearchResult(
                title=p.get("name", ""),
                url=p["url"],
                snippet=p.get("summary") or p.get("snippet", ""),
            )
            for p in pages[:max_results]
        ]


class DDGSource:
    """DuckDuckGo 文本搜索：免 key，海外源覆盖好；国内网络不保证可达。"""

    name = "ddg"

    def search(self, query: str, max_results: int = 6) -> list[SearchResult]:
        from ddgs import DDGS

        rows = DDGS().text(query, max_results=max_results)
        return [
            SearchResult(title=r.get("title", ""), url=r["href"], snippet=r.get("body", ""))
            for r in rows
        ]


def available_sources() -> list:
    """按 D13 优先级给出当前环境可用的源：博查（有 key 才有）在前，DDG 兜底。"""
    out = []
    key = os.getenv("BOCHA_API_KEY")
    if key:
        out.append(BochaSource(key))
    out.append(DDGSource())
    return out


def search_with_fallback(query: str, max_results: int = 6) -> tuple[str, list[SearchResult]]:
    """依次尝试各源，返回 (源名, 结果)。全部失败返回 ("none", [])——
    调用方据此降级（模型知识作答并如实标注），而不是崩掉整个 run。"""
    for src in available_sources():
        try:
            rows = src.search(query, max_results)
            if rows:
                return src.name, rows
        except Exception:
            continue  # 单源故障不致命，换下一个源
    return "none", []
