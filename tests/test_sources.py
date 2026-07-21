"""证据源层(T5)单测：RepoSource 取代 BochaSource/DDGSource 后的行为。

覆盖点对应 D15 落地的关键改动：
  - available_sources()/search_with_fallback() 的签名与返回形状没变
    （graph.py 的调用点因此不用动一行，只是这层内部实现换了血）；
  - RepoSource 把 grep_repo 的 "path:行号:内容" 文本行还原成
    SearchResult(title/url/snippet)，url 字段变成 file:line 引用；
  - 自然语言 query（不一定是合法正则）不应该让 Worker 直接崩掉。
"""

from pathlib import Path

import pytest

from repo_audit.sources import RepoSource, SearchResult, available_sources, search_with_fallback

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "class Widget:\n    def render(self):\n        return 'ok'\n"
    )
    return tmp_path


def test_available_sources_returns_repo_source(sample_repo):
    sources = available_sources(sample_repo)
    assert len(sources) == 1
    assert isinstance(sources[0], RepoSource)
    assert sources[0].name == "repo"


def test_available_sources_defaults_to_cwd(monkeypatch, sample_repo):
    """没显式传 root 时兜底用 cwd——State 还没接 repo_root 字段前的过渡行为。
    黑盒验证：不看内部属性，直接确认它真的能在 cwd 指向的仓库里搜到东西。"""
    monkeypatch.chdir(sample_repo)
    results = available_sources()[0].search("Widget")
    assert len(results) == 1


def test_repo_source_search_returns_search_result_shape(sample_repo):
    results = RepoSource(sample_repo).search("Widget")
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, SearchResult)
    assert result.title == "pkg/core.py"
    assert result.url == "pkg/core.py:1"
    assert "Widget" in result.snippet


def test_repo_source_search_natural_language_query_does_not_crash(sample_repo):
    """query 常是 Planner 拆出的自然语言子问题，可能不是合法正则
    （括号/问号很常见）——不合法正则应退化为字面量搜索，而不是抛异常。"""
    results = RepoSource(sample_repo).search("这个类(Widget)是干嘛的？")
    assert results == []  # 字面量搜不到，但绝不能抛异常


def test_repo_source_search_respects_max_results(sample_repo):
    many = sample_repo / "many.py"
    many.write_text("\n".join(f"def f{i}(): pass  # needle" for i in range(10)))
    results = RepoSource(sample_repo).search("needle", max_results=3)
    assert len(results) == 3


def test_search_with_fallback_finds_repo_evidence(sample_repo):
    name, results = search_with_fallback("Widget", root=sample_repo)
    assert name == "repo"
    assert len(results) == 1
    assert results[0].url == "pkg/core.py:1"


def test_search_with_fallback_no_match_returns_none_sentinel(sample_repo):
    name, results = search_with_fallback("totally_absent_token_zzz", root=sample_repo)
    assert name == "none"
    assert results == []


def test_search_with_fallback_missing_root_is_not_fatal(tmp_path):
    """仓库不存在/不可读——单源故障不致命，走 D6 既有的降级契约。"""
    missing = tmp_path / "does_not_exist"
    name, results = search_with_fallback("anything", root=missing)
    assert name == "none"
    assert results == []


# ──────────────────────────────────────────────────────────────
# 用本仓库自身当 fixture：证据源在真实、混杂的仓库上也能工作
# ──────────────────────────────────────────────────────────────

def test_repo_source_on_real_repo_finds_stategraph():
    results = RepoSource(REPO_ROOT).search("StateGraph")
    assert results
    assert any("graph.py" in r.title for r in results)
    assert all(r.url.rsplit(":", 1)[-1].isdigit() for r in results)  # url 形如 "path/file.py:123"
