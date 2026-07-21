"""仓库工具四件套(T3)单测。

两层 fixture 缺一不可：
  - tmp_path 造的最小合成仓库——用来精确断言行号/截断阈值/护栏这些
    "差一位就是假引用"的细节，需要完全可控的输入；
  - 本仓库自身(decision-engine/repo-audit)——用来证明工具在一个真实、
    混杂(有 .git/.venv/多种文件类型)的仓库上跑得通，不是只在玩具数据上
    工作。任务要求"用本仓库自身当 fixture"，就是奔着这层去的。

rg 分支的测试策略：这台机器上没有真的装 ripgrep(`shutil.which("rg")`
在 subprocess 能找到的 PATH 里查不到——交互式 shell 里的 `rg` 其实是
Claude Code 自带的 shell 函数壳，不是独立二进制，子进程调用看不到它)。
所以"纯 Python 回退路径"是这台机器上真实会走到的代码，直接测；
"rg 路径"用 monkeypatch 伪造 shutil.which + subprocess.run 来测命令构造
与输出解析是否正确——不依赖测试机是否装了 rg，两条路径的覆盖都不看运气。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from decision_engine.repo_tools import (
    PathEscapeError,
    grep_repo,
    read_file,
    repo_stats,
    repo_tree,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_repo(tmp_path):
    """最小合成仓库：够小、够可控，用来精确断言行号/截断/护栏。"""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    (tmp_path / "README.md").write_text("# Sample\n\nThis is a fixture repo.\n")
    (tmp_path / "src" / "pkg" / "core.py").write_text(
        "\n".join(f"line {i}" for i in range(1, 11)) + "\n"
    )
    (tmp_path / "src" / "pkg" / "util.py").write_text(
        "def helper():\n    return 42\n\n\ndef needle_marker():\n    pass\n"
    )
    return tmp_path


# ──────────────────────────────────────────────────────────────
# 路径护栏
# ──────────────────────────────────────────────────────────────

def test_read_file_rejects_relative_escape(sample_repo):
    with pytest.raises(PathEscapeError):
        read_file(sample_repo, "../outside.txt")


def test_read_file_rejects_absolute_escape(sample_repo):
    with pytest.raises(PathEscapeError):
        read_file(sample_repo, "/etc/passwd")


def test_read_file_rejects_symlink_escape(sample_repo, tmp_path_factory):
    """resolve() 会展开符号链接——root 内部一个指向 root 外部的软链接，
    看似"相对路径没有 .."，实际解析后一样跳出根目录，必须照样被拒。"""
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_text("top secret\n")
    link = sample_repo / "escape_link"
    link.symlink_to(secret)

    with pytest.raises(PathEscapeError):
        read_file(sample_repo, "escape_link")


def test_read_file_within_root_is_allowed(sample_repo):
    text = read_file(sample_repo, "README.md")
    assert "Sample" in text


# ──────────────────────────────────────────────────────────────
# 行号正确性
# ──────────────────────────────────────────────────────────────

def test_read_file_line_numbers_and_range(sample_repo):
    text = read_file(sample_repo, "src/pkg/core.py", start=3, end=5)
    lines = text.splitlines()
    assert len(lines) == 3
    lineno, content = lines[0].split("\t")
    assert lineno.strip() == "3"
    assert content == "line 3"
    assert lines[-1].endswith("line 5")


def test_read_file_full_range_when_start_end_omitted(sample_repo):
    text = read_file(sample_repo, "src/pkg/core.py")
    lines = text.splitlines()
    assert lines[0].endswith("line 1")
    assert lines[-1].endswith("line 10")
    assert len(lines) == 10


def test_read_file_missing_file_raises(sample_repo):
    with pytest.raises(FileNotFoundError):
        read_file(sample_repo, "does_not_exist.py")


# ──────────────────────────────────────────────────────────────
# 截断生效
# ──────────────────────────────────────────────────────────────

def test_read_file_line_limit_truncates(sample_repo):
    text = read_file(sample_repo, "src/pkg/core.py", line_limit=3)
    assert "已截断" in text
    body = text.split("\n\n…")[0]
    assert len(body.splitlines()) == 3


def test_read_file_char_limit_truncates(sample_repo):
    text = read_file(sample_repo, "src/pkg/core.py", char_limit=5)
    assert "已截断" in text


def test_grep_repo_result_limit_truncates(sample_repo):
    (sample_repo / "many.txt").write_text(
        "\n".join(f"needle {i}" for i in range(10)) + "\n"
    )
    out = grep_repo(sample_repo, r"needle", max_results=2)
    assert "已截断" in out
    body = out.split("\n\n…")[0]
    assert len(body.splitlines()) == 2


def test_grep_repo_exact_boundary_not_marked_truncated(sample_repo):
    """命中数恰好等于上限时不应误报"已截断"——探针(limit+1)机制的验证点。"""
    (sample_repo / "exact.txt").write_text(
        "\n".join(f"needle {i}" for i in range(3)) + "\n"
    )
    out = grep_repo(sample_repo, r"needle", glob="exact.txt", max_results=3)
    assert "已截断" not in out
    assert len(out.splitlines()) == 3


def test_repo_tree_line_limit_truncates(sample_repo):
    text = repo_tree(sample_repo, max_depth=2, line_limit=2)
    assert "已截断" in text


# ──────────────────────────────────────────────────────────────
# repo_tree：噪音过滤 + 深度
# ──────────────────────────────────────────────────────────────

def test_repo_tree_depth_limits_recursion(sample_repo):
    """fixture 实际层级是 root/src/pkg/core.py——三层。逐级验证:
    max_depth=1 只看到 root 的直接子项(src/、README.md)；
    max_depth=2 多看到一层(pkg/)；core.py 在 pkg 内部，属于第 3 层，
    要 max_depth=3 才可见——深度数与"目录层级"必须严格对应，差一层
    Planner 拿到的地图就会漏掉或多出一整层结构。"""
    depth1 = repo_tree(sample_repo, max_depth=1)
    assert "src" in depth1
    assert "pkg" not in depth1
    assert "core.py" not in depth1

    depth2 = repo_tree(sample_repo, max_depth=2)
    assert "pkg" in depth2
    assert "core.py" not in depth2

    depth3 = repo_tree(sample_repo, max_depth=3)
    assert "core.py" in depth3


def test_repo_tree_skips_noise_dirs(sample_repo):
    text = repo_tree(sample_repo, max_depth=3)
    assert ".git" not in text


# ──────────────────────────────────────────────────────────────
# grep_repo：纯 Python 回退路径(这台机器的真实代码路径)
# ──────────────────────────────────────────────────────────────

def test_grep_repo_pure_python_finds_match(sample_repo, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)  # 强制走纯 Python 分支
    out = grep_repo(sample_repo, r"needle_marker")
    assert "src/pkg/util.py:5:def needle_marker():" in out


def test_grep_repo_no_match_returns_sentinel(sample_repo, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = grep_repo(sample_repo, r"nonexistent_token_xyz")
    assert out == "(无匹配)"


def test_grep_repo_glob_filter(sample_repo, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = grep_repo(sample_repo, r"Sample", glob="*.md")
    assert "README.md" in out

    out2 = grep_repo(sample_repo, r"Sample", glob="*.py")
    assert out2 == "(无匹配)"


def test_grep_repo_invalid_regex_raises(sample_repo):
    with pytest.raises(ValueError):
        grep_repo(sample_repo, r"(unclosed")


# ──────────────────────────────────────────────────────────────
# grep_repo：rg 路径(mock subprocess，不依赖测试机是否真装了 rg)
# ──────────────────────────────────────────────────────────────

def test_grep_repo_uses_rg_when_available(sample_repo, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)
    captured = {}

    def fake_run(cmd, capture_output, text, timeout, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

        class _Result:
            returncode = 0
            stdout = "src/pkg/util.py:5:def needle_marker():\n"
            stderr = ""

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = grep_repo(sample_repo, r"needle_marker", glob="*.py")

    assert "src/pkg/util.py:5:def needle_marker():" in out
    assert captured["cwd"] == sample_repo.resolve()
    assert "needle_marker" in captured["cmd"]
    assert "--glob" in captured["cmd"]


def test_grep_repo_rg_failure_falls_back_to_pure_python(sample_repo, monkeypatch):
    """rg 存在但调用失败(权限/损坏等)——照样兜底到纯 Python，不让工具挂掉。"""
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    def broken_run(*args, **kwargs):
        raise OSError("simulated rg crash")

    monkeypatch.setattr(subprocess, "run", broken_run)
    out = grep_repo(sample_repo, r"needle_marker")
    assert "src/pkg/util.py:5:def needle_marker():" in out


def test_grep_repo_rg_no_match_returncode_is_not_error(sample_repo, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    def fake_run(cmd, capture_output, text, timeout, cwd=None):
        class _Result:
            returncode = 1  # rg: 1 = 无匹配，不是错误
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = grep_repo(sample_repo, r"needle_marker")
    assert out == "(无匹配)"


@pytest.mark.skipif(shutil.which("rg") is None, reason="需要真实安装 rg 才能跑集成测试")
def test_grep_repo_real_rg_integration(sample_repo):
    """机器上真的有 rg 时才跑：证明不是只在 mock 世界里正确。"""
    out = grep_repo(sample_repo, r"needle_marker")
    assert "util.py" in out


# ──────────────────────────────────────────────────────────────
# repo_stats
# ──────────────────────────────────────────────────────────────

def test_repo_stats_counts_and_language_breakdown(sample_repo):
    stats = repo_stats(sample_repo)
    assert stats.total_files >= 3  # README.md + core.py + util.py（.git 内容不计入）
    assert stats.language_breakdown.get("Python", 0) >= 2
    assert stats.language_breakdown.get("Markdown", 0) >= 1


def test_repo_stats_entry_hints_include_readme_head(sample_repo):
    stats = repo_stats(sample_repo)
    assert any("Sample" in hint for hint in stats.entry_hints)


def test_repo_stats_render_is_string(sample_repo):
    stats = repo_stats(sample_repo)
    assert isinstance(stats.render(), str)
    assert str(stats) == stats.render()


# ──────────────────────────────────────────────────────────────
# 用本仓库自身当 fixture——端到端在真实仓库上不炸
# ──────────────────────────────────────────────────────────────

def test_repo_tree_on_real_repo():
    text = repo_tree(REPO_ROOT, max_depth=2)
    assert "src" in text
    assert ".venv" not in text


def test_repo_stats_on_real_repo():
    stats = repo_stats(REPO_ROOT)
    assert stats.total_files > 0
    assert "Python" in stats.language_breakdown


def test_grep_repo_on_real_repo_finds_stategraph():
    out = grep_repo(REPO_ROOT, r"StateGraph", glob="*.py")
    assert "graph.py" in out


def test_read_file_on_real_repo_pyproject():
    text = read_file(REPO_ROOT, "pyproject.toml", start=1, end=3)
    lineno = text.splitlines()[0].split("\t")[0].strip()
    assert lineno == "1"
