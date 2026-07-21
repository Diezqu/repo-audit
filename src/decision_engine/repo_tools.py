"""仓库工具四件套(T3)：Worker 取证的全部手段。

方案 E 的检索选型是"结构化导航，不用向量库"(DECISIONS.md D15、五问弹药 #2)——
代码是精确标识符的世界，grep 找函数名/类名零误差；embedding 反而把代码结构切碎，
召回模糊。这四个函数就是那句话的全部实现：
  repo_tree  给 Planner 一张地图(读全貌，决定拆哪些子任务)
  repo_stats 给 Planner 语言构成 + 入口线索(地图的文字摘要版)
  grep_repo  给 Worker 按关键字定位候选行(取证的主力工具)
  read_file  给 Worker/Verifier 按 file:line 精确回读上下文(核验的唯一依据)

四个函数共享两条护栏，只写一遍、四处复用：
  1. 路径护栏 _resolve_within——任何来自外部(LLM 输出)的 rel_path，
     必须先 resolve() 到绝对路径，再校验 is_relative_to(root)，越界直接
     抛异常。这不是可选加固：rel_path 本质上是模型的输出，模型可能(无论是
     幻觉还是被仓库内容注入)产出 "../../../etc/passwd" 这样的字符串，护栏
     必须挡在"打开文件"这个动作之前，而不是指望模型自觉。
  2. 截断护栏 _finalize——单次工具调用的返回值有上限(行数与字符数双保险)，
     超限时在结果末尾追加"已截断"提示。这同样不是美化：Worker 池是并行的，
     一个工具调用吐出几万行会直接把那个 Worker 的上下文和整条 run 的成本
     拖垮，上限是成本护栏；提示语是让上游知道"没看全，需要更精确的查询"，
     而不是误以为文件/仓库只有这么大。
"""

import fnmatch
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class PathEscapeError(Exception):
    """rel_path 解析后跳出了仓库根目录——路径穿越，直接拒绝、不做静默纠正。"""


# ── 共享常量 ──────────────────────────────────────────────────────

# 单次返回的上限：行数与字符数都卡，任何一个先到都触发截断。
# 500 行 / 2 万字符是经验值——够 Worker 看清一个中等文件或几十条 grep 命中，
# 又不至于让一次工具调用吃掉半个上下文窗口。
MAX_LINES = 500
MAX_CHARS = 20_000
TRUNCATION_NOTE = "\n\n…（已截断，超出单次返回上限，请缩小范围或提高检索精度后重试）"

# 遍历时跳过的目录/文件噪音：虚拟环境、依赖、构建产物、VCS 元数据。
# 不接入 .gitignore 解析(需要额外三方库做 glob 语义，且本项目"不新增第三方
# 依赖")，用这份手工白名单覆盖绝大多数真实仓库的噪音源——够用，且零依赖。
_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", ".tox",
    "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "dist", "build", ".idea", ".vscode",
}
_SKIP_FILES = {".DS_Store"}


# ── 护栏 1：路径 ──────────────────────────────────────────────────

def _resolve_within(root: Path, rel_path: str) -> Path:
    """把 rel_path 解析到绝对路径，并断言它仍在 root 内部，否则拒绝。

    resolve() 同时处理两类逃逸：
      - 字面上的 "../../etc/passwd"（is_relative_to 会直接判否）；
      - 符号链接逃逸——root 内部一个看似无害的软链接如果指向 root 外部，
        resolve() 会把它展开成真实目标路径，同样会被 is_relative_to 挡下。
    传入绝对路径的 rel_path 也安全：pathlib 的 "/" 运算符遇到绝对右操作数会
    直接丢弃左边的 root，但下面的 is_relative_to 校验照样会因为结果不在
    root 之下而拒绝——护栏生效的关键始终是最后这一次校验，不依赖 "/"
    运算符本身的行为细节。
    """
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    if not candidate.is_relative_to(root):
        raise PathEscapeError(f"越界路径：{rel_path!r} 解析后跳出仓库根目录 {root}")
    return candidate


# ── 护栏 2：截断 ──────────────────────────────────────────────────

def _finalize(lines: list[str], line_limit: int = MAX_LINES, char_limit: int = MAX_CHARS) -> str:
    """行数/字符数任一超限就截断，并统一追加提示——四个工具共用同一份实现，
    保证"已截断"这四个字在任何工具的输出里都是同一种、可被下游识别的信号。
    """
    lines_truncated = len(lines) > line_limit
    text = "\n".join(lines[:line_limit])
    chars_truncated = len(text) > char_limit
    text = text[:char_limit]
    if lines_truncated or chars_truncated:
        text += TRUNCATION_NOTE
    return text


def _iter_files(root: Path):
    """遍历 root 下所有文件，跳过噪音目录——repo_stats 与 grep 的纯 Python
    回退共用这一份枚举逻辑，"什么算仓库内容"只定义一次，不允许两处各写一套
    互相不一致的过滤规则。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name in _SKIP_FILES:
                continue
            yield Path(dirpath) / name


# ── repo_tree ────────────────────────────────────────────────────

def repo_tree(root: str | Path, max_depth: int = 2, *, line_limit: int = MAX_LINES) -> str:
    """目录树文本，深度从 root 算起(max_depth=2 即展示子目录与孙目录两层)。

    Planner 靠它建立"这个仓库大概长什么样"的第一印象，所以噪音目录
    (.git/__pycache__/.venv…)必须过滤，否则一个装好依赖的仓库前几百行全是
    site-packages，地图直接失效。
    """
    root = Path(root).resolve()
    lines = [f"{root.name}/"]

    def _walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            raw_entries = list(dir_path.iterdir())
        except PermissionError:
            return
        entries = [
            e for e in raw_entries
            if e.name not in _SKIP_FILES and not (e.is_dir() and e.name in _SKIP_DIRS)
        ]
        entries.sort(key=lambda e: (e.is_file(), e.name.lower()))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(f"{prefix}{connector}{name}")
            if entry.is_dir():
                _walk(entry, prefix + ("    " if is_last else "│   "), depth + 1)

    _walk(root, "", 1)
    return _finalize(lines, line_limit)


# ── read_file ────────────────────────────────────────────────────

def read_file(
    root: str | Path,
    rel_path: str,
    start: int | None = None,
    end: int | None = None,
    *,
    line_limit: int = MAX_LINES,
    char_limit: int = MAX_CHARS,
) -> str:
    """带行号返回文件内容，行号 1-based、start/end 为闭区间。

    行号格式必须严格对齐 "file:L起-L止" 这个引用格式——它是 Evidence 的
    source_url 与 Verifier 回读核验的唯一接口，行号错一位，引用就是假的。
    """
    path = _resolve_within(Path(root), rel_path)
    if not path.is_file():
        raise FileNotFoundError(f"不是文件或不存在：{rel_path}")

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(raw_lines)
    lo = 1 if start is None else max(1, start)
    hi = total if end is None else min(total, end)
    if lo > hi:
        return "(空范围：start 大于 end，或文件为空)"

    numbered = [f"{lo + i:>6}\t{line}" for i, line in enumerate(raw_lines[lo - 1 : hi])]
    return _finalize(numbered, line_limit, char_limit)


# ── grep_repo ────────────────────────────────────────────────────

def grep_repo(
    root: str | Path,
    pattern: str,
    glob: str | None = None,
    *,
    max_results: int = MAX_LINES,
) -> str:
    """优先 subprocess 调 ripgrep；机器没有 rg 时回退纯 Python 逐行匹配。

    两条路径产出完全相同的格式(rel/path:行号:内容)，调用方不需要知道当前
    机器有没有装 rg——"回退"意味着功能对等，不是残废替代品。pattern 统一按
    Python 正则语义预校验(两条路径共用同一次校验)，非法正则在真正开始扫描
    前就报错，而不是让 rg 和纯 Python 各给一种报错格式。
    """
    root = Path(root).resolve()
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"非法正则表达式：{pattern!r}（{exc}）") from exc

    if shutil.which("rg"):
        try:
            return _grep_with_rg(root, pattern, glob, max_results)
        except (subprocess.SubprocessError, OSError):
            pass  # rg 存在但调用本身失败(权限/损坏等罕见情况)——照样兜底，不让工具挂掉
    return _grep_pure_python(root, pattern, glob, max_results)


def _grep_with_rg(root: Path, pattern: str, glob: str | None, max_results: int) -> str:
    cmd = [
        "rg", "--line-number", "--no-heading", "--color=never",
        "--max-count", str(max_results),  # 单文件内的安全阀；跨文件总量仍由 _finalize 兜底
    ]
    if glob:
        cmd += ["--glob", glob]
    cmd += ["--", pattern]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=root)
    if proc.returncode not in (0, 1):  # 1 = 无匹配，是正常结果不是错误
        raise subprocess.SubprocessError((proc.stderr or "rg 非零退出").strip())
    lines = [ln for ln in proc.stdout.splitlines() if ln]
    if not lines:
        return "(无匹配)"
    return _finalize(lines, max_results)


def _grep_pure_python(root: Path, pattern: str, glob: str | None, max_results: int) -> str:
    """rg 不可用时的等价实现。多抓一条(limit+1)作为"还有更多"的探针——
    否则命中数恰好等于上限时会被误判成"已经是全部"，截断提示就漏报了。
    """
    regex = re.compile(pattern)
    matches: list[str] = []
    probe = max_results + 1
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        if glob and not fnmatch.fnmatch(rel, glob):
            continue
        try:
            with path.open("r", encoding="utf-8", errors="strict") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if regex.search(line):
                        matches.append(f"{rel}:{lineno}:{line.rstrip(chr(10))}")
                        if len(matches) >= probe:
                            break
        except (UnicodeDecodeError, OSError):
            continue  # 二进制/不可读文件——静默跳过，不让一个坏文件拖垮整次 grep
        if len(matches) >= probe:
            break

    if not matches:
        return "(无匹配)"
    return _finalize(matches, max_results)


# ── repo_stats ───────────────────────────────────────────────────

_LANG_BY_EXT = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin",
    ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++", ".cc": "C++",
    ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".md": "Markdown", ".rst": "reStructuredText",
    ".yml": "YAML", ".yaml": "YAML", ".json": "JSON", ".toml": "TOML",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".sql": "SQL",
}
_LANG_BY_NAME = {
    "Makefile": "Makefile", "Dockerfile": "Dockerfile",
    "LICENSE": "Text", "LICENCE": "Text",
}


@dataclass(frozen=True)
class RepoStats:
    """repo_stats 的返回值。结构化而非纯字符串——沿用 D3 的原则(证据是结构化
    对象，不是纯文本)：统计数据本身该是结构化对象，渲染成文本是消费方
    (Planner 的 prompt 拼接)的需要，不该反过来让"文本"成为唯一形态，否则
    程序化读取(比如回归测试里断言语言占比)就得反过来解析字符串。
    """

    total_files: int
    language_breakdown: dict[str, int]
    entry_hints: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"文件总数：{self.total_files}", "语言构成："]
        lines += [f"  {lang}: {count}" for lang, count in self.language_breakdown.items()]
        if self.entry_hints:
            lines.append("入口线索：")
            for hint in self.entry_hints:
                lines.append(f"  {hint}")
        return "\n".join(lines)

    def __str__(self) -> str:  # 直接塞进 prompt 时可以 str(stats) 或 f"{stats}"
        return self.render()


def _find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def repo_stats(root: str | Path, *, hint_lines: int = 12) -> RepoStats:
    """语言构成 + 文件数 + 入口线索(README 头部 / pyproject 概要)。

    入口线索只挑这两个文件不是遗漏——D15 明确 v1 只服务 Python 仓库，
    README 说明"这是什么"、pyproject 说明"依赖与打包方式"，对 Python 仓库
    这两个文件基本就是"如何开始读这个项目"的全部线索。
    """
    root = Path(root).resolve()
    counts: dict[str, int] = {}
    total = 0
    for path in _iter_files(root):
        total += 1
        lang = (
            _LANG_BY_NAME.get(path.name)
            or _LANG_BY_EXT.get(path.suffix.lower())
            or ("无扩展名" if not path.suffix else "其他")
        )
        counts[lang] = counts.get(lang, 0) + 1
    ranked = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    hints: list[str] = []
    readme = _find_first(root, ("README.md", "README.rst", "README.txt", "README"))
    if readme is not None:
        head = readme.read_text(encoding="utf-8", errors="replace").splitlines()[:hint_lines]
        hints.append(f"{readme.name} 头部：\n    " + "\n    ".join(head))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        head = pyproject.read_text(encoding="utf-8", errors="replace").splitlines()[:hint_lines]
        hints.append("pyproject.toml 概要：\n    " + "\n    ".join(head))

    return RepoStats(total_files=total, language_breakdown=ranked, entry_hints=hints)
