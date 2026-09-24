"""Model tier plumbing: reads the two-tier routing config from environment variables.

Which node uses which tier is an architecture decision made in the graph,
not here — this module only constructs clients.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()


@dataclass(frozen=True)
class ModelTier:
    name: str  # "cheap" | "flagship"——仅供可观测性打标用，不参与路由逻辑
    model: str
    api_key: str
    base_url: str

    def client(self, **kwargs) -> ChatOpenAI:
        # T8 成本归因：把"这次调用属于哪一档"随 ChatOpenAI 实例一起绑定成
        # tags/metadata。LangChain 的 Runnable 会把构造时绑定的 tags/metadata
        # 自动并入之后每一次 invoke() 的运行时 config，Langfuse 的
        # CallbackHandler 从同一份运行时 config 里把它们原样抄进 trace——
        # 不需要在 planner/worker/synthesizer 调用点各自手动传。
        #
        # 为什么标 tier 而不是等 Langfuse 自动抓到的 model 名去反推：
        # Langfuse 本来就会自动记录 model 名和 token 用量，但"按档聚合
        # 成本"如果要靠"model 名 → 属于哪一档"反推，就得单独维护一张映射表，
        # 换供应商/改配置（config.py 顶部就是"全走环境变量、换供应商不改
        # 代码"）这张表马上过期。tier 标签在调用发起的源头直接打上去，
        # 归因就不依赖任何映射表，Langfuse 那边按 tags 分组/筛选即可拿到
        # "旗舰档花了多少、便宜档花了多少"。
        #
        # 用 kwargs.pop 而不是直接覆盖：避免未来某个调用点自己传了
        # tags/metadata 时被这里悄悄吃掉——合并而不是覆盖。
        tags = [self.name, *kwargs.pop("tags", [])]
        metadata = {"model_tier": self.name, **kwargs.pop("metadata", {})}
        kwargs.setdefault("timeout", 60)
        kwargs.setdefault("max_retries", 1)
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )


def _tier(prefix: str) -> ModelTier:
    missing = [
        f"{prefix}_{k}"
        for k in ("MODEL", "API_KEY", "BASE_URL")
        if not os.getenv(f"{prefix}_{k}")
    ]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)} (see .env.example)")
    return ModelTier(
        name=prefix.lower(),
        model=os.environ[f"{prefix}_MODEL"],
        api_key=os.environ[f"{prefix}_API_KEY"],
        base_url=os.environ[f"{prefix}_BASE_URL"],
    )


def cheap_tier() -> ModelTier:
    return _tier("CHEAP")


def flagship_tier() -> ModelTier:
    return _tier("FLAGSHIP")


def run_mode() -> str:
    """Only an explicit demo setting permits fabricated output."""
    mode = os.getenv("REPO_AUDIT_MODE", "real").strip().lower()
    if mode not in {"real", "demo"}:
        raise ValueError("REPO_AUDIT_MODE must be 'real' or 'demo'")
    return mode


def _real_tiers() -> tuple[ModelTier, ModelTier]:
    """Validate both providers before any model can be called."""
    return cheap_tier(), flagship_tier()


def try_cheap_tier() -> ModelTier | None:
    if run_mode() == "demo":
        return None
    return _real_tiers()[0]


def try_flagship_tier() -> ModelTier | None:
    if run_mode() == "demo":
        return None
    return _real_tiers()[1]


def verifier_enabled() -> bool:
    """Run deterministic citation checks by default; opt out explicitly."""
    return os.getenv("VERIFIER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off", ""}


def langfuse_handler() -> CallbackHandler | None:
    """Skip tracing in demo, or when any Langfuse setting is absent.

    Check the environment before constructing CallbackHandler: its constructor
    may initialize an exporter and background thread.
    """
    if run_mode() == "demo" or not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    ):
        return None
    return CallbackHandler()
