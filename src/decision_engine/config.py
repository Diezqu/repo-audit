"""Model tier plumbing: reads the two-tier routing config from environment variables.

Which node uses which tier is an architecture decision made in the graph,
not here — this module only constructs clients.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass(frozen=True)
class ModelTier:
    model: str
    api_key: str
    base_url: str

    def client(self, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
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
        model=os.environ[f"{prefix}_MODEL"],
        api_key=os.environ[f"{prefix}_API_KEY"],
        base_url=os.environ[f"{prefix}_BASE_URL"],
    )


def cheap_tier() -> ModelTier:
    return _tier("CHEAP")


def flagship_tier() -> ModelTier:
    return _tier("FLAGSHIP")


def try_cheap_tier() -> ModelTier | None:
    """有配置就给客户端，没配就给 None——上层节点据此退回假数据模式，
    这样骨架和 CI 在没有任何密钥的环境里也能完整跑通。"""
    try:
        return cheap_tier()
    except RuntimeError:
        return None


def try_flagship_tier() -> ModelTier | None:
    try:
        return flagship_tier()
    except RuntimeError:
        return None
