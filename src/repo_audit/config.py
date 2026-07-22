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


def verifier_enabled() -> bool:
    """Verifier 节点的开关（D3）：读 VERIFIER_ENABLED 环境变量，默认关闭。

    这个开关就是将来"Verifier 开/关 ablation"的闸门本体——README 里点名
    的核心数字（同一评测集在开关两种状态下各跑一遍的引用错误率差值）就是
    拿这个开关一开一关跑出来的，不是另外发明一套双跑机制，评测阶段直接
    复用它。默认关闭是因为 graph._judge_claim 的判定规则本体（分工红线
    §3：Ziyang 亲手写）今晚还没有定案——提前默认打开只会让每条 claim 在
    存在性检查通过后都撞上 NotImplementedError（被节点兜底成
    verdict=None），既没有意义，又会让人误以为核验已经在生效。

    与上面 try_cheap_tier/try_flagship_tier 是同一条"环境变量即配置来源"
    的原则，但返回形状故意不同：那两个要么给一个能构造好的 ModelTier、
    要么给 None（下游据此整体切换到假数据模式，"能不能构造出一个客户端"
    是它们要回答的问题）；这里只是一个布尔开关，天生没有"构造失败"这一说
    ——所以直接返回 bool、不叫 try_verifier_enabled，命名如实反映两者是
    不同性质的读取结果，不是偷懒少写一个 try。
    """
    return os.getenv("VERIFIER_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def langfuse_handler() -> CallbackHandler | None:
    """T8 可观测性埋点：三个 Langfuse env（PUBLIC_KEY/SECRET_KEY/HOST）任一
    为空就返回 None，调用方拿到 None 就不传 callbacks，整条链路与接入
    Langfuse 之前完全一样。

    这与上面 try_cheap_tier/try_flagship_tier 的假数据模式是同一条设计
    原则（D6"骨架无密钥可跑"的延伸）：没有配置时是"优雅缺席"——链路零
    副作用地退回无观测状态，而不是初始化到一半再报错或卡住。

    三个变量在这里显式逐个查、而不是直接 new 一个 CallbackHandler() 再看
    它能不能工作：CallbackHandler() 内部会调 langfuse.get_client()，只要
    进程里还没有任何 Langfuse 客户端存在，这一步就会隐式创建一个默认的
    Langfuse() 单例（同样读这三个 env var）——那个单例具体做了什么（是否
    尝试连网、是否起后台上报线程）由 langfuse 包内部决定，没有文档承诺
    "缺 key 时保证零副作用"。提前一步在这里用纯 Python 的 and 短路掉，
    才能把"零副作用"这个承诺放在我们自己审计得到的代码里，而不是寄望于
    第三方库不会在缺 key 时做多余的事。
    """
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    ):
        return None
    return CallbackHandler()
