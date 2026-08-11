"""Model-message governance for agent runner requests.

This module owns model-facing message shaping and tool-result content normalization.
It may return copied messages or persisted-result placeholders, but it must not
mutate an existing session history list in place.

中文：本模块负责 agent runner 请求中“面向模型的消息治理”。
它主要负责：
1. 整理即将发给模型的消息列表；
2. 规范化工具调用结果内容；
3. 在上下文超限时做裁剪、压缩、占位替换等处理。

它可以返回：
- 复制出来的新消息列表；
- 已持久化工具结果的占位内容；
但绝不能原地修改已经存在的会话历史列表，否则会污染持久化 transcript。
"""

# from __future__ import annotations
# 作用：延迟求值类型注解（PEP 563）。
# 开启后，类型注解不会在定义时立即解析成真实对象，
# 可以减少循环导入问题，也允许在注解中使用尚未定义的类型名。
from __future__ import annotations

# dataclass：用于通过类注解自动生成 __init__、__repr__ 等样板代码。
from dataclasses import dataclass

# Path：面向对象的文件路径类型，比字符串路径更安全、更易用。
from pathlib import Path

# TYPE_CHECKING：只在类型检查工具（如 mypy/pyright）运行时为 True，
#                真实运行时为 False，常用于避免循环导入。
# Any：表示任意类型。
# cast：类型转换辅助函数，只用于类型检查器，不影响运行时值。
from typing import TYPE_CHECKING, Any, cast

# loguru 的 logger，用于记录日志。
from loguru import logger

# 从项目工具函数模块导入一组辅助函数。
from nanobot.utils.helpers import (
    # 估算单条消息大约占用多少 token。
    estimate_message_tokens,
    # 估算一整组 prompt/messages + tools 定义大约占用多少 token。
    # 通常会结合 provider/model 的 tokenizer 或估算逻辑。
    estimate_prompt_tokens_chain,
    # 寻找“合法消息起始位置”。
    # 因为某些 provider 要求历史消息必须从 user 消息开始，
    # 或者不能以 tool result 开头等，所以需要一个合法切片起点。
    find_legal_message_start,
    # 如果工具结果过大，可能会把它持久化到文件/存储中，
    # 然后返回一个较短的占位说明或截断内容。
    maybe_persist_tool_result,
    # 按字符数截断文本。
    truncate_text,
)

# 确保工具结果非空；如果为空，会根据工具名生成默认输出。
from nanobot.utils.runtime import ensure_nonempty_tool_result

# TYPE_CHECKING 在运行时是 False，
# 因此下面两个 import 只服务于类型提示，不会真正导入，避免循环依赖。
if TYPE_CHECKING:
    # 工具注册表类型，用于获取工具定义。
    from nanobot.agent.tools.registry import ToolRegistry
    # LLM provider 基类类型，用于访问 provider 能力和 token 估算。
    from nanobot.providers.base import LLMProvider

# 安全缓冲 token 数。
# 计算输入预算时会额外减去这个 buffer，避免贴着模型上限导致溢出。
SNIP_SAFETY_BUFFER = 1024

# 工具结果“可被 in-flight compact”的最小字符数。
# 小于这个长度的工具结果通常不值得压缩。
MICROCOMPACT_MIN_CHARS = 500

# in-flight 压缩目标比例。
# 当上下文超预算时，不一定只压到刚好等于 budget，
# 而是压到 budget 的 85%，留一些余量。
INFLIGHT_COMPACT_TARGET_RATIO = 0.85

# 可参与 in-flight compact 的工具名集合。
# frozenset：不可变集合，创建后不能修改，适合常量配置。
COMPACTABLE_TOOLS = frozenset({
    "read_file",           # 读文件结果可能很大
    "exec",                # 命令执行输出可能很大
    "grep",                # 搜索结果可能很大
    "find_files",          # 文件查找结果可能很多
    "web_search",          # 网页搜索结果可能很大
    "web_fetch",           # 网页抓取正文可能很大
    "list_dir",            # 目录列表可能很大
    "list_exec_sessions",  # exec 会话列表也可能较大
})

# read_file is the recovery path for persisted results; exempting it prevents persist->read->persist loops.
# 中文：read_file 是读取“已持久化工具结果”的恢复路径；
# 如果把 read_file 的结果也 offload/持久化，可能会形成：
# 持久化 -> 用 read_file 读持久化结果 -> read_file 结果又被持久化 的循环。
# 因此这里豁免 read_file，不做工具结果 offload。
TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS = frozenset({"read_file"})

# 当历史中的 tool result 缺失/丢失/中断时，用来补位的默认内容。
BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"

# 需要被剔除的 assistant 占位文本集合。
# 这类文本通常是上下文压缩后留下的无意义占位，不应再发给模型。
PLACEHOLDER_TEXTS = frozenset({
    "[Previous assistant message omitted.]",
})


def _tool_call_name_is_valid(tool_call: Any) -> bool:
    """Whether a persisted OpenAI-style tool_call carries a usable name.

    Mirrors ``ToolCallRequest.has_valid_name`` for the dict shape stored in
    message history: a degenerate call with ``name=None`` / ``""`` cannot be
    executed and is rejected by upstream APIs if replayed.

    中文：判断一个持久化后的 OpenAI 风格 tool_call 是否带有可用名称。

    这里镜像了 ``ToolCallRequest.has_valid_name`` 的逻辑，
    用于检查保存在 message history 中的 dict 结构：
    如果一个 tool call 的 name 是 None 或空字符串，那么它不可执行，
    并且如果重放给上游 API，也会被拒绝。
    """
    # 如果 tool_call 不是 dict，直接认为非法。
    # 这里用 isinstance 做运行时类型检查。
    if not isinstance(tool_call, dict):
        return False

    # cast 不会改变运行时值，只是告诉类型检查器：
    # “我确认 tool_call 是 dict[str, Any] 类型”。
    tool_call_data = cast(dict[str, Any], tool_call)

    # OpenAI 风格 tool_call 通常形如：
    # {
    #   "id": "...",
    #   "type": "function",
    #   "function": {"name": "...", "arguments": "..."}
    # }
    # 所以先尝试从 function 字段取。
    fn = tool_call_data.get("function")

    # 如果 function 是 dict，则从 function.name 取名字；
    # 否则兼容某些直接放在顶层的 name 字段。
    # isinstance(fn, dict) 是类型守卫。
    # cast(...) 只是辅助类型检查。
    name = cast(dict[str, Any], fn).get("name") if isinstance(fn, dict) else tool_call_data.get("name")

    # 名字必须是非空字符串才算有效。
    # bool(name) 会排除空字符串 ""。
    return isinstance(name, str) and bool(name)


# @dataclass(slots=True)
# dataclass：根据字段注解自动生成构造函数等。
# slots=True：让生成的类使用 __slots__，而不是 __dict__。
# 好处：节省内存、属性访问略快；缺点：不能动态添加未声明属性。
@dataclass(slots=True)
class ContextGovernanceConfig:
    """
    中文：上下文治理配置对象。

    它封装了 ContextGovernor 在处理消息时需要的运行环境信息：
    - provider：模型提供方；
    - model：模型名；
    - tools：工具注册表；
    - workspace：工作区路径，用于持久化大型工具结果；
    - session_key：会话标识；
    - max_tool_result_chars：单条工具结果允许保留的最大字符数；
    - context_window_tokens：模型上下文窗口大小；
    - context_block_limit：显式上下文块限制；
    - max_tokens：模型最大输出 token；
    - inflight_start_index：从哪个消息索引开始允许 in-flight compact。
    """

    # LLMProvider 实例，提供模型调用、token 估算等能力。
    provider: LLMProvider

    # 当前模型名，例如 "gpt-4.1"、"claude-..." 等。
    model: str

    # 工具注册表，用于获取当前可用工具的 JSON schema 定义。
    tools: ToolRegistry

    # 工作区路径。
    # Path | None 是联合类型，表示可能是 Path，也可能是 None。
    # 这里用于持久化过大的 tool result。
    workspace: Path | None

    # 会话 key，用于区分不同 session。
    # 如果没有 session 概念，可以为 None。
    session_key: str | None

    # 单条工具结果最多保留多少字符。
    # 超过这个值时可能被持久化、截断或压缩。
    max_tool_result_chars: int

    # 模型上下文窗口总 token 数。
    # None 表示未知或未配置。
    context_window_tokens: int | None = None

    # 某些 provider 可能有单独的 context block limit。
    # 如果设置了，就优先用它作为输入预算。
    context_block_limit: int | None = None

    # 本次请求允许的最大输出 token 数。
    # None 表示使用 provider 默认值。
    max_tokens: int | None = None

    # inflight compact 只处理该索引及之后的消息。
    # 用于避免压缩历史早期固定消息或系统消息之后的不可动区域。
    inflight_start_index: int = 0


class ContextGovernor:
    """Prepare model-copy messages while preserving persisted history.

    中文：准备一份“面向模型的消息副本”，同时不破坏已持久化的历史记录。
    """

    def prepare_for_model(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        """
        中文：把原始历史 messages 整理成可以安全发给模型的 messages。

        参数：
        - config：上下文治理配置；
        - messages：原始消息历史；
        - compacted_tool_call_ids：已经被压缩过的 tool_call_id 集合。
          这个集合会被本方法就地更新（新增本次压缩过的 id）。

        返回：
        - 一个新的或经过治理的消息列表，可发给模型。
        """

        # 第一步：移除 assistant 占位消息。
        # 例如 "[Previous assistant message omitted.]" 这种无意义内容。
        updated = self.strip_placeholder_assistant_messages(messages)

        # 第二步：移除格式非法的 tool_calls。
        # 例如 tool call name 为 None、空字符串、非字符串等。
        updated = self.strip_malformed_tool_calls(updated)

        # 第三步：丢弃孤儿 tool result。
        # 即没有对应 assistant.tool_calls 声明的 tool 消息。
        updated = self.drop_orphan_tool_results(updated)

        # 第四步：为缺失 tool result 的 tool_call 补一个错误结果。
        # 某些 provider 要求每个 tool_call 必须有对应 tool result。
        updated = self.backfill_missing_tool_results(updated)

        # 第五步：应用工具结果预算。
        # 过大的 tool result 会被持久化/截断/规范化。
        updated = self.apply_tool_result_budget(config, updated)

        # 第六步：如果当前请求仍然可能超上下文，
        # 则对“正在执行链路中”的大 tool result 做压缩。
        updated = self.compact_inflight_overflow(config, updated, compacted_tool_call_ids)

        # 第七步：如果还是超，就从历史头部开始裁剪旧消息。
        updated = self.snip_history(config, updated)

        # 第八步：裁剪后可能又产生孤儿 tool result，再清理一次。
        updated = self.drop_orphan_tool_results(updated)

        # 第九步：裁剪后可能又出现缺失 tool result，再补一次。
        return self.backfill_missing_tool_results(updated)

    @staticmethod
    def input_budget(config: ContextGovernanceConfig) -> int:
        """
        中文：计算本次请求可用于输入（prompt + tools）的 token 预算。

        逻辑：
        1. 如果没有 context_window_tokens，返回 0；
        2. 估算/获取最大输出 token；
        3. 如果有显式 context_block_limit，则直接用它；
        4. 否则：context_window_tokens - max_output - 安全 buffer。
        """

        # 如果不知道上下文窗口大小，就无法计算预算。
        # 在 Python 中，None、0 都会在这个 if 中视为假值。
        if not config.context_window_tokens:
            return 0

        # getattr(obj, name, default)：安全获取属性。
        # 这里等价于：
        #   generation = getattr(config.provider, "generation", None)
        #   provider_max_tokens = getattr(generation, "max_tokens", 4096)
        #
        # 如果 config.provider 没有 generation 属性，
        # 内层 getattr 得到 None，然后 getattr(None, "max_tokens", 4096)
        # 不会报错，而是返回默认值 4096。
        provider_max_tokens = getattr(
            getattr(config.provider, "generation", None),
            "max_tokens",
            4096,
        )

        # 计算最大输出 token。
        # 优先级：
        # 1. config.max_tokens 如果是 int，就用它；
        # 2. 否则如果 provider_max_tokens 是 int，就用 provider 默认；
        # 3. 否则兜底 4096。
        #
        # 这里使用了 Python 条件表达式：
        #   a if condition else b
        max_output = config.max_tokens if isinstance(config.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )

        # 计算输入预算。
        # config.context_block_limit or (...)：
        # 如果 context_block_limit 是非 None/非 0 的值，则使用它；
        # 否则用公式计算。
        budget = config.context_block_limit or (
            config.context_window_tokens - max_output - SNIP_SAFETY_BUFFER
        )

        # 预算必须大于 0，否则返回 0。
        return budget if budget > 0 else 0

    @staticmethod
    def normalize_tool_result(
        config: ContextGovernanceConfig,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        """
        中文：规范化某个工具调用结果。

        主要做三件事：
        1. 确保结果非空；
        2. 如果结果过大，尝试持久化到 workspace；
        3. 如果仍是过长字符串，则截断到 max_tool_result_chars。
        """

        # 确保工具结果不是空值。
        # 如果工具返回 None/空字符串等，该函数可能生成一段说明文本，
        # 避免模型看到空结果产生困惑。
        result = ensure_nonempty_tool_result(tool_name, result)

        # read_file 是恢复持久化结果的路径，不能再被 offload。
        # 否则会出现 persist -> read -> persist 的循环。
        if tool_name in TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS:
            return result

        # 尝试持久化过大的工具结果。
        # maybe_persist_tool_result 可能返回：
        # - 原始内容（如果不大）；
        # - 占位说明（例如“结果已保存到某文件”）；
        # - 截断后的内容。
        try:
            content = maybe_persist_tool_result(
                config.workspace,           # 工作区路径，可能为 None
                config.session_key,         # 当前会话 key
                tool_call_id,               # tool call id，用于命名/索引
                result,                     # 原始结果
                max_chars=config.max_tool_result_chars,  # 最大字符数阈值
            )
        except Exception:
            # 持久化失败不能阻断主流程。
            # logger.exception 会记录异常堆栈。
            logger.exception(
                "Tool result persist failed for {} in {}; using raw result",
                tool_call_id,
                config.session_key or "default",
            )
            # 回退使用原始结果。
            content = result

        # 如果经过持久化/处理后仍是字符串，并且仍然超长，
        # 就强制截断到配置上限。
        if isinstance(content, str) and len(content) > config.max_tool_result_chars:
            return truncate_text(content, config.max_tool_result_chars)

        # 否则直接返回处理后的内容。
        return content

    @staticmethod
    def strip_placeholder_assistant_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove assistant messages that are compaction placeholders.

        Messages like ``[Previous assistant message omitted.]`` carry no useful
        context for the model and can cause it to repeatedly attempt tool calls
        that previously failed, producing malformed responses in a loop.
        Consecutive same-role messages that result from removal are handled
        downstream by the provider's merge-consecutive logic. Only the
        model-facing copy is repaired; the persisted transcript is untouched
        (a copy is returned, or the same list object when nothing changes).

        中文：删除作为“压缩占位符”的 assistant 消息。

        像 ``[Previous assistant message omitted.]`` 这类消息对模型没有价值，
        反而可能诱导模型重复尝试此前失败的工具调用，形成错误循环。

        删除后可能出现连续同角色消息，例如连续两条 user 消息，
        这个问题交由下游 provider 的“合并连续同角色消息”逻辑处理。

        只修复“面向模型的副本”，不修改持久化 transcript：
        - 如果没有变化，返回原列表对象；
        - 如果有变化，返回新列表。
        """

        # updated 初始为 None。
        # 这是一种“惰性复制”模式：
        # 只要没有发现需要删除的消息，就不复制列表，直接返回原 messages。
        updated: list[dict[str, Any]] | None = None

        # enumerate 同时拿到索引 idx 和消息 msg。
        for idx, msg in enumerate(messages):
            # 如果不是 assistant 消息：
            if msg.get("role") != "assistant":
                # 如果已经进入了复制模式，就把这条消息加入新列表。
                if updated is not None:
                    updated.append(msg)
                # 否则还没发生删除，不需要复制/追加。
                continue

            # 取出 assistant 消息 content。
            # OpenAI 风格 content 可能是字符串，也可能是多模态 list。
            content = msg.get("content", "")

            # 只在 content 是字符串时提取文本；否则视为空字符串。
            text = content if isinstance(content, str) else ""

            # 判断文本去掉首尾空白后是否是已知占位文本。
            is_placeholder = text.strip() in PLACEHOLDER_TEXTS

            # 判断该 assistant 消息是否带 tool_calls。
            # bool(msg.get("tool_calls"))：
            # - None/[] 为 False；
            # - 非空 list 为 True。
            has_tool_calls = bool(msg.get("tool_calls"))

            # 只有“纯占位文本且没有 tool_calls”的 assistant 消息才删除。
            # 如果带 tool_calls，不能随便删，否则后续 tool result 会孤立。
            if is_placeholder and not has_tool_calls:
                # 第一次发现要删除的消息时，创建新列表，
                # 并把删除点之前的消息复制进去。
                if updated is None:
                    updated = list(messages[:idx])

                # 记录调试日志。
                # {!r} 表示用 repr 输出，便于看到引号和转义。
                logger.debug(
                    "Stripping placeholder assistant message from history: {!r}",
                    text[:60],
                )

                # continue 表示不把当前消息加入 updated，即删除它。
                continue

            # 如果已经进入复制模式，则保留当前消息。
            if updated is not None:
                updated.append(msg)

        # 如果 updated 仍然是 None，说明没有任何消息被删除。
        if updated is None:
            return messages

        # 返回清理后的新列表。
        return updated

    @staticmethod
    def strip_malformed_tool_calls(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop persisted assistant tool_calls whose name is missing/non-string.

        A degenerate tool call (``name=None`` or ``""``) that slipped into the
        saved history before this guard existed gets replayed on every turn and
        makes upstream APIs reject the whole request
        (``messages.content.N.tool_use.name: Input should be a valid string``),
        permanently wedging the session. Removing the bad call here lets the
        existing orphan-result cleanup drop its now-dangling tool result, so a
        polluted session self-heals on its next turn. The persisted transcript
        is left untouched; only the model-facing copy is repaired (a copy is
        returned, or the same list object when nothing changes).

        中文：删除持久化历史中 name 缺失/非字符串的 assistant tool_calls。

        如果一个退化 tool call（name=None 或 ""）在这个守卫存在之前进入历史，
        它会在每一轮被重放，并导致上游 API 拒绝整个请求，例如：
        ``messages.content.N.tool_use.name: Input should be a valid string``，
        从而让会话永久卡死。

        在这里删除坏 tool call 后，后续的孤儿 tool result 清理逻辑
        会顺带删除它对应的悬空 tool result，让被污染的会话在下一轮自愈。

        持久化 transcript 不变；只修复面向模型的副本：
        - 没有变化时返回原列表；
        - 有变化时返回新列表。
        """

        # updated 为 None 表示尚未发生任何修改。
        updated: list[dict[str, Any]] | None = None

        # 遍历所有消息。
        for idx, msg in enumerate(messages):
            # 只处理 assistant 消息，因为只有 assistant 会带 tool_calls。
            if msg.get("role") != "assistant":
                # 如果已经在复制模式，则保留非 assistant 消息。
                if updated is not None:
                    updated.append(msg)
                continue

            # 取出 tool_calls 字段。
            calls = msg.get("tool_calls")

            # 如果没有 tool_calls，保留原消息。
            if not calls:
                if updated is not None:
                    updated.append(msg)
                continue

            # 列表推导式：只保留 name 合法的 tool_call。
            # cast(list[Any], calls) 只是告诉类型检查器 calls 是 list。
            kept = [tc for tc in cast(list[Any], calls) if _tool_call_name_is_valid(tc)]

            # 如果保留数量等于原数量，说明没有非法 tool_call。
            if len(kept) == len(calls):
                if updated is not None:
                    updated.append(msg)
                continue

            # 第一次发现需要修改时，复制当前消息之前的所有消息。
            # [dict(m) for m in messages[:idx]]：
            # - 对列表做浅拷贝；
            # - 对每条消息 dict 做浅拷贝，避免后续修改影响原 message。
            if updated is None:
                updated = [dict(m) for m in messages[:idx]]

            # 记录告警：删除了多少个非法 tool_call。
            logger.warning(
                "Stripping {} malformed tool_call(s) with missing/non-string "
                "name from assistant history before request",
                len(calls) - len(kept),
            )

            # 创建当前 assistant 消息的副本，避免修改原对象。
            repaired = dict(msg)

            # 如果还有合法 tool_calls，就替换为 kept。
            if kept:
                repaired["tool_calls"] = kept
            else:
                # 如果全部 tool_call 都非法，直接删除 tool_calls 字段。
                # dict.pop(key, default)：如果 key 不存在也不会报错。
                repaired.pop("tool_calls", None)

            # An assistant turn with neither content nor any valid tool call is
            # itself invalid upstream; drop it entirely in that case.
            # 中文：如果一条 assistant 消息既没有 content，也没有任何有效 tool call，
            # 那么它对上游 API 来说本身就是无效的；这种情况下整条删除。

            # 判断是否还有文本内容。
            has_content = bool(repaired.get("content"))

            # 如果没有 kept，也没有 content，则不 append，相当于删除整条消息。
            if not kept and not has_content:
                continue

            # 否则保留修复后的 assistant 消息。
            updated.append(repaired)

        # 如果没有发生任何修改，返回原列表。
        if updated is None:
            return messages

        # 返回修复后的新列表。
        return updated

    @staticmethod
    def drop_orphan_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop invalid tool results before history is sent back to providers.

        中文：在历史发送给 provider 之前，删除非法/孤立的 tool result。
        """

        # declared：assistant.tool_calls 中声明过的 tool_call id 集合。
        declared: set[str] = set()

        # fulfilled：已经有对应 tool result 的 tool_call id 集合。
        # 用于防止同一个 tool_call_id 出现多个重复 tool result。
        fulfilled: set[str] = set()

        # updated None 表示还没有发生删除/修改。
        updated: list[dict[str, Any]] | None = None

        # 遍历消息。
        for idx, msg in enumerate(messages):
            # 获取角色。
            role = msg.get("role")

            # 如果是 assistant 消息，收集它声明的 tool_call id。
            if role == "assistant":
                # msg.get("tool_calls") or []：
                # 如果 tool_calls 为 None/False，则用空列表。
                for tc in cast(list[Any], msg.get("tool_calls") or []):
                    # 只处理 dict 类型 tool_call。
                    if isinstance(tc, dict):
                        tool_call = cast(dict[str, Any], tc)

                        # 如果有 id，则加入 declared。
                        if tool_call.get("id"):
                            declared.add(str(tool_call["id"]))

            # 如果是 tool result 消息，需要验证它是否合法。
            if role == "tool":
                # tool result 必须通过 tool_call_id 关联到 assistant.tool_calls。
                tid = msg.get("tool_call_id")

                # 转成字符串；如果没有则为空字符串。
                tid_str = str(tid) if tid else ""

                # 删除条件：
                # 1. 没有 tool_call_id；
                # 2. tool_call_id 不在 assistant 声明集合中；
                # 3. 该 tool_call_id 已经有 tool result 了，重复。
                if not tid_str or tid_str not in declared or tid_str in fulfilled:
                    # 第一次删除时，复制此前所有消息。
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]

                    # continue：不 append 当前 tool 消息，即删除。
                    continue

                # 标记这个 tool_call_id 已被满足。
                fulfilled.add(tid_str)

            # 如果已经进入复制模式，则保留当前消息。
            # 注意这里用 dict(msg) 复制当前消息，避免后续修改影响原对象。
            if updated is not None:
                updated.append(dict(msg))

        # 没有删除任何消息时返回原列表。
        if updated is None:
            return messages

        # 返回清理后的消息副本。
        return updated

    @staticmethod
    def backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert synthetic error results for assistant tool_calls with missing tool outputs.

        中文：为缺少 tool 输出的 assistant.tool_calls 插入合成的错误结果。

        很多模型 API 要求：
        每个 assistant.tool_call 后面必须有一个对应的 role=tool 消息。
        如果缺失，需要补一个占位错误结果。
        """

        # declared：记录所有 assistant 声明过的 tool_call。
        # 每个元素是 (assistant消息索引, tool_call_id, tool_name)。
        declared: list[tuple[int, str, str]] = []

        # fulfilled：已经有 tool result 的 tool_call_id 集合。
        fulfilled: set[str] = set()

        # 遍历消息，收集声明与满足情况。
        for idx, msg in enumerate(messages):
            role = msg.get("role")

            # assistant 消息：收集 tool_calls。
            if role == "assistant":
                for tc in cast(list[Any], msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        # 默认工具名为空字符串。
                        name = ""

                        tool_call = cast(dict[str, Any], tc)

                        # 只处理有 id 的 tool call。
                        if tool_call.get("id"):
                            # OpenAI 风格工具名一般在 function.name。
                            func = tool_call.get("function")

                            # 如果 function 是 dict，则提取 name。
                            if isinstance(func, dict):
                                func_data = cast(dict[str, Any], func)
                                raw_name = func_data.get("name", "")

                                # 如果 name 不是字符串，也强转成字符串，
                                # 尽量保证 tool result 的 name 字段合法。
                                name = raw_name if isinstance(raw_name, str) else str(raw_name)

                            # 记录声明。
                            declared.append((idx, str(tool_call["id"]), name))

            # tool 消息：记录已经满足的 tool_call_id。
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        # 列表推导式：找出所有声明了但没有对应 tool result 的 call。
        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]

        # 如果没有缺失，直接返回原消息列表。
        if not missing:
            return messages

        # 创建消息列表副本。
        # 注意：list(messages) 只是浅拷贝列表本身，不拷贝每条 dict。
        # 但本函数只 insert 新消息，不修改旧 dict，所以安全。
        updated = list(messages)

        # offset：因为每插入一条补位 tool result，后续插入位置要向后偏移。
        offset = 0

        # 遍历每个缺失的 tool call。
        for assistant_idx, call_id, name in missing:
            # 初步插入位置是 assistant 消息后一条。
            insert_at = assistant_idx + 1 + offset

            # 如果 assistant 后面已经跟着若干 tool result，
            # 则把补位消息插到这些 tool result 之后，
            # 保持 assistant -> tool(s) 的连续结构。
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1

            # 插入合成 tool result。
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": BACKFILL_CONTENT,
            })

            # 每插入一条，后续插入点加 1。
            offset += 1

        # 返回补齐后的消息列表。
        return updated

    def apply_tool_result_budget(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        中文：对所有 tool result 应用内容预算/规范化。

        主要是把过大的 tool result：
        - 持久化到外部；
        - 替换成占位说明；
        - 或截断到最大字符数。
        """

        # 初始假设不修改原列表。
        updated = messages

        # 遍历每条消息。
        for idx, message in enumerate(messages):
            # 只处理 role=tool 的工具结果消息。
            if message.get("role") != "tool":
                continue

            # 规范化当前 tool result。
            normalized = self.normalize_tool_result(
                config,
                # 如果没有 tool_call_id，用 tool_索引 兜底。
                str(message.get("tool_call_id") or f"tool_{idx}"),
                # 如果没有工具名，用 "tool" 兜底。
                str(message.get("name") or "tool"),
                # 原始工具结果内容。
                message.get("content"),
            )

            # 只有内容发生变化才复制并修改。
            if normalized != message.get("content"):
                # updated is messages：
                # 用 is 判断是否是同一个对象，而不是仅判断值相等。
                # 第一次需要修改时，复制整个消息列表及每条 dict。
                if updated is messages:
                    updated = [dict(m) for m in messages]

                # 修改副本中的 tool content。
                updated[idx]["content"] = normalized

        # 返回原列表或副本。
        return updated

    def compact_inflight_overflow(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Compact in-flight tool results only when the request would overflow.

        中文：只有当请求会超出上下文预算时，才压缩当前 in-flight 的工具结果。
        """

        # 计算输入预算。
        budget = self.input_budget(config)

        # 如果没有预算，就不做 in-flight compact。
        if budget <= 0:
            return messages

        # 获取工具定义，用于 token 估算。
        # 很多模型不仅 messages 占 token，tools schema 也占 token。
        tools = config.tools.get_definitions()

        # 先把“此前已经记录为压缩过”的 tool result 应用压缩占位文本。
        updated = self._apply_recorded_compactions(messages, compacted_tool_call_ids)

        # 估算当前 prompt 的 token 数。
        # estimate_prompt_tokens_chain 可能根据 provider 选择不同估算方式。
        # 返回值：
        # - estimate：估算 token 数；
        # - source：估算来源/方法说明，用于日志。
        estimate, source = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            updated,
            tools,
        )

        # 如果没有超预算，直接返回。
        if estimate <= budget:
            return updated

        # 压缩目标不是刚好压到 budget，而是压到 budget 的 85%，留余量。
        target = int(budget * INFLIGHT_COMPACT_TARGET_RATIO)

        # 找出可以压缩的 tool result 候选。
        candidates = self._inflight_compaction_candidates(
            config,
            updated,
            compacted_tool_call_ids,
        )

        # 没有候选则不压缩。
        if not candidates:
            return updated

        # enumerate 遍历候选。
        # candidate_idx 用于判断是否是最后一个候选。
        for candidate_idx, (idx, tool_call_id) in enumerate(candidates):
            # 是否是最新/最后一个候选。
            is_newest_candidate = candidate_idx == len(candidates) - 1

            # 如果已经是最后一个候选，并且当前 estimate 已经小于等于 budget，
            # 就不必再压最后一个。这样可以尽量保留最新工具结果。
            if is_newest_candidate and estimate <= budget:
                break

            # 如果这个 tool_call_id 已经压缩过，跳过。
            if tool_call_id in compacted_tool_call_ids:
                continue

            # 第一次需要实际修改时，复制消息列表和消息 dict。
            if updated is messages:
                updated = [dict(m) for m in messages]

            # 记录该 tool_call_id 已压缩。
            # 注意：这里会原地修改传入的 set。
            compacted_tool_call_ids.add(tool_call_id)

            # 把该位置 tool result 内容替换为压缩提示。
            self._compact_tool_result_at(updated, idx)

            # 重新估算 token。
            estimate, source = estimate_prompt_tokens_chain(
                config.provider,
                config.model,
                updated,
                tools,
            )

            # 如果已经达到目标预算，停止继续压缩。
            if estimate <= target:
                break

        # 记录 debug 日志。
        logger.debug(
            "In-flight context compaction for {}: prompt={} budget={} target={} via {}, ids={}",
            config.session_key or "default",
            estimate,
            budget,
            target,
            source,
            len(compacted_tool_call_ids),
        )

        # 返回处理后的消息。
        return updated

    def snip_history(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        中文：当上下文仍然超预算时，从历史头部开始裁剪旧消息。

        基本策略：
        1. 保留所有 system 消息；
        2. 从最新的非 system 消息开始向前保留；
        3. 保留到预算耗尽为止；
        4. 保证返回的历史尾部是“合法”的，例如尽量从 user 消息开始。
        """

        # 如果没有消息，或者没有上下文窗口配置，直接返回。
        if not messages or not config.context_window_tokens:
            return messages

        # 计算输入预算。
        budget = self.input_budget(config)

        # 没有预算则返回。
        if budget <= 0:
            return messages

        # 获取工具定义，用于估算 token。
        tools = config.tools.get_definitions()

        # 估算当前完整消息列表的 token。
        estimate, _ = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            messages,
            tools,
        )

        # 如果没有超预算，不需要裁剪。
        if estimate <= budget:
            return messages

        # 分离 system 消息和非 system 消息。
        # dict(msg)：复制每条消息，避免后续修改影响原消息。
        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]

        # 如果没有非 system 消息，无法裁剪，直接返回原消息。
        if not non_system:
            return messages

        # 估算所有 system 消息本身的 token。
        # sum(...)：对生成器表达式求和。
        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)

        # 估算“只包含 system 消息 + tools 定义”时的固定 token 开销。
        # 这里取 max(system_tokens, fixed_tokens)，
        # 因为 tools schema 等固定开销可能比 system 文本本身更大。
        fixed_tokens, _ = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            system_messages,
            tools,
        )

        # 剩余可用于非 system 历史消息的 token 预算。
        # max(0, ...) 保证不会是负数。
        remaining_budget = max(0, budget - max(system_tokens, fixed_tokens))

        # kept：准备保留的非 system 消息。
        kept: list[dict[str, Any]] = []

        # kept_tokens：当前已保留消息的估算 token 总数。
        kept_tokens = 0

        # reversed(non_system)：从最新到最旧遍历。
        # 这样可以优先保留最近上下文。
        for message in reversed(non_system):
            # 估算当前消息 token。
            msg_tokens = estimate_message_tokens(message)

            # 如果已经有保留消息，并且加入当前消息会超预算，就停止。
            # 注意：如果 kept 为空，即使第一条超预算，也会先保留它。
            # 这是为了避免把全部非 system 历史都裁掉。
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break

            # 保留当前消息。
            kept.append(message)

            # 累加 token。
            kept_tokens += msg_tokens

        # 因为刚才从新到旧 append，所以需要 reverse 回时间顺序。
        kept.reverse()

        # 最终返回：
        # system_messages + 合法处理后的历史尾部。
        return system_messages + self._legal_history_tail(kept, non_system)

    @staticmethod
    def _tool_result_compaction_message(message: dict[str, Any]) -> str:
        """
        中文：生成工具结果被压缩后的提示文本。

        该提示告诉模型：
        - 之前的某个工具结果因为太大被压缩；
        - 不要原样重复调用同一个工具；
        - 应该缩小路径/查询范围、换工具，或告知用户无法完成。
        """

        # 获取工具名，默认 "tool"。
        name = message.get("name", "tool")

        # 返回压缩提示。
        return (
            f"Error: The previous {name} result was compacted to fit context because it was too "
            "large. Do not repeat the same call unchanged. Retry with a narrower path, query, "
            "range, or result limit, use another tool, or tell the user the task cannot fit in "
            "the available context."
        )

    def _legal_history_tail(
        self,
        kept: list[dict[str, Any]],
        non_system: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        中文：对裁剪后保留的历史尾部做“合法化”。

        目标是确保发给模型的消息序列符合 provider 要求，
        例如不能从 tool result 开始，最好从 user 消息开始。
        """

        # fallback：兜底历史。
        # 如果 kept 非空就用 kept；
        # 否则如果 non_system 非空，就用最后一条消息；
        # 否则空列表。
        fallback = kept if kept else (non_system[-1:] if non_system else [])

        # Python 的 or 会返回第一个“真值”操作数。
        # 空列表 [] 是假值，非空列表是真值。
        # 优先使用 kept 中从某个 user 消息开始的尾部；
        # 如果没有，就使用完整 non_system 中最后一个 user 消息开始的尾部；
        # 再不行用 fallback。
        kept = self._user_tail(kept) or self._user_tail(non_system, last=True) or fallback

        # 找到合法消息起始索引。
        start = find_legal_message_start(kept)

        # 如果 start 是非 0 索引，则切掉前面非法部分；
        # 如果 start 为 0 或 None/假值，则返回完整 kept。
        return kept[start:] if start else kept

    @staticmethod
    def _user_tail(messages: list[dict[str, Any]], *, last: bool = False) -> list[dict[str, Any]]:
        """
        中文：从消息列表中找到一段以 user 消息开头的尾部。

        参数：
        - messages：消息列表；
        - last：
          - False：从前往后找第一个 user；
          - True：从后往前找最后一个 user。

        返回：
        - 从该 user 消息开始到末尾的列表；
        - 如果找不到 user，返回空列表。
        """

        # 函数定义中的 * 表示后面的 last 只能作为关键字参数传入。
        # 也就是说必须写 _user_tail(msgs, last=True)，不能位置传参。

        # 如果 last=True，从后往前遍历；否则从前往后。
        # range(len(messages) - 1, -1, -1)：
        # 从最后一个索引到 0，步长 -1。
        indexes = range(len(messages) - 1, -1, -1) if last else range(len(messages))

        # 遍历索引。
        for idx in indexes:
            # 找到 role=user 的消息。
            if messages[idx].get("role") == "user":
                # 返回从该 user 消息开始的切片。
                return messages[idx:]

        # 找不到返回空列表。
        return []

    def _apply_recorded_compactions(
        self,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        """
        中文：把已经记录为压缩过的 tool result 替换为压缩提示文本。

        compacted_tool_call_ids 可能来自此前轮次，
        这里确保这些 tool result 在模型可见消息中已经是压缩态。
        """

        # 如果没有已压缩 id，直接返回原列表。
        if not compacted_tool_call_ids:
            return messages

        # updated 初始指向原列表。
        updated = messages

        # 遍历所有消息。
        for idx, msg in enumerate(messages):
            # 只处理 tool result。
            if msg.get("role") != "tool":
                continue

            # 获取 tool_call_id。
            tool_call_id = msg.get("tool_call_id")

            # 如果没有 id，或者 id 不在已压缩集合中，跳过。
            if not tool_call_id or str(tool_call_id) not in compacted_tool_call_ids:
                continue

            # 生成该 tool result 的压缩提示。
            compaction_message = self._tool_result_compaction_message(msg)

            # 如果当前 content 已经是压缩提示，不需要修改。
            if msg.get("content") == compaction_message:
                continue

            # 第一次修改时复制整个列表及每条 dict。
            if updated is messages:
                updated = [dict(m) for m in messages]

            # 替换副本中的 content。
            updated[idx]["content"] = compaction_message

        # 返回原列表或副本。
        return updated

    def _inflight_compaction_candidates(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[tuple[int, str]]:
        """
        中文：找出可以参与 in-flight compact 的 tool result 候选。

        返回：
        - [(消息索引, tool_call_id), ...]
        """

        # compactable：候选列表。
        compactable: list[tuple[int, str]] = []

        # 遍历消息。
        for idx, msg in enumerate(messages):
            # 不处理 inflight_start_index 之前的消息。
            # 这些通常是固定前缀/早期上下文，不参与动态压缩。
            if idx < config.inflight_start_index:
                continue

            # 只处理 role=tool，并且工具名在白名单里的消息。
            if msg.get("role") != "tool" or msg.get("name") not in COMPACTABLE_TOOLS:
                continue

            # 获取 tool_call_id。
            tool_call_id = msg.get("tool_call_id")

            # 如果没有 id，或者已经压缩过，跳过。
            if not tool_call_id or str(tool_call_id) in compacted_tool_call_ids:
                continue

            # 获取内容。
            content = msg.get("content")

            # 只压缩字符串内容，并且长度达到最小阈值。
            # 太小的结果压缩收益低。
            if not isinstance(content, str) or len(content) < MICROCOMPACT_MIN_CHARS:
                continue

            # 加入候选。
            compactable.append((idx, str(tool_call_id)))

        # 返回候选列表。
        return compactable

    def _compact_tool_result_at(self, messages: list[dict[str, Any]], idx: int) -> None:
        """
        中文：把指定索引位置的 tool result 内容替换为压缩提示。

        注意：调用方必须确保 messages 已经是副本，
        因为这里会直接原地修改 messages[idx]["content"]。
        """

        # 直接替换 content。
        messages[idx]["content"] = self._tool_result_compaction_message(messages[idx])