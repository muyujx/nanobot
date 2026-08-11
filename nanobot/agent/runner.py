"""Shared execution loop for tool-using agents."""
# 模块文档字符串：说明本模块是用于使用工具的 AI Agent 的共享执行循环。

from __future__ import annotations
# 导入 annotations 以启用 PEP 563 (延迟计算注解)。
# 这允许在类型提示中使用尚未定义的类名（前向引用），并减少导入时的内存和时间开销。

import asyncio
# 导入 asyncio 模块，用于编写并发代码（异步 I/O、事件循环、协程等）。

import inspect
# 导入 inspect 模块，用于获取实时对象的信息（如检查函数签名、参数等，常用于动态回调处理）。

import os
# 导入 os 模块，提供与操作系统交互的接口（如读取环境变量 NANOBOT_LLM_TIMEOUT_S）。

from collections.abc import Awaitable, Callable, Iterable
# 从 collections.abc 导入抽象基类，用于类型提示。
# 相比 typing 模块中的对应类，Python 3.9+ 推荐使用 collections.abc 中的类，性能更好且更现代。
# Awaitable: 表示可 await 的对象（如协程）。
# Callable: 表示可调用对象（如函数、方法）。
# Iterable: 表示可迭代对象（如列表、元组）。

from copy import deepcopy
# 导入 deepcopy，用于创建对象的深拷贝，确保修改副本时不会影响原始对象（常用于隔离状态）。

from dataclasses import dataclass, field
# 导入 dataclass 装饰器（用于自动生成 __init__ 等方法）和 field（用于配置数据类字段）。

from pathlib import Path
# 导入 Path，提供面向对象的文件系统路径操作，比传统的 os.path 更现代、更易用。

from typing import Any, cast
# 导入 Any（表示任意类型）和 cast（用于静态类型检查器的类型转换提示，运行时无实际作用）。

from loguru import logger
# 导入 loguru 的 logger 实例，用于在代码中输出结构化、带颜色的日志信息。

from nanobot.agent.context_governance import (
    ContextGovernanceConfig, # 上下文治理配置数据类
    ContextGovernor, # 上下文治理器，负责裁剪、压缩或修复对话历史以适配 LLM 上下文窗口
)
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
# 导入 Agent 生命周期钩子相关的类，允许外部在 Agent 运行的各个阶段（如执行工具前/后）注入自定义逻辑。

from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result
# 导入工具注册表（管理可用工具）和判断工具执行结果是否为错误的辅助函数。

from nanobot.providers.base import (
    LLMProvider, # LLM 提供者基类，定义了与各种大模型 API 交互的标准接口
    LLMResponse, # LLM 响应数据类，封装模型返回的内容、工具调用、Token 使用量等
    ProviderCallContext, # 提供者调用上下文，用于传递特定于提供者的状态或配置
    ProviderConversationState, # 提供者对话状态，某些模型（如 Claude）需要维护特定的对话状态标识
    ToolCallRequest, # 工具调用请求数据类，封装模型请求调用的工具名称和参数
)
from nanobot.providers.conversation_state import (
    ProviderConversationStateController, # 对话状态控制器，管理多轮对话中的状态流转和消息合并
    allows_conversation_message_merge, # 辅助函数，判断当前消息是否允许与下一条同角色消息合并
)
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META, # 运行时上下文消息的元数据键名常量
    detach_runtime_context, # 从消息内容中分离出运行时注入的上下文块（如系统提示、隐藏指令）
    reattach_runtime_context, # 将分离出的上下文块重新附加到合并后的消息内容中
)
from nanobot.session.history_visibility import is_hidden_history_message
# 导入辅助函数，用于判断某条历史消息是否被标记为“隐藏”（不对用户显示，但会发给模型）。

from nanobot.utils.helpers import (
    IncrementalThinkExtractor, # 增量思考提取器，用于在流式输出中逐步提取模型的“思考过程”
    build_assistant_message, # 构建标准格式的 Assistant 消息字典
    estimate_message_tokens, # 估算单条消息的 Token 数量
    estimate_prompt_tokens_chain, # 估算整个消息链（Prompt）的 Token 数量
    extract_reasoning, # 从响应中提取推理/思考内容（针对支持思考链的模型如 o1）
    strip_reasoning_tags, # 剥离推理内容中的特定 XML/Markdown 标签
    strip_think, # 剥离 <think> 等思考标签
)
from nanobot.utils.llm_runtime import LLMRuntime
# 导入 LLM 运行时配置类，封装了模型名称、生成参数（温度、最大 Token 等）和上下文窗口大小。

from nanobot.utils.prompt_templates import render_template
# 导入模板渲染函数，用于将 Jinja2 或其他模板引擎的模板字符串渲染为最终的 Prompt。

from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE, # 当模型最终返回空白内容时使用的默认提示消息
    build_budget_exhausted_finalization_message, # 构建当 Token 预算耗尽时的最终总结提示消息
    build_finalization_retry_message, # 构建强制要求模型给出最终答案的重试提示消息
    build_goal_continue_message, # 构建鼓励模型继续完成目标的提示消息
    build_length_recovery_message, # 构建当输出被长度截断时，要求模型继续生成的恢复消息
    is_blank_text, # 判断文本是否为空白（仅包含空格、换行等）
    repeated_external_lookup_error, # 检查是否重复调用了相同的外部查询工具（防止死循环）
    repeated_workspace_violation_error, # 检查是否重复尝试访问工作区之外的违规路径（防止死循环）
)

# ==================== 类型别名与常量定义 ====================

GoalContinueMessage = str | Callable[[], str | None]
# 定义类型别名 GoalContinueMessage。
# 使用 Python 3.10+ 的联合类型语法 `|`。
# 它可以是一个普通的字符串，或者是一个不接受参数并返回字符串或 None 的可调用对象（如 lambda 函数）。

ProgressCallback = Callable[[str], Awaitable[None]]
# 进度回调类型：接受一个字符串参数（增量内容），返回一个 Awaitable（异步协程），无返回值。

RetryWaitCallback = Callable[[str], Awaitable[None]]
# 重试等待回调类型：当触发重试时调用，接受重试原因字符串，返回异步协程。

CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]
# 检查点回调类型：接受一个包含当前状态字典的参数，用于持久化或记录 Agent 的中间状态。

InjectionCallback = Callable[..., Awaitable[Iterable[Any] | None]]
# 注入回调类型：接受任意参数（...），返回一个异步协程，该协程最终返回一个可迭代对象（包含要注入的消息）或 None。

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
# 默认错误消息：当调用 AI 模型发生未知错误时，向用户展示的兜底提示。

_ARREARAGE_ERROR_MESSAGE = (
    "The AI provider rejected the request because the API key is out of quota or the "
    "account is in arrears. Please top up / check the billing status of your API key and try again."
)
# 欠费/配额耗尽错误消息：专门针对 API 密钥余额不足或配额用完的特定错误提示。

_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
# 模型错误占位符：当模型彻底崩溃无法生成内容时，持久化到历史记录中的占位文本。

_MAX_EMPTY_RETRIES = 2
# 最大空响应重试次数：如果模型连续返回空白内容，最多重试 2 次。

_MAX_LENGTH_RECOVERIES = 3
# 最大长度恢复次数：如果模型输出因达到 max_tokens 被截断，最多尝试自动拼接恢复 3 次。

_MAX_INJECTIONS_PER_TURN = 3
# 每轮最大注入消息数：限制单次回调最多注入 3 条用户消息，防止恶意或错误的回调导致上下文爆炸。

_MAX_INJECTION_CYCLES = 5
# 最大注入循环数：限制在一个完整的 Agent 运行周期内，最多触发 5 次外部消息注入，防止无限循环。


def _restore_outer_whitespace(content: str, original: str | None) -> str:
    """Restore boundary whitespace stripped while cleaning one recovered segment."""
    # 辅助函数：恢复在清理恢复片段时被剥离的边界空白字符。
    # 参数:
    #   content: 清理后的核心内容。
    #   original: 清理前的原始内容。
    if not original:
        # 如果没有提供原始字符串，直接返回处理后的内容。
        return content
    leading_size = len(original) - len(original.lstrip())
    # 计算原始字符串前导空白字符的数量。
    # lstrip() 会移除左侧所有空白字符，两者长度之差即为前导空白数。
    trailing_size = len(original) - len(original.rstrip())
    # 计算原始字符串尾部空白字符的数量。
    # rstrip() 会移除右侧所有空白字符，两者长度之差即为尾部空白数。
    leading = original[:leading_size]
    # 截取原始字符串的前导空白部分。
    trailing = original[-trailing_size:] if trailing_size else ""
    # 截取原始字符串的尾部空白部分。如果尾部没有空白，则返回空字符串。
    return f"{leading}{content}{trailing}"
    # 将前导空白、清理后的内容、尾部空白重新拼接并返回，确保格式（如缩进、换行）不丢失。


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""
    # 使用 dataclass 装饰器自动生成 __init__, __repr__, __eq__ 等方法。
    # slots=True (Python 3.10+) 会为类创建 __slots__，阻止创建 __dict__ 和 __weakref__，
    # 从而显著减少内存占用并加快属性访问速度，非常适合频繁实例化的配置类。

    initial_messages: list[dict[str, Any]]
    # 初始消息列表：Agent 开始运行时的对话历史（包含系统提示和用户首轮输入）。

    tools: ToolRegistry
    # 工具注册表实例：包含当前 Agent 可以调用的所有工具的定义和执行逻辑。

    runtime: LLMRuntime
    # LLM 运行时配置：包含模型名称、温度、Top-P 等生成参数。

    max_iterations: int
    # 最大迭代次数：Agent 执行循环（思考->调用工具->观察）的最大允许次数，防止死循环。

    max_tool_result_chars: int
    # 工具结果最大字符数：限制工具返回结果的长度，超长部分会被截断，防止撑爆上下文窗口。

    hook: AgentHook | None = None
    # 生命周期钩子实例：允许外部监听和干预 Agent 的运行过程。默认为 None。

    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    # 自定义错误消息：当发生一般性错误时展示给用户的文本。

    max_iterations_message: str | None = None
    # 达到最大迭代次数时的自定义提示消息。

    concurrent_tools: bool = False
    # 是否并发执行工具：如果为 True，且工具标记为线程/并发安全，则使用 asyncio.gather 并发调用多个工具。

    fail_on_tool_error: bool = False
    # 工具出错时是否直接失败：如果为 True，任何工具执行抛出异常都会导致整个 Agent 运行中断。

    workspace: Path | None = None
    # 工作区路径：限制文件操作工具只能在指定的目录范围内执行，提供安全沙箱。

    session_key: str | None = None
    # 会话唯一标识符：用于日志追踪和状态隔离。

    context_block_limit: int | None = None
    # 上下文块限制：针对某些特定模型（如 Claude 的 cache control）的上下文块数量限制。

    provider_retry_mode: str = "standard"
    # 提供者重试模式：控制底层 LLM API 在遇到网络错误或限流时的重试策略。

    progress_callback: ProgressCallback | None = None
    # 进度回调函数：用于在非流式模式下，将模型的增量生成内容实时推送给调用方。

    stream_progress_deltas: bool = True
    # 是否流式传输进度增量：控制是否将细粒度的文本块通过 progress_callback 发送。

    retry_wait_callback: RetryWaitCallback | None = None
    # 重试等待回调：在触发重试前调用，可用于通知 UI 显示“正在重试...”或执行退避等待。

    checkpoint_callback: CheckpointCallback | None = None
    # 检查点回调：在关键状态变更（如工具执行完毕）时触发，用于保存断点或更新前端状态。

    injection_callback: InjectionCallback | None = None
    # 注入回调：允许外部在 Agent 运行中途动态注入新的用户消息（如人类干预）。

    llm_timeout_s: float | None = None
    # LLM 请求超时时间（秒）：单次 LLM API 调用的最大等待时间。

    goal_active_predicate: Callable[[], bool] | None = None
    # 目标激活谓词：一个返回布尔值的函数，用于判断当前 Agent 是否仍有未完成的目标（用于决定是否自动继续）。

    goal_continue_message: GoalContinueMessage | None = None
    # 目标继续消息：当目标仍激活但模型停止生成时，自动注入的提示消息。

    finalize_on_max_iterations: bool = True
    # 达到最大迭代时是否强制总结：如果为 True，会在循环结束时调用模型生成最终总结，而不是直接截断。

    provider_state: ProviderConversationState | None = None
    # 提供者状态：用于恢复某些需要维护服务端状态的模型的对话上下文。


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""
    # Agent 运行结果数据类，封装单次执行的所有最终产出和统计信息。

    final_content: str | None
    # 最终内容：Agent 运行结束后，呈现给用户的最终文本回复。

    messages: list[dict[str, Any]]
    # 完整的消息历史：包含所有交互记录（用户、助手、工具调用、工具结果），用于持久化。

    tools_used: list[str] = field(default_factory=list)
    # 使用的工具列表：记录本次运行中成功调用过的工具名称。
    # 使用 field(default_factory=list) 是为了避免所有实例共享同一个列表对象（Python 中可变默认参数的常见陷阱）。

    usage: dict[str, int] = field(default_factory=dict)
    # Token 使用量统计：包含 prompt_tokens, completion_tokens, total_tokens 等键值对。

    stop_reason: str = "completed"
    # 停止原因：说明 Agent 为何结束运行（如 "completed", "max_iterations", "tool_error", "cancelled"）。

    error: str | None = None
    # 错误信息：如果运行异常终止，这里会包含具体的错误描述。

    tool_events: list[dict[str, str]] = field(default_factory=list)
    # 工具事件日志：记录每次工具调用的状态、耗时或错误详情，用于审计或调试。

    had_injections: bool = False
    # 是否发生过外部注入：标记在运行过程中是否通过 injection_callback 动态插入了新消息。

    # Terminal tail to emit when the preceding final-content prefix was already streamed.
    # 待处理的流式内容：当最终内容的前缀已经被流式传输时，这里保存需要追加的尾部内容（常用于长度恢复场景）。
    pending_stream_content: str | None = None

    provider_state: ProviderConversationState | None = field(default=None, repr=False)
    # 提供者状态快照：运行结束时的底层模型状态，用于下一次对话的无缝衔接。
    # repr=False 表示在打印对象时忽略此字段，因为状态对象可能很大且包含敏感或冗余信息。


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""
    # AgentRunner 类：负责运行具备工具调用能力的 LLM 核心循环。
    # 设计原则：只关注 LLM 交互、工具执行和上下文管理，不涉及具体的业务逻辑或 UI 渲染（产品层关注点）。

    def __init__(self) -> None:
        # 初始化方法。
        self.context_governor = ContextGovernor()
        # 实例化 ContextGovernor，用于管理、压缩或修复对话上下文，
        # 确保发送给 LLM 的消息符合上下文窗口限制和特定模型的格式要求。

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        # 静态方法：合并左右两部分消息内容。
        # 返回类型可以是纯字符串，或者是包含多个内容块（如文本、图像）的字典列表（多模态格式）。
        if isinstance(left, str) and isinstance(right, str):
            # 如果左右都是纯字符串，则用两个换行符连接它们；如果左侧为空，则直接返回右侧。
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            # 内部辅助函数：将任意类型的值转换为标准的内容块列表格式。
            if isinstance(value, list):
                # 如果已经是列表，遍历其中的元素。
                return [
                    cast(dict[str, Any], item)
                    # cast 仅用于告诉 mypy 等静态检查器“我确信 item 是字典”，运行时无操作。
                    if isinstance(item, dict)
                    # 如果元素已经是字典（如 {"type": "image", ...}），直接保留。
                    else {"type": "text", "text": str(item)}
                    # 否则，将其转换为标准的文本块字典。
                    for item in cast(list[Any], value)
                ]
            if value is None:
                # 如果值为 None，返回空列表。
                return []
            # 对于其他单一值（如纯字符串），包装成单个文本块字典。
            return [{"type": "text", "text": str(value)}]

        # 将左侧和右侧的内容都转换为块列表，然后相加合并。
        return _to_blocks(left) + _to_blocks(right)

    @classmethod
    def _append_injected_messages(
        cls,
        messages: list[dict[str, Any]],
        injections: list[dict[str, Any]],
    ) -> None:
        """Append injected user messages while preserving role alternation."""
        # 类方法：将外部注入的用户消息追加到消息列表中，同时尽量保持角色交替（User/Assistant）的规则。
        for injection in injections:
            # 遍历每一个待注入的消息。
            if (
                messages
                # 1. 消息列表不能为空。
                and injection.get("role") == "user"
                # 2. 注入的消息必须是 user 角色。
                and messages[-1].get("role") == "user"
                # 3. 历史消息的最后一条也必须是 user 角色（连续两个 user 消息通常需要合并）。
                and not is_hidden_history_message(injection)
                # 4. 注入的消息不能是隐藏的历史消息。
                and not is_hidden_history_message(messages[-1])
                # 5. 历史最后一条消息也不能是隐藏的。
                and allows_conversation_message_merge(messages[-1])
                # 6. 根据特定模型的规则，允许这两条 user 消息进行内容合并。
            ):
                # 如果满足以上所有条件，则执行深度合并逻辑，而不是简单追加。
                merged = dict(messages[-1])
                # 浅拷贝最后一条历史消息，作为合并的基础。

                left_meta = merged.get("_meta")
                right_meta = injection.get("_meta")
                # 获取两条消息的元数据字典（_meta 通常包含内部状态、路由信息等）。

                left_meta_dict = cast(dict[str, Any], left_meta) if isinstance(left_meta, dict) else None
                right_meta_dict = (
                    cast(dict[str, Any], right_meta) if isinstance(right_meta, dict) else None
                )
                # 安全地将元数据转换为字典类型，如果不是字典则置为 None。

                left_marker = (
                    left_meta_dict.get(RUNTIME_CONTEXT_MESSAGE_META)
                    if left_meta_dict is not None
                    else None
                )
                right_marker = (
                    right_meta_dict.get(RUNTIME_CONTEXT_MESSAGE_META)
                    if right_meta_dict is not None
                    else None
                )
                # 提取运行时上下文标记（Marker），用于定位消息中动态注入的上下文块。

                left_marker_dict = (
                    cast(dict[str, Any], left_marker) if isinstance(left_marker, dict) else None
                )
                right_marker_dict = (
                    cast(dict[str, Any], right_marker) if isinstance(right_marker, dict) else None
                )
                # 确保 Marker 也是字典类型。

                empty_sources: list[str] = []
                empty_blocks: list[dict[str, Any]] = []
                # 初始化空的来源和块列表，用于处理没有 Marker 的情况。

                detached_left = (
                    detach_runtime_context(merged.get("content"), left_marker_dict)
                    if left_marker_dict is not None
                    else (merged.get("content"), empty_sources, empty_blocks)
                )
                # 从左侧消息内容中分离出纯文本内容和运行时上下文块。
                # 如果没有 Marker，则认为没有分离出任何特殊块，直接使用原内容。

                detached_right = (
                    detach_runtime_context(injection.get("content"), right_marker_dict)
                    if right_marker_dict is not None
                    else (injection.get("content"), empty_sources, empty_blocks)
                )
                # 同理，从右侧（注入）消息内容中分离出纯文本和上下文块。

                if detached_left is not None and detached_right is not None:
                    # 如果两侧的分离操作都成功（未返回 None）。
                    left_content, left_sources, left_blocks = detached_left
                    right_content, right_sources, right_blocks = detached_right
                    # 解包分离出的纯内容、来源标识和上下文块。

                    merged_content = cls._merge_message_content(left_content, right_content)
                    # 调用前面定义的合并方法，将两侧的纯文本内容合并。

                    context_blocks = [*left_blocks, *right_blocks]
                    # 使用解包操作符 `*` 将两侧的上下文块列表拼接成一个新列表。

                    if context_blocks:
                        # 如果合并后存在需要重新注入的上下文块。
                        merged_content, marker = reattach_runtime_context(
                            merged_content,
                            [*left_sources, *right_sources],
                            context_blocks,
                        )
                        # 将上下文块重新附加到合并后的文本中，并生成新的 Marker。

                        internal_meta = dict(left_meta_dict) if left_meta_dict is not None else {}
                        # 创建一个新的元数据字典，初始复制左侧的元数据。
                        if right_meta_dict is not None:
                            for key, value in right_meta_dict.items():
                                internal_meta.setdefault(key, value)
                            # 遍历右侧元数据，如果键不存在则添加（左侧优先级更高）。

                        internal_meta[RUNTIME_CONTEXT_MESSAGE_META] = marker
                        # 将新生成的 Marker 写入内部元数据。
                        merged["_meta"] = internal_meta
                        # 更新合并后消息的 _meta 字段。

                    merged["content"] = merged_content
                    # 将合并并处理好的内容写回消息字典。
                else:
                    # 如果分离操作失败（例如格式不支持），则退化为简单的文本合并。
                    merged["content"] = cls._merge_message_content(
                        merged.get("content"),
                        injection.get("content"),
                    )
                messages[-1] = merged
                # 用合并后的新消息替换历史列表中的最后一条消息。
                continue
                # 跳过后续的 append 操作，进入下一个注入消息的处理。

            # 如果不满足合并条件，则直接将注入的消息追加到历史列表末尾。
            messages.append(injection)

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        conversation_state: ProviderConversationStateController | None = None,
        phase: str = "after error",
        iteration: int | None = None,
        allow_goal_continue: bool = False,
    ) -> tuple[bool, int]:
        """Drain pending injections. Returns (should_continue, updated_cycles).

        If injections are found and we haven't exceeded _MAX_INJECTION_CYCLES,
        append them to *messages* (and emit a checkpoint if *assistant_message*
        and *iteration* are both provided) and return (True, cycles+1) so the
        caller continues the iteration loop.  Otherwise return (False, cycles).
        """
        # 函数文档字符串（翻译）：排空待处理的注入消息。返回元组 (是否应该继续循环, 更新后的循环计数)。
        # 如果找到注入消息且未超过 _MAX_INJECTION_CYCLES 限制，将它们追加到 *messages* 中
        # （如果同时提供了 *assistant_message* 和 *iteration*，则发出一个检查点），
        # 并返回 (True, cycles+1) 以便调用者继续迭代循环。否则返回 (False, cycles)。

        injections: list[dict[str, Any]] = []
        # 初始化一个空列表，用于存储获取到的注入消息。
        real_injection = False
        # 标记变量：用于区分是真实的用户注入，还是系统自动生成的“目标继续”消息。

        if injection_cycles < _MAX_INJECTION_CYCLES:
            # 检查当前的注入循环次数是否小于最大允许次数，防止无限注入死循环。
            injections = await self._drain_injections(spec)
            # 调用内部方法实际获取外部注入的消息列表。
            real_injection = bool(injections)
            # 如果获取到了真实的注入消息，将标记设为 True。

        if not injections and allow_goal_continue and assistant_message is not None:
            # 如果没有真实的注入消息，但允许目标继续（allow_goal_continue），且存在助手消息。
            predicate = spec.goal_active_predicate
            # 获取判断目标是否仍然激活的谓词函数。
            if predicate is not None and predicate():
                # 如果谓词存在且调用后返回 True（表示目标仍未完成）。
                injections = [self._build_goal_continue_message(spec)]
                # 自动构建一条“继续完成目标”的提示消息作为注入内容。

        if not injections:
            # 如果经过上述逻辑后仍然没有任何注入消息，则直接返回不继续，并保持原有循环计数。
            return False, injection_cycles

        if real_injection:
            # 如果是真实的用户注入，将循环计数器加 1。
            injection_cycles += 1

        if assistant_message is not None:
            # 如果提供了助手消息（意味着当前轮次模型已经生成了回复）。
            messages.append(assistant_message)
            # 将助手消息先追加到历史消息列表中。
            if iteration is not None:
                # 如果提供了当前迭代次数，说明需要记录一个完整的检查点。
                checkpoint: dict[str, Any] = {
                    "phase": "final_response", # 阶段标识：最终响应
                    "iteration": iteration,    # 当前迭代轮次
                    "model": spec.runtime.model, # 使用的模型名称
                    "assistant_message": assistant_message, # 助手消息内容
                    "completed_tool_results": [], # 已完成的工具结果（此处为空）
                    "pending_tool_calls": [],   # 待处理的工具调用（此处为空）
                }
                if conversation_state is not None:
                    # 如果存在对话状态控制器。
                    checkpoint["provider_state"] = conversation_state.checkpoint(
                        messages
                    )
                    # 获取当前底层提供商的对话状态快照并加入检查点。
                await self._emit_checkpoint(
                    spec,
                    checkpoint,
                )
                # 触发检查点回调，将状态持久化或通知前端。

        self._append_injected_messages(messages, injections)
        # 调用前面定义的类方法，将注入的消息智能地合并或追加到消息历史中。

        if real_injection:
            # 记录真实注入的日志。
            logger.info(
                "Injected {} follow-up message(s) {} ({}/{})",
                len(injections), phase, injection_cycles, _MAX_INJECTION_CYCLES,
            )
        else:
            # 记录系统自动继续目标的日志。
            logger.info("Injected sustained-goal continuation {}", phase)

        return True, injection_cycles
        # 返回 True 指示主循环应该继续迭代，并返回更新后的注入循环计数。

    def _build_goal_continue_message(self, spec: AgentRunSpec) -> dict[str, str]:
        # 构建“目标继续”提示消息的辅助方法。
        custom = spec.goal_continue_message
        # 获取配置中自定义的继续消息（可能是字符串或可调用对象）。
        if callable(custom):
            # 使用 Python 内置函数 callable() 检查 custom 是否为可调用对象（如函数、lambda）。
            try:
                custom = custom()
                # 如果是，则调用它获取实际的字符串内容。
            except Exception:
                # 捕获回调执行过程中的任何异常，防止导致整个 Agent 崩溃。
                logger.exception("goal_continue_message callback failed")
                custom = None
                # 发生异常时，将 custom 重置为 None，后续会使用默认模板。
        return build_goal_continue_message(custom)
        # 调用工具函数生成标准格式的字典消息，传入 custom（若为 None 则使用默认值）。

    async def _drain_injections(self, spec: AgentRunSpec) -> list[dict[str, Any]]:
        """Drain pending user messages via the injection callback.

        Returns normalized user messages (capped by
        ``_MAX_INJECTIONS_PER_TURN``), or an empty list when there is
        nothing to inject. Messages beyond the cap are logged so they
        are not silently lost.
        """
        # 函数文档字符串（翻译）：通过注入回调排空待处理的用户消息。
        # 返回标准化的用户消息列表（受 ``_MAX_INJECTIONS_PER_TURN`` 限制），
        # 或者在没有什么可注入时返回空列表。超出限制的消息会被记录日志，以免被无声地丢弃。

        if spec.injection_callback is None:
            # 如果没有配置注入回调，直接返回空列表。
            return []

        try:
            signature = inspect.signature(spec.injection_callback)
            # 使用 inspect.signature 获取回调函数的签名对象。
            # 这是 Python 反射机制的核心用法，用于在运行时动态检查函数的参数结构。
            accepts_limit = (
                "limit" in signature.parameters
                # 检查签名参数中是否包含名为 "limit" 的参数。
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    # 或者检查是否有任何参数的类型是 VAR_KEYWORD（即 **kwargs）。
                    for parameter in signature.parameters.values()
                )
            )
            # accepts_limit 为 True 表示该回调函数支持接收 limit 关键字参数。
            # 这种设计允许系统向后兼容旧版本的不接受 limit 参数的回调函数。

            if accepts_limit:
                # 如果支持 limit 参数，则传入最大注入数量限制。
                items = await spec.injection_callback(limit=_MAX_INJECTIONS_PER_TURN)
            else:
                # 否则，不传参直接调用。
                items = await spec.injection_callback()
        except Exception:
            # 捕获调用回调时可能发生的任何异常。
            logger.exception("injection_callback failed")
            return []
            # 记录异常堆栈并返回空列表，保证主流程继续。

        if not items:
            # 如果回调返回了假值（如 None 或空列表），直接返回空列表。
            return []

        injected_messages: list[dict[str, Any]] = []
        # 初始化列表，用于存放标准化后的消息。
        for item in items:
            # 遍历回调返回的每一个元素。
            if item is None:
                # 忽略 None 值。
                continue
            if isinstance(item, dict):
                # 如果元素已经是字典格式。
                message_item = cast(dict[str, Any], item)
                # 使用 cast 告诉类型检查器这是一个字典（运行时无影响）。
                if message_item.get("role") == "user" and "content" in message_item:
                    # 检查它是否是一个合法的 user 角色消息且包含 content 字段。
                    if self._has_injection_content(message_item.get("content")):
                        # 进一步检查 content 是否有实际内容。
                        injected_messages.append(message_item)
                continue
                # 处理完字典后继续下一个循环。

            # 如果不是字典，尝试提取 content 属性（兼容某些自定义的消息对象）。
            content = getattr(item, "content") if hasattr(item, "content") else str(item)
            # getattr 配合 hasattr 是安全获取对象属性的标准做法。如果都没有，则强制转为字符串。
            if self._has_injection_content(content):
                # 如果有内容，将其包装成标准的 user 消息字典格式。
                injected_messages.append({"role": "user", "content": content})

        if len(injected_messages) > _MAX_INJECTIONS_PER_TURN:
            # 如果标准化后的消息数量超过了单轮最大限制。
            dropped = len(injected_messages) - _MAX_INJECTIONS_PER_TURN
            # 计算被丢弃的消息数量。
            logger.warning(
                "Injection callback returned {} messages, capping to {} ({} dropped)",
                len(injected_messages), _MAX_INJECTIONS_PER_TURN, dropped,
            )
            # 记录警告日志。
            injected_messages = injected_messages[:_MAX_INJECTIONS_PER_TURN]
            # 使用列表切片截断列表，只保留前 N 条消息。

        return injected_messages

    @staticmethod
    def _has_injection_content(content: Any) -> bool:
        # 静态辅助方法：判断给定的 content 是否包含有效的实质内容。
        if content is None:
            return False
        if isinstance(content, str):
            # 如果是字符串，去除首尾空白后检查是否为空。
            return bool(content.strip())
        if isinstance(content, list):
            # 如果是列表（多模态内容块），检查列表是否非空。
            return bool(cast(list[Any], content))
        return True
        # 对于其他类型（如字典、对象），默认认为有内容。

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        # Agent 运行的主入口方法。接收运行配置 spec，返回运行结果 AgentRunResult。
        hook = spec.hook or AgentHook()
        # 获取生命周期钩子实例，如果 spec 中未提供，则使用默认的空钩子 AgentHook()。
        messages = list(spec.initial_messages)
        # 浅拷贝初始消息列表，避免修改外部传入的原始数据。
        context = AgentRunHookContext(messages=deepcopy(messages))
        # 创建钩子上下文对象，传入消息的深拷贝，确保钩子内部的修改不会影响主流程。

        try:
            # 开始 try 块，包裹核心执行逻辑。
            await hook.before_run(context)
            # 触发“运行前”钩子。
            result = await self._run_core(spec, hook, messages)
            # 调用核心执行循环，获取最终结果。
        except asyncio.CancelledError as exc:
            # 捕获 asyncio 特有的取消异常。当外部调用 task.cancel() 时会抛出此异常。
            context.messages = deepcopy(messages)
            context.stop_reason = "cancelled"
            context.error = None
            context.exception = exc
            # 更新上下文状态为“已取消”。
            raise
            # 【关键】必须使用裸 raise 重新抛出 CancelledError，否则 asyncio 事件循环会认为任务已正常完成，
            # 导致取消信号无法正确向上传播，引发严重的并发 bug。
        except Exception as exc:
            # 捕获所有其他常规异常。
            context.messages = deepcopy(messages)
            context.stop_reason = "error"
            context.error = f"Error: {type(exc).__name__}: {exc}"
            context.exception = exc
            # 更新上下文状态为“错误”，并记录异常信息。
            await hook.on_error(context)
            # 触发“发生错误”钩子，允许外部进行清理或告警。
            raise
            # 重新抛出异常，让调用方感知到运行失败。
        else:
            # 【Python 特性】else 块仅在 try 块没有抛出任何异常（即正常完成）时执行。
            context.messages = deepcopy(result.messages)
            context.final_content = result.final_content
            context.tools_used = list(result.tools_used)
            context.usage = dict(result.usage)
            context.stop_reason = result.stop_reason
            context.error = result.error
            context.tool_events = deepcopy(result.tool_events)
            context.had_injections = result.had_injections
            context.exception = None
            # 将成功运行的结果数据同步到钩子上下文中。

            if context.error is not None:
                # 如果结果中包含了非致命的错误信息（例如工具执行失败但 Agent 继续运行了）。
                await hook.on_error(context)
                # 依然触发错误钩子。
            await hook.after_run(context)
            # 触发“运行后”钩子。
            return result
            # 返回最终的运行结果。
        finally:
            # 【Python 特性】finally 块无论 try 块是正常结束、抛出异常还是被取消，都一定会执行。
            # 通常用于资源清理。
            context.messages = deepcopy(messages)
            if context.exception is None:
                # 如果没有发生异常，正常触发 finally 钩子。
                await hook.on_finally(context)
            else:
                # 如果发生了异常，尝试触发 finally 钩子，但要用 try-except 包裹。
                try:
                    await hook.on_finally(context)
                except Exception:
                    # 防止清理逻辑本身的异常掩盖了原始的运行异常。
                    logger.exception(
                        "AgentHook.on_finally error after {}",
                        context.stop_reason or "run exception",
                    )

    async def _run_core(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        # 核心执行循环方法。实现了“思考 -> 调用工具 -> 观察结果 -> 继续思考”的 ReAct 范式。
        final_content: str | None = None
        # 最终回复内容。
        tools_used: list[str] = []
        # 记录成功使用的工具名称。
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        # 累计 Token 使用量统计。
        error: str | None = None
        # 错误信息。
        stop_reason = "completed"
        # 停止原因，默认为正常完成。
        tool_events: list[dict[str, str]] = []
        # 工具调用事件日志。
        external_lookup_counts: dict[str, int] = {}
        # 外部查询工具调用计数器（用于防止对同一目标进行死循环查询）。
        # Per-turn throttle for repeated attempts against the same outside target.
        # 注释翻译：每轮节流，防止对同一外部目标进行重复尝试。
        workspace_violation_counts: dict[str, int] = {}
        # 工作区违规计数器（用于防止模型不断尝试读取沙箱外的文件）。
        empty_content_retries = 0
        # 空内容重试计数器。
        # Segments from one uninterrupted length-recovery chain. Tool work or
        # injected user input starts a new logical answer and clears the chain.
        # 注释翻译：来自一个不间断的长度恢复链的片段。工具工作或注入的用户输入会开始一个新的逻辑答案并清除该链。
        length_recovery_parts: list[str] = []
        # 存储因长度截断而分段的回复内容，以便后续拼接。
        had_injections = False
        # 标记是否发生过外部消息注入。
        injection_cycles = 0
        # 注入循环计数器。
        compacted_tool_call_ids: set[str] = set()
        # 已经被压缩/裁剪的工具调用 ID 集合（用于上下文治理）。
        pending_stream_content: str | None = None
        # 待处理的流式内容尾部。

        conversation_state = ProviderConversationStateController(
            provider=spec.runtime.provider,
            model=spec.runtime.model,
            messages=messages,
            state=spec.provider_state,
        )
        # 初始化对话状态控制器，用于管理特定 LLM 提供商（如 Claude）需要的多轮对话状态标识。

        governance_config = ContextGovernanceConfig(
            provider=spec.runtime.provider,
            model=spec.runtime.model,
            tools=spec.tools,
            workspace=spec.workspace,
            session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            context_window_tokens=spec.runtime.context_window_tokens,
            context_block_limit=spec.context_block_limit,
            max_tokens=spec.runtime.generation.max_tokens,
            inflight_start_index=len(spec.initial_messages),
        )
        # 初始化上下文治理配置，定义了如何处理和裁剪历史消息以适配模型的上下文窗口。

        for iteration in range(spec.max_iterations):
            # 开始主迭代循环，最多运行 spec.max_iterations 次。

            # Keep the persisted conversation untouched. Context governance
            # may repair or compact historical messages for the model, but
            # those synthetic edits must not shift the append boundary used
            # later when the caller saves only the new turn. A governance
            # failure must stop the run instead of sending an ungoverned copy.
            # 注释翻译：保持持久化的对话不变。上下文治理可能会为模型修复或压缩历史消息，
            # 但这些合成编辑不能移动追加边界（调用方稍后仅保存新轮次时会使用该边界）。
            # 治理失败必须停止运行，而不是发送未经治理的副本。
            messages_for_model = self.context_governor.prepare_for_model(
                governance_config,
                messages,
                compacted_tool_call_ids,
            )
            # 调用上下文治理器，生成专门发给模型的（可能经过裁剪或格式化的）消息列表。

            context = AgentHookContext(
                iteration=iteration,
                messages=messages,
                session_key=spec.session_key,
            )
            # 为当前迭代创建钩子上下文。
            await hook.before_iteration(context)
            # 触发“迭代前”钩子。

            provider_context = conversation_state.prepare_request(
                messages,
                context_window_tokens=spec.runtime.context_window_tokens,
                model_messages=messages_for_model,
            )
            # 准备特定于提供商的请求上下文（如附加 system prompt 或 cache 控制头）。

            response = await self._request_model(
                spec,
                messages_for_model,
                hook,
                context,
                conversation_state=conversation_state,
                provider_context=provider_context,
            )
            # 核心调用：向 LLM 发送请求并获取响应（包含流式处理、重试、错误恢复等逻辑）。

            conversation_state.observe_response(response, messages)
            # 让状态控制器观察模型的响应，更新内部状态（如提取特定的 state token）。

            context.response = response
            context.tool_calls = list(response.tool_calls)
            # 将响应和提取出的工具调用列表同步到钩子上下文。

            original_content = response.content
            # 保存模型返回的原始文本内容，用于后续可能的空白恢复或长度拼接。

            reasoning_text, cleaned_content = extract_reasoning(
                response.reasoning_content,
                response.thinking_blocks,
                response.content,
            )
            # 从响应中分离出“推理/思考过程”和“最终清洗后的回复内容”（针对 o1 等思维链模型）。

            response.content = cleaned_content
            # 将清洗后的内容覆盖回 response 对象。

            raw_usage = self._usage_or_estimate(spec, messages_for_model, response)
            # 获取本次请求的 Token 使用量。如果 API 没返回，则通过本地估算得出。
            context.usage = dict(raw_usage)
            # 记录到当前迭代的上下文中。
            self._accumulate_usage(usage, raw_usage)
            # 累加到全局的总 usage 统计中。

            if reasoning_text and not context.streamed_reasoning:
                # 如果存在推理文本，且之前没有流式输出过推理过程。
                await hook.emit_reasoning(reasoning_text)
                await hook.emit_reasoning_end()
                context.streamed_reasoning = True
                # 通过钩子一次性发射完整的推理文本，并标记已发射。

            if response.should_execute_tools:
                # 【核心分支】如果模型决定调用工具（should_execute_tools 为 True）。
                context.tool_calls = list(response.tool_calls)
                if hook.wants_streaming():
                    # 如果当前配置了流式输出，在工具执行前结束当前的流式文本块。
                    await hook.on_stream_end(context, resuming=True)

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                # 构建标准的 Assistant 消息字典，包含文本和 OpenAI 格式的 tool_calls。
                # 列表推导式 `[tc.to_openai_tool_call() for tc in ...]` 用于批量转换对象格式。

                assistant_message = conversation_state.project_response_message(
                    assistant_message,
                    response,
                )
                # 让状态控制器对消息进行投影/修改（例如添加特定的 provider 标记）。

                messages.append(assistant_message)
                # 将助手消息追加到历史对话中。

                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools", # 阶段：等待工具执行
                        "iteration": iteration,
                        "model": spec.runtime.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )
                # 发出检查点，通知外部系统“模型已给出指令，即将执行工具”。

                await hook.before_execute_tools(context)
                # 触发“执行工具前”钩子。

                # 准备执行工具（代码截断于此，下一部分将展示工具执行及结果处理逻辑）
                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    response.tool_calls,
                    external_lookup_counts,
                    workspace_violation_counts,
                    hook,
                    context,
                )
                # 调用 _execute_tools 方法并发或串行执行模型请求的所有工具。
                # 返回三个值：
                # 1. results: 工具执行结果列表（顺序与 tool_calls 对应）。
                # 2. new_events: 工具执行的事件日志（状态、耗时、错误详情等）。
                # 3. fatal_error: 如果配置了 fail_on_tool_error 且工具抛出异常，这里会捕获该致命错误对象。

                tool_events.extend(new_events)
                # 将本次迭代产生的新工具事件追加到全局的 tool_events 列表中。

                tools_used.extend(
                    tool_call.name
                    for tool_call, event in zip(response.tool_calls, new_events)
                    if event.get("status") == "ok"
                )
                # 使用生成器表达式配合 zip 函数，并行遍历工具调用和对应的事件。
                # 只有当事件状态为 "ok"（成功执行）时，才将工具名称追加到 tools_used 列表中。

                context.tool_results = list(results)
                context.tool_events = list(new_events)
                # 将工具结果和事件同步到当前的钩子上下文中，供 after_execute_tools 等钩子使用。

                completed_tool_results: list[dict[str, Any]] = []
                # 初始化一个列表，用于存放构建好的、准备发回给模型的 tool 角色消息。

                for tool_call, result in zip(response.tool_calls, results):
                    # 遍历每一个工具调用及其对应的执行结果。
                    tool_message = {
                        "role": "tool", # 消息角色必须是 "tool"，以符合 OpenAI 等标准 API 规范。
                        "tool_call_id": tool_call.id, # 必须与模型请求时的 tool_call_id 严格对应。
                        "name": tool_call.name, # 工具名称。
                        "content": self.context_governor.normalize_tool_result(
                            governance_config,
                            tool_call.id,
                            tool_call.name,
                            result,
                        ),
                        # 调用上下文治理器的 normalize_tool_result 方法。
                        # 作用：将工具返回的任意 Python 对象（如字典、列表、异常）序列化为字符串，
                        # 并根据 max_tool_result_chars 配置进行截断，防止超长结果撑爆 LLM 上下文窗口。
                    }
                    messages.append(tool_message)
                    # 将构建好的 tool 消息追加到主对话历史中。
                    completed_tool_results.append(tool_message)
                    # 同时将其加入 completed_tool_results 列表，用于后续发送检查点。

                if fatal_error is not None:
                    # 如果在工具执行过程中发生了致命错误（例如 spec.fail_on_tool_error 为 True 且工具抛出异常）。
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    # 格式化错误信息。
                    final_content = error
                    stop_reason = "tool_error"
                    # 设置最终的回复内容和停止原因。
                    self._append_final_message(messages, final_content)
                    # 将错误信息作为最终的 assistant 消息追加到历史中。
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    # 同步状态到钩子上下文。
                    await hook.after_iteration(context)
                    # 触发“迭代后”钩子。
                    should_continue, injection_cycles = await self._try_drain_injections(
                        spec, messages, None, injection_cycles,
                        phase="after tool error",
                    )
                    # 尝试排空外部注入消息。即使发生致命错误，也允许外部注入干预（例如人类接管）。
                    if should_continue:
                        had_injections = True
                        length_recovery_parts.clear()
                        continue
                        # 如果外部注入了新消息，清空长度恢复链，并使用 continue 跳过后续逻辑，直接进入下一轮迭代。
                    break
                    # 如果没有注入消息，则使用 break 彻底终止主循环。

                checkpoint_model_messages = (
                    self.context_governor.prepare_for_model(
                        governance_config,
                        messages,
                        compacted_tool_call_ids,
                    )
                    if response.provider_state is not None
                    else None
                )
                # 如果底层 Provider 需要维护特定的对话状态（如 Claude 的 state token），
                # 则重新调用 prepare_for_model 获取包含最新工具结果的模型消息视图，用于生成检查点快照。

                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed", # 阶段标识：工具执行完毕
                        "iteration": iteration,
                        "model": spec.runtime.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results, # 包含刚才所有工具的结果
                        "pending_tool_calls": [], # 待处理调用已清空
                        "provider_state": conversation_state.checkpoint(
                            messages,
                            model_messages=checkpoint_model_messages,
                        ),
                        # 保存当前 Provider 的对话状态快照。
                    },
                )
                # 发出检查点，通知外部系统工具已执行完毕，可以更新 UI 或持久化状态。

                empty_content_retries = 0
                length_recovery_parts.clear()
                # 工具执行成功并产生了新上下文，重置空内容重试计数，并清空之前的长度截断恢复链。

                # Checkpoint 1: drain injections after tools, before next LLM call
                # 原注释翻译：检查点 1：在工具执行后、下一次 LLM 调用前排空注入消息。
                _drained, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after tool execution",
                )
                if _drained:
                    had_injections = True
                # 尝试获取外部注入。如果成功获取并追加了消息，标记 had_injections 为 True。

                await hook.after_iteration(context)
                # 触发“迭代后”钩子。
                continue
                # 【关键控制流】使用 continue 直接跳过后面的“直接回复”处理逻辑，
                # 进入 for 循环的下一轮迭代，让 LLM 根据刚才的工具结果继续思考。

            if response.has_tool_calls:
                # 如果模型返回了工具调用格式（has_tool_calls 为 True），
                # 但 response.should_execute_tools 为 False（通常是因为 finish_reason 不是 "tool_calls"，
                # 例如被安全过滤器拦截，或者 API 返回了异常的 finish_reason）。
                logger.warning(
                    "Ignoring tool calls under finish_reason='{}' for {}",
                    response.finish_reason,
                    spec.session_key or "default",
                )
                # 记录警告日志，说明我们忽略了这些工具调用，将其视为普通的文本回复处理。

            clean = hook.finalize_content(context, response.content)
            # 调用钩子的 finalize_content 方法。
            # 作用：允许外部逻辑对模型输出的原始文本进行最终的清洗、格式化或脱敏处理。

            if (
                response.finish_reason
                not in {"error", "length", "refusal", "content_filter"}
                and is_blank_text(clean)
            ):
                # 如果模型没有报错、没有因长度截断、没有拒绝回答、没有触发内容过滤器，
                # 但是最终清洗后的内容却是纯空白（is_blank_text 为 True）。
                empty_content_retries += 1
                # 空内容重试计数器加 1。
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    # 如果重试次数还未达到上限（默认 2 次）。
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                        # 如果是流式模式，通知流结束（不恢复）。
                    await hook.after_iteration(context)
                    continue
                    # 触发迭代后钩子，并使用 continue 重新发起一次相同的 LLM 请求，期望模型这次能输出内容。

                # 如果达到了最大重试次数，模型依然返回空白。
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_content_retries,
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)

                retry_messages = self._finalization_retry_messages(messages_for_model)
                # 构建特殊的重试消息列表，通常会在末尾追加一条强烈的系统提示，
                # 强制要求模型：“不要再调用工具，必须立即给出最终的文字总结”。

                response = await self._request_finalization_retry(
                    spec,
                    messages_for_model,
                    transcript=messages,
                    conversation_state=conversation_state,
                )
                # 发起“最终总结重试”请求。这会临时禁用工具调用，逼迫模型输出文本。

                retry_usage = self._usage_or_estimate(spec, retry_messages, response)
                self._accumulate_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                # 获取重试请求的 Token 消耗，累加到全局统计中，并合并到当前迭代的 raw_usage 中。

                context.response = response
                context.usage = dict(raw_usage)
                context.tool_calls = list(response.tool_calls)
                # 用重试后的新响应覆盖上下文中的旧响应。

                original_content = response.content
                clean = hook.finalize_content(context, response.content)
                # 再次对重试后生成的内容进行清洗。

            if response.finish_reason == "length":
                # 如果模型的 finish_reason 是 "length"，表示输出达到了 max_tokens 限制被强制截断。
                if len(length_recovery_parts) < _MAX_LENGTH_RECOVERIES:
                    # 如果当前的恢复片段数量还未达到最大恢复次数（默认 3 次）。
                    length_recovery_parts.append(
                        _restore_outer_whitespace(clean or "", original_content)
                    )
                    # 将当前截断的片段（恢复其首尾空白字符后）追加到恢复链列表中。

                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        len(length_recovery_parts),
                        _MAX_LENGTH_RECOVERIES,
                    )
                    if hook.wants_streaming():
                        context.stream_continues_current_message = True
                        await hook.on_stream_end(context, resuming=True)
                        # 如果是流式模式，标记流将继续当前消息，并通知流结束（带恢复标记）。

                    messages.append(conversation_state.project_response_message(
                        build_assistant_message(
                            clean,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        ),
                        response,
                    ))
                    # 将当前截断的片段作为 assistant 消息追加到历史中。

                    messages.append(build_length_recovery_message(clean or ""))
                    # 【核心恢复机制】追加一条特殊的 user 角色消息（例如：“请继续你刚才未说完的话”），
                    # 引导模型在下一轮迭代中接着当前的断点继续生成。

                    await hook.after_iteration(context)
                    continue
                    # 触发迭代后钩子，并使用 continue 进入下一轮循环，让模型继续生成。

            # Some streaming providers recover with a complete response but no
            # content deltas. When an earlier length segment is already visible,
            # emit this terminal segment into the same stream; otherwise the
            # regular full response would duplicate the visible prefix.
            # 原注释翻译：某些流式提供商在恢复时会返回完整的响应，但不包含内容增量（deltas）。
            # 当较早的长度片段已经对前端可见时，将此终端片段发射到同一个流中；
            # 否则，常规的完整响应会重复显示已经可见的前缀。
            if (
                length_recovery_parts
                and hook.wants_streaming()
                and not context.streamed_content
                and response.finish_reason != "error"
                and not is_blank_text(clean)
            ):
                await hook.on_stream(
                    context,
                    _restore_outer_whitespace(clean or "", original_content),
                )
                # 调用钩子的 on_stream 方法，将恢复后的尾部内容增量发送给流式处理器。
                context.streamed_content = True
                # 标记当前上下文已经有内容被流式传输过了。

            assistant_message: dict[str, Any] | None = None
            # 初始化最终的助手消息变量。
            if response.finish_reason != "error" and not is_blank_text(clean):
                # 如果模型没有报错，且最终内容不为空白。
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                # 构建标准的 Assistant 消息字典。
                assistant_message = conversation_state.project_response_message(
                    assistant_message,
                    response,
                )
                # 让状态控制器对消息进行投影/修改（例如添加特定的 provider 标记）。

            # Check for mid-turn injections BEFORE signaling stream end.
            # If injections are found we keep the stream alive (resuming=True)
            # so streaming channels don't prematurely finalize the card.
            # 原注释翻译：在发出流结束信号之前检查中途注入。如果找到注入，我们保持流活跃（resuming=True），
            # 以免流式通道过早地最终化（关闭）卡片。
            should_continue, injection_cycles = await self._try_drain_injections(
                spec, messages, assistant_message, injection_cycles,
                conversation_state=conversation_state,
                phase="after final response",
                iteration=iteration,
                allow_goal_continue=(
                    response.finish_reason not in {"refusal", "content_filter"}
                ),
                # 如果模型不是因为拒绝回答或内容过滤器而停止，则允许目标继续逻辑。
            )
            if should_continue:
                had_injections = True
                # 如果检测到注入消息，标记发生过注入。

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=should_continue)
                # 触发流结束钩子。如果 should_continue 为 True，表示流还会继续。

            if should_continue:
                # 如果外部注入了新消息，或者目标继续被激活。
                length_recovery_parts.clear()
                # 清空长度恢复链，因为新的对话轮次开始了。
                await hook.after_iteration(context)
                continue
                # 触发迭代后钩子，并使用 continue 跳过后续的 break 逻辑，进入下一轮迭代。

            if response.finish_reason == "error":
                # 【错误处理】如果模型请求失败（如 API 报错、网络异常等）。
                if LLMProvider.is_arrearage_response(response):
                    # 检查是否是欠费/配额耗尽的特定错误。
                    final_content = _ARREARAGE_ERROR_MESSAGE
                else:
                    # 否则使用常规错误消息。
                    # Python 特性：`or` 运算符的短路求值，从左到右取第一个真值。
                    final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                self._append_model_error_placeholder(messages)
                # 在历史记录中追加一个模型错误占位符，确保历史消息的完整性。
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                # 触发迭代后钩子。
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after LLM error",
                )
                if should_continue:
                    had_injections = True
                    length_recovery_parts.clear()
                    continue
                    # 如果外部注入了新消息，继续循环。
                break
                # 否则使用 break 彻底终止主循环。

            if is_blank_text(clean):
                # 【空白兜底】如果经过所有重试后，最终内容依然是纯空白。
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                # 将默认的空回复提示追加到历史中。
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after empty response",
                )
                if should_continue:
                    had_injections = True
                    length_recovery_parts.clear()
                    continue
                break

            messages.append(
                assistant_message
                or conversation_state.project_response_message(
                    build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ),
                    response,
                )
            )
            # 将最终的助手消息追加到主对话历史中。
            # 如果 assistant_message 为 None（理论上前面已经处理，此处为双保险），则重新构建并投影。

            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response", # 阶段标识：最终响应
                    "iteration": iteration,
                    "model": spec.runtime.model,
                    "assistant_message": messages[-1], # 获取刚刚追加的最后一条消息
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                    "provider_state": conversation_state.checkpoint(messages),
                    # 保存最终的 Provider 状态快照。
                },
            )
            # 发出最终响应检查点。

            if length_recovery_parts:
                # 如果存在长度截断恢复链（即模型之前因为太长被截断过）。
                final_content = (
                    "".join(length_recovery_parts)
                    + _restore_outer_whitespace(clean or "", original_content)
                ).strip()
                # 【核心拼接】使用 "".join() 将之前保存的所有片段和当前的最后一部分拼接成一个完整的字符串。
                # 然后使用 .strip() 去除首尾多余的空白。
            else:
                final_content = clean

            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
            # 【正常结束】触发迭代后钩子，并使用 break 正常退出主循环。

        else:
            # 【Python 特性】for...else 结构。
            # else 块仅在 for 循环“自然完成”（即遍历完 range 且没有被 break 中断）时执行。
            # 这里表示 Agent 达到了 max_iterations 限制，但仍未产生最终回复。
            stop_reason = "max_iterations"
            # Drain any remaining injections so they are appended to the
            # conversation history instead of being re-published as
            # independent inbound messages by _dispatch's finally block.
            # We include them before the no-tools finalization pass so the
            # final response can account for every known follow-up.
            # 原注释翻译：排空任何剩余的注入，以便它们被追加到对话历史中，
            # 而不是被 _dispatch 的 finally 块重新发布为独立的入站消息。
            # 我们在无工具最终化传递之前包含它们，以便最终响应可以考虑每个已知的后续跟进。
            drained_after_max_iterations, injection_cycles = await self._try_drain_injections(
                spec, messages, None, injection_cycles,
                phase="after max_iterations",
            )
            if drained_after_max_iterations:
                had_injections = True

            terminal_content = None
            if spec.finalize_on_max_iterations:
                # 如果配置了达到最大迭代时强制总结。
                terminal_content = await self._try_finalize_after_max_iterations(
                    spec,
                    hook,
                    messages,
                    usage,
                    conversation_state,
                )
                # 尝试调用模型（禁用工具）生成最终的总结陈词。
            if terminal_content is None:
                # 如果总结失败或未配置总结。
                terminal_content = self._max_iterations_fallback(spec)
                # 使用兜底的固定文案（如：“已达到最大执行次数...”）。

            if length_recovery_parts:
                # 如果之前有长度截断的片段。
                terminal_tail = f"\n\n{terminal_content.lstrip()}"
                # 构建尾部内容，去除左侧空白。
                final_content = (
                    "".join(length_recovery_parts).rstrip() + terminal_tail
                ).strip()
                # 将之前的片段与兜底文案拼接。
                pending_stream_content = terminal_tail
                # 记录待处理的流式内容尾部（因为之前的片段可能已经流式输出了，这里只需要输出新增的尾部）。
            else:
                final_content = terminal_content

            self._append_final_message(messages, terminal_content)
            # 将最终的兜底或总结消息追加到历史中。

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            had_injections=had_injections,
            pending_stream_content=pending_stream_content,
            provider_state=conversation_state.finish(messages),
        )
        # 封装所有运行期间的状态和数据，返回最终的 AgentRunResult 对象。

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        # 辅助方法：构建发送给 LLM Provider 的通用参数字典。
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": spec.runtime.model,
            "retry_mode": spec.provider_retry_mode,
            "on_retry_wait": spec.retry_wait_callback,
        }
        # 初始化基础参数字典，包含消息历史、工具定义、模型名称和重试配置。

        generation = spec.runtime.generation
        # 获取生成参数配置对象。
        kwargs["temperature"] = generation.temperature
        kwargs["max_tokens"] = generation.max_tokens
        kwargs["reasoning_effort"] = generation.reasoning_effort
        # 将温度、最大 Token 数、推理努力程度（针对 o1 等模型）写入参数字典。
        return kwargs

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
        *,
        malformed_retry: bool = False,
        conversation_state: ProviderConversationStateController,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        # 核心方法：向 LLM Provider 发送请求，处理超时、流式输出、进度回调和底层重试。

        timeout_s: float | None = spec.llm_timeout_s
        # 获取配置的 LLM 请求超时时间（秒）。
        if timeout_s is None:
            # Default to a finite timeout to avoid per-session lock starvation when an LLM
            # request hangs indefinitely (e.g. gateway/network stall).
            # Set NANOBOT_LLM_TIMEOUT_S=0 to disable.
            # 原注释翻译：默认使用有限的超时时间，以避免当 LLM 请求无限期挂起（例如网关/网络停滞）时
            # 出现会话锁饥饿。设置 NANOBOT_LLM_TIMEOUT_S=0 以禁用。
            raw = os.environ.get("NANOBOT_LLM_TIMEOUT_S", "300").strip()
            # 从环境变量读取超时配置，默认 300 秒。
            try:
                timeout_s = float(raw)
                # 尝试转换为浮点数。
            except (TypeError, ValueError):
                timeout_s = 300.0
                # 如果转换失败，使用默认值。
        if timeout_s <= 0:
            timeout_s = None
            # 如果超时时间小于等于 0，表示禁用超时检查。

        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=spec.tools.get_definitions(),
        )
        # 构建请求参数字典，传入工具注册表提供的工具定义（JSON Schema 列表）。

        wants_streaming = hook.wants_streaming()
        # 检查当前钩子是否支持/要求流式输出。

        progress_callback = spec.progress_callback
        wants_progress_streaming = (
            not wants_streaming
            # 如果不是真正的流式模式（例如某些前端只想要进度条而不是打字机效果）。
            and spec.stream_progress_deltas
            and progress_callback is not None
            and getattr(spec.runtime.provider, "supports_progress_deltas", False) is True
            # 检查配置是否开启进度增量、回调是否存在、且 Provider 是否支持进度增量。
        )
        # 判断是否需要使用“进度流式”模式（一种轻量级的流式替代方案）。

        progress_state: dict[str, bool] | None = None
        # 初始化进度状态字典，用于在进度流式模式下跟踪推理标签的状态。
        active_hosted_tools: dict[str, dict[str, Any]] = {}
        # 初始化活跃托管工具字典。某些 Provider（如 OpenAI 的 Assistant API）会在服务端托管工具执行。
        # 这里用于跟踪这些托管工具的状态。

        async def _provider_tool_event(event: dict[str, Any]) -> None:
            # 内部异步回调函数：用于处理 Provider 级别发出的工具事件。
            if event.get("kind") != "hosted_tool":
                return
                # 只处理托管工具事件。
            await hook.on_provider_tool_event(context, event)
            # 触发钩子，允许外部监听托管工具的执行状态。
            call_id = event.get("call_id")
            if not call_id:
                return
            call_id = str(call_id)
            if event.get("phase") == "start":
                active_hosted_tools[call_id] = dict(event)
                # 如果工具开始执行，记录到活跃字典中。
            elif event.get("phase") in {"end", "error"}:
                active_hosted_tools.pop(call_id, None)
                # 如果工具结束或出错，从活跃字典中移除。

        if wants_streaming:
            # 【分支 1：真正的流式输出模式】
            thinking_buf = ""
            # 初始化思考内容缓冲区，用于在流式输出中逐步收集模型的思考过程。
            async def _stream(delta: str) -> None:
                # 定义内部异步回调函数：用于处理流式输出中的“正文内容”增量（delta）。
                if delta:
                    # 如果增量不为空，标记当前上下文已经有内容被流式传输。
                    context.streamed_content = True
                await hook.on_stream(context, delta)
                # 触发钩子的 on_stream 方法，将增量内容推送给外部（如前端渲染打字机效果）。

            async def _thinking(delta: str) -> None:
                # 定义内部异步回调函数：用于处理流式输出中的“思考/推理过程”增量。
                nonlocal thinking_buf
                # 【Python 特性】使用 nonlocal 关键字声明变量。
                # 允许在这个嵌套函数内部，修改外部函数 _request_model 中的局部变量 thinking_buf。
                if not delta:
                    return
                    # 如果没有增量，直接返回。
                prev_clean = strip_reasoning_tags(thinking_buf)
                # 在追加新内容之前，先清理当前缓冲区中的推理标签（如 <think> 等）。
                thinking_buf += delta
                # 将新的增量追加到缓冲区。
                new_clean = strip_reasoning_tags(thinking_buf)
                # 清理追加后的完整缓冲区。
                incremental = new_clean[len(prev_clean):]
                # 通过字符串切片，计算出本次新增的“纯净”推理文本。
                if incremental:
                    # 如果有实质性的新增推理文本。
                    context.streamed_reasoning = True
                    # 标记推理内容已经流式传输过。
                    await hook.emit_reasoning(incremental)
                    # 触发钩子，发射推理内容增量。

            async def _stream_recover() -> None:
                # 定义内部异步回调函数：用于处理流式输出的网络恢复事件。
                await hook.on_stream_end(context, resuming=True)
                # 触发流结束钩子，但标记为 resuming=True，表示流稍后会继续。

            coro = spec.runtime.provider.chat_stream_with_retry(
                **kwargs,
                provider_context=provider_context,
                on_content_delta=_stream,
                on_thinking_delta=_thinking,
                on_tool_call_delta=_provider_tool_event,
                on_stream_recover=_stream_recover,
            )
            # 调用 Provider 的流式聊天接口（带重试机制）。
            # 【Python 特性】使用 **kwargs 将之前构建的字典解包为关键字参数传入。
            # 将返回的协程对象（Coroutine）赋值给变量 coro，暂不执行，等待后续统一进行超时包装。

        elif wants_progress_streaming:
            # 【分支 2：进度流式输出模式】（非真正的打字机流式，而是通过进度回调推送增量）。
            stream_buf = ""
            # 初始化流内容缓冲区。
            think_extractor = IncrementalThinkExtractor()
            # 实例化增量思考提取器，用于在流中动态识别和提取 <think> 标签内的内容。
            progress_state = {"reasoning_open": False}
            # 初始化进度状态字典，用于跟踪推理标签是否处于打开状态。

            async def _stream_progress(delta: str) -> None:
                # 定义内部异步回调函数：处理进度流式模式下的内容增量。
                nonlocal stream_buf
                # 允许修改外部变量 stream_buf。
                if not delta:
                    return
                prev_clean = strip_think(stream_buf)
                # 清理当前缓冲区中的 <think> 标签，获取纯正文。
                stream_buf += delta
                # 追加新内容。
                new_clean = strip_think(stream_buf)
                # 清理追加后的缓冲区。
                incremental = new_clean[len(prev_clean):]
                # 计算新增的纯正文内容。

                if await think_extractor.feed(stream_buf, hook.emit_reasoning):
                    # 将完整的缓冲区喂给 think_extractor。
                    # 如果提取器发现了新的推理内容，它会调用 hook.emit_reasoning，并返回 True。
                    context.streamed_reasoning = True
                    progress_state["reasoning_open"] = True
                    # 更新状态：推理内容已流式传输，且推理标签处于打开状态。

                if incremental:
                    # 如果有新增的纯正文内容。
                    if progress_state["reasoning_open"]:
                        # 如果之前推理标签是打开的，现在正文开始了，说明推理结束。
                        await hook.emit_reasoning_end()
                        # 触发推理结束钩子。
                        progress_state["reasoning_open"] = False
                        # 更新状态。
                    context.streamed_content = True
                    # 标记内容已流式传输。
                    callback = progress_callback
                    if callback is not None:
                        await callback(incremental)
                        # 调用外部配置的进度回调函数，推送正文增量。

            coro = spec.runtime.provider.chat_stream_with_retry(
                **kwargs,
                provider_context=provider_context,
                on_content_delta=_stream_progress,
                on_tool_call_delta=_provider_tool_event,
            )
            # 调用流式接口，但只绑定内容增量和工具调用增量回调。

        else:
            # 【分支 3：普通非流式模式】
            coro = spec.runtime.provider.chat_with_retry(
                **kwargs,
                provider_context=provider_context,
            )
            # 调用普通的异步聊天接口（带重试机制），等待完整的响应。

        # Streaming requests also have provider-level idle timeouts
        # (NANOBOT_STREAM_IDLE_TIMEOUT_S), but a stream that keeps producing
        # very slow deltas can still run forever. Use a more generous wall-clock
        # timeout for streaming while preserving NANOBOT_LLM_TIMEOUT_S=0 as an
        # opt-out for all LLM wall-clock timeouts.
        # 原注释翻译：流式请求也有 Provider 级别的空闲超时（NANOBOT_STREAM_IDLE_TIMEOUT_S），
        # 但一个持续产生非常缓慢增量的流仍然可能永远运行。为流式使用更宽裕的绝对时钟超时，
        # 同时保留 NANOBOT_LLM_TIMEOUT_S=0 作为所有 LLM 绝对时钟超时的退出选项。
        is_streaming_request = wants_streaming or wants_progress_streaming
        # 判断当前是否为流式请求。
        outer_timeout_s = (
            max(300.0, timeout_s * 2)
            if is_streaming_request and timeout_s is not None
            else timeout_s
        )
        # 计算外部超时时间。如果是流式请求，给予至少 300 秒或配置超时两倍的宽裕时间。
        try:
            response = (
                await coro if outer_timeout_s is None
                else await asyncio.wait_for(coro, timeout=outer_timeout_s)
            )
            # 【Python 特性】使用 asyncio.wait_for(coro, timeout=...) 包装协程，实现异步超时控制。
            # 如果 outer_timeout_s 为 None，则直接 await 协程，不设超时限制。
        except asyncio.TimeoutError:
            # 捕获 asyncio 超时异常。
            if outer_timeout_s is None:
                response = LLMResponse(
                    content="Error calling LLM: stream stalled",
                    finish_reason="error",
                    error_kind="timeout",
                )
            else:
                response = LLMResponse(
                    content=f"Error calling LLM: timed out after {outer_timeout_s:g}s",
                    finish_reason="error",
                    error_kind="timeout",
                )
                # 构建一个表示超时的 LLMResponse 对象，让上层逻辑能够统一处理错误。

        # chat_stream_with_retry may recover internally, so only fail unfinished
        # hosted calls after the provider returns its final error response.
        # 原注释翻译：chat_stream_with_retry 可能会在内部恢复，所以只有在 Provider 返回最终的错误响应后，
        # 才使未完成的托管调用失败。
        if response.finish_reason == "error":
            # 如果最终的响应状态是错误。
            for event in list(active_hosted_tools.values()):
                # 遍历所有仍在活跃状态的托管工具事件。
                await _provider_tool_event({
                    **event,
                    "phase": "error",
                    "result": None,
                    "error": response.content
                    or "Model request failed before the provider-hosted tool completed.",
                })
                # 手动构造并触发一个 error 阶段的事件，清理这些悬挂的工具状态。
                # 【Python 特性】使用 **event 解包字典，并覆盖/新增 phase, result, error 键。

        if progress_state and progress_state.get("reasoning_open"):
            # 如果在进度流式模式下，推理标签在流结束时仍处于打开状态。
            await hook.emit_reasoning_end()
            # 强制触发推理结束钩子，确保状态闭合。

        dropped, all_dropped, original_finish_reason = (
            self._drop_malformed_tool_calls(response)
        )
        # 调用辅助方法，剥离响应中缺失名称或格式错误的畸形工具调用。
        # 返回：被丢弃的数量、是否全部被丢弃、原始的 finish_reason。

        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and not malformed_retry
        ):
            # 如果所有工具调用都是畸形的被丢弃了，且模型原本是想调用工具的，且这还不是重试请求。
            logger.warning(
                "Retrying LLM request after all {} malformed tool call(s) were dropped",
                dropped,
            )
            retry_messages = self._malformed_tool_call_retry_messages(
                messages, response.content,
            )
            # 构建特殊的重试消息，告知模型它之前的工具调用格式错误，要求重新正确调用。
            return await self._request_model(
                spec, retry_messages, hook, context,
                malformed_retry=True,
                conversation_state=conversation_state,
                provider_context=conversation_state.independent_request_context(
                    context_window_tokens=spec.runtime.context_window_tokens,
                ),
            )
            # 递归调用 _request_model 发起重试，并标记 malformed_retry=True。

        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and malformed_retry
        ):
            # 如果重试之后，模型依然返回全是畸形的工具调用。
            logger.warning(
                "Malformed tool calls persisted after retry; falling back to no-tools request",
            )
            fallback_messages = self._malformed_tool_call_retry_messages(
                messages, response.content,
            )
            return await self._request_no_tools(
                spec,
                fallback_messages,
                provider_context=conversation_state.independent_request_context(
                    context_window_tokens=spec.runtime.context_window_tokens,
                ),
            )
            # 【最终兜底】放弃工具调用，回退到“无工具请求”模式，逼迫模型直接用文本回复。

        return response
        # 返回最终清洗和验证过的 LLMResponse 对象。

    @staticmethod
    def _drop_malformed_tool_calls(
        response: LLMResponse,
    ) -> tuple[int, bool, str | None]:
        """Strip tool calls whose name is missing/non-string from the response.

        Returns (dropped_count, all_dropped, original_finish_reason).

        A degenerate call (name=None or "") cannot be executed, and if it were
        persisted into the assistant message it would be replayed on every
        subsequent turn, causing upstream validation errors
        (``tool_use.name: Input should be a valid string``) that permanently
        wedge the session. Dropping it here keeps it out of execution, the
        assistant message, and the saved history in one place.
        """
        # 静态方法文档字符串（翻译）：从响应中剥离名称缺失/非字符串的工具调用。
        # 返回 (丢弃数量, 是否全部丢弃, 原始 finish_reason)。
        # 退化的调用（name=None 或 ""）无法执行，如果将其持久化到 assistant 消息中，
        # 它会在后续每一轮中被重放，导致上游验证错误（``tool_use.name: 输入应为有效字符串``），
        # 从而使会话永久卡死。在这里丢弃它，可以一次性防止其进入执行、assistant 消息和保存的历史。

        calls = getattr(response, "tool_calls", None)
        # 使用 getattr 安全地获取 response 对象的 tool_calls 属性。如果属性不存在，则返回 None。
        if not calls:
            # 如果没有工具调用。
            return (0, False, getattr(response, "finish_reason", None))

        valid = [tc for tc in calls if tc.has_valid_name()]
        # 【Python 特性】列表推导式：遍历 calls，只保留通过 has_valid_name() 验证的有效工具调用。
        if len(valid) == len(calls):
            # 如果所有调用都是有效的。
            return (0, False, getattr(response, "finish_reason", None))

        dropped = len(calls) - len(valid)
        # 计算被丢弃的畸形调用数量。
        original_finish_reason = getattr(response, "finish_reason", None)
        # 保存原始的 finish_reason。
        logger.warning(
            "Dropped {} malformed tool call(s) with missing/non-string name "
            "from LLM response (finish_reason={!r})",
            dropped,
            original_finish_reason,
        )
        # 记录警告日志。{!r} 格式化占位符会调用对象的 repr() 方法，通常用于给字符串加上引号。
        response.tool_calls = valid
        # 用过滤后的有效列表覆盖 response 中的原列表。

        # The opaque candidate still contains every raw function_call item.
        # Advancing it after dropping even one call would replay an unmatched
        # call without a corresponding tool output on the next request.
        # 原注释翻译：不透明的候选状态（Provider state）仍然包含每个原始的 function_call 项。
        # 在丢弃哪怕一个调用后推进该状态，都会在下一次请求中重放一个没有对应工具输出的未匹配调用。
        response.provider_state = None
        # 【关键防御】清空 Provider 状态，防止底层模型 API 重放这些无效的调用记录。

        if not valid:
            # 如果过滤后没有任何有效的工具调用。
            response.finish_reason = "stop"
            # 将 finish_reason 强制改为 "stop"，让上层逻辑将其视为普通的文本停止，而不是工具调用。
        return (dropped, not valid, original_finish_reason)
        # 返回统计结果和原始状态。

    @staticmethod
    def _malformed_tool_call_retry_messages(
        messages: list[dict[str, Any]],
        assistant_text: str | None,
    ) -> list[dict[str, Any]]:
        # 静态方法：构建用于畸形工具调用重试的消息列表。
        retry_messages = list(messages)
        # 复制当前的消息历史。
        note = (
            "The previous model response attempted to call tools, but every tool call "
            "was malformed: the tool_use blocks had missing or non-string tool names. "
            "Do not answer with a promise to use tools. Either call the required tools again "
            "using valid tool names from the provided tool list and JSON object inputs, or give "
            "a final answer only if no tool is required."
        )
        # 构建强烈的系统提示文本，指出之前的错误并要求纠正。
        if assistant_text:
            # 如果模型在调用工具前还输出了一些文本。
            note += (
                f"\n\nPrevious assistant text before the malformed calls:\n"
                f"{assistant_text}"
            )
            # 将之前的文本也附加到提示中，帮助模型回忆上下文。
        retry_messages.append({"role": "user", "content": note})
        # 将这条提示作为 user 消息追加到历史末尾。
        return retry_messages

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        transcript: list[dict[str, Any]],
        conversation_state: ProviderConversationStateController,
    ) -> LLMResponse:
        # 异步方法：发起“最终总结重试”请求（用于模型输出空白时逼迫其说话）。
        retry_messages = self._finalization_retry_messages(messages)
        # 构建包含强烈总结提示的消息列表。
        provider_context = conversation_state.prepare_request(
            transcript,
            context_window_tokens=spec.runtime.context_window_tokens,
            supplemental_messages=[retry_messages[-1]],
        )
        # 准备请求上下文。
        response = await self._request_no_tools(
            spec,
            retry_messages,
            provider_context=provider_context,
        )
        # 调用无工具请求接口。
        conversation_state.observe_response(
            response,
            transcript,
            adopt_candidate_state=False,
        )
        # 让状态控制器观察响应，但不采纳候选状态（因为这是临时重试）。
        return response

    @staticmethod
    def _finalization_retry_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 静态方法：构建最终总结重试的消息列表。
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        # 追加标准的总结提示消息。
        return retry_messages
    async def _try_finalize_after_max_iterations(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
        usage: dict[str, int],
        conversation_state: ProviderConversationStateController,
    ) -> str | None:
        # 异步方法：尝试在达到最大迭代次数（预算耗尽）后，强制模型生成最终总结。
        retry_messages = self._budget_exhausted_finalization_messages(messages)
        # 构建包含“预算已耗尽，请立即总结”提示的消息列表。
        try:
            response = await self._request_no_tools(
                spec,
                retry_messages,
                provider_context=conversation_state.independent_request_context(
                    context_window_tokens=spec.runtime.context_window_tokens,
                ),
            )
            # 发起一个禁用工具的 LLM 请求，逼迫模型直接输出文本总结。
        except Exception:
            # 捕获请求过程中可能发生的任何异常。
            logger.exception(
                "Budget-exhausted finalization failed for {}; using fallback",
                spec.session_key or "default",
            )
            # 记录异常日志，说明最终总结失败，将使用兜底文案。
            return None

        raw_usage = self._usage_or_estimate(spec, retry_messages, response)
        self._accumulate_usage(usage, raw_usage)
        # 获取这次总结请求的 Token 消耗，并累加到全局统计中。

        if response.finish_reason == "error" or response.has_tool_calls:
            # 如果总结请求依然报错，或者模型还在试图调用工具（说明模型没有听从指令）。
            logger.warning(
                "Budget-exhausted finalization returned finish_reason='{}' "
                "with {} tool call(s) for {}; using fallback",
                response.finish_reason,
                len(response.tool_calls),
                spec.session_key or "default",
            )
            # 记录警告日志。
            return None
            # 返回 None，让调用方使用固定的兜底文案。

        context = AgentHookContext(
            iteration=spec.max_iterations,
            messages=messages,
            response=response,
            usage=dict(raw_usage),
            session_key=spec.session_key,
        )
        # 为这次总结请求创建一个临时的钩子上下文。
        clean = hook.finalize_content(context, response.content)
        # 调用钩子对总结内容进行清洗。
        if is_blank_text(clean):
            # 如果总结内容依然是空白。
            return None
        return clean
        # 返回成功的总结文本。

    async def _request_no_tools(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        # 异步方法：发起一个完全禁用工具的 LLM 请求。
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=None,
            # 【关键】将 tools 参数显式设置为 None，从底层 API 层面禁用工具调用功能。
        )
        return await spec.runtime.provider.chat_with_retry(
            **kwargs,
            provider_context=provider_context,
        )
        # 调用普通的聊天接口并返回响应。

    @staticmethod
    def _budget_exhausted_finalization_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # 静态方法：构建预算耗尽时的最终总结消息列表。
        retry_messages = list(messages)
        retry_messages.append(build_budget_exhausted_finalization_message())
        # 追加标准的预算耗尽提示消息。
        return retry_messages

    @staticmethod
    def _max_iterations_fallback(spec: AgentRunSpec) -> str:
        # 静态方法：获取达到最大迭代次数时的兜底提示文案。
        if spec.max_iterations_message:
            # 如果用户在配置中自定义了提示消息模板。
            return spec.max_iterations_message.format(
                max_iterations=spec.max_iterations,
            )
            # 使用 Python 字符串的 .format() 方法填充最大迭代次数变量。
        return render_template(
            "agent/max_iterations_message.md",
            strip=True,
            max_iterations=spec.max_iterations,
        )
        # 否则，调用模板渲染函数，从默认的 Markdown 模板文件中加载文案。

    def _usage_or_estimate(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> dict[str, int]:
        # 方法：获取 Token 使用量。优先使用 API 返回的真实数据，否则进行本地估算。
        usage = self._usage_dict(response.usage)
        # 将响应中的 usage 对象转换为标准的整数字典。
        total = self._usage_total(usage)
        # 计算总 Token 数。
        if total > 0:
            # 如果 API 返回了有效的使用量数据。
            usage["total_tokens"] = total
            usage.setdefault("provider_tokens", total)
            # 确保 total_tokens 和 provider_tokens 键存在。
            return usage

        if response.finish_reason == "error":
            # 如果请求报错，且没有使用量数据，直接返回空字典。
            return {}

        return self._estimate_response_usage(spec, messages, response)
        # 否则，调用本地估算方法计算 Token 数。

    def _estimate_response_usage(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> dict[str, int]:
        # 方法：在 API 未返回 Token 数时，通过本地分词器或启发式规则估算 Token 消耗。
        try:
            tools = spec.tools.get_definitions()
        except Exception:
            tools = None
            # 尝试获取工具定义，如果失败则置为 None。
        prompt_tokens, _ = estimate_prompt_tokens_chain(
            spec.runtime.provider,
            spec.runtime.model,
            messages,
            tools,
        )
        # 调用辅助函数，估算整个输入提示（Prompt）的 Token 数量。

        assistant_message = build_assistant_message(
            response.content or "",
            tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        # 构建模型回复的完整消息字典。
        completion_tokens = estimate_message_tokens(assistant_message)
        # 调用辅助函数，估算输出内容（Completion）的 Token 数量。

        total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)
        # 计算总 Token 数，确保不为负数。
        if total_tokens <= 0:
            return {}

        return {
            "prompt_tokens": max(0, prompt_tokens),
            "completion_tokens": max(0, completion_tokens),
            "total_tokens": total_tokens,
            "estimated_tokens": total_tokens,
            # 标记这是估算出来的数据。
        }

    @staticmethod
    def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
        # 静态方法：将任意格式的 usage 数据清洗为标准的整数字典。
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
                # 尝试将值转换为整数，None 视为 0。
            except (TypeError, ValueError):
                continue
                # 如果转换失败，跳过该键。
        return result

    @staticmethod
    def _usage_total(usage: dict[str, int]) -> int:
        # 静态方法：从 usage 字典中安全地提取总 Token 数。
        return max(0, usage.get("total_tokens", 0) or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        ))
        # 【Python 特性】使用 `or` 的短路特性：如果 total_tokens 不存在或为 0，则手动相加 prompt 和 completion。

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        # 静态方法：将新的使用量累加到目标字典中。
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value
            # 遍历并累加。使用 .get(key, 0) 处理键不存在的情况。

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        # 静态方法：合并两个使用量字典，返回一个新的字典（不修改原字典）。
        merged = dict(left)
        # 浅拷贝左侧字典。
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        external_lookup_counts: dict[str, int],
        workspace_violation_counts: dict[str, int],
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        # 异步方法：执行模型请求的所有工具调用。负责并发调度和结果收集。
        hook = hook or AgentHook()
        context = context or AgentHookContext(iteration=0, messages=[])
        # 提供默认的钩子和上下文对象，防止空指针。

        batches = self._partition_tool_batches(spec, tool_calls)
        # 调用辅助方法，根据工具的并发安全性将工具调用分割成多个批次。

        tool_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
        # 初始化列表，用于收集所有工具的执行结果、事件和可能的异常。

        for batch in batches:
            # 遍历每一个工具批次。
            if spec.concurrent_tools and len(batch) > 1:
                # 如果配置允许并发执行工具，且当前批次有多个工具。
                batch_results = await asyncio.gather(*(
                    self._run_tool(
                        spec,
                        tool_call,
                        external_lookup_counts,
                        workspace_violation_counts,
                        hook,
                        context,
                    )
                    for tool_call in batch
                ))
                # 【Python 特性】使用 asyncio.gather() 并发等待批次内的所有协程完成。
                # 生成器表达式为每个 tool_call 创建一个 _run_tool 协程。
                # 前面的 * 将生成器解包为多个独立的参数传递给 gather。
                tool_results.extend(batch_results)
                # 将批次结果追加到总列表中。
            else:
                # 如果不允许并发，或者批次只有一个工具，则串行执行。
                batch_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
                for tool_call in batch:
                    result = await self._run_tool(
                        spec,
                        tool_call,
                        external_lookup_counts,
                        workspace_violation_counts,
                        hook,
                        context,
                    )
                    # 逐个 await 执行工具。
                    tool_results.append(result)
                    batch_results.append(result)

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        # 初始化最终的返回容器。

        for result, event, error in tool_results:
            # 遍历所有收集到的结果。
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
                # 记录第一个发生的致命错误。
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        external_lookup_counts: dict[str, int],
        workspace_violation_counts: dict[str, int],
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        # 异步方法：执行单个工具调用的完整生命周期（包含安全检查、参数准备、执行、异常捕获）。
        hook = hook or AgentHook()
        context = context or AgentHookContext(iteration=0, messages=[])

        hint = "\n\n[Analyze the error above and try a different approach.]"
        # 定义一个提示后缀，当工具出错时附加给模型，引导其改变策略。

        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        # 检查是否重复调用了相同的外部查询工具（防止模型陷入死循环查询）。
        if lookup_error:
            # 如果检测到重复查询。
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + hint, event, RuntimeError(lookup_error)
            return lookup_error + hint, event, None
            # 拦截本次调用，直接返回错误信息给模型，不实际执行工具。

        prepare_call = cast(
            Callable[[str, Any], object] | None,
            getattr(spec.tools, "prepare_call", None),
        )
        # 尝试从工具注册表中获取 prepare_call 方法（用于预处理工具参数或进行安全校验）。
        # 使用 cast 和 getattr 进行安全的动态方法获取。
        tool, params, prep_error = None, tool_call.arguments, None
        if callable(prepare_call):
            # 如果存在预处理方法。
            prepared = prepare_call(tool_call.name, tool_call.arguments)
            if isinstance(prepared, tuple):
                # 如果预处理方法返回了一个元组。
                prepared_tuple = cast(tuple[object, ...], prepared)
                if len(prepared_tuple) == 3:
                    tool, params, prep_error = cast(tuple[Any, Any, str | None], prepared_tuple)
                    # 解包出：工具对象、预处理后的参数、预处理错误信息。

        if prep_error:
            # 如果预处理阶段发现了错误（例如路径越权、参数非法）。
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
                # 截取错误信息的核心部分作为日志详情。
            }
            handled = self._classify_violation(
                raw_text=prep_error,
                soft_payload=prep_error + hint,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            # 调用安全违规分类器，判断这是否是 SSRF 或工作区越权等安全边界问题。
            if handled is not None:
                return handled
                # 如果分类器处理了该错误（例如添加了安全警告），直接返回处理后的结果。
            return prep_error + hint, event, (
                RuntimeError(prep_error) if spec.fail_on_tool_error else None
            )
            # 否则返回普通的错误信息。

        await hook.before_execute_tool(context, tool_call, tool, params)
        # 触发“执行单个工具前”钩子。
        try:
            if tool is not None:
                # 如果预处理阶段直接返回了工具对象实例。
                result = await tool.execute(**params)
                # 直接调用工具对象的 execute 方法。
                # 【Python 特性】使用 **params 将字典解包为关键字参数。
            else:
                # 否则，通过工具注册表根据名称查找并执行工具。
                result = await spec.tools.execute(tool_call.name, params)
        except asyncio.CancelledError:
            # 捕获任务取消异常。
            raise
            # 必须重新抛出，保证异步取消信号正常传播。
        except Exception as exc:
            # 捕获工具执行过程中抛出的任何常规异常。
            await hook.on_execute_tool_error(context, tool_call, tool, params, exc)
            # 触发工具执行错误钩子。
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            payload = f"Error: {type(exc).__name__}: {exc}"
            handled = self._classify_violation(
                raw_text=str(exc),
                # Preserve legacy exception payloads without the retry hint.
                # 原注释翻译：保留不带重试提示的传统异常负载。
                soft_payload=payload,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            if handled is not None:
                return handled
            if spec.fail_on_tool_error:
                return payload, event, exc
                # 如果配置了工具出错即失败，将原始异常对象作为 fatal_error 返回。
            return payload, event, None
            # 否则将错误信息转为字符串返回给模型，让模型自行纠正。

        if is_tool_error_result(result):
            # 如果工具执行没有抛出异常，但返回的结果被标记为“错误结果”。
            await hook.on_execute_tool_error(context, tool_call, tool, params, result)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            handled = self._classify_violation(
                raw_text=result,
                soft_payload=result + hint,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            if handled is not None:
                return handled
            if spec.fail_on_tool_error:
                return result + hint, event, RuntimeError(result)
            return result + hint, event, None

        await hook.after_execute_tool(context, tool_call, tool, params, result)
        # 触发“执行单个工具后”钩子。

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        # 构建用于日志和事件追踪的简短详情字符串。
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}, None
        # 返回成功的结果。
    # SSRF is a hard security block at the tool boundary, but the agent turn
    # should recover conversationally instead of aborting the runtime.
    # 原注释翻译：SSRF（服务器端请求伪造）是工具边界处的硬性安全拦截，但 Agent 的对话轮次应该以对话方式恢复，而不是中止运行时。
    _SSRF_MARKERS: tuple[str, ...] = (
        "internal/private url detected",
        "private/internal address",
        "private address",
    )
    # 【Python 特性】tuple[str, ...] 类型提示表示一个包含任意数量字符串的元组。
    # 这里定义了用于检测 SSRF 错误信息的关键词标记。

    _SSRF_BOUNDARY_NOTE: str = (
        "This is a non-bypassable security boundary. Stop trying to access "
        "private/internal URLs. Do not retry with curl, wget, encoded IPs, "
        "alternate DNS, redirects, proxies, or another tool. Ask the user for "
        "local files, logs, screenshots, or an explicit safe public URL instead. "
        "If the user explicitly trusts this private URL, ask them to whitelist "
        "the exact IP/CIDR via tools.ssrfWhitelist."
    )
    # 定义当检测到 SSRF 攻击时，追加给模型的强烈安全警告提示语。

    # Non-SSRF boundary markers returned to the LLM as recoverable tool errors.
    # 原注释翻译：非 SSRF 边界标记，作为可恢复的工具错误返回给 LLM。
    _WORKSPACE_VIOLATION_MARKERS: tuple[str, ...] = (
        "outside the configured workspace",
        "outside allowed directory",
        "working_dir is outside",
        "working_dir could not be resolved",
        "path outside working dir",
        "path traversal detected",
    )
    # 定义工作区越权（如路径遍历、访问沙箱外文件）的关键词标记。

    @classmethod
    def _is_ssrf_violation(cls, text: str) -> bool:
        # 类方法：检查给定的错误文本是否匹配 SSRF 安全违规。
        if not text:
            return False
        lowered = text.lower()
        # 将文本转换为小写，进行大小写不敏感的匹配。
        return any(marker in lowered for marker in cls._SSRF_MARKERS)
        # 【Python 特性】使用 any() 函数配合生成器表达式。
        # 只要 cls._SSRF_MARKERS 中有任何一个标记包含在 lowered 中，就立即返回 True。

    @classmethod
    def _is_workspace_violation(cls, text: str) -> bool:
        """True when *text* looks like any policy boundary rejection."""
        # 类方法文档字符串（翻译）：当 *text* 看起来像任何策略边界拒绝时返回 True。
        if not text:
            return False
        lowered = text.lower()
        if cls._is_ssrf_violation(lowered):
            return True
            # SSRF 也属于广义的工作区/策略违规。
        return any(marker in lowered for marker in cls._WORKSPACE_VIOLATION_MARKERS)
        # 检查工作区越权标记。

    def _classify_violation(
        self,
        *,
        raw_text: str,
        soft_payload: str,
        event: dict[str, str],
        tool_call: ToolCallRequest,
        workspace_violation_counts: dict[str, int],
    ) -> tuple[Any, dict[str, str], BaseException | None] | None:
        """Classify safety-boundary failures, or return ``None`` to pass through."""
        # 方法文档字符串（翻译）：对安全边界失败进行分类，或者返回 ``None`` 以直接透传。
        # 【Python 特性】参数列表中的 `*` 表示其后的所有参数都必须作为关键字参数传递（Keyword-Only Arguments），
        # 这有助于提高方法调用的可读性并防止参数顺序错误。

        if self._is_ssrf_violation(raw_text):
            # 如果检测到 SSRF 违规。
            logger.warning(
                "Tool {} blocked by SSRF guard; returning non-retryable tool error: {}",
                tool_call.name,
                raw_text.replace("\n", " ").strip()[:200],
            )
            # 记录警告日志。
            event["detail"] = self._event_detail("ssrf_violation: ", raw_text)
            # 更新事件日志的详情字段。
            return self._ssrf_soft_payload(raw_text), event, None
            # 返回追加了安全警告的软负载（soft payload），不抛出致命异常，让模型通过对话恢复。

        if self._is_workspace_violation(raw_text):
            # 如果检测到工作区越权违规。
            escalation = repeated_workspace_violation_error(
                tool_call.name,
                tool_call.arguments,
                workspace_violation_counts,
            )
            # 检查是否重复触发了相同的工作区违规（防止模型陷入不断尝试越权的死循环）。
            event["detail"] = self._event_detail("workspace_violation: ", raw_text)
            if escalation is not None:
                # 如果检测到重复违规，需要升级警告。
                logger.warning(
                    "Tool {} hit workspace boundary repeatedly; escalating hint",
                    tool_call.name,
                )
                event["detail"] = self._event_detail(
                    "workspace_violation_escalated: ",
                    raw_text,
                )
                return escalation, event, None
                # 返回升级后的错误提示。
            return soft_payload, event, None
            # 首次违规，返回普通的错误提示。

        return None
        # 如果不是安全边界违规，返回 None，交由常规的工具错误逻辑处理。

    @classmethod
    def _ssrf_soft_payload(cls, raw_text: str) -> str:
        # 类方法：构建 SSRF 错误的软负载文本。
        text = raw_text.strip() or "Error: request blocked by SSRF guard"
        # 清理原始错误文本，如果为空则使用默认提示。
        return f"{text}\n\n{cls._SSRF_BOUNDARY_NOTE}"
        # 将原始错误与强烈的安全边界警告拼接在一起返回。

    @staticmethod
    def _event_detail(prefix: str, text: str, limit: int = 160) -> str:
        # 静态方法：格式化并截断事件详情字符串。
        return (prefix + text.replace("\n", " ").strip())[:limit]
        # 替换换行符为空格，添加前缀，并使用切片 [:limit] 截断到最大长度，防止日志过长。

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict[str, Any],
    ) -> None:
        # 异步方法：触发检查点回调。
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)
            # 如果配置了检查点回调，则 await 调用它，将当前状态 payload 传递给外部系统。

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        # 静态方法：将最终的回复内容安全地追加到消息历史中。
        if not content:
            return
            # 如果内容为空，直接返回。
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            # 如果历史列表不为空，且最后一条消息已经是无工具调用的 assistant 消息。
            if messages[-1].get("content") == content:
                return
                # 如果内容完全相同，避免重复追加。
            messages[-1] = build_assistant_message(content)
            # 否则，直接用新的最终内容覆盖最后一条消息（通常用于错误兜底或强制总结）。
            return
        messages.append(build_assistant_message(content))
        # 否则，构建新的 assistant 消息并追加到列表末尾。

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        # 静态方法：在模型彻底崩溃时，追加一个错误占位符消息。
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
            # 如果最后一条已经是 assistant 消息，则不重复追加占位符。
        messages.append(build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))
        # 追加预定义的错误占位符文本。

    def _partition_tool_batches(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        # 方法：根据工具的并发安全性，将工具调用列表分割成多个执行批次。
        if not spec.concurrent_tools:
            # 如果配置不允许并发执行工具。
            return [[tool_call] for tool_call in tool_calls]
            # 【Python 特性】列表推导式：将每个工具调用包装成只有一个元素的列表，强制串行执行。

        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        # 初始化总批次列表和当前正在收集的批次列表。

        for tool_call in tool_calls:
            # 遍历每一个工具调用。
            get_tool = cast(Callable[[str], Any] | None, getattr(spec.tools, "get", None))
            # 尝试获取工具注册表的 get 方法。
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            # 如果 get 方法可调用，则根据名称获取工具对象实例。
            can_batch = bool(tool and tool.concurrency_safe)
            # 判断该工具是否可以安全地与其他工具并发执行（通过检查 tool.concurrency_safe 属性）。

            if can_batch:
                # 如果该工具支持并发。
                current.append(tool_call)
                continue
                # 将其加入当前批次，并继续检查下一个工具。

            # 如果该工具不支持并发（例如文件写入、数据库事务等）。
            if current:
                # 如果当前批次中已经积累了支持并发的工具。
                batches.append(current)
                current = []
                # 将当前批次保存到总列表中，并清空当前批次。
            batches.append([tool_call])
            # 将这个不支持并发的工具单独作为一个批次添加，确保它被串行执行。

        if current:
            # 循环结束后，如果当前批次还有剩余的并发安全工具。
            batches.append(current)
            # 将最后一个批次添加到总列表中。

        return batches
        # 返回分割好的批次列表，供 _execute_tools 进行调度。
