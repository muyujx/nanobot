"""Agent loop: the core processing engine."""
# 模块文档字符串：Agent 循环：核心处理引擎。
# 这个文件包含了 AI Agent 的核心事件循环、状态管理和消息处理逻辑。

# pyright: reportPrivateUsage=false
# Pyright 配置指令：禁用对访问私有成员（以单下划线 _ 开头的变量或方法）的警告。
# 允许在外部或子类中灵活访问内部实现细节。

from __future__ import annotations
# 启用 PEP 563 延迟注解计算（Postponed Evaluation of Annotations）。
# 作用：1. 允许在类定义完成前使用前向引用（例如在方法参数中引用尚未定义完的类自身）。
# 2. 将类型注解存储为字符串，减少运行时的内存开销和导入时间。

import asyncio
# 导入 asyncio 库：Python 的异步 I/O 基础设施。
# 提供事件循环（Event Loop）、协程（Coroutines）、任务（Tasks）、锁（Locks）、队列（Queues）等并发原语。

import dataclasses
# 导入 dataclasses 模块：提供 @dataclass 装饰器。
# 用于自动生成类的 __init__、__repr__、__eq__ 等方法，减少样板代码。

import inspect
# 导入 inspect 模块：用于检查 Python 对象的内部信息。
# 常用于获取函数签名（signature）、参数类型、源码、判断对象是否为协程等。

import os
# 导入 os 模块：提供与操作系统交互的接口。
# 常用于读取环境变量（如 os.environ.get）、操作文件路径等。

import time
# 导入 time 模块：提供时间相关的函数。
# 常用于获取时间戳（time.time）、高精度计时（time.perf_counter）、休眠等。

import weakref
# 导入 weakref 模块：提供弱引用功能。
# 弱引用不会增加对象的引用计数，常用于缓存或映射（如 WeakValueDictionary），
# 防止因长生命周期容器（如全局字典）持有对象引用而导致的内存泄漏。

from collections.abc import Coroutine, Iterable, Mapping
# 从 collections.abc 导入抽象基类，用于类型提示（Type Hinting）：
# Coroutine: 协程类型（可被 await 的对象）。
# Iterable: 可迭代对象类型（如列表、元组、生成器）。
# Mapping: 映射/字典类型（如 dict）。

from contextlib import AbstractContextManager, ExitStack, nullcontext, suppress
# 从 contextlib 导入上下文管理器相关工具：
# AbstractContextManager: 上下文管理器的抽象基类（实现 __enter__ 和 __exit__）。
# ExitStack: 动态管理多个上下文管理器，无需深度嵌套 with 语句，非常适合在运行时决定需要打开多少资源。
# nullcontext: 空的上下文管理器。当某个条件不满足、不需要上下文管理时，用作 with 语句的占位符，避免 if-else 分支。
# suppress: 用于抑制（忽略）指定的异常，替代 try...except...pass 的冗长写法。

from dataclasses import dataclass, field
# 导入 dataclass 装饰器和 field 函数。
# field 用于自定义数据类字段的生成行为，最常用的是 default_factory，
# 用于解决 Python 中“可变默认参数（如 [] 或 {}）在多个实例间共享”的经典陷阱。

from enum import Enum, auto
# 导入 Enum 用于创建枚举类，保证值的唯一性和可读性。
# auto 用于让枚举成员自动分配唯一的整数值（1, 2, 3...）。

from functools import partial
# 导入 partial：用于“冻结”函数的部分参数，返回一个新的可调用对象（偏函数）。
# 常用于回调函数中提前绑定一些上下文参数。

from pathlib import Path
# 导入 Path：提供面向对象的文件系统路径操作。
# 替代传统的 os.path 字符串操作，支持 / 运算符拼接路径，更加 Pythonic。

from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar, cast
# 导入 typing 模块中的类型提示工具：
# TYPE_CHECKING: 一个特殊的常量，在运行时始终为 False，但在静态类型检查（如 mypy/pyright）时为 True。
#                用于将仅用于类型提示的导入包裹在 if TYPE_CHECKING: 块中，避免运行时的循环导入和不必要的内存占用。
# Any: 表示任意类型，放弃类型检查。
# Awaitable: 表示可以被 await 的对象（如协程、Future）。
# Callable: 表示可调用对象（函数、方法等），可指定参数和返回值类型，如 Callable[[int], str]。
# TypeVar: 用于定义泛型类型变量，使函数或类能够支持泛型编程。
# cast: 强制类型转换。仅在静态类型检查时生效，运行时直接返回原对象，用于“安抚”类型检查器。

from loguru import logger
# 导入 loguru 的 logger：一个功能强大、开箱即用且支持颜色高亮的第三方日志记录库。

# ==================== 内部模块导入 ====================
from nanobot.agent import context as agent_context
# 导入 agent 的上下文管理模块，并重命名为 agent_context 避免与局部变量冲突。

from nanobot.agent import model_presets as preset_helpers
# 导入模型预设（Model Presets）辅助工具模块。

from nanobot.agent.autocompact import AutoCompact
# 导入 AutoCompact：自动压缩/归档旧会话历史的组件，防止上下文窗口溢出。

from nanobot.agent.automation_turns import publish_next_deferred_turn
# 导入自动化轮次发布函数，用于处理延迟执行的自动化任务（如定时任务）。

from nanobot.agent.context import ContextBuilder
# 导入 ContextBuilder：负责构建发送给 LLM 的系统提示词、历史记忆和上下文信息。

from nanobot.agent.cron_turns import CronTurnCoordinator
# 导入 CronTurnCoordinator：定时任务（Cron Jobs）协调器，管理基于时间的触发器。

from nanobot.agent.hook import AgentHook, AgentTurnHookFactory
# 导入 Agent 钩子（Hook）接口和工厂，用于在 Agent 运行的各个阶段（如工具调用前后）注入自定义逻辑。

from nanobot.agent.memory import Consolidator
# 导入 Consolidator：记忆整合器，负责将长期记忆或摘要注入到上下文中。

from nanobot.agent.model_runtime import ModelRuntimeResolver
# 导入 ModelRuntimeResolver：模型运行时解析器，负责管理当前使用的 LLM 提供商、模型名称、上下文窗口大小等配置。

from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
# 导入 AgentRunner：实际执行 LLM 调用和工具执行的核心运行器。
# AgentRunSpec：运行器的配置规范数据类。
# _MAX_INJECTIONS_PER_TURN：单轮对话中允许注入的最大消息数量限制。

from nanobot.agent.subagent import SubagentManager
# 导入 SubagentManager：子 Agent 管理器，负责生成和管理并行的子 Agent 任务。

from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
# 导入请求上下文工具：
# RequestContext: 封装当前请求的元数据（如 channel, chat_id, workspace）。
# bind_request_context / reset_request_context: 使用 contextvars 将请求上下文绑定到当前协程，并在结束后重置。

from nanobot.agent.tools.exec_session import ExecSessionManager
# 导入 ExecSessionManager：管理代码执行（如 Python/Shell 沙盒）的会话生命周期。

from nanobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
# 导入文件状态管理工具：追踪单次会话中文件的读写操作，防止重复操作或冲突。

from nanobot.agent.tools.message import MessageTool
# 导入 MessageTool：一个特殊的工具，允许 Agent 主动向用户发送消息（而不是作为工具调用的返回值）。

from nanobot.agent.tools.registry import ToolRegistry
# 导入 ToolRegistry：工具注册表，存储所有可用的工具及其定义（JSON Schema）。

from nanobot.agent.tools.runtime_control import AgentRuntimeControl
# 导入 AgentRuntimeControl：允许 Agent 通过工具调用动态修改自身的运行时配置（如切换模型）。

from nanobot.agent.tools.self import MyTool
# 导入 MyTool：提供关于 Agent 自身状态查询的工具。

from nanobot.agent.turn_delivery import (
    TurnDelivery,
    TurnDeliveryFactory,
)
# 导入消息投递组件：
# TurnDelivery: 负责将 Agent 的响应流式传输或发送到具体的渠道（如 CLI, Telegram, Slack）。
# TurnDeliveryFactory: 根据消息来源创建对应的 TurnDelivery 实例。

from nanobot.agent.turn_delivery import TurnRoute as TurnRoute
# 导入 TurnRoute：定义消息的路由信息（如目标 channel, chat_id）。

from nanobot.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
# 导入钩子规范定义和构建函数，用于组装复杂的生命周期钩子链。

from nanobot.bus.events import InboundMessage, OutboundMessage
# 导入消息总线事件模型：
# InboundMessage: 入站消息（用户发给 Agent 的消息）。
# OutboundMessage: 出站消息（Agent 回复给用户的消息）。

from nanobot.bus.outbound_events import StreamedResponseEvent
# 导入流式响应事件，用于标记该回复是否通过流式传输（Streaming）发送。

from nanobot.bus.queue import MessageBus
# 导入 MessageBus：核心消息总线，负责在 Agent、网关、外部渠道之间异步路由消息。

from nanobot.bus.runtime_events import RuntimeEventBus
# 导入 RuntimeEventBus：运行时事件总线，用于发布状态变更（如模型切换、会话持久化完成）给前端 UI。

from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
# 导入命令路由系统：
# CommandContext: 命令执行上下文。
# CommandRouter: 解析和分发斜杠命令（如 /stop, /new）。
# register_builtin_commands: 注册系统内置的默认命令。

from nanobot.config.schema import AgentDefaults, ModelPresetConfig
# 导入配置 Schema 模型：
# AgentDefaults: Agent 的默认配置参数。
# ModelPresetConfig: 模型预设配置（如定义 "gpt-4o" 预设的具体参数）。

from nanobot.providers.base import LLMProvider, ProviderConversationState
# 导入 LLM 提供商基础接口：
# LLMProvider: 统一的 LLM API 调用抽象层。
# ProviderConversationState: 某些提供商（如 Anthropic/Claude）支持的对话状态恢复机制。

from nanobot.providers.factory import ProviderSnapshot
# 导入 ProviderSnapshot：提供商配置的快照，用于序列化或持久化当前使用的 Provider 状态。

from nanobot.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    RuntimeContextProvider,
    append_runtime_context,
    resolve_runtime_context,
    runtime_context_blocks_from_metadata,
)
# 导入运行时上下文注入工具：用于在提示词中动态注入当前时间、工作区路径、活跃文件等环境信息。

from nanobot.security.workspace_access import (
    WorkspaceScopeResolver,
    bind_workspace_scope,
    reset_workspace_scope,
)
# 导入工作区安全与沙盒作用域管理工具，限制 Agent 的文件读写范围。

from nanobot.session import turn_continuation
# 导入 turn_continuation：处理跨轮次的状态延续（如长文本分段生成、内部状态机继续）。

from nanobot.session.automation_turns import automation_history_overrides
# 导入自动化轮次的历史记录覆盖逻辑，防止自动化消息污染用户的历史视图。

from nanobot.session.goal_state import (
    goal_state_runtime_lines,
    runner_wall_llm_timeout_s,
    sustained_goal_active,
)
# 导入目标状态管理：支持 Agent 执行长期/持续性的目标（Sustained Goals），并动态调整超时时间。

from nanobot.session.history_visibility import HIDDEN_HISTORY_META
# 导入隐藏历史元数据标记，用于在历史记录中标记不应展示给最终用户的内部系统消息。

from nanobot.session.keys import UNIFIED_SESSION_KEY, remember_last_channel
# 导入会话键管理：
# UNIFIED_SESSION_KEY: 统一会话模式下的全局唯一键。
# remember_last_channel: 记录用户最后交互的渠道，以便在统一会话中正确路由回复。

from nanobot.session.manager import (
    Session,
    SessionManager,
    replay_max_messages_for_context,
)
# 导入会话管理器：
# Session: 单个会话的数据模型（包含历史消息、元数据、Provider 状态）。
# SessionManager: 负责会话的加载、保存、缓存和生命周期管理。
# replay_max_messages_for_context: 根据上下文窗口计算需要回放的最大历史消息数。

from nanobot.session.model_selection import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    model_preset_from_metadata,
)
# 导入会话级别的模型选择逻辑，允许单个会话绑定特定的模型预设。

from nanobot.triggers.local_turns import LocalTriggerTurnCoordinator
# 导入本地触发器协调器：处理基于本地文件系统或事件触发的自动化任务。

from nanobot.utils.cancellation import task_is_cancelling
# 导入任务取消检测工具，用于优雅地处理 asyncio 任务的取消信号。

from nanobot.utils.document import reference_non_image_attachments
# 导入文档附件处理工具：将非图片附件（如 PDF、TXT）转换为 LLM 可读的文本引用。

from nanobot.utils.helpers import image_placeholder_text
# 导入辅助函数：生成图片占位符文本（当图片无法直接传入 LLM 时）。

from nanobot.utils.helpers import truncate_text as truncate_text_fn
# 导入文本截断函数，防止超长的工具返回结果撑爆上下文窗口。

from nanobot.utils.llm_runtime import LLMRuntime
# 导入 LLMRuntime：不可变的运行时配置数据类，包含当前使用的 Provider、模型、Token 限制等。

from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
)
# 导入运行时常量：EMPTY_FINAL_RESPONSE_MESSAGE，当 Agent 决定不回复时使用的默认占位符。

# ==================== 仅在类型检查时导入的模块 ====================
if TYPE_CHECKING:
    # TYPE_CHECKING 在运行时为 False，因此这些导入不会在运行时执行，避免了循环导入和性能损耗。
    # 它们仅用于为 IDE 和 mypy/pyright 提供类型提示。
    from nanobot.agent.tools.mcp import MCPConnection
    # MCP (Model Context Protocol) 连接对象类型。
    from nanobot.config.schema import (
        ChannelsConfig,
        Config,
        MCPServerConfig,
        ProviderConfig,
        ToolsConfig,
    )
    # 全局配置、渠道配置、MCP 服务器配置、Provider 配置、工具配置的 Schema 类型。
    from nanobot.cron.service import CronService
    # 定时任务后台服务类型。
    from nanobot.triggers.local_store import LocalTriggerStore
    # 本地触发器持久化存储类型。

# 定义一个泛型类型变量 _T，用于在方法签名中表示任意返回类型。
_T = TypeVar("_T")

# 定义一个常量字符串，用于在 Provider 状态元数据中标记子 Agent 的任务 ID。
_SUBAGENT_PROVIDER_TASK_META = "subagent_provider_task_id"


class TurnKind(Enum):
    """
    枚举类：定义 Agent 轮次（Turn）的类型。
    """
    USER = auto()   # 用户发起的轮次（包含人类用户的输入，或外部渠道的消息）
    SYSTEM = auto() # 系统发起的轮次（如定时任务、子 Agent 返回结果、内部自动化触发）


@dataclass
class TurnContext:
    """
    数据类：封装单次 Agent 轮次（Turn）执行过程中的所有上下文状态。
    它作为参数在 Agent 的各个处理阶段（如 restore, build, run, save）之间传递。
    """
    msg: InboundMessage
    # 触发本次轮次的原始入站消息对象。
    
    session_key: str
    # 当前会话的唯一标识符（如 "cli:direct" 或 "telegram:12345"）。
    
    turn_id: str
    # 本次轮次的唯一 ID（通常由 session_key 和纳秒级时间戳组成），用于日志追踪和事件关联。
    
    runtime: LLMRuntime | None
    # 当前轮次使用的 LLM 运行时配置（包含 Provider、模型名称、上下文窗口大小等）。
    # 使用 `| None` 表示在 BUILD 阶段之前可能尚未解析出运行时配置。
    
    kind: TurnKind
    # 轮次类型（USER 或 SYSTEM）。
    
    delivery: TurnDelivery
    # 消息投递对象，负责将 Agent 的输出流式传输或发送到具体的渠道（如 Telegram, CLI）。
    
    original_user_text: str | None = None
    # 用户输入的原始纯文本。如果是系统消息或内部续传，则为 None。
    
    session: Session | None = None
    # 当前会话的持久化对象（包含历史记录、元数据）。在 RESTORE 阶段被填充。

    history: list[dict[str, Any]] = field(default_factory=list)
    # 提取出的、准备发送给 LLM 的历史消息列表。
    # 【Python 特殊用法】: 使用 `field(default_factory=list)` 而不是 `default=[]`。
    # 这是因为 Python 的默认参数是在函数定义时求值的，如果使用 `[]`，所有 TurnContext 实例将共享同一个列表，导致严重的数据污染。
    
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    # 构建好的、包含 System Prompt 和最新用户消息的完整初始消息列表，直接传给 LLM。
    
    provider_state: ProviderConversationState | None = field(default=None, repr=False)
    # 某些 LLM Provider（如 Anthropic）支持的对话状态对象，用于在多次请求间维持内部缓存（如 Prompt Caching）。
    # `repr=False` 表示在打印该对象时忽略此字段，避免输出过长的二进制/JSON 数据。
    
    request_context: RequestContext | None = None
    # 绑定到当前协程的请求上下文，包含工作区路径、渠道信息等，供工具调用时读取。
    
    runtime_context_blocks: list[RuntimeContextBlock] = field(default_factory=list)
    # 动态解析出的运行时上下文块（如当前时间、活跃文件列表），将被注入到 System Prompt 中。
    
    attributes: dict[str, Any] = field(default_factory=dict)
    # 附加的自定义属性字典，用于在钩子（Hooks）或事件总线中传递额外数据。

    final_content: str | None = None
    # Agent 最终生成的完整文本响应内容。
    
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    # 本次轮次中产生的所有消息（包括用户输入、Assistant 思考、工具调用、工具结果）。
    
    stop_reason: str = ""
    # LLM 停止生成的原因（如 "stop", "max_iterations", "tool_calls", "error"）。
    
    had_injections: bool = False
    # 标记在本次轮次中是否发生了“消息注入”（如用户在 Agent 思考时发送了新消息，或子 Agent 返回了结果）。
    
    streamed_content: bool = False
    # 标记本次响应是否已经通过流式传输（Streaming）发送给了用户。
    
    input_persisted_early: bool = False
    # 标记用户的输入消息是否在轮次开始前就被提前持久化到了 Session 历史中（常用于命令处理或防崩溃恢复）。
    
    save_skip: int = 0
    # 在保存历史记录时，需要跳过的消息数量（例如跳过已经提前持久化的用户消息）。

    outbound: OutboundMessage | None = None
    # 最终组装好的、准备发送到消息总线的出站消息对象。
    
    suppress_response: bool = False
    # 标记是否应该抑制（丢弃）最终的响应（例如执行了 /stop 命令，或工具已经主动发送了消息）。

    on_progress: Callable[..., Awaitable[None]] | None = None
    # 进度回调函数（异步），用于向 UI 报告工具调用的进度。
    
    on_stream: Callable[[str], Awaitable[None]] | None = None
    # 流式输出回调函数（异步），每当 LLM 生成一个文本 delta 时调用。
    
    on_stream_end: Callable[..., Awaitable[None]] | None = None
    # 流式结束回调函数（异步），当一段流式输出完成时调用。
    
    on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None
    # 运行时准入回调，当确定了本次使用的 LLM 模型和配置后触发。
    
    on_retry_wait: Callable[[str], Awaitable[None]] | None = None
    # 重试等待回调，当 LLM API 限流或报错，Agent 决定等待重试时触发。

    pending_queue: asyncio.Queue[InboundMessage] | None = None
    # 待处理消息队列。如果用户在 Agent 思考时发送了新消息，新消息会被放入此队列，
    # 以便在当前轮次结束时作为“注入消息”无缝衔接给 LLM。
    
    pending_summary: str | None = None
    # 如果会话历史太长被压缩（AutoCompact），这里存储压缩后的摘要文本。

    ephemeral: bool = False
    # 标记本次轮次是否为“临时/短暂”的（如系统内部心跳、不记录到持久化历史中的操作）。
    
    run_extra_hooks_for_ephemeral: bool = False
    # 标记即使是临时轮次，是否也要强制运行额外的钩子。
    
    hooks: list[AgentHook] = field(default_factory=list)
    # 本次轮次专属的钩子列表。
    
    hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)
    # 本次轮次专属的钩子工厂列表。
    
    turn_scopes: list[AbstractContextManager[Any]] = field(default_factory=list)
    # 本次轮次需要管理的上下文管理器列表（如临时的环境变量覆盖、沙盒挂载），将在轮次结束时自动清理。
    
    tools: ToolRegistry | None = None
    # 本次轮次可用的工具注册表（允许在特定轮次中禁用或启用特定工具）。

    turn_wall_started_at: float = field(default_factory=time.time)
    # 整个轮次（包含排队等待时间）开始的绝对时间戳（Wall-clock time）。
    
    visible_run_started_at: float | None = None
    # 实际开始调用 LLM 或执行工具的时间戳（用于计算对用户可见的延迟）。
    
    turn_latency_ms: int | None = None
    # 本次轮次的总延迟（毫秒），用于日志记录和 UI 展示。

    def require_runtime(self) -> LLMRuntime:
        """
        获取由 BUILD 阶段建立的运行时配置。
        如果尚未初始化，则抛出 RuntimeError，确保阶段执行的顺序性。
        """
        if self.runtime is None:
            raise RuntimeError("turn runtime is not initialized; BUILD must run before this stage")
        return self.runtime

    def require_session(self) -> Session:
        """
        获取由 RESTORE 阶段建立的会话对象。
        如果尚未初始化，则抛出 RuntimeError，确保阶段执行的顺序性。
        """
        if self.session is None:
            raise RuntimeError("turn session is not initialized; RESTORE must run before this stage")
        return self.session


class AgentLoop:
    """
    Agent 循环是核心处理引擎。

    它的主要职责：
    1. 从消息总线（MessageBus）接收消息。
    2. 结合历史、记忆、技能构建上下文（Context）。
    3. 调用 LLM 生成响应或工具调用指令。
    4. 执行工具调用（Tool Calls）。
    5. 将最终响应发送回消息总线。
    """

    @property
    def current_iteration(self) -> int:
        """
        【属性装饰器】获取当前 Agent 循环的迭代次数（即工具调用的轮数）。
        使用 @property 允许像访问属性一样调用方法（如 loop.current_iteration 而不是 loop.current_iteration()）。
        """
        return self._current_iteration

    @property
    def tool_names(self) -> list[str]:
        """获取当前注册的所有工具的名称列表。"""
        return self.tools.tool_names

    @property
    def last_usage(self) -> Mapping[str, int]:
        """获取最新的聚合 Token 使用情况（如 prompt_tokens, completion_tokens），通过运行时控制快照暴露。"""
        return self._last_usage

    @property
    def provider(self) -> LLMProvider:
        """获取当前选中的 LLM 提供商实例（用于未来的轮次准入）。"""
        return self.runtime_resolver.runtime.provider

    @property
    def model(self) -> str:
        """获取当前选中的 LLM 模型名称（如 "gpt-4o"）。"""
        return self.runtime_resolver.runtime.model

    @property
    def context_window_tokens(self) -> int:
        """获取当前配置的上下文窗口大小限制（Token 数）。"""
        return self.runtime_resolver.runtime.context_window_tokens

    @property
    def model_presets(self) -> Mapping[str, ModelPresetConfig]:
        """获取所有已配置的模型预设字典，供 UI 选择和展示。"""
        return self.runtime_resolver.model_presets

    @property
    def model_preset(self) -> str | None:
        """获取当前激活的模型预设名称。"""
        return self.runtime_resolver.model_preset

    @model_preset.setter
    def model_preset(self, name: str | None) -> None:
        """
        【Setter 装饰器】允许通过赋值操作（如 loop.model_preset = "gpt-4o"）来切换模型预设。
        内部实际调用 set_model_preset 方法。
        """
        self.set_model_preset(name)

    def llm_runtime(self) -> LLMRuntime:
        """
        解析并返回用于准入下一个轮次的不可变默认运行时配置。
        如果解析出的新配置与旧配置不同（如模型改变），则发布运行时选择事件通知 UI。
        """
        previous = self.runtime_resolver.runtime
        # 调用 admit() 获取当前默认的运行时快照
        runtime = self.runtime_resolver.admit()
        if (
            runtime.model != previous.model
            or runtime.model_preset != previous.model_preset
            or runtime.snapshot_signature != previous.snapshot_signature
        ):
            # 如果模型、预设或提供商快照签名发生变化，发布事件
            self._publish_runtime_selection(runtime)
        return runtime

    def dream_runtime(self) -> LLMRuntime | None:
        """
        解析用于 Dream（梦境/后台思考）模式的可选预设配置，但不改变默认配置。
        Dream 模式通常用于 Agent 在空闲时进行长期记忆整合或后台推理。
        """
        if not self.dream_model_preset:
            return None
        return self.runtime_resolver.resolve_preset(self.dream_model_preset)

    # ==================== 类级别常量 ====================
    # 用于在 Session 的 metadata 字典中存储内部状态的键名。
    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint" # 存储中断时的运行时检查点（用于崩溃恢复）
    _PENDING_USER_TURN_KEY = "pending_user_turn"   # 标记是否有等待处理的用户轮次
    _PROVIDER_STATE_CHECKPOINT_VERSION_KEY = "provider_state_checkpoint_version" # Provider 状态检查点版本号键
    _PROVIDER_STATE_CHECKPOINT_VERSION = "v1"      # 当前 Provider 状态检查点版本号

    def __init__(
        self,
        bus: MessageBus,
        # 消息总线实例，用于在 Agent、网关和外部渠道之间路由入站和出站消息。
        provider: LLMProvider,
        # 默认的 LLM 提供商实例（如 OpenAI, Anthropic, Ollama 等），提供基础的模型调用能力。
        workspace: Path,
        # 工作区路径（Pathlib Path 对象），Agent 在此目录下执行文件读写和代码运行。
        model: str | None = None,
        # 初始使用的模型名称（如 "gpt-4o"）。如果为 None，则由 provider 决定默认模型。
        max_iterations: int | None = None,
        # 单次对话轮次中，允许 Agent 连续调用工具的最大迭代次数，防止死循环。
        max_concurrent_subagents: int | None = None,
        # 允许同时并发运行的子 Agent 最大数量。
        context_window_tokens: int | None = None,
        # LLM 的上下文窗口大小（Token 数量限制），用于计算历史消息截断和压缩策略。
        context_block_limit: int | None = None,
        # 注入到 Prompt 中的运行时上下文块（如文件列表、系统信息）的最大数量限制。
        max_tool_result_chars: int | None = None,
        # 工具返回结果的最大字符数。超过此长度的结果将被截断，防止撑爆上下文窗口。
        fail_on_tool_error: bool | None = None,
        # 如果为 True，当工具执行抛出异常时，直接终止当前轮次并向用户报错；否则将错误信息作为工具结果返回给 LLM 让其自我修正。
        provider_retry_mode: str = "standard",
        # 提供商 API 调用失败时的重试策略（如 "standard", "aggressive" 等）。
        tool_hint_max_length: int | None = None,
        # 工具提示（Tool Hint）在 UI 中显示的最大字符长度。
        cron_service: CronService | None = None,
        # 定时任务（Cron）后台服务实例，用于处理基于时间的自动化触发器。
        restrict_to_workspace: bool = False,
        # 安全沙盒配置：是否严格限制 Agent 的文件操作只能在 workspace 目录内进行。
        session_manager: SessionManager | None = None,
        # 会话管理器实例，负责会话的持久化、加载和缓存。如果为 None，则使用默认的基于文件的管理器。
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        # Model Context Protocol (MCP) 服务器配置字典，用于连接外部工具和数据源。
        channels_config: ChannelsConfig | None = None,
        # 渠道（如 Telegram, Discord, CLI）的配置信息，用于消息路由和格式化。
        timezone: str | None = None,
        # Agent 所在的时区（如 "Asia/Shanghai"），用于在 Prompt 中注入准确的当前时间。
        session_ttl_minutes: int = 0,
        # 会话的存活时间（Time-To-Live，分钟）。超过此时间未活动的会话将被自动归档或压缩以节省资源。0 表示不自动过期。
        consolidation_ratio: float = 0.5,
        # 记忆整合触发比例。当历史 Token 占用达到上下文窗口的此比例时，触发后台记忆压缩/摘要。
        hooks: list[AgentHook] | None = None,
        # 全局 Agent 钩子列表，在每次轮次的特定生命周期阶段（如工具调用前）执行。
        hook_factories: list[AgentTurnHookFactory] | None = None,
        # 钩子工厂列表，用于根据每次轮次的上下文动态生成特定的钩子。
        unified_session: bool = False,
        # 是否启用“统一会话”模式。启用后，同一用户在不同渠道（如 CLI 和 Web）的对话将共享同一个 Session 和历史记录。
        disabled_skills: list[str] | None = None,
        # 被禁用的技能（Skill）或工具名称列表。
        tools_config: ToolsConfig | None = None,
        # 工具系统的全局配置对象，包含各个工具的开关和参数。
        image_generation_provider_config: ProviderConfig | None = None,
        # 图像生成提供商的单一配置（向后兼容）。
        image_generation_provider_configs: dict[str, ProviderConfig] | None = None,
        # 图像生成提供商配置字典，支持配置多个不同的图像生成后端（如 DALL-E, Midjourney）。
        provider_snapshot_loader: Callable[..., ProviderSnapshot] | None = None,
        # 回调函数：用于加载或刷新 LLM 提供商的配置快照。
        provider_signature: tuple[object, ...] | None = None,
        # 提供商配置的签名（元组），用于检测配置是否发生变化，从而决定是否需要重新初始化 Provider。
        model_presets: dict[str, ModelPresetConfig] | None = None,
        # 模型预设配置字典（如 "fast", "smart", "cheap" 对应的模型和参数组合）。
        preset_catalog_loader: preset_helpers.PresetCatalogLoader | None = None,
        # 回调函数：用于动态加载预设目录或远程预设列表。
        model_preset: str | None = None,
        # 初始激活的模型预设名称。
        dream_model_preset: str | None = None,
        # 用于 Dream（后台/空闲思考）模式的特定模型预设。
        preset_snapshot_loader: preset_helpers.PresetSnapshotLoader | None = None,
        # 回调函数：用于加载预设的快照状态。
        runtime_events: RuntimeEventBus | None = None,
        # 运行时事件总线，用于向外部（如 Web UI）推送状态变更事件（如 "模型已切换", "正在思考"）。
        turn_delivery_factory: TurnDeliveryFactory | None = None,
        # 消息投递工厂，负责根据入站消息创建对应的出站消息投递通道（如流式输出到 Telegram）。
        runtime_model_publisher: Callable[[str, str | None], None] | None = None,
        # 回调函数：当运行时模型发生变更时调用，用于通知外部系统（如更新 UI 上的模型显示）。
        restart_mode: str = "auto",
        # 重启模式配置，决定在发生严重错误或配置更新时 Agent 循环如何重启。
        local_trigger_store: LocalTriggerStore | None = None,
        # 本地触发器存储后端，用于持久化基于文件系统或本地事件的自动化规则。
        idle_compact_check_interval_seconds: int = 0,
        # 空闲会话压缩检查的间隔时间（秒）。Agent 循环在空闲时会定期扫描并压缩过期的会话历史。
    ):
        # 延迟导入 ToolsConfig 以避免循环导入问题，并在未提供配置时创建默认实例。
        from nanobot.config.schema import ToolsConfig

        # 【Python 特殊用法】: `or` 短路逻辑
        # 如果传入了 tools_config 则使用它，否则实例化一个空的默认 ToolsConfig。
        _tc = tools_config or ToolsConfig()
        # 获取 Agent 的默认配置参数集合（如默认的 max_iterations 等）。
        defaults = AgentDefaults()
        
        # 将传入的核心依赖绑定到实例属性。
        self.bus = bus
        
        # 初始化消息投递和运行时事件总线。
        if turn_delivery_factory is not None:
            # 如果外部传入了自定义的投递工厂，进行一致性校验。
            if turn_delivery_factory.bus is not bus:
                raise ValueError("turn delivery factory must use the agent message bus")
            if (
                runtime_events is not None
                and turn_delivery_factory.runtime_events is not runtime_events
            ):
                raise ValueError("turn delivery factory must use the agent runtime event bus")
            self.turn_delivery_factory = turn_delivery_factory
            self.runtime_events = turn_delivery_factory.runtime_events
        else:
            # 否则，使用传入的 runtime_events 或创建一个新的默认实例。
            self.runtime_events = runtime_events or RuntimeEventBus()
            # 使用 MessageBus 和 RuntimeEventBus 初始化默认的 TurnDeliveryFactory。
            self.turn_delivery_factory = TurnDeliveryFactory(bus, self.runtime_events)
            
        # 提取事件发布器，用于后续快速发布运行时事件。
        self.runtime_event_publisher = self.turn_delivery_factory.runtime_event_publisher
        self.channels_config = channels_config
        self.restart_mode = restart_mode
        self._runtime_model_publisher = runtime_model_publisher
        self.workspace = workspace
        
        # 确定初始使用的模型：优先使用传入的 model，否则调用 provider 获取其默认模型。
        initial_model = model or provider.get_default_model()
        
        # 确定最大工具迭代次数：优先使用传入值，否则使用默认配置。
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        
        # 确定初始上下文窗口大小。
        initial_context_window = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        
        # 获取配置的模型预设字典，如果未提供则为空字典。
        configured_presets = model_presets or {}
        
        # 初始化模型运行时解析器（ModelRuntimeResolver）。
        # 它负责管理当前的 LLMRuntime（包含 provider, model, context_window 等不可变配置），
        # 并支持在运行时动态切换模型或预设。
        self.runtime_resolver = ModelRuntimeResolver(
            # 捕获初始的 LLMRuntime 快照。
            LLMRuntime.capture(
                provider,
                initial_model,
                context_window_tokens=initial_context_window,
                snapshot_signature=provider_signature,
            ),
            model_presets=configured_presets,
            preset_catalog_loader=preset_catalog_loader,
            configured_default_preset=model_preset,
            provider_snapshot_loader=provider_snapshot_loader,
            preset_snapshot_loader=preset_snapshot_loader,
        )
        self.dream_model_preset = dream_model_preset
        self.context_block_limit = context_block_limit
        
        # 设置工具返回结果的最大字符数限制。
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        
        # 设置工具提示的最大长度。
        self.tool_hint_max_length = (
            tool_hint_max_length if tool_hint_max_length is not None
            else defaults.tool_hint_max_length
        )
        
        # 保存工具配置及常用的子配置（web 搜索、代码执行）。
        self.tools_config = _tc
        self.web_config = _tc.web
        self.exec_config = _tc.exec
        
        # 处理图像生成提供商配置。
        # 将传入的字典转换为新的字典以避免修改外部传入的原始对象。
        self._image_generation_provider_configs = dict(image_generation_provider_configs or {})
        if (
            image_generation_provider_config is not None
            and "openrouter" not in self._image_generation_provider_configs
        ):
            # 向后兼容：如果提供了单一的 image_generation_provider_config，
            # 且字典中没有 "openrouter" 键，则将其作为 "openrouter" 的默认配置注入。
            self._image_generation_provider_configs["openrouter"] = image_generation_provider_config
            
        self.cron_service = cron_service
        self.local_trigger_store = local_trigger_store
        self.restrict_to_workspace = restrict_to_workspace
        
        # 初始化工作区作用域解析器，用于处理文件读写的安全沙盒边界。
        self.workspace_scopes = WorkspaceScopeResolver(
            default_workspace=workspace,
            default_restrict_to_workspace=restrict_to_workspace,
        )
        
        # 记录 Agent 循环启动的绝对时间。
        self._start_time = time.time()
        # 初始化一个字典，用于记录最近一次轮次的 Token 使用量统计。
        self._last_usage: dict[str, int] = {}
        # 保存全局钩子和钩子工厂列表，如果未提供则初始化为空列表。
        self._extra_hooks: list[AgentHook] = hooks or []
        self._hook_factories: list[AgentTurnHookFactory] = hook_factories or []

        # 初始化上下文构建器，负责组装 System Prompt、提取工作区文件结构等。
        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        
        # 初始化会话管理器，如果未传入则使用默认的文件存储会话管理器。
        self.sessions = session_manager or SessionManager(workspace)
        # 为会话管理器设置文件容量归档回调，当会话历史文件过大时触发记忆归档。
        self.sessions.set_file_cap_archiver(self.context.memory.raw_archive)
        
        # 初始化工具注册表，用于存储和查找所有可用的工具。
        self.tools = ToolRegistry()
        
        # 初始化文件状态存储。每个逻辑会话拥有一个文件读写追踪器。
        # 由于工具注册表是全局共享的，工具在执行时通过 contextvars 获取当前会话的活跃状态。
        self._file_state_store = FileStateStore()
        
        # 初始化代码执行会话管理器，管理 Python/Shell 等沙盒进程的生命周期。
        self._exec_session_manager = ExecSessionManager()
        
        # 初始化 AgentRunner，它是实际调用 LLM 并执行工具循环的核心组件。
        self.runner = AgentRunner()
        
        # 初始化子 Agent 管理器，负责派生和监控并行的子 Agent 任务。
        self.subagents = SubagentManager(
            workspace=workspace,
            bus=bus,
            tools_config=_tc,
            max_tool_result_chars=self.max_tool_result_chars,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
            max_iterations=self.max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            fail_on_tool_error=fail_on_tool_error,
            # 【Python 特殊用法】: lambda 表达式
            # 传入一个 lambda 函数，用于动态获取特定会话的 LLM 墙超时时间。
            llm_wall_timeout_for_session=lambda sk: runner_wall_llm_timeout_s(self.sessions, sk),
        )
        
        self._unified_session = unified_session
        self._running = False # 标记 Agent 主循环是否正在运行的状态标志。
        self._mcp_servers = mcp_servers or {} # MCP 服务器配置字典。
        self._mcp_stacks: dict[str, MCPConnection] = {} # 存储已连接的 MCP 上下文管理器栈。
        self._mcp_connecting = False # 标记是否正在进行 MCP 连接操作，防止并发连接。
        self._runtime_context_providers: list[RuntimeContextProvider] = [] # 动态运行时上下文提供者列表。
        self._active_tasks: dict[str, set[asyncio.Task[Any]]] = {} # 按 session_key 跟踪当前正在执行的 asyncio 任务集合。
        self._discarding_sessions: set[str] = set() # 记录正在被丢弃/清理的会话键，防止在清理期间被重新激活。
        self._background_tasks: set[asyncio.Task[Any]] = set() # 跟踪所有后台任务（如记忆整合），以便在关闭时优雅取消。
        self._close_mcp_lock = asyncio.Lock() # 异步锁，确保关闭 MCP 连接时的线程/协程安全。
        
        # 【Python 特殊用法】: weakref.WeakValueDictionary
        # 创建一个弱引用字典来存储每个 session 的 asyncio.Lock。
        # 作用：当某个 session 长时间不活跃且从内存中清除时，对应的 Lock 对象也会自动被垃圾回收，
        # 避免全局字典无限增长导致内存泄漏。
        self._session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        
        # 为每个 session 维护一个待处理消息队列，用于实现“中途消息注入”（Mid-turn message injection）。
        # 当 session 正在处理一个耗时任务时，用户发来的新消息会被放入此队列，
        # 并在当前 LLM 思考或工具执行完毕后，作为新的上下文注入给 LLM。
        self._pending_queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        
        # 存储被延迟执行的自动化轮次消息（如定时任务在 Agent 繁忙时触发）。
        self._deferred_automation_turns: dict[str, list[InboundMessage]] = {}
        
        # 初始化定时任务（Cron）协调器。
        self._cron_turns = CronTurnCoordinator(
            publish_inbound=self.bus.publish_inbound, # 将消息发布回总线的回调
            dispatch=self._dispatch,                 # 直接分发消息给处理逻辑的回调
            is_running=lambda: self._running,        # 检查主循环是否存活的 lambda
            deferred_queues=self._deferred_automation_turns, # 延迟队列引用
        )
        
        # 初始化本地触发器协调器（处理基于本地文件变更等事件的自动化）。
        self._local_trigger_turns = LocalTriggerTurnCoordinator(
            publish_inbound=self.bus.publish_inbound,
            dispatch=self._dispatch,
            is_running=lambda: self._running,
            deferred_queues=self._deferred_automation_turns,
        )
        
        # 将所有的自动化协调器打包成元组，方便后续统一遍历处理。
        self._automation_turn_coordinators = (
            ("cron", self._cron_turns),
            ("local trigger", self._local_trigger_turns),
        )
        
        # 【环境变量读取】: 获取最大并发请求数限制。
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 表示无限制；默认值为 3。
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        
        # 【Python 特殊用法】: asyncio.Semaphore (信号量)
        # 如果 _max > 0，则创建一个信号量，用于限制同时处理的入站消息总数，防止系统过载。
        # 如果 <= 0，则设为 None，后续代码中使用 nullcontext() 替代。
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        
        # 初始化记忆整合器（Consolidator），负责在上下文过长时调用 LLM 进行历史摘要和压缩。
        self.consolidator = Consolidator(
            store=self.context.memory,
            sessions=self.sessions,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            consolidation_ratio=consolidation_ratio,
            unified_session=unified_session,
        )
        
        # 初始化自动压缩组件，负责定期清理和归档过期的闲置会话。
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self._idle_compact_check_interval_s = idle_compact_check_interval_seconds
        # 【Python 特殊用法】: time.monotonic()
        # 记录下一次空闲压缩检查的时间点。monotonic 时钟不受系统时间调整（如 NTP 同步）的影响，适合计算时间间隔。
        self._next_idle_compact_check_at = time.monotonic()
        
        # 如果指定了初始模型预设，则应用它，但不立即发布更新事件（因为系统还在初始化中）。
        if model_preset:
            self.set_model_preset(model_preset, publish_update=False)
            
        # 注册系统默认的工具集（如文件读写、代码执行、网络搜索等）。
        self._register_default_tools(provider_snapshot_loader=provider_snapshot_loader)
        
        # 初始化当前迭代计数器为 0。
        self._current_iteration: int = 0
        
        # 初始化命令路由器，并注册内置的斜杠命令（如 /stop, /help, /new）。
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)
    @classmethod
    def from_config(
        cls,
        config: Config,
        bus: MessageBus | None = None,
        **extra: Any,
    ) -> AgentLoop:
        """
        【类方法 / 工厂模式】从配置对象创建 AgentLoop 实例。
        
        @classmethod: 类方法装饰器。使得该方法可以通过类名直接调用（如 AgentLoop.from_config(...)），
                      第一个参数 cls 代表类本身，而不是实例 self。常用于实现替代构造函数。
        **extra: Any: 关键字参数收集器。允许调用者传入任意数量的额外命名参数，
                      这些参数会被收集到字典 extra 中，并在最后透传给 __init__，
                      从而允许外部覆盖或扩展从标准配置派生的参数（如注入自定义的 cron_service）。
        """
        # 延迟导入 Provider 工厂，避免在模块加载时产生循环依赖。
        from nanobot.providers.factory import make_provider

        # 如果未提供消息总线，则实例化一个默认的内存消息总线。
        if bus is None:
            bus = MessageBus()
            
        # 获取 Agent 的默认配置块。
        defaults = config.agents.defaults
        
        # 【Python 特殊用法】: dict.pop(key, default)
        # 从 extra 字典中提取并移除 "provider" 键。
        # 如果调用者通过 extra 传入了自定义的 provider，则使用它；
        # 否则（返回 None），使用 make_provider(config) 根据配置文件动态创建默认的 Provider。
        provider = extra.pop("provider", None) or make_provider(config)
        
        # 解析配置中的预设（Preset），获取默认的模型名称和上下文窗口大小。
        resolved = config.resolve_preset()
        # 同样使用 pop 和 or 短路逻辑，允许外部通过 extra 覆盖模型和上下文窗口配置。
        model = extra.pop("model", None) or resolved.model
        context_window_tokens = extra.pop("context_window_tokens", None) or resolved.context_window_tokens
        
        # 提取快照加载器回调函数。
        provider_snapshot_loader = extra.pop("provider_snapshot_loader", None)
        preset_snapshot_loader = extra.pop("preset_snapshot_loader", None) or preset_helpers.make_preset_snapshot_loader(
            config,
            provider_snapshot_loader,
        )
        
        # 调用类的构造函数（cls 即 AgentLoop），将解析出的配置和 extra 中剩余的参数解包传入。
        return cls(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=model,
            max_iterations=defaults.max_tool_iterations,
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            context_window_tokens=context_window_tokens,
            context_block_limit=defaults.context_block_limit,
            max_tool_result_chars=defaults.max_tool_result_chars,
            fail_on_tool_error=defaults.fail_on_tool_error,
            provider_retry_mode=defaults.provider_retry_mode,
            tool_hint_max_length=defaults.tool_hint_max_length,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            idle_compact_check_interval_seconds=defaults.idle_compact_check_interval_seconds,
            consolidation_ratio=defaults.consolidation_ratio,
            tools_config=config.tools,
            model_presets=preset_helpers.configured_model_presets(config),
            model_preset=defaults.model_preset,
            dream_model_preset=defaults.dream.model_override,
            restart_mode=config.gateway.restart_mode,
            provider_snapshot_loader=provider_snapshot_loader,
            preset_snapshot_loader=preset_snapshot_loader,
            # 【Python 特殊用法】: **extra 解包
            # 将 extra 字典中剩余的所有键值对解包为关键字参数传递给 __init__。
            **extra,
        )

    def _sync_subagent_runtime_limits(self) -> None:
        """
        保持子 Agent 的运行时限制与主循环的可变设置同步。
        例如，如果用户在运行时动态修改了最大工具迭代次数，子 Agent 也需要遵守新的限制。
        """
        self.subagents.max_iterations = self.max_iterations

    def invalidate_runtime_config(self) -> None:
        """
        使运行时配置失效，并通知客户端（如 Web UI）刷新其模型目录。
        通常在配置文件发生热更新或 Provider 状态改变时调用。
        """
        self.runtime_resolver.invalidate()
        self._publish_runtime_selection(self.runtime_resolver.runtime)

    def runtime_for_session(
        self,
        session: Session,
        *,
        recover_removed: bool = True,
    ) -> LLMRuntime:
        """
        解析特定会话绑定的不可变运行时配置。
        如果会话元数据中指定了特定的模型预设，则使用该预设；否则回退到全局默认配置。
        * (星号) 强制要求后续参数必须作为关键字参数传递，提高代码可读性。
        """
        # 从会话元数据中提取绑定的模型预设名称。
        name = model_preset_from_metadata(session.metadata)
        if name is None:
            # 如果没有绑定特定预设，返回全局默认的 LLM 运行时。
            return self.llm_runtime()
        try:
            # 尝试解析指定的预设配置。
            return self.runtime_resolver.resolve_preset(name)
        except KeyError:
            # 如果预设已被删除或不存在：
            if not recover_removed or name in self.runtime_resolver.model_presets:
                raise # 如果不允许恢复，或者预设实际上存在（引发其他错误），则抛出异常。
                
            # 记录警告日志，说明会话引用了已删除的预设，将回退到默认配置。
            logger.warning(
                "Session '{}' references removed model preset '{}'; falling back to default",
                session.key,
                name,
            )
            # 从会话元数据中移除无效的预设键。
            session.metadata.pop(SESSION_MODEL_PRESET_METADATA_KEY, None)
            # 持久化更新后的会话。
            self.sessions.save(session)
            # 返回全局默认运行时。
            return self.llm_runtime()

    def set_session_model_preset(
        self,
        session_key: str,
        name: str,
    ) -> LLMRuntime:
        """
        验证并持久化单个会话的模型预设选择。
        允许用户通过命令或 UI 为特定对话切换模型。
        """
        # 解析并验证预设名称是否有效，如果无效这里会抛出异常。
        runtime = self.runtime_resolver.resolve_preset(name)
        # 获取或创建会话对象。
        session = self.sessions.get_or_create(session_key)
        # 将新的预设名称写入会话元数据。
        session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = runtime.model_preset
        # 保存会话到持久化存储。
        self.sessions.save(session)
        return runtime

    def _publish_runtime_selection(
        self,
        runtime: LLMRuntime,
        *,
        publish_update: bool = True,
    ) -> None:
        """
        向外部系统（如 UI 或网关）发布运行时模型变更事件。
        """
        if not publish_update:
            return
        # 如果配置了自定义的模型发布回调，则调用它。
        if self._runtime_model_publisher is not None:
            self._runtime_model_publisher(runtime.model, runtime.model_preset)
        # 通过运行时事件总线发布模型变更事件。
        self.runtime_event_publisher.runtime_model_changed(
            runtime.model,
            runtime.model_preset,
        )

    def set_model_preset(
        self,
        name: str | None,
        *,
        publish_update: bool = True,
    ) -> LLMRuntime:
        """
        选择命名的默认运行时预设，用于未来的所有轮次。
        """
        old_model = self.model
        # 在解析器中切换默认预设。
        runtime = self.runtime_resolver.select_preset(name)
        # 发布变更事件。
        self._publish_runtime_selection(runtime, publish_update=publish_update)
        logger.info(
            "Runtime model switched for next turn: {} -> {}",
            old_model,
            runtime.model,
        )
        return runtime

    def set_runtime_model(self, model: str) -> LLMRuntime:
        """在当前 Provider 上直接切换模型（不使用预设）。"""
        return self.runtime_resolver.select_model(model)

    def set_runtime_context_window(self, context_window_tokens: int) -> LLMRuntime:
        """修改未来轮次的上下文 Token 限制。"""
        return self.runtime_resolver.select_context_window(context_window_tokens)

    def _register_default_tools(
        self,
        *,
        provider_snapshot_loader: Callable[..., ProviderSnapshot] | None,
    ) -> None:
        """
        通过插件加载器注册系统默认的工具集（如文件读写、代码执行、网络搜索等）。
        """
        from nanobot.agent.tools.context import ToolContext
        from nanobot.agent.tools.loader import ToolLoader

        # 构建工具上下文对象，包含所有工具在执行时可能需要的依赖和配置。
        ctx = ToolContext(
            config=self.tools_config,
            workspace=str(self.workspace),
            bus=self.bus,
            subagent_manager=self.subagents,
            cron_service=self.cron_service,
            exec_session_manager=self._exec_session_manager,
            sessions=self.sessions,
            provider_snapshot_loader=provider_snapshot_loader,
            image_generation_provider_configs=self._image_generation_provider_configs,
            timezone=self.context.timezone or "UTC",
            workspace_sandbox=self.workspace_scopes.sandbox_status,
            runtime_events=self.runtime_events,
        )
        
        # 实例化工具加载器并执行加载，将工具注册到 self.tools 注册表中。
        loader = ToolLoader()
        registered = loader.load(ctx, self.tools)

        # 特殊处理 MyTool：它需要显式的运行时控制能力，并且受配置开关控制。
        if self.tools_config.my.enable:
            self.tools.register(
                MyTool(
                    runtime_control=AgentRuntimeControl(self),
                    modify_allowed=self.tools_config.my.allow_set,
                )
            )
            registered.append("my")

        logger.info("Registered {} tools: {}", len(registered), registered)

    async def _connect_mcp(self) -> None:
        """连接配置的所有 MCP (Model Context Protocol) 服务器。"""
        await agent_context.connect_mcp(self, self.tools)

    def register_runtime_context_provider(
        self,
        provider: RuntimeContextProvider,
    ) -> Callable[[], None]:
        """
        注册一个每轮执行的动态上下文提供者，并返回一个取消订阅的回调函数。
        
        【Python 特殊用法】: 闭包 (Closure)
        内部定义的 _unsubscribe 函数捕获了外部的 provider 变量。
        当调用返回的 _unsubscribe 时，它会从列表中移除对应的 provider。
        """
        if provider in self._runtime_context_providers:
            return lambda: None # 如果已注册，返回一个空操作函数。
        self._runtime_context_providers.append(provider)

        def _unsubscribe() -> None:
            # suppress(ValueError) 用于在 provider 不在列表中时忽略异常，避免程序崩溃。
            with suppress(ValueError):
                self._runtime_context_providers.remove(provider)

        return _unsubscribe

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        """提交一个定时任务触发的轮次。"""
        return await self._cron_turns.submit(msg)

    async def submit_local_trigger_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        """提交一个本地触发器（如文件变更）触发的轮次。"""
        return await self._local_trigger_turns.submit(msg)

    def pending_cron_job_ids_for_session(self, session_key: str) -> set[str]:
        """获取特定会话当前挂起的定时任务 ID 集合。"""
        return self._cron_turns.pending_job_ids_for_session(session_key)

    def pending_local_trigger_ids_for_session(self, session_key: str) -> set[str]:
        """获取特定会话当前挂起的本地触发器 ID 集合。"""
        return self._local_trigger_turns.pending_trigger_ids_for_session(session_key)

    async def _publish_next_deferred_automation_turn(self, session_key: str) -> None:
        """
        发布下一个被延迟的自动化轮次。
        当 Agent 完成当前繁忙的任务后，检查是否有排队等待的自动化消息需要处理。
        """
        await publish_next_deferred_turn(
            deferred_queues=self._deferred_automation_turns,
            publish_inbound=self.bus.publish_inbound,
            session_key=session_key,
        )

    def _persist_user_message_early(
        self,
        msg: InboundMessage,
        session: Session,
        runtime_context_blocks: list[RuntimeContextBlock] | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        在轮次正式开始前，提前持久化触发该轮次的用户消息。
        这在处理斜杠命令或防止系统崩溃导致用户输入丢失时非常有用。
        返回 True 表示消息已成功持久化。
        """
        # 检查元数据，判断是否应该跳过用户消息的持久化（例如内部续传消息）。
        if not turn_continuation.should_persist_user_message(msg.metadata):
            return False
            
        # 提取并过滤有效的媒体文件路径。
        media_paths = [
            path
            for path in (msg.media or [])
            if isinstance(cast(object, path), str) and path
        ]
        content_value = cast(object, msg.content)
        # 检查消息是否包含纯文本或媒体附件。
        has_text = isinstance(content_value, str) and content_value.strip()
        
        if has_text or media_paths or runtime_context_blocks:
            # 【Python 3.9+ 特殊用法】: 字典合并运算符 `|`
            # 将 media 字典与 agent_context 提取的额外元数据合并。
            extra: dict[str, Any] = ({"media": list(media_paths)} if media_paths else {}) | agent_context.session_extra(msg.metadata)
            extra.update(kwargs) # 合并调用者传入的额外参数。
            
            text = content_value if isinstance(content_value, str) else ""
            # 处理自动化历史覆盖逻辑（例如隐藏系统自动生成的提示词）。
            text_override, automation_extra = automation_history_overrides(msg.metadata)
            if text_override is not None:
                text = text_override
            extra.update(automation_extra)
            
            # 将运行时上下文块（如当前时间、工作区信息）追加到文本中。
            text, runtime_context_meta = append_runtime_context(
                text,
                runtime_context_blocks or (),
            )
            if runtime_context_meta is not None:
                extra[RUNTIME_CONTEXT_HISTORY_META] = runtime_context_meta
                
            # 将消息添加到会话历史中。
            session.add_message("user", text, **extra)
            # 标记该会话有一个等待处理的用户轮次（用于崩溃恢复）。
            self._mark_pending_user_turn(session)
            # 保存会话到磁盘/数据库。
            self.sessions.save(session)
            return True
        return False

    def _build_initial_messages(self, ctx: TurnContext) -> list[dict[str, Any]]:
        """
        为 LLM 轮次构建初始的消息列表（包含 System Prompt、历史记录和最新用户输入）。
        """
        assert ctx.session is not None
        # 获取当前消息对应的工作区作用域（处理多工作区或沙盒路径映射）。
        scope = self.workspace_scopes.for_message(ctx.msg, ctx.session.metadata)
        # 调用 ContextBuilder 组装完整的消息列表。
        return self.context.build_messages(
            history=ctx.history,
            current_message=ctx.msg.content,
            media=ctx.msg.media if ctx.kind is TurnKind.USER and ctx.msg.media else None,
            channel=ctx.delivery.route.channel,
            session_summary=ctx.pending_summary, # 如果历史被压缩，传入摘要文本。
            workspace=scope.project_path,
            runtime_context_blocks=ctx.runtime_context_blocks,
            include_memory=ctx.session.policy.persist, # 是否包含长期记忆。
            include_memory_recent_history=not ctx.ephemeral, # 临时轮次不包含近期历史记忆。
            session_key=ctx.session.key,
            unified_session=self._unified_session,
        )

    def _request_context_for_turn(self, ctx: TurnContext) -> RequestContext:
        """
        为当前轮次构建请求上下文（RequestContext），供工具在执行时读取环境变量和元数据。
        """
        assert ctx.session is not None
        scope = self.workspace_scopes.for_turn(
            channel=ctx.delivery.route.channel,
            message_metadata=ctx.msg.metadata,
            session_metadata=ctx.session.metadata,
        )
        return RequestContext(
            channel=ctx.delivery.route.channel,
            chat_id=ctx.delivery.route.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            session_key=ctx.session_key,
            original_user_text=ctx.original_user_text,
            runtime=ctx.runtime,
            metadata=dict(ctx.msg.metadata or {}),
            attributes=dict(ctx.attributes),
            sender_id=ctx.msg.sender_id,
            turn_id=ctx.turn_id,
            workspace=scope.project_path,
        )

    async def _resolve_runtime_context_for_turn(
        self,
        ctx: TurnContext,
    ) -> list[RuntimeContextBlock]:
        """解析当前轮次的运行时上下文块。"""
        assert ctx.request_context is not None
        return await self._resolve_runtime_context_for_request(
            ctx.request_context,
            ctx.tools or self.tools,
        )

    async def _resolve_runtime_context_for_request(
        self,
        request: RequestContext,
        tools: ToolRegistry,
    ) -> list[RuntimeContextBlock]:
        """
        聚合所有注册的运行时上下文提供者，并解析出需要注入到 Prompt 中的上下文块。
        """
        # 合并工具级别和全局级别的上下文提供者。
        providers = [
            *tools.get_runtime_context_providers(),
            *self._runtime_context_providers,
        ]
        # 从请求元数据中恢复已有的上下文块。
        blocks = runtime_context_blocks_from_metadata(request.metadata)
        # 调用所有提供者异步生成新的上下文块并追加。
        blocks.extend(await resolve_runtime_context(providers, request))
        return blocks

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """
        直接从主 run() 循环中分发并执行斜杠命令（如 /stop, /new），并将结果发布到总线。
        这绕过了常规的 LLM 调用流程。
        """
        ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
        result = await dispatch_fn(ctx)
        if result:
            await self.bus.publish_outbound(result)
        else:
            logger.warning("Command '{}' matched but dispatch returned None", raw)

    async def _cancel_active_tasks(self, key: str) -> int:
        """
        取消并等待特定 session_key 的所有活跃工作（包括主任务、子 Agent、代码执行会话）。
        返回被取消的任务总数。
        """
        # 从活跃任务字典中弹出（移除并返回）该 session 的所有任务集合。
        tasks = tuple(self._active_tasks.pop(key, set()))
        # 遍历任务，调用 cancel() 并统计成功取消的数量。
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            # 等待任务实际结束，抑制 CancelledError 和其他异常。
            with suppress(asyncio.CancelledError, Exception):
                await t
        # 取消该 session 关联的所有子 Agent。
        sub_cancelled = await self.subagents.cancel_by_session(key)
        # 终止该 session 拥有的所有代码执行沙盒进程。
        exec_cancelled = await self._exec_session_manager.terminate_by_owner(key)
        return cancelled + sub_cancelled + exec_cancelled

    async def discard_session(self, key: str) -> None:
        """
        停止特定 session 的所有活跃工作，并从缓存中丢弃该会话。
        通常在用户执行 /new (重置会话) 或会话过期时调用。
        """
        self._discarding_sessions.add(key) # 标记为正在丢弃，防止并发重建。
        try:
            self.sessions.invalidate(key) # 从会话管理器缓存中移除。
            await self._cancel_active_tasks(key) # 取消所有关联任务。
        finally:
            self._discarding_sessions.discard(key) # 清理标记。

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """
        返回用于任务路由和中途消息注入的有效 session_key。
        如果启用了统一会话模式（Unified Session），则所有非覆盖消息都路由到全局唯一的 Key。
        """
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    def _remember_unified_session_route(
        self,
        session: Session,
        msg: InboundMessage,
        *,
        is_user_turn: bool,
    ) -> None:
        """
        在统一会话模式下，记住最新的用户侧路由（Channel 和 Chat ID）。
        这确保当 Agent 在后台生成回复时，能够正确地将消息发送到用户最后活跃的渠道。
        """
        if (
            not self._unified_session
            or session.key != UNIFIED_SESSION_KEY
            or not is_user_turn
            or msg.channel in {"cli", "system"} # 忽略 CLI 和系统内部消息
            or msg.sender_id == "subagent"      # 忽略子 Agent 的消息
        ):
            return
        _, automation_metadata = automation_history_overrides(msg.metadata)
        if automation_metadata:
            return # 忽略自动化生成的消息。
        # 将最新的 channel 和 chat_id 写入会话元数据。
        remember_last_channel(session.metadata, msg.channel, msg.chat_id)

    @staticmethod
    def _replay_token_budget(runtime: LLMRuntime) -> int:
        """
        【静态方法】根据上下文窗口大小，推导用于回放会话历史的 Token 预算。
        需要预留足够的空间给 System Prompt、工具定义和 LLM 的最大输出 Token。
        """
        if runtime.context_window_tokens <= 0:
            return 0
        max_output = runtime.generation.max_tokens
        try:
            reserved_output = int(max_output)
        except (TypeError, ValueError):
            reserved_output = 4096 # 默认预留 4K Token 给输出。
            
        # 预算 = 总窗口 - 预留输出 - 1024 (System Prompt 和工具定义的安全边距)
        budget = runtime.context_window_tokens - max(1, reserved_output) - 1024
        # 确保预算至少为 128，或者在极端情况下为总窗口的一半。
        return budget if budget > 0 else max(128, runtime.context_window_tokens // 2)
    async def _run_agent_loop(
        self,
        initial_messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        runtime: LLMRuntime,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        original_user_text: str | None = None,
        pending_queue: asyncio.Queue[InboundMessage] | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        turn_scopes: list[AbstractContextManager[Any]] | None = None,
        tools: ToolRegistry | None = None,
        request_context: RequestContext | None = None,
        provider_state: ProviderConversationState | None = None,
    ) -> tuple[str | None, list[str], list[dict[str, Any]], str, bool]:
        """
        【核心方法】运行 Agent 迭代循环（即 LLM 调用 -> 工具执行 -> 结果回传 -> 再次调用 的循环）。

        *on_stream*: 在流式传输期间，每当生成一个文本片段（delta）时调用此回调。
        *on_stream_end(resuming, merge_next)*: 当一个流式会话结束时调用。
        ``resuming=True`` 表示当前轮次还在继续（LLM 还要继续生成或调用工具）。
        ``merge_next=True`` 表示下一个文本片段属于同一条用户可见的助手消息（避免被拆分成多条消息）。

        返回一个五元组：
        (final_content, tools_used, messages, stop_reason, had_injections)
        - final_content: 最终的文本回复内容
        - tools_used: 本轮次中使用的工具名称列表
        - messages: 本轮次产生的所有消息（用于持久化）
        - stop_reason: 停止原因（如 "stop", "max_iterations", "error" 等）
        - had_injections: 是否在处理过程中注入了新的消息
        """
        # 同步子 Agent 的运行时限制，确保与主循环的配置一致。
        self._sync_subagent_runtime_limits()

        # ==================== 内部闭包函数 ====================

        async def _checkpoint(payload: dict[str, Any]) -> None:
            """
            【闭包 / 回调函数】在工具执行过程中保存检查点（Checkpoint）。
            如果 Agent 在执行工具时崩溃或被取消，可以从检查点恢复部分上下文。
            """
            if session is None:
                return # 临时会话不需要保存检查点。
                
            # 复制一份 payload，避免修改原始数据。
            public_payload = dict(payload)
            # 提取并移除私有的 provider_state（Provider 的内部状态）。
            private_state = public_payload.pop("provider_state", None)
            public_payload.pop(self._PROVIDER_STATE_CHECKPOINT_VERSION_KEY, None)
            
            # 如果 payload 中包含 provider_state 且类型正确，将其存入 session 对象。
            if "provider_state" in payload and (
                private_state is None
                or isinstance(private_state, ProviderConversationState)
            ):
                session.provider_state = private_state
                # 记录检查点的版本号，用于恢复时验证兼容性。
                public_payload[self._PROVIDER_STATE_CHECKPOINT_VERSION_KEY] = (
                    self._PROVIDER_STATE_CHECKPOINT_VERSION
                )
            # 将公开的检查点数据写入 session 的元数据。
            self._set_runtime_checkpoint(session, public_payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """
            【闭包 / 回调函数】从待处理队列中排空（Drain）后续消息。
            
            当队列中没有立即可用的消息，但本轮次派生的子 Agent 仍在运行时，
            会阻塞等待至少一个结果到达（或超时）。
            这能保持运行器循环存活，使后续的子 Agent 完成事件按顺序被消费，
            而不是被单独分发（避免上下文断裂）。
            
            * (星号) 强制 limit 必须作为关键字参数传递。
            """
            if pending_queue is None:
                return []

            async def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                """
                将待处理的入站消息转换为 LLM 格式的 user 消息字典。
                处理图片附件、运行时上下文注入和子 Agent 结果的隐藏标记。
                """
                content = pending_msg.content
                image_paths = pending_msg.media if pending_msg.media else None
                
                # 如果有图片，将非图片附件转换为文本引用，图片保留为路径列表。
                if image_paths:
                    content, image_paths = reference_non_image_attachments(
                        content,
                        image_paths,
                    )
                    image_paths = image_paths or None
                    
                # 构建符合 LLM API 格式的用户消息内容（可能包含文本和图片）。
                user_content = self.context.build_user_content(
                    content,
                    image_paths=image_paths,
                )
                row: dict[str, Any] = {"role": "user", "content": user_content}
                
                # 提取消息元数据。
                metadata_value = cast(object, pending_msg.metadata)
                metadata = (
                    pending_msg.metadata
                    if isinstance(metadata_value, dict)
                    else {}
                )
                
                # 如果不是系统内部消息，需要解析运行时上下文并注入。
                if pending_msg.channel != "system":
                    scope = self.workspace_scopes.for_turn(
                        channel=pending_msg.channel,
                        message_metadata=metadata,
                        session_metadata=session.metadata if session is not None else None,
                    )
                    # 为这条注入的消息构建独立的请求上下文。
                    pending_request = RequestContext(
                        channel=pending_msg.channel,
                        chat_id=pending_msg.chat_id,
                        message_id=metadata.get("message_id"),
                        session_key=active_session_key,
                        original_user_text=pending_msg.content,
                        runtime=runtime,
                        metadata=dict(metadata),
                        attributes=dict(request_ctx.attributes),
                        sender_id=pending_msg.sender_id,
                        turn_id=request_ctx.turn_id,
                        workspace=scope.project_path,
                    )
                    # 异步解析运行时上下文块。
                    blocks = await self._resolve_runtime_context_for_request(
                        pending_request,
                        effective_tools,
                    )
                    # 将上下文块追加到消息内容中。
                    row["content"], runtime_marker = append_runtime_context(
                        user_content,
                        blocks,
                    )
                    if runtime_marker is not None:
                        row["_meta"] = {
                            RUNTIME_CONTEXT_MESSAGE_META: runtime_marker,
                        }
                        
                # 如果是子 Agent 返回的结果，添加隐藏标记，使其不出现在用户可见的历史中。
                if (
                    pending_msg.sender_id == "subagent"
                    and metadata.get("injected_event") == "subagent_result"
                ):
                    subagent_marker: dict[str, Any] = {"kind": "subagent_result"}
                    task_id = metadata.get("subagent_task_id")
                    if isinstance(task_id, str) and task_id:
                        subagent_marker["subagent_task_id"] = task_id
                        row["subagent_task_id"] = task_id
                    # 标记该消息为隐藏历史，前端 UI 和记忆系统会忽略它。
                    row[HIDDEN_HISTORY_META] = subagent_marker
                    row["injected_event"] = "subagent_result"
                return row

            items: list[dict[str, Any]] = []
            # 尝试非阻塞地从队列中取出消息，直到达到 limit 上限或队列为空。
            while len(items) < limit:
                try:
                    items.append(await _to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            # 【关键逻辑】如果队列中没有消息，但本轮次派生的子 Agent 仍在运行：
            # 阻塞等待子 Agent 的结果，而不是直接返回空列表。
            # 这能让主循环保持存活，等待子 Agent 完成后再继续 LLM 对话，
            # 避免子 Agent 结果被单独分发导致上下文断裂。
            if (not items
                    and session is not None
                    and self.subagents.get_running_count_by_session(session.key) > 0):
                try:
                    # 阻塞等待最多 300 秒（5 分钟）。
                    msg = await asyncio.wait_for(pending_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for sub-agent completion in session {}",
                        session.key,
                    )
                    return items # 超时后返回已收集的消息（可能为空）。
                items.append(await _to_user_message(msg))
                # 超时后继续尝试排空队列中剩余的消息。
                while len(items) < limit:
                    try:
                        items.append(await _to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items

        # ==================== 主执行逻辑 ====================

        # 确定当前会话的有效 Key。
        active_session_key = session.key if session else session_key
        
        # 解析当前轮次的工作区作用域（沙盒边界）。
        effective_scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=metadata,
            session_metadata=session.metadata if session is not None else None,
        )
        # 确定使用的工具注册表。
        effective_tools = tools or self.tools
        
        # 构建或使用传入的请求上下文。
        request_ctx = request_context or RequestContext(
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=active_session_key,
            original_user_text=original_user_text,
            runtime=runtime,
            metadata=dict(metadata or {}),
            workspace=effective_scope.project_path,
        )
        
        # 【Python 特殊用法】: Context Variables (上下文变量) 绑定
        # 使用 bind_file_states / bind_request_context / bind_workspace_scope
        # 将当前会话的文件状态、请求上下文和工作区作用域绑定到当前 asyncio 任务。
        # 工具在执行时可以通过 contextvars 获取这些值，而无需显式传递参数。
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        request_token = bind_request_context(request_ctx)
        workspace_token = bind_workspace_scope(effective_scope)
        
        # 【Python 特殊用法】: ExitStack
        # 创建一个上下文管理器栈，用于管理本轮次中需要清理的资源。
        # 无论本轮次是正常结束还是抛出异常，turn_scope_stack.close() 都会确保所有资源被清理。
        turn_scope_stack = ExitStack()
        
        # 【闭包 / 延迟计算】
        # 因为 create_goal 可能会在本轮次中创建目标元数据，所以这个函数需要延迟调用。
        def _goal_continue() -> str | None:
            """如果存在活跃目标，返回提示 LLM 继续工作的文本；否则返回 None。"""
            _goal_lines = goal_state_runtime_lines(session.metadata if session is not None else None)
            if not _goal_lines:
                return None
            return (
                "You have an active sustained goal:\n\n"
                + "\n".join(_goal_lines)
                + "\n\nPlease continue working toward the objective using your tools, "
                "or call update_goal with action='complete' if the work is truly finished."
            )

        session_metadata = session.metadata if session is not None else None
        
        try:
            # 将调用者传入的轮次作用域（如临时的环境变量覆盖）推入上下文栈。
            for scope in turn_scopes or ():
                turn_scope_stack.enter_context(scope)
                
            # 【构建钩子链】
            # 根据当前轮次的配置，构建一个包含所有生命周期钩子（Hooks）的组合对象。
            # 这些钩子会在工具调用前、工具调用后、流式输出等关键时刻被触发。
            hook = build_agent_turn_hook(AgentTurnHookSpec(
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                channel=channel,
                chat_id=chat_id,
                message_id=message_id,
                metadata=metadata,
                attributes=dict(request_ctx.attributes),
                session_key=active_session_key,
                workspace=effective_scope.project_path,
                tool_hint_max_length=self.tool_hint_max_length,
                # 【Python 特殊用法】: lambda 和 setattr
                # 每次迭代开始时，更新 self._current_iteration 属性。
                on_iteration=lambda iteration: setattr(self, "_current_iteration", iteration),
                registered_hook_factories=self._hook_factories,
                turn_hook_factories=list(hook_factories or []),
                registered_hooks=self._extra_hooks,
                turn_hooks=list(hooks or []),
                ephemeral=ephemeral,
                run_extra_hooks_for_ephemeral=run_extra_hooks_for_ephemeral,
            ))
            
            # 【核心调用】运行 AgentRunner，执行实际的 LLM 调用和工具循环。
            result = await self.runner.run(AgentRunSpec(
                initial_messages=initial_messages,
                tools=effective_tools,
                runtime=runtime,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                hook=hook,
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True, # 允许并发执行多个工具调用。
                workspace=effective_scope.project_path,
                session_key=session.key if session else None,
                context_block_limit=self.context_block_limit,
                provider_retry_mode=self.provider_retry_mode,
                progress_callback=on_progress,
                stream_progress_deltas=on_stream is not None, # 如果配置了流式回调，启用流式输出。
                retry_wait_callback=on_retry_wait,
                checkpoint_callback=_checkpoint, # 传入检查点保存回调。
                injection_callback=_drain_pending, # 传入消息注入回调。
                # 【动态超时计算】
                # 持续性目标可能会合理地超过标准的 LLM 超时时间；
                # 空闲超时仍然由流式 Provider 中的 NANOBOT_STREAM_IDLE_TIMEOUT_S 限制。
                llm_timeout_s=runner_wall_llm_timeout_s(
                    self.sessions,
                    session.key if session is not None else session_key,
                    metadata=session_metadata,
                    message_metadata=metadata,
                ),
                # 【Python 特殊用法】: lambda 谓词函数
                # 检查当前会话是否有活跃的持续性目标。
                goal_active_predicate=lambda: sustained_goal_active(session.metadata) if session is not None else False,
                goal_continue_message=_goal_continue,
                # 判断在达到最大迭代次数时是否应该生成最终回复（而不是直接截断）。
                finalize_on_max_iterations=turn_continuation.should_finalize_on_max_iterations(
                    pending_queue_available=pending_queue is not None and session is not None,
                    session_metadata=session_metadata,
                    message_metadata=metadata,
                ),
                provider_state=provider_state,
            ))
        finally:
            # 【Python 特殊用法】: finally 块
            # 无论 try 块中的代码是正常结束、抛出异常还是被取消，finally 中的清理代码都会执行。
            # 关闭上下文栈，释放所有绑定的资源。
            turn_scope_stack.close()
            # 重置 Context Variables，防止内存泄漏或状态污染。
            reset_workspace_scope(workspace_token)
            reset_request_context(request_token)
            reset_file_states(file_state_token)
            
        # 记录最新的 Token 使用量。
        self._last_usage = result.usage
        
        # 如果不是临时会话，更新 Provider 的内部状态。
        if session is not None and not ephemeral:
            session.provider_state = result.provider_state
            
        # 【处理最大迭代次数停止】
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # 判断是否需要流式输出最终的预算响应。
            should_stream = turn_continuation.should_stream_budget_response(
                stop_reason=result.stop_reason,
                pending_queue_available=pending_queue is not None and session is not None,
                session_metadata=session_metadata,
                message_metadata=metadata,
            )
            # 将最终内容推送到流式通道，确保流式渠道（如飞书）能更新卡片，而不是留空。
            if on_stream and on_stream_end and should_stream:
                stream_content = (
                    result.pending_stream_content
                    if result.pending_stream_content is not None
                    else result.final_content or ""
                )
                await on_stream(stream_content)
                await on_stream_end(resuming=False) # resuming=False 表示本轮次真正结束。
        elif result.stop_reason == "error":
            # 记录 LLM 返回的错误信息（仅前 200 字符）。
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
            
        # 返回五元组结果。
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    def _check_expired_sessions_if_due(self) -> None:
        """
        扫描空闲会话，但不会比配置的间隔更频繁。
        用于触发自动压缩和归档过期的会话。
        """
        now = time.monotonic()
        # 如果还没到下一次检查的时间点，直接返回。
        if now < self._next_idle_compact_check_at:
            return
        # 更新下一次检查的时间点。
        self._next_idle_compact_check_at = now + self._idle_compact_check_interval_s
        # 执行过期检查，传入调度后台任务的回调和运行时解析器。
        self.auto_compact.check_expired(
            self.schedule_background,
            self.runtime_for_session,
            active_session_keys=self._pending_queues.keys(),
        )

    async def run(self) -> None:
        """
        【主事件循环】运行 Agent 循环，将消息作为任务分发，以保持对 /stop 命令的响应性。
        
        这是一个无限循环，从消息总线中消费入站消息，并为每条消息创建一个独立的 asyncio.Task。
        使用任务（而不是直接 await）是为了让主循环保持响应，能够及时处理 /stop 等控制命令。
        """
        self._running = True # 设置运行标志。
        try:
            # 连接配置的所有 MCP 服务器。
            await self._connect_mcp()
            logger.info("Agent loop started")

            # 【主循环】
            while self._running:
                try:
                    # 从消息总线消费入站消息，设置 1 秒超时。
                    # 使用超时是为了让主循环有机会定期检查空闲会话和运行状态。
                    msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 超时是正常的，定期检查过期会话后继续循环。
                    self._check_expired_sessions_if_due()
                    continue
                except asyncio.CancelledError:
                    # 【Python 特殊用法】: 处理 CancelledError
                    # 保留真正的任务取消，以便关闭操作能够正常完成。
                    # 只忽略从集成层泄漏出来的非任务级 CancelledError 信号。
                    if not self._running or task_is_cancelling():
                        raise # 重新抛出，让关闭流程继续。
                    logger.warning(
                        "Ignoring leaked CancelledError while consuming inbound messages"
                    )
                    continue
                except Exception as e:
                    # 捕获消费消息时的其他异常，记录日志后继续循环，避免主循环崩溃。
                    logger.warning("Error consuming inbound message: {}, continuing...", e)
                    continue

                # 去除消息内容两端的空白字符。
                raw = msg.content.strip()
                # 计算有效的 session_key（处理统一会话模式）。
                effective_key = self._effective_session_key(msg)
                
                # 【处理运行时控制命令】
                # 检查消息是否是特殊的运行时控制命令（如 /stop, /model, /compact 等）。
                # 如果是，直接处理并跳过后续的常规流程。
                if await agent_context.handle_runtime_control(self, msg, self.tools):
                    continue
                    
                # 【检查会话存在性】
                # 如果消息要求会话必须已存在（require_existing_session=True），
                # 但当前缓存中没有该会话，则忽略此消息。
                if (
                    msg.require_existing_session
                    and self.sessions.get_cached(effective_key) is None
                ):
                    continue
                    
                # 【处理优先级命令】
                # 检查消息是否是优先级命令（如 /stop）。
                # 优先级命令需要立即执行，不能排队等待常规任务完成。
                if self.commands.is_priority(raw):
                    await self._dispatch_command_inline(
                        msg, effective_key, raw,
                        self.commands.dispatch_priority,
                    )
                    continue
                    
                # 【处理自动化轮次延迟】
                # 检查是否需要延迟自动化轮次（如定时任务、本地触发器）。
                # 如果当前会话正在处理其他任务，自动化消息会被放入延迟队列。
                deferred = False
                for label, coordinator in self._automation_turn_coordinators:
                    if coordinator.defer_if_active(
                        msg,
                        session_key=effective_key,
                        active_session_keys=self._pending_queues.keys(),
                    ):
                        logger.info(
                            "Deferred {} turn for active session {}",
                            label,
                            effective_key,
                        )
                        deferred = True
                        break
                if deferred:
                    continue
                    
                # 【处理中途消息注入】
                # 如果该会话已经有一个活跃的待处理队列（即正在处理该会话的任务），
                # 将消息路由到该队列进行中途注入，而不是创建竞争任务。
                if effective_key in self._pending_queues:
                    # 非优先级命令不能被排队注入，需要直接分发。
                    if self.commands.is_dispatchable_command(raw):
                        await self._dispatch_command_inline(
                            msg, effective_key, raw,
                            self.commands.dispatch,
                        )
                        continue
                        
                    pending_msg = msg
                    # 如果 effective_key 与 msg.session_key 不同，需要替换 session_key_override。
                    if effective_key != msg.session_key:
                        # 【Python 特殊用法】: dataclasses.replace
                        # 创建一个消息对象的浅拷贝，只修改指定的字段（session_key_override）。
                        # 这比手动构造新对象更安全、更简洁。
                        pending_msg = dataclasses.replace(
                            msg,
                            session_key_override=effective_key,
                        )
                    try:
                        # 尝试将消息放入队列。
                        self._pending_queues[effective_key].put_nowait(pending_msg)
                    except asyncio.QueueFull:
                        # 队列已满，记录警告并回退到创建新任务。
                        logger.warning(
                            "Pending queue full for session {}, falling back to queued task",
                            effective_key,
                        )
                    else:
                        # 成功路由到待处理队列。
                        logger.info(
                            "Routed follow-up message to pending queue for session {}",
                            effective_key,
                        )
                        continue
                        
                # 【创建并注册任务】
                # 在分发前计算有效的会话 Key。
                # 这确保 /stop 命令在启用统一会话时也能正确找到任务。
                # 创建一个新的 asyncio.Task 来处理这条消息。
                task = asyncio.create_task(self._dispatch(msg))
                # 将任务添加到该会话的活跃任务集合中。
                active_tasks = self._active_tasks.setdefault(effective_key, set())
                active_tasks.add(task)
                # 【Python 特殊用法】: add_done_callback
                # 当任务完成（无论成功、失败还是被取消）时，从集合中移除它。
                task.add_done_callback(active_tasks.discard)
        finally:
            # 【Python 特殊用法】: finally 块确保清理
            # MCP stdio 传输使用 AnyIO 的取消作用域；必须从打开它们的任务中关闭它们。
            await self.close_mcp()

    async def _dispatch(self, msg: InboundMessage) -> None:
        """
        【消息分发】处理单条消息：同一会话串行处理，不同会话并发处理。
        
        这是每条消息的入口点。使用会话级别的锁（Session Lock）确保同一会话的消息不会并发执行，
        但不同会话的消息可以同时处理。
        """
        # 计算有效的会话 Key。
        session_key = self._effective_session_key(msg)
        # 如果需要，替换消息中的 session_key_override。
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
            
        # 获取该会话的锁（使用弱引用字典管理，防止内存泄漏）。
        lock = self._get_session_lock(session_key)
        # 【Python 特殊用法】: nullcontext
        # 如果配置了并发限制（_concurrency_gate），使用信号量；否则使用 nullcontext()。
        # nullcontext() 是一个空的上下文管理器，不执行任何操作，但允许统一使用 with 语句。
        gate = self._concurrency_gate or nullcontext()

        # 创建一个未路由的投递对象。
        delivery = self.turn_delivery_factory.unrouted(msg, session_key)
        pending: asyncio.Queue[InboundMessage] | None = None
        
        try:
            # 【Python 特殊用法】: async with 和多个上下文管理器
            # 同时获取会话锁和并发限制信号量。
            async with lock, gate:
                # 只有拥有会话锁的任务才能发布活跃的中途注入队列。
                # 创建一个新的待处理队列（最多容纳 20 条消息）。
                pending = asyncio.Queue(maxsize=20)
                self._pending_queues[session_key] = pending
                try:
                    # 创建完整的消息投递对象（包含流式输出回调）。
                    delivery = self.turn_delivery_factory.create(
                        msg,
                        session_key,
                        enable_stream=True,
                    )
                    # 【核心调用】处理消息，执行 Agent 循环。
                    response = await self._process_message(
                        msg,
                        on_stream=delivery.on_stream,
                        on_stream_end=delivery.on_stream_end,
                        pending_queue=pending,
                        delivery=delivery,
                    )
                    # 检查是否需要内部续传（即本轮次未完成，需要继续执行）。
                    continuing = turn_continuation.internal_continuation_pending(msg.metadata)
                    # 完成投递，发送最终响应。
                    await delivery.complete(
                        response,
                        publish_completion=not continuing, # 如果是续传，不发布完成事件。
                    )
                    # 通知所有自动化协调器该轮次已完成。
                    for _, coordinator in self._automation_turn_coordinators:
                        coordinator.complete(msg, response=response)
                        
                except asyncio.CancelledError:
                    # 【处理任务取消】
                    # 通知自动化协调器发生了取消错误。
                    for _, coordinator in self._automation_turn_coordinators:
                        coordinator.complete(msg, error=asyncio.CancelledError())
                    logger.info("Task cancelled for session {}", session_key)
                    try:
                        # 尝试中止流式输出。
                        await delivery.abort_stream()
                    except Exception:
                        logger.debug(
                            "Could not close stream for cancelled session {}",
                            session_key,
                            exc_info=True,
                        )
                    # 【保留部分上下文】
                    # 保留被中断轮次的部分上下文，避免用户丢失在 /stop 前积累的工具结果和助手消息。
                    # 检查点已经在工具执行期间由 _emit_checkpoint 持久化到会话元数据；
                    # 现在将其具体化到会话历史中，使其在下一轮对话中可见。
                    if session_key in self._discarding_sessions:
                        raise # 如果会话正在被丢弃，不需要恢复检查点。
                    try:
                        key = self._effective_session_key(msg)
                        session = self.sessions.get_or_create(key)
                        # 尝试从检查点恢复运行时状态。
                        if self._restore_runtime_checkpoint(session):
                            # 清除待处理的用户轮次标记。
                            self._clear_pending_user_turn(session)
                            # 保存会话。
                            self.sessions.save(session)
                            logger.info(
                                "Restored partial context for cancelled session {}",
                                key,
                            )
                    except Exception:
                        logger.debug(
                            "Could not restore checkpoint for cancelled session {}",
                            session_key,
                            exc_info=True,
                        )
                    raise # 重新抛出 CancelledError，让调用者知道任务被取消。
                    
                except Exception as exc:
                    # 【处理异常】
                    logger.exception("Error processing message for session {}", session_key)
                    # 标记投递失败。
                    await delivery.fail(
                        publish_completion=not turn_continuation.internal_continuation_pending(
                            msg.metadata
                        )
                    )
                    # 通知自动化协调器发生了错误。
                    for _, coordinator in self._automation_turn_coordinators:
                        coordinator.complete(msg, error=exc)
                        
                finally:
                    # 【清理待处理队列】
                    # 排空待处理队列中剩余的消息，并重新发布到总线，
                    # 使它们作为新的入站消息被处理，而不是被静默丢弃。
                    # 只移除自己的队列；后续等待锁的任务不能窃取清理所有权。
                    queue = None
                    if self._pending_queues.get(session_key) is pending:
                        queue = self._pending_queues.pop(session_key, None)
                    else:
                        queue = pending
                        
                    if queue is not None:
                        leftover = 0
                        # 循环取出队列中的所有消息。
                        while True:
                            try:
                                item = queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            # 将消息重新发布到总线。
                            await self.bus.publish_inbound(item)
                            leftover += 1
                        if leftover:
                            logger.info(
                                "Re-published {} leftover message(s) to bus for session {}",
                                leftover, session_key,
                            )
                    # 如果不是内部续传，标记投递为空闲。
                    if not turn_continuation.internal_continuation_pending(msg.metadata):
                        await delivery.idle()
                    # 发布下一个延迟的自动化轮次。
                    await self._publish_next_deferred_automation_turn(session_key)
        finally:
            # 如果 pending 为 None（说明在获取锁前就发生了异常），也需要执行清理。
            if pending is None:
                await delivery.idle()
                await self._publish_next_deferred_automation_turn(session_key)
    async def close_mcp(self) -> None:
        """
        【资源清理】停止所有活跃工作，然后关闭执行会话、子 Agent 和 MCP 资源。
        
        即使任务排空（Drain）过程被取消中断，资源清理也必须继续运行。
        网关关闭时会为此协程设置超时限制，因此将清理阶段放在 ``finally`` 中，
        可以防止超时的后台任务在事件循环关闭后仍然保留子进程传输（Subprocess Transports）。
        """
        # 【并发安全】
        # Agent 循环可能在 run() 中自行关闭，同时网关关闭也会执行一次最终的关闭操作。
        # 使用锁序列化这两个所有者，防止它们并发地关闭相同的子进程传输。
        close_lock = getattr(self, "_close_mcp_lock", None)
        if close_lock is None:
            close_lock = self._close_mcp_lock = asyncio.Lock()
        # 【Python 特殊用法】: async with
        # 异步上下文管理器，等待获取锁后再执行关闭操作。
        async with close_lock:
            await self._close_mcp_unlocked()

    async def _close_mcp_unlocked(self) -> None:
        """
        实际执行关闭逻辑的内部方法（调用者需已持有锁）。
        """
        errors: list[BaseException] = [] # 收集清理过程中发生的所有异常。
        
        # 获取所有活跃的任务组。
        active_task_groups = getattr(self, "_active_tasks", {})
        # 【Python 特殊用法】: 集合推导式和元组解包
        # 将所有任务组中的任务展平（Flatten）为一个元组。
        active_tasks = tuple({task for tasks in active_task_groups.values() for task in tasks})
        active_task_groups.clear() # 清空任务组字典。
        
        # 获取当前正在执行的任务（即调用此方法的任务）。
        current_task = asyncio.current_task()
        # 过滤掉当前任务，避免取消自己。
        active_tasks = tuple(task for task in active_tasks if task is not current_task)
        
        # 取消所有未完成的活跃任务。
        for task in active_tasks:
            if not task.done():
                task.cancel()
                
        try:
            # 等待所有已取消的任务实际结束。
            if active_tasks:
                # 【Python 特殊用法】: asyncio.gather 和 return_exceptions=True
                # 并发等待所有任务完成，即使某些任务抛出异常也不会中断其他任务的等待。
                await asyncio.gather(*active_tasks, return_exceptions=True)
            # 等待所有后台任务完成。
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
        except BaseException as exc:
            # 捕获所有异常（包括 KeyboardInterrupt、SystemExit 等）。
            errors.append(exc)
        finally:
            # 清空后台任务集合。
            self._background_tasks.clear()

        # 定义清理步骤的元组，按顺序执行。
        cleanup_steps = (
            self.subagents.close,                    # 关闭子 Agent 管理器。
            self._exec_session_manager.close_all,    # 关闭所有代码执行会话。
            lambda: agent_context.close_mcp(self),   # 关闭 MCP 连接。
        )
        # 依次执行每个清理步骤。
        for cleanup in cleanup_steps:
            try:
                await cleanup()
            except BaseException as exc:
                errors.append(exc) # 收集异常，但不中断后续清理。
                
        # 【异常聚合】
        # 如果只有一个异常，直接抛出。
        if len(errors) == 1:
            raise errors[0]
        # 如果有多个异常，使用 ExceptionGroup（Python 3.11+）聚合抛出。
        if errors:
            raise BaseExceptionGroup("failed to close agent resources", errors)

    def schedule_background(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        【后台任务调度】将一个协程调度为受跟踪的后台任务。
        后台任务会在 Agent 关闭时被排空（等待完成或取消）。
        
        典型用途：记忆整合、会话压缩等不需要阻塞用户响应的操作。
        """
        # 创建 asyncio.Task 来执行协程。
        task = asyncio.create_task(coro)
        # 将任务添加到后台任务集合中。
        self._background_tasks.add(task)
        # 【Python 特殊用法】: add_done_callback
        # 当任务完成时，自动从集合中移除，防止集合无限增长。
        task.add_done_callback(self._background_tasks.discard)

    def stop(self) -> None:
        """
        【停止 Agent 循环】设置运行标志为 False，使主循环退出。
        """
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue[InboundMessage] | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        tools: ToolRegistry | None = None,
        runtime: LLMRuntime | None = None,
        delivery: TurnDelivery | None = None,
        on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> OutboundMessage | None:
        """
        【消息处理入口】处理单条入站消息并返回响应。
        
        这是 Agent 处理消息的完整流程编排方法，按顺序执行以下阶段：
        1. RESTORE - 恢复会话和检查点
        2. COMPACT - 压缩过长的历史
        3. COMMAND - 检查是否是斜杠命令
        4. BUILD   - 构建上下文和初始消息
        5. RUN     - 执行 Agent 循环（LLM + 工具）
        6. SAVE    - 持久化结果
        7. RESPOND - 组装出站消息
        """
        # 确定轮次类型：系统消息或用户消息。
        kind = TurnKind.SYSTEM if msg.channel == "system" else TurnKind.USER
        
        # 【确定会话 Key】
        if kind is TurnKind.SYSTEM:
            # 系统消息的 chat_id 格式可能是 "channel:chat_id"，需要解析。
            destination = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            key = session_key or msg.session_key_override or f"{destination[0]}:{destination[1]}"
        else:
            key = session_key or msg.session_key
            
        # 如果未提供投递对象，创建一个新的。
        if delivery is None:
            delivery = self.turn_delivery_factory.create(msg, key)
        elif delivery.session_key != key:
            # 验证投递对象的会话 Key 与处理会话一致。
            raise ValueError("turn delivery session does not match the processing session")
            
        # 如果未提供流式回调，使用投递对象的默认回调。
        if on_stream is None:
            on_stream = delivery.on_stream
        if on_stream_end is None:
            on_stream_end = delivery.on_stream_end
            
        # 记录轮次开始时间。
        t0 = time.time()
        
        # 【构建轮次上下文】
        # 创建一个 TurnContext 对象，封装本轮次的所有状态。
        ctx = TurnContext(
            msg=msg,
            session=None, # 将在 RESTORE 阶段填充。
            session_key=key,
            turn_id=f"{key}:{time.time_ns()}", # 使用纳秒时间戳生成唯一 ID。
            runtime=runtime,
            kind=kind,
            delivery=delivery,
            original_user_text=(
                None
                if kind is TurnKind.SYSTEM
                or turn_continuation.internal_continuation_inbound(msg.metadata)
                else msg.content
            ),
            turn_wall_started_at=t0,
            visible_run_started_at=turn_continuation.internal_continuation_run_started_at(
                msg.metadata,
            ),
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            on_runtime_admitted=on_runtime_admitted,
            pending_queue=pending_queue,
            ephemeral=ephemeral,
            run_extra_hooks_for_ephemeral=run_extra_hooks_for_ephemeral,
            hooks=list(hooks or []),
            hook_factories=list(hook_factories or []),
            tools=tools,
            attributes=dict(attributes or {}),
        )
        
        # 【流式回调包装】
        # 即使最终文本来自非流式恢复，也可能存在流式回调。
        # 只有最后一个完成的片段才能抑制常规的出站消息。
        if ctx.on_stream is not None:
            stream_callback = ctx.on_stream
            stream_end_callback = ctx.on_stream_end
            
            # 【Python 特殊用法】: inspect.signature
            # 检查 stream_end 回调是否接受 merge_next 参数。
            # 使用 inspect 模块在运行时检查函数签名，实现向后兼容。
            stream_end_accepts_merge_next = False
            if stream_end_callback is not None:
                try:
                    stream_end_signature = inspect.signature(stream_end_callback)
                    stream_end_accepts_merge_next = (
                        "merge_next" in stream_end_signature.parameters
                        or any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in stream_end_signature.parameters.values()
                        )
                    )
                except (TypeError, ValueError):
                    pass # 如果无法获取签名，假设不支持 merge_next。
                    
            segment_streamed_content = False # 标记当前片段是否有流式内容。

            async def _tracked_stream(delta: str) -> None:
                """
                【闭包 / 包装函数】跟踪流式输出，记录是否有实际内容被流式传输。
                nonlocal 关键字允许修改外层函数的变量。
                """
                nonlocal segment_streamed_content
                if delta:
                    segment_streamed_content = True
                await stream_callback(delta)

            async def _tracked_stream_end(
                *,
                resuming: bool = False,
                merge_next: bool = False,
            ) -> None:
                """
                【闭包 / 包装函数】跟踪流式结束，更新上下文状态。
                """
                nonlocal segment_streamed_content
                ctx.streamed_content = segment_streamed_content
                segment_streamed_content = False # 重置标记，为下一个片段做准备。
                if stream_end_callback is not None:
                    if merge_next and stream_end_accepts_merge_next:
                        await stream_end_callback(resuming=resuming, merge_next=True)
                    else:
                        await stream_end_callback(resuming=resuming)

            # 用包装后的回调替换原始回调。
            ctx.on_stream = _tracked_stream
            ctx.on_stream_end = _tracked_stream_end

        # ==================== 执行轮次阶段 ====================
        
        # 阶段 1: RESTORE - 恢复会话和检查点。
        await self._run_turn_stage(ctx, "restore", self._restore_turn)
        
        # 阶段 2: COMPACT - 压缩过长的会话历史。
        await self._run_turn_stage(ctx, "compact", self._compact_session)
        
        # 阶段 3: COMMAND - 检查是否是斜杠命令。
        # 如果返回 True，表示是命令，直接返回结果，跳过后续阶段。
        if await self._run_turn_stage(ctx, "command", self._dispatch_command):
            return ctx.outbound
            
        # 阶段 4: BUILD - 构建上下文和初始消息。
        await self._run_turn_stage(ctx, "build", self._build_turn)
        
        # 阶段 5: RUN - 执行 Agent 循环（LLM 调用 + 工具执行）。
        await self._run_turn_stage(ctx, "run", self._run_turn)
        
        # 阶段 6: SAVE - 持久化结果。
        await self._run_turn_stage(ctx, "save", self._persist_turn)
        
        # 阶段 7: RESPOND - 组装出站消息。
        await self._run_turn_stage(ctx, "respond", self._prepare_outbound)
        
        return ctx.outbound

    async def _run_turn_stage(
        self,
        ctx: TurnContext,
        name: str,
        handler: Callable[[TurnContext], Awaitable[_T]],
    ) -> _T:
        """
        【阶段执行器】运行单个轮次阶段，并记录执行时间和异常。
        
        这是一个泛型方法，使用 TypeVar _T 来推断 handler 的返回类型。
        所有阶段都通过此方法执行，确保统一的错误处理和日志记录。
        """
        # 【Python 特殊用法】: time.perf_counter()
        # 使用高精度计时器（Performance Counter），比 time.time() 更适合测量短时间间隔。
        started_at = time.perf_counter()
        try:
            # 执行阶段处理器。
            result = await handler(ctx)
        except Exception:
            # 计算执行时长（毫秒）。
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.debug(
                "[turn {}] Stage {} failed after {:.1f}ms",
                ctx.turn_id,
                name,
                duration_ms,
            )
            raise # 重新抛出异常，让上层处理。
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            "[turn {}] Stage {} completed in {:.1f}ms",
            ctx.turn_id,
            name,
            duration_ms,
        )
        return result

    def _assemble_outbound(
        self,
        msg: InboundMessage,
        final_content: str,
        stop_reason: str,
        had_injections: bool,
        streamed_content: bool,
        *,
        log_content: bool = True,
        turn_latency_ms: int | None = None,
    ) -> OutboundMessage | None:
        """
        【出站消息组装】从轮次结果组装最终的出站消息。
        
        返回 None 表示不需要发送消息（例如工具已经主动发送了消息）。
        """
        # 【MessageTool 抑制逻辑】
        # 检查是否有 MessageTool 实例，并且它在本轮次中已经发送了消息。
        # 如果工具已经发送了消息，且没有注入新消息，则抑制常规的出站消息。
        # 【Python 特殊用法】: 海象运算符 (Walrus Operator) :=
        # 在条件表达式中同时赋值和判断，避免重复调用 self.tools.get("message")。
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        # 记录响应日志（可选隐藏内容）。
        if log_content:
            preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
            logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        else:
            logger.info("Response to {}:{}: [content hidden]", msg.channel, msg.sender_id)

        event = None
        # 复制消息元数据。
        meta = dict(msg.metadata or {})
        
        # 如果有流式内容且不是错误，添加 StreamedResponseEvent 事件。
        if streamed_content and stop_reason not in {"error", "tool_error"}:
            event = StreamedResponseEvent()
            
        # 记录轮次延迟。
        if turn_latency_ms is not None:
            meta["latency_ms"] = int(turn_latency_ms)

        # 组装并返回出站消息对象。
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            event=event,
            metadata=meta,
        )

    async def _restore_turn(self, ctx: TurnContext) -> None:
        """
        【阶段 1: RESTORE】恢复检查点 / 待处理的用户轮次；引用非图片附件。
        
        这个阶段负责：
        1. 处理消息中的非图片附件（如 PDF、TXT），将其转换为文本引用。
        2. 获取或创建会话对象。
        3. 恢复之前崩溃或取消时保存的检查点。
        """
        msg = ctx.msg

        # 【处理非图片附件】
        # 如果是用户消息且包含媒体附件，将非图片附件转换为文本引用。
        if ctx.kind is TurnKind.USER and msg.media:
            new_content, image_paths = reference_non_image_attachments(
                msg.content,
                msg.media,
            )
            # 【Python 特殊用法】: dataclasses.replace
            # 创建消息对象的副本，只修改 content 和 media 字段。
            ctx.msg = dataclasses.replace(msg, content=new_content, media=image_paths)
            msg = ctx.msg

        # 【获取或创建会话】
        if ctx.session is None:
            if msg.require_existing_session:
                # 如果要求会话必须存在，从缓存中获取。
                ctx.session = self.sessions.get_cached(ctx.session_key)
                if ctx.session is None:
                    raise RuntimeError("required session is not active")
            else:
                # 否则获取或创建新会话。
                ctx.session = self.sessions.get_or_create(ctx.session_key)
                
        session = ctx.session
        # 如果会话策略不允许持久化，标记为临时会话。
        ctx.ephemeral = ctx.ephemeral or not session.policy.persist
        
        # 【处理工具限制】
        # 如果会话策略禁用了某些工具，创建一个受限的工具注册表。
        tools = ctx.tools or self.tools
        if session.policy.disabled_tools:
            restricted = ToolRegistry()
            for name in tools.tool_names:
                tool = tools.get(name)
                if name not in session.policy.disabled_tools and tool:
                    restricted.register(tool)
            tools = restricted
        ctx.tools = tools

        # 【记录日志】
        if ctx.kind is TurnKind.SYSTEM:
            logger.info("Processing system message from {}", msg.sender_id)
        elif session.policy.log_content:
            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)
        else:
            logger.info("Processing message from {}:{}: [content hidden]", msg.channel, msg.sender_id)

        # 【记住统一会话路由】
        # 在统一会话模式下，记住用户最后交互的渠道。
        self._remember_unified_session_route(
            session,
            msg,
            is_user_turn=ctx.original_user_text is not None,
        )
        
        # 通知投递对象轮次已开始。
        await ctx.delivery.started()
        
        # 如果是用户消息，持久化消息的工作区作用域。
        if ctx.kind is TurnKind.USER:
            self.workspace_scopes.persist_message_scope(session, msg)

        # 【恢复运行时检查点】
        # 如果存在之前保存的检查点（例如上次崩溃或取消时的状态），恢复到会话历史中。
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
            
        # 【恢复待处理的用户轮次】
        # 如果上次崩溃时只持久化了用户消息但没有生成回复，补充一条错误消息。
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

    async def _compact_session(self, ctx: TurnContext) -> None:
        """
        【阶段 2: COMPACT】压缩过长的会话历史。
        
        如果会话历史太长，AutoCompact 会将其压缩为摘要，
        并返回待处理的摘要文本，用于后续的上下文构建。
        """
        session = ctx.require_session()
        # 调用 AutoCompact 准备会话，可能触发压缩。
        ctx.session, pending = self.auto_compact.prepare_session(
            session,
            ctx.session_key,
        )
        # 存储待处理的摘要文本。
        ctx.pending_summary = pending

    async def _dispatch_command(self, ctx: TurnContext) -> bool:
        """
        【阶段 3: COMMAND】检查并分发斜杠命令。
        
        返回 True 表示消息是命令，已处理完毕；
        返回 False 表示不是命令，需要继续执行后续的 BUILD/RUN/SAVE 阶段。
        """
        # 系统消息不处理命令。
        if ctx.kind is TurnKind.SYSTEM:
            return False
            
        session = ctx.require_session()
        raw = ctx.msg.content.strip()
        
        # 检查是否是自动化消息（自动化消息不处理用户命令）。
        _, automation_metadata = automation_history_overrides(ctx.msg.metadata)
        is_user_turn = (
            ctx.original_user_text is not None
            and not automation_metadata
            and ctx.msg.channel != "system"
            and ctx.msg.sender_id != "subagent"
        )
        
        # 构建命令上下文。
        cmd_ctx = CommandContext(
            msg=ctx.msg,
            session=session,
            key=ctx.session_key,
            raw=raw,
            loop=self,
            runtime=ctx.runtime,
            is_user_turn=is_user_turn,
            turn_scopes=ctx.turn_scopes,
        )
        
        # 分发命令。
        result = await self.commands.dispatch(cmd_ctx)
        
        if result is not None:
            # 命令执行成功，设置出站消息。
            ctx.outbound = result
            
            # 【快捷命令的特殊处理】
            # 快捷命令跳过 BUILD 和 SAVE 阶段，因此需要在这里手动持久化轮次，
            # 以便 WebUI 在 _turn_end 事件后刷新历史时能看到消息。
            # 使用 _command 标记消息，使 get_history 能将其从 LLM 上下文中过滤掉。
            # /new 命令被排除，因为它会清空整个会话。
            if cmd_ctx.raw.lower() != "/new":
                # 提前持久化用户消息。
                ctx.input_persisted_early = self._persist_user_message_early(
                    ctx.msg, session, _command=True
                )
                # 添加助手的命令响应到历史。
                session.add_message(
                    "assistant", result.content, _command=True
                )
                # 清除待处理的用户轮次标记。
                self._clear_pending_user_turn(session)
                # 保存会话。
                self.sessions.save(session)
                
                # 如果不是临时会话，发布轮次持久化事件。
                if not ctx.ephemeral:
                    await self.runtime_event_publisher.session_turn_persisted(
                        ctx.msg,
                        ctx.session_key,
                        turn_id=ctx.turn_id,
                        attributes=ctx.attributes,
                    )
            return True # 是命令，已处理。
        return False # 不是命令，继续后续阶段。
    async def _build_turn(self, ctx: TurnContext) -> None:
        """
        【阶段 4: BUILD】构建轮次的运行时配置、历史记录和初始消息。
        
        这个阶段负责：
        1. 解析当前会话使用的 LLM 运行时（模型、上下文窗口等）。
        2. 提取会话历史（受 Token 预算限制）。
        3. 处理子 Agent 的后续消息持久化。
        4. 构建发送给 LLM 的初始消息列表（包含 System Prompt）。
        """
        session = ctx.require_session()
        
        # 【解析运行时配置】
        runtime = ctx.runtime
        if runtime is None:
            # 如果调用者没有指定运行时，从会话元数据中解析。
            runtime = self.runtime_for_session(session)
            ctx.runtime = runtime
            
        # 【Dream 模式日志】
        # 如果 session_key 以 "dream:" 开头，说明是 Dream（后台思考）模式。
        if ctx.session_key.startswith("dream:"):
            logger.info(
                "Dream run using model={} (preset={})",
                runtime.model,
                runtime.model_preset or "default",
            )
            
        # 【运行时准入回调】
        # 通知调用者已确定使用的运行时配置。
        if ctx.on_runtime_admitted is not None:
            await ctx.on_runtime_admitted(runtime)
            
        # 【计算历史回放的最大消息数】
        # 根据上下文窗口大小计算最多能回放多少条历史消息。
        replay_max_messages = replay_max_messages_for_context(
            runtime.context_window_tokens
        )
        
        # 【记忆整合】
        # 如果不是临时会话，检查是否需要进行记忆整合（压缩长历史为摘要）。
        if not ctx.ephemeral:
            await self.consolidator.maybe_consolidate_by_tokens(
                session,
                runtime=runtime,
                replay_max_messages=replay_max_messages,
            )
            
        # 【判断是否是子 Agent 消息】
        is_subagent = ctx.kind is TurnKind.SYSTEM and ctx.msg.sender_id == "subagent"

        # 【MessageTool 轮次开始标记】
        # 如果是用户消息且存在 MessageTool，标记轮次开始。
        if ctx.kind is TurnKind.USER and (message_tool := self.tools.get("message")):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # 【构建历史提取参数】
        _hist_kwargs: dict[str, Any] = {
            "max_messages": replay_max_messages,      # 最大消息数限制。
            "max_tokens": self._replay_token_budget(runtime), # Token 预算限制。
            "extend_to_user": is_subagent,            # 子 Agent 消息需要扩展到用户消息边界。
        }
        # 【提取会话历史】
        ctx.history = session.get_history(**_hist_kwargs)
        
        # 【处理 Provider 状态】
        # 某些 LLM Provider（如 Anthropic）支持在多次请求间维持内部状态（如 Prompt Caching）。
        stored_state = session.provider_state
        subagent_followup_persisted = False
        
        # 【子 Agent 后续消息处理】
        if is_subagent:
            # 保留持久的内部投递作为助手记录，但将此次完成呈现为新的后续输入给模型。
            # 不支持 assistant-prefill 的 Provider 会丢弃末尾的助手消息，
            # 因此使用持久化的记录作为当前提示会隐藏独立分发的子 Agent 结果。
            subagent_followup_persisted = self._persist_subagent_followup(
                session,
                ctx.msg,
            )
            if subagent_followup_persisted:
                logger.debug("Subagent result persisted for session {}", ctx.session_key)
                # 在任何可能失败的 Provider 兼容性或提示组装工作之前，
                # 建立一个持久的、可回放的基线。兼容的分阶段状态会在下面的第二次原子保存中替换它。
                session.provider_state = None
                self.sessions.save(session)
            ctx.input_persisted_early = True
            
        # 【记录运行时到投递对象】
        ctx.delivery.record_runtime(runtime)

        # 【构建请求上下文】
        ctx.request_context = self._request_context_for_turn(ctx)
        
        # 【解析运行时上下文块】
        # 仅对用户消息解析运行时上下文（如当前时间、工作区文件列表等）。
        if ctx.kind is TurnKind.USER:
            ctx.runtime_context_blocks = await self._resolve_runtime_context_for_turn(ctx)
            
        staged_provider_state = False
        
        # 【处理 Provider 状态恢复】
        # 如果存在存储的 Provider 状态，且当前 Provider 支持恢复该状态。
        if stored_state is not None and runtime.provider.can_resume_conversation_state(
            stored_state,
            runtime.model,
        ):
            # 构建当前消息（用于追加到 Provider 状态中）。
            current_provider_message = self.context.build_current_message(
                ctx.msg.content,
                media=ctx.msg.media if ctx.kind is TurnKind.USER and ctx.msg.media else None,
                runtime_context_blocks=ctx.runtime_context_blocks,
            )
            
            # 【子 Agent 任务 ID 标记】
            task_id = ctx.msg.metadata.get("subagent_task_id") if is_subagent else None
            already_staged = False
            if isinstance(task_id, str) and task_id:
                # 在消息的 _meta 中添加子 Agent 任务 ID 标记。
                internal_meta = current_provider_message.get("_meta")
                current_provider_message["_meta"] = {
                    **(
                        cast(dict[str, Any], internal_meta)
                        if isinstance(internal_meta, dict)
                        else {}
                    ),
                    _SUBAGENT_PROVIDER_TASK_META: task_id,
                }
                # 检查该任务 ID 是否已经被暂存过（避免重复添加）。
                already_staged = any(
                    isinstance(message.get("_meta"), dict)
                    and cast(dict[str, Any], message["_meta"]).get(
                        _SUBAGENT_PROVIDER_TASK_META
                    )
                    == task_id
                    for message in stored_state.pending_messages
                )
                
            # 【设置 Provider 状态】
            # 如果已经暂存过，直接使用存储的状态；否则追加当前消息。
            ctx.provider_state = (
                stored_state
                if already_staged
                else stored_state.with_pending_messages([
                    *stored_state.pending_messages,
                    current_provider_message,
                ])
            )
            
            # 【持久化分阶段的 Provider 状态】
            if (
                not ctx.ephemeral
                and (ctx.kind is TurnKind.USER or subagent_followup_persisted)
            ):
                session.provider_state = ctx.provider_state
                staged_provider_state = True
        elif stored_state is not None:
            # 如果 Provider 不支持恢复状态，清空它。
            session.provider_state = None
            
        # 【提前持久化用户消息】
        if ctx.kind is TurnKind.USER:
            ctx.input_persisted_early = self._persist_user_message_early(
                ctx.msg,
                session,
                runtime_context_blocks=ctx.runtime_context_blocks,
            )
            # 如果分阶段了 Provider 状态但没有持久化用户消息，回滚 Provider 状态。
            if staged_provider_state and not ctx.input_persisted_early:
                session.provider_state = stored_state
        elif subagent_followup_persisted and staged_provider_state:
            # 在提示组装和第一个模型检查点之前，将可回放基线升级为可恢复状态。
            self.sessions.save(session)
            
        # 【构建初始消息列表】
        # 这是发送给 LLM 的完整消息列表，包含 System Prompt、历史记录和当前消息。
        ctx.initial_messages = self._build_initial_messages(ctx)

        # 【设置默认回调】
        # 如果调用者没有提供进度回调，使用投递对象的默认回调。
        if ctx.on_progress is None:
            ctx.on_progress = ctx.delivery.progress_callback()
        if ctx.on_retry_wait is None:
            ctx.on_retry_wait = ctx.delivery.retry_wait_callback()

    async def _run_turn(self, ctx: TurnContext) -> None:
        """
        【阶段 5: RUN】执行 Agent 循环（LLM 调用 + 工具执行）。
        
        这是最核心的阶段，调用 _run_agent_loop 执行实际的 LLM 推理和工具调用循环。
        """
        runtime = ctx.require_runtime()
        
        # 【记录可见运行开始时间】
        if ctx.visible_run_started_at is None:
            ctx.visible_run_started_at = time.time()
            
        # 通知投递对象轮次正在运行。
        await ctx.delivery.running(started_at=ctx.visible_run_started_at)
        
        # 【核心调用】运行 Agent 循环。
        result = await self._run_agent_loop(
            ctx.initial_messages,
            runtime=runtime,
            on_progress=ctx.on_progress,
            on_stream=ctx.on_stream,
            on_stream_end=ctx.on_stream_end,
            on_retry_wait=ctx.on_retry_wait,
            session=ctx.session,
            channel=ctx.delivery.route.channel,
            chat_id=ctx.delivery.route.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            metadata=ctx.msg.metadata,
            session_key=ctx.session_key,
            original_user_text=ctx.original_user_text,
            pending_queue=ctx.pending_queue,
            ephemeral=ctx.ephemeral,
            run_extra_hooks_for_ephemeral=ctx.run_extra_hooks_for_ephemeral,
            hooks=ctx.hooks,
            hook_factories=ctx.hook_factories,
            turn_scopes=ctx.turn_scopes,
            tools=ctx.tools,
            request_context=ctx.request_context,
            provider_state=ctx.provider_state,
        )
        
        # 【解包结果】
        final_content, _, all_msgs, stop_reason, had_injections = result
        
        # 将结果存储到上下文中。
        ctx.final_content = final_content
        ctx.all_messages = all_msgs
        ctx.stop_reason = stop_reason
        ctx.had_injections = had_injections
        
        # 【处理轮次续传】
        # 如果是用户消息，检查是否需要继续执行（例如内部状态机需要多轮）。
        if ctx.kind is TurnKind.USER:
            await turn_continuation.maybe_continue_turn(ctx)

    async def _persist_turn(self, ctx: TurnContext) -> None:
        """
        【阶段 6: SAVE】持久化轮次结果到会话历史。
        
        这个阶段负责：
        1. 计算轮次延迟。
        2. 保存所有新消息到会话历史（截断过长的工具结果）。
        3. 触发后台记忆整合（如果需要）。
        4. 清除检查点和待处理标记。
        5. 发布轮次持久化事件。
        """
        runtime = ctx.require_runtime()
        session = ctx.require_session()
        
        # 【准备保存边界】
        # 计算哪些消息需要保存（跳过已经提前持久化的消息）。
        turn_continuation.prepare_save_boundary(ctx)

        # 【处理空回复】
        # 如果是用户消息且最终回复为空，使用默认的空回复占位符。
        if (
            ctx.kind is TurnKind.USER
            and (ctx.final_content is None or not ctx.final_content.strip())
            and not ctx.suppress_response
        ):
            ctx.final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        # 【计算轮次延迟】
        # 根据消息类型选择不同的延迟计算起点：
        # - 系统消息或内部续传：从可见运行开始时间计算。
        # - 普通用户消息：从轮次墙钟开始时间计算（包含排队等待时间）。
        latency_started_at = (
            ctx.visible_run_started_at
            if (
                ctx.kind is TurnKind.SYSTEM
                or turn_continuation.internal_continuation_inbound(ctx.msg.metadata)
            )
            and ctx.visible_run_started_at is not None
            else ctx.turn_wall_started_at
        )
        ctx.turn_latency_ms = max(0, int((time.time() - latency_started_at) * 1000))
        
        # 【保存轮次消息】
        self._save_turn(
            session, ctx.all_messages, ctx.save_skip,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        
        # 记录延迟到投递对象。
        ctx.delivery.record_latency(ctx.turn_latency_ms)
        
        # 【后台任务】
        if not ctx.ephemeral:
            # 强制执行文件容量限制，归档过大的会话文件。
            session.enforce_file_cap(
                # 【Python 特殊用法】: functools.partial
                # 使用 partial 预绑定 session_key 参数，创建一个新函数。
                on_archive=partial(self.context.memory.raw_archive, session_key=ctx.session_key)
            )
            # 调度后台记忆整合任务（不阻塞当前响应）。
            self.schedule_background(
                self.consolidator.maybe_consolidate_by_tokens(
                    session,
                    runtime=runtime,
                    replay_max_messages=replay_max_messages_for_context(
                        runtime.context_window_tokens
                    ),
                )
            )
            
        # 【清除状态标记】
        self._clear_pending_user_turn(session)   # 清除待处理的用户轮次标记。
        self._clear_runtime_checkpoint(session)  # 清除运行时检查点。
        
        # 【保存会话】
        self.sessions.save(session)
        
        # 【发布持久化事件】
        if not ctx.ephemeral:
            await self.runtime_event_publisher.session_turn_persisted(
                ctx.msg,
                ctx.session_key,
                turn_id=ctx.turn_id,
                attributes=ctx.attributes,
            )

    async def _prepare_outbound(self, ctx: TurnContext) -> None:
        """
        【阶段 7: RESPOND】组装最终的出站消息。
        
        根据轮次类型和抑制标记，决定是否发送响应以及如何组装响应。
        """
        # 【抑制响应】
        # 如果标记为抑制响应（例如工具已经主动发送了消息），不发送出站消息。
        if ctx.suppress_response:
            ctx.outbound = None
            return
            
        # 【系统消息响应】
        # 系统消息使用后台响应格式。
        if ctx.kind is TurnKind.SYSTEM:
            ctx.outbound = ctx.delivery.background_response(
                ctx.final_content,
                stop_reason=ctx.stop_reason,
                streamed=ctx.streamed_content,
                latency_ms=ctx.turn_latency_ms,
            )
            return
            
        # 【用户消息响应】
        # 使用标准的出站消息组装方法。
        ctx.outbound = self._assemble_outbound(
            ctx.msg,
            cast(str, ctx.final_content), # 【Python 特殊用法】: cast 强制类型转换，安抚类型检查器。
            ctx.stop_reason,
            ctx.had_injections,
            ctx.streamed_content,
            log_content=ctx.require_session().policy.log_content,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        
        # 【临时会话的额外元数据】
        # 临时会话的出站消息需要携带停止原因，供调用者判断。
        if ctx.ephemeral and ctx.outbound is not None:
            ctx.outbound.metadata["_stop_reason"] = ctx.stop_reason

    def _sanitize_persisted_blocks(
        self,
        content: list[object],
        *,
        should_truncate_text: bool = False,
    ) -> list[object]:
        """
        【数据清理】在写入会话历史之前，移除易变的多媒体载荷。
        
        主要处理：
        1. 将 base64 编码的图片替换为占位符文本（避免历史文件过大）。
        2. 截断过长的文本块（如果 should_truncate_text=True）。
        """
        filtered: list[object] = []
        for block in content:
            # 如果不是字典类型，直接保留。
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            block_data = cast(dict[str, Any], block)
            # 【处理 base64 图片】
            # 检查是否是 image_url 类型且 URL 以 "data:image/" 开头（base64 编码）。
            image_url = cast(dict[str, Any], block_data.get("image_url", {}))
            if block_data.get("type") == "image_url" and str(
                image_url.get("url", "")
            ).startswith("data:image/"):
                # 从 _meta 中提取原始文件路径。
                internal_meta = cast(dict[str, Any], block_data.get("_meta") or {})
                path = cast(str, internal_meta.get("path", ""))
                # 替换为占位符文本。
                filtered.append(
                    {"type": "text", "text": image_placeholder_text(path)}
                )
                continue

            # 【处理文本块】
            if block_data.get("type") == "text" and isinstance(
                block_data.get("text"),
                str,
            ):
                text = cast(str, block_data["text"])
                # 如果需要截断且文本过长，执行截断。
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                # 保留其他字段，只更新文本内容。
                filtered.append({**block_data, "text": text})
                continue

            # 其他类型的块直接保留。
            filtered.append(block_data)

        return filtered

    def _save_turn(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        skip: int,
        *,
        turn_latency_ms: int | None = None,
    ) -> None:
        """
        【会话历史保存】将新轮次的消息保存到会话中，截断过大的工具结果。
        
        这个方法负责：
        1. 验证工具调用 ID 的有效性（防止未声明的工具结果污染 Provider 请求）。
        2. 截断过长的工具返回结果。
        3. 清理多媒体载荷（如 base64 图片）。
        4. 为每条消息添加时间戳。
        """
        from datetime import datetime

        # 【构建已声明的工具调用 ID 集合】
        # 遍历会话中所有已有的助手消息，提取其 tool_calls 中的 ID。
        declared_tool_call_ids = {
            str(tc["id"])
            for m in session.messages
            if m.get("role") == "assistant"
            for tc_value in cast(Iterable[object], m.get("tool_calls") or [])
            if isinstance(tc_value, dict)
            for tc in (cast(dict[str, Any], tc_value),)
            if tc.get("id")
        }
        
        # 【构建已履行的工具调用 ID 集合】
        # 遍历会话中所有已有的 tool 角色消息，提取其 tool_call_id。
        fulfilled_tool_call_ids = {
            str(m["tool_call_id"])
            for m in session.messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        
        last_assistant_idx: int | None = None # 记录最后一条助手消息的索引，用于添加延迟标记。
        
        # 【遍历新消息】
        # skip 参数用于跳过已经提前持久化的消息。
        for m in messages[skip:]:
            entry = dict(m) # 复制消息字典，避免修改原始数据。
            
            # 【提取运行时上下文元数据】
            internal_meta = cast(object, entry.pop("_meta", None))
            runtime_context_meta = (
                cast(dict[str, Any], internal_meta).get(
                    RUNTIME_CONTEXT_MESSAGE_META
                )
                if isinstance(internal_meta, dict)
                else None
            )
            
            role, content = entry.get("role"), entry.get("content")
            
            # 【跳过空的助手消息】
            # 空的助手消息会污染会话上下文。
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
                
            # 【处理工具结果消息】
            if role == "tool":
                tool_call_id = entry.get("tool_call_id")
                tool_call_id_str = str(tool_call_id) if tool_call_id else ""
                
                # 【验证工具调用 ID】
                # 未声明的工具结果会破坏未来的 Provider 请求，必须丢弃。
                if (
                    not tool_call_id_str
                    or tool_call_id_str not in declared_tool_call_ids
                    or tool_call_id_str in fulfilled_tool_call_ids
                ):
                    logger.warning(
                        "Dropping invalid tool result {} from session {} during persistence",
                        tool_call_id_str or "(missing id)",
                        session.key,
                    )
                    continue
                    
                # 标记该工具调用 ID 已履行。
                fulfilled_tool_call_ids.add(tool_call_id_str)
                
                # 【截断过长的工具结果】
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    # 如果内容是列表（多模态），清理并截断其中的文本块。
                    filtered = self._sanitize_persisted_blocks(
                        cast(list[object], content),
                        should_truncate_text=True,
                    )
                    if not filtered:
                        # 如果过滤后为空，保留一个占位符，以保持 tool_call/result 配对。
                        filtered = [
                            {"type": "text", "text": "[tool result omitted during persistence]"}
                        ]
                    entry["content"] = filtered
                    
            # 【处理用户消息】
            elif role == "user":
                if isinstance(content, list):
                    # 清理多模态内容（移除 base64 图片等）。
                    filtered = self._sanitize_persisted_blocks(
                        cast(list[object], content),
                    )
                    if not filtered:
                        continue # 如果过滤后为空，跳过此消息。
                    entry["content"] = filtered
                    
                # 如果有运行时上下文元数据，添加到消息中。
                if isinstance(runtime_context_meta, dict):
                    entry[RUNTIME_CONTEXT_HISTORY_META] = runtime_context_meta
                    
            # 【添加时间戳】
            entry.setdefault("timestamp", datetime.now().isoformat())
            
            # 【追加到会话历史】
            session.messages.append(entry)
            
            # 【记录助手消息索引和工具调用 ID】
            if role == "assistant":
                last_assistant_idx = len(session.messages) - 1
                # 将新声明的工具调用 ID 添加到集合中。
                declared_tool_call_ids.update(
                    str(tc["id"])
                    for tc_value in cast(
                        Iterable[object],
                        entry.get("tool_calls") or [],
                    )
                    if isinstance(tc_value, dict)
                    for tc in (cast(dict[str, Any], tc_value),)
                    if tc.get("id")
                )
                
        # 【添加延迟标记】
        # 将轮次延迟添加到最后的助手消息中。
        if turn_latency_ms is not None and last_assistant_idx is not None:
            session.messages[last_assistant_idx]["latency_ms"] = int(turn_latency_ms)
            
        # 更新会话的最后更新时间。
        session.updated_at = datetime.now()
    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """
        【子 Agent 消息持久化】在提示组装之前持久化子 Agent 的后续消息，保持历史记录的持久性。
        
        返回 True 表示新条目已追加；
        返回 False 表示后续消息被去重（相同的 ``subagent_task_id`` 已存在于会话中），
        或者没有值得持久化的内容。
        """
        # 如果消息内容为空，不需要持久化。
        if not msg.content:
            return False
            
        # 【提取子 Agent 任务 ID】
        metadata_value = cast(object, msg.metadata)
        task_id = (
            msg.metadata.get("subagent_task_id")
            if isinstance(metadata_value, dict)
            else None
        )
        
        # 【去重检查】
        # 检查会话中是否已经存在相同 task_id 的子 Agent 结果。
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False # 已存在，跳过。
            
        # 【添加助手消息】
        # 将子 Agent 的结果作为助手消息添加到会话历史。
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",  # 标记为子 Agent 结果。
            subagent_task_id=task_id,          # 记录任务 ID，用于去重。
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """
        【设置检查点】将最新的进行中轮次状态持久化到会话元数据中。
        
        检查点用于在 Agent 崩溃或被取消时恢复部分上下文。
        """
        # 将检查点数据存入会话元数据。
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        # 立即保存会话，确保检查点被持久化。
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        """
        【标记待处理轮次】标记该会话有一个等待处理的用户轮次。
        
        如果 Agent 在处理用户消息时崩溃，可以通过此标记检测到未完成的轮次。
        """
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        """
        【清除待处理标记】清除待处理的用户轮次标记。
        
        在轮次成功完成或被取消恢复后调用。
        """
        # 【Python 特殊用法】: dict.pop(key, default)
        # 使用 pop 安全地移除键，如果键不存在则返回 None 而不抛出异常。
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        """
        【清除检查点】清除运行时检查点。
        
        在轮次成功完成后调用，避免下次启动时误恢复旧检查点。
        """
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        """
        【静态方法】生成检查点消息的唯一键，用于去重和匹配。
        
        返回一个包含消息关键字段的元组，用于比较两条消息是否相同。
        """
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """
        【恢复检查点】在新请求之前，将未完成的轮次具体化到会话历史中。
        
        当 Agent 在工具执行过程中崩溃或被取消时，检查点保存了部分完成的上下文。
        此方法将这些部分完成的消息恢复到会话历史，避免用户丢失已完成的工作。
        
        返回 True 表示成功恢复了检查点；False 表示没有检查点可恢复。
        """
        from datetime import datetime

        # 【读取检查点数据】
        checkpoint = cast(
            object,
            session.metadata.get(self._RUNTIME_CHECKPOINT_KEY),
        )
        if not isinstance(checkpoint, dict):
            return False # 没有有效的检查点。
        checkpoint_data = cast(dict[str, Any], checkpoint)

        # 【提取检查点中的消息】
        # 助手消息（LLM 的回复或工具调用请求）。
        assistant_message = cast(object, checkpoint_data.get("assistant_message"))
        # 已完成的工具结果列表。
        completed_tool_results = cast(
            Iterable[object],
            checkpoint_data.get("completed_tool_results") or [],
        )
        # 待处理的工具调用列表（尚未执行或正在执行的工具）。
        pending_tool_calls = cast(
            Iterable[object],
            checkpoint_data.get("pending_tool_calls") or [],
        )

        # 【构建恢复的消息列表】
        restored_messages: list[dict[str, Any]] = []
        
        # 恢复助手消息。
        if isinstance(assistant_message, dict):
            restored = dict(cast(dict[str, Any], assistant_message))
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
            
        # 恢复已完成的工具结果。
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(cast(dict[str, Any], message))
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
                
        # 【处理待处理的工具调用】
        # 对于尚未完成的工具调用，生成一条错误消息作为工具结果。
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_data = cast(dict[str, Any], tool_call)
            tool_id = tool_call_data.get("id")
            function_data = cast(
                dict[str, Any],
                tool_call_data.get("function") or {},
            )
            name = function_data.get("name") or "tool"
            # 生成一条错误消息，说明任务在工具完成前被中断。
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # 【去重：检查会话历史中是否已包含这些消息】
        # 计算恢复消息与会话历史的重叠部分，避免重复追加。
        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        # 从最大可能的重叠大小开始向下搜索。
        for size in range(max_overlap, 0, -1):
            # 取会话历史的最后 size 条消息。
            existing = session.messages[-size:]
            # 取恢复消息的前 size 条。
            restored = restored_messages[:size]
            # 逐条比较消息键是否相同。
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
                
        # 只追加重叠部分之后的消息。
        appended_messages = restored_messages[overlap:]
        session.messages.extend(appended_messages)
        
        # 【检查 Provider 状态是否需要同步】
        assistant_message_data = (
            cast(dict[str, Any], assistant_message)
            if isinstance(assistant_message, dict)
            else None
        )
        # 检查检查点中的 Provider 状态版本是否匹配。
        provider_state_is_synchronized = (
            checkpoint_data.get(self._PROVIDER_STATE_CHECKPOINT_VERSION_KEY)
            == self._PROVIDER_STATE_CHECKPOINT_VERSION
        )
        
        # 【判断检查点的阶段】
        phase = checkpoint_data.get("phase")
        
        # 精确的最终响应：检查点处于 "final_response" 阶段，
        # 且有助手消息，没有工具结果和待处理的工具调用。
        exact_final_response = (
            phase == "final_response"
            and assistant_message_data is not None
            and assistant_message_data.get("role") == "assistant"
            and not bool(checkpoint_data.get("completed_tool_results"))
            and not bool(checkpoint_data.get("pending_tool_calls"))
        )
        
        # 精确的已完成工具：检查点处于 "tools_completed" 阶段，
        # 且有助手消息，没有待处理的工具调用。
        exact_completed_tools = (
            phase == "tools_completed"
            and assistant_message_data is not None
            and assistant_message_data.get("role") == "assistant"
            and not bool(checkpoint_data.get("pending_tool_calls"))
        )
        
        # 【决定是否清空 Provider 状态】
        # 只有当 Provider 状态已同步，且检查点处于精确的最终响应或已完成工具阶段时，
        # 才保留 Provider 状态。否则清空它，避免状态不一致。
        if not (
            provider_state_is_synchronized
            and (exact_final_response or exact_completed_tools)
        ):
            session.provider_state = None

        # 清除待处理标记和检查点。
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """
        【恢复待处理的用户轮次】关闭一个在崩溃前只持久化了用户消息的轮次。
        
        如果 Agent 在持久化用户消息后、生成回复前崩溃，
        会话历史中会有一条用户消息但没有对应的助手回复。
        此方法检测这种情况，并补充一条错误消息作为助手回复。
        
        返回 True 表示修复了待处理的轮次；False 表示没有待处理的轮次。
        """
        from datetime import datetime

        # 检查是否有待处理的用户轮次标记。
        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        # 【检查最后一条消息是否是用户消息】
        # 如果是，说明 Agent 在生成回复前崩溃了。
        if session.messages and session.messages[-1].get("role") == "user":
            # 补充一条错误消息作为助手回复。
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            # 清空 Provider 状态，避免状态不一致。
            session.provider_state = None
            # 更新会话的最后更新时间。
            session.updated_at = datetime.now()

        # 清除待处理标记。
        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        ephemeral: bool = False,
        _run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        tools: ToolRegistry | None = None,
        persist_user_message: bool = True,
        runtime: LLMRuntime | None = None,
        on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> OutboundMessage | None:
        """
        【直接处理接口】直接处理一条外部消息并返回出站载荷。
        
        这个方法提供了不经过消息总线的直接调用接口，
        适用于测试、API 调用或需要在代码中直接触发 Agent 的场景。
        
        注意：此方法与总线轮次共享会话锁，确保串行执行。
        """
        # 【验证渠道】
        # "system" 渠道保留给内部消息，不允许外部直接调用。
        if channel == "system":
            raise ValueError("channel 'system' is reserved for internal messages")
            
        # 确保 MCP 连接已建立。
        await self._connect_mcp()
        
        # 【构建消息元数据】
        metadata: dict[str, Any] = {}
        if not persist_user_message:
            # 如果不需要持久化用户消息，添加跳过标记。
            metadata[turn_continuation.SKIP_USER_PERSIST_META] = True
            
        # 【构建入站消息对象】
        msg = InboundMessage(
            channel=channel, sender_id=sender_id, chat_id=chat_id,
            content=content, media=media or [], metadata=metadata,
        )
        
        # 【获取会话锁】
        # 共享分发锁，确保直接调用与总线轮次串行执行。
        lock = self._get_session_lock(session_key)
        
        try:
            # 【Python 特殊用法】: async with
            # 获取会话锁后执行处理。
            async with lock:
                # 【构建关键字参数字典】
                # 只包含非 None 的参数，避免覆盖默认值。
                kwargs: dict[str, Any] = {
                    "session_key": session_key,
                    "on_progress": on_progress,
                    "on_stream": on_stream,
                    "on_stream_end": on_stream_end,
                    "ephemeral": ephemeral,
                }
                if _run_extra_hooks_for_ephemeral:
                    kwargs["run_extra_hooks_for_ephemeral"] = True
                if hooks is not None:
                    kwargs["hooks"] = hooks
                if hook_factories is not None:
                    kwargs["hook_factories"] = hook_factories
                if tools is not None:
                    kwargs["tools"] = tools
                if runtime is not None:
                    kwargs["runtime"] = runtime
                if on_runtime_admitted is not None:
                    kwargs["on_runtime_admitted"] = on_runtime_admitted
                if attributes is not None:
                    kwargs["attributes"] = dict(attributes)
                    
                # 【核心调用】处理消息。
                return await self._process_message(
                    msg,
                    **kwargs, # 解包关键字参数。
                )
        finally:
            # 【清理】无论处理成功还是失败，都发布空闲状态事件。
            await self.runtime_event_publisher.run_status_changed(msg, session_key, "idle")
            # 清除轮次状态。
            self.runtime_event_publisher.clear_turn(session_key)

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """
        【会话锁管理】返回共享的会话锁，同时允许空闲的会话条目过期。
        
        使用弱引用字典（WeakValueDictionary）存储锁，
        当会话不再被引用时，锁会自动被垃圾回收，防止内存泄漏。
        """
        # 尝试从弱引用字典中获取现有的锁。
        lock = self._session_locks.get(session_key)
        if lock is None:
            # 如果锁不存在，创建一个新的。
            lock = asyncio.Lock()
            # 存入弱引用字典。
            self._session_locks[session_key] = lock
        return lock
