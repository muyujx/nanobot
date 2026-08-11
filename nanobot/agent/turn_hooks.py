"""Turn-scoped hook assembly for agent runs.

中文：本模块用于为 Agent 的一次运行回合（turn）组装“回合作用域内”的钩子链。
"""

# 从 __future__ 导入 annotations 特性。
# 作用：让类型注解延迟求值，避免在类或函数定义时立即解析复杂类型。
# 这样可以使用更新的类型注解写法，也更适合递归类型、字符串类型注解等场景。
from __future__ import annotations

# 从 collections.abc 导入 Awaitable 和 Callable。
# Awaitable：表示一个可以被 await 的对象，通常用于描述异步函数或协程的返回类型。
# Callable：表示一个可调用对象，比如函数、方法、实现了 __call__ 的类实例等。
from collections.abc import Awaitable, Callable

# 从 dataclasses 导入 dataclass 和 field。
# dataclass：装饰器，用于把普通类自动变成数据类，自动生成 __init__、__repr__ 等。
# field：用于给数据类字段配置默认值、默认工厂等高级行为。
from dataclasses import dataclass, field

# 从 pathlib 导入 Path。
# Path 是面向对象的文件路径类型，用于表示文件路径或目录路径。
from pathlib import Path

# 从 typing 导入 Any。
# Any 表示任意类型，通常用于不确定类型或者允许任意值的场景。
from typing import Any

# 从 loguru 导入 logger。
# loguru 是一个日志库，logger 用于输出调试、信息、异常等日志。
from loguru import logger

# 从 nanobot.agent.hook 模块导入钩子相关类型。
from nanobot.agent.hook import (
    # AgentHook：Agent 钩子的基础类型或接口，表示一个可以挂载到 Agent 生命周期中的钩子对象。
    AgentHook,
    # AgentTurnHookContext：一次 Agent 回合的上下文对象，保存当前回合的参数和环境信息。
    AgentTurnHookContext,
    # AgentTurnHookFactory：回合钩子工厂类型，通常是一个可调用对象，接收上下文并返回钩子或 None。
    AgentTurnHookFactory,
    # CompositeHook：组合钩子，用于把多个钩子包装成一个钩子，依次或统一调用。
    CompositeHook,
)

# 从 nanobot.agent.progress_hook 导入 AgentProgressHook。
# AgentProgressHook：负责处理进度、流式输出、迭代次数等信息的钩子。
from nanobot.agent.progress_hook import AgentProgressHook


# @dataclass(slots=True) 表示这是一个数据类。
# slots=True 的作用：
# 1. 自动生成 __slots__，减少实例内存占用。
# 2. 禁止给实例动态添加未在字段中声明的属性。
# 3. 对属性访问有一定优化，同时约束更严格。
@dataclass(slots=True)
class AgentTurnHookSpec:
    """Inputs needed to build the hook chain for one agent turn.

    中文：构建一个 Agent 回合钩子链所需要的输入参数。
    """

    # on_progress：进度回调函数。
    # 类型是 Callable[..., Awaitable[None]] | None：
    # - Callable[...] 表示这是一个可调用对象。
    # - ... 表示可以接收任意数量和类型的参数。
    # - Awaitable[None] 表示调用后返回一个可 await 的对象，最终结果类型是 None。
    # - | None 表示这个字段也可以为 None。
    # 默认值为 None，表示调用方可以不传进度回调。
    on_progress: Callable[..., Awaitable[None]] | None = None

    # on_stream：流式输出回调函数。
    # 类型是 Callable[[str], Awaitable[None]] | None：
    # - Callable[[str], ...] 表示这个函数接收一个 str 参数。
    # - Awaitable[None] 表示返回值可以被 await，最终返回 None。
    # - | None 表示可以为空。
    # 通常用于实时输出模型生成的文本片段。
    on_stream: Callable[[str], Awaitable[None]] | None = None

    # on_stream_end：流式输出结束时的回调函数。
    # 类型是 Callable[..., Awaitable[None]] | None：
    # - 可以接收任意参数。
    # - 返回一个可 await 的对象，最终结果为 None。
    # - 可以为 None。
    # 通常用于在流式输出完成后做清理、通知或收尾工作。
    on_stream_end: Callable[..., Awaitable[None]] | None = None

    # channel：消息渠道。
    # 默认值为 "cli"，表示命令行环境。
    # 可能还会有其他渠道，例如 web、api、discord 等。
    channel: str = "cli"

    # chat_id：会话 ID。
    # 默认值为 "direct"，表示直接对话或直接调用场景。
    # 用于区分不同聊天、会话或请求上下文。
    chat_id: str = "direct"

    # message_id：消息 ID。
    # 类型是 str | None，表示可以是字符串，也可以是 None。
    # 默认值为 None，表示调用方可以不提供消息 ID。
    message_id: str | None = None

    # metadata：附加元数据。
    # 类型是 dict[str, Any] | None：
    # - dict[str, Any] 表示键为字符串、值为任意类型的字典。
    # - | None 表示可以为 None。
    # 默认值为 None，调用方可以传入额外的上下文信息。
    metadata: dict[str, Any] | None = None

    # session_key：会话标识。
    # 类型是 str | None。
    # 默认值为 None。
    # 用于标记当前 Agent 运行属于哪个会话，方便持久化、恢复、隔离上下文。
    session_key: str | None = None

    # workspace：工作目录。
    # 类型是 Path | None。
    # 默认值为 None。
    # 如果 Agent 需要读写文件、执行工具或访问项目目录，这个字段可以指定工作路径。
    workspace: Path | None = None

    # tool_hint_max_length：工具提示最大长度。
    # 默认值为 40。
    # 用于控制展示工具提示信息时的最大字符长度，避免日志或 UI 过长。
    tool_hint_max_length: int = 40

    # on_iteration：每轮迭代回调。
    # 类型是 Callable[[int], None] | None：
    # - Callable[[int], None] 表示接收一个 int 参数，返回 None。
    # - | None 表示可以为空。
    # 通常用于监听 Agent 当前执行到第几轮迭代。
    on_iteration: Callable[[int], None] | None = None

    # registered_hook_factories：已注册的回合钩子工厂列表。
    # 类型是 list[AgentTurnHookFactory]。
    # field(default_factory=list) 的作用：
    # - 不能直接写默认值 []，因为所有实例会共享同一个列表，导致状态污染。
    # - default_factory=list 表示每次创建实例时调用 list() 生成一个新的空列表。
    registered_hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)

    # turn_hook_factories：当前回合专用的钩子工厂列表。
    # 同样使用 field(default_factory=list)，确保每个实例拥有独立列表。
    # 与 registered_hook_factories 相比，这里更偏向临时、本次回合有效的钩子工厂。
    turn_hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)

    # registered_hooks：已注册的现成钩子列表。
    # 类型是 list[AgentHook]。
    # default_factory=list 保证每个实例拥有独立空列表。
    # 这些钩子不是通过工厂创建，而是直接追加到钩子链中。
    registered_hooks: list[AgentHook] = field(default_factory=list)

    # turn_hooks：当前回合专用的现成钩子列表。
    # 类型是 list[AgentHook]。
    # default_factory=list 保证每个实例拥有独立空列表。
    # 这些钩子通常只对当前回合有效。
    turn_hooks: list[AgentHook] = field(default_factory=list)

    # ephemeral：是否是临时运行。
    # True 表示这次 Agent 运行是短暂的、一次性的、可能不需要完整钩子链。
    # False 表示普通运行。
    ephemeral: bool = False

    # run_extra_hooks_for_ephemeral：临时运行是否仍然执行额外钩子。
    # 默认 False。
    # 如果 ephemeral=True 且这个字段为 False，则只返回 progress_hook。
    # 如果 ephemeral=True 且这个字段为 True，则仍然继续构建完整钩子链。
    run_extra_hooks_for_ephemeral: bool = False

    # attributes：附加属性。
    # 类型是 dict[str, Any] | None。
    # 默认值为 None。
    # 与 metadata 类似，但可能用于内部属性传递或钩子自定义状态。
    attributes: dict[str, Any] | None = None


# build_agent_turn_hook 是一个函数。
# 参数 spec 的类型是 AgentTurnHookSpec，包含构建钩子链所需的全部配置。
# 返回值类型是 AgentHook，表示最终构建出来的单个钩子对象。
# 即使内部是多个钩子，也可能通过 CompositeHook 包装成一个 AgentHook。
def build_agent_turn_hook(spec: AgentTurnHookSpec) -> AgentHook:
    """Build the hook chain used by ``AgentRunner`` for one turn.

    中文：构建 ``AgentRunner`` 在一个回合中使用的钩子链。
    """

    # 创建 AgentProgressHook 实例。
    # 这个钩子负责处理进度、流式输出、流结束、会话标识、工具提示长度、迭代回调等。
    # 变量 progress_hook 保存这个进度钩子对象。
    progress_hook = AgentProgressHook(
        # 把 spec 中的 on_progress 传给进度钩子。
        # 如果 spec.on_progress 是 None，进度钩子内部需要自行判断是否调用。
        on_progress=spec.on_progress,
        # 把 spec 中的 on_stream 传给进度钩子。
        # 用于流式输出模型生成内容。
        on_stream=spec.on_stream,
        # 把 spec 中的 on_stream_end 传给进度钩子。
        # 用于流式输出结束后的处理。
        on_stream_end=spec.on_stream_end,
        # 把 spec 中的 session_key 传给进度钩子。
        # 用于标识当前会话。
        session_key=spec.session_key,
        # 把 spec 中的 tool_hint_max_length 传给进度钩子。
        # 用于限制工具提示长度。
        tool_hint_max_length=spec.tool_hint_max_length,
        # 把 spec 中的 on_iteration 传给进度钩子。
        # 用于监听 Agent 的迭代次数。
        on_iteration=spec.on_iteration,
    )

    # 如果满足两个条件：
    # 1. spec.ephemeral 为 True，表示这是临时运行。
    # 2. spec.run_extra_hooks_for_ephemeral 为 False，表示临时运行不需要额外钩子。
    # 则直接返回 progress_hook，不构建更复杂的钩子链。
    if spec.ephemeral and not spec.run_extra_hooks_for_ephemeral:
        # 返回唯一的进度钩子。
        return progress_hook

    # 创建 AgentTurnHookContext 上下文对象。
    # 这个对象会传递给各个钩子工厂，让工厂根据当前回合信息决定是否创建钩子。
    # 变量 turn_context 保存当前回合上下文。
    turn_context = AgentTurnHookContext(
        # 传入进度回调。
        # 钩子或工厂可以在需要时调用这个回调报告进度。
        on_progress=spec.on_progress,
        # 传入工作目录。
        # 如果工具或钩子需要访问文件路径，可以使用这个字段。
        workspace=spec.workspace,
        # 传入消息渠道。
        # 例如 cli、web、api 等。
        channel=spec.channel,
        # 传入会话 ID。
        # 用于区分不同聊天或请求。
        chat_id=spec.chat_id,
        # 传入消息 ID。
        # 可能用于追踪某条具体消息。
        message_id=spec.message_id,
        # 传入会话 key。
        # 用于会话级别的上下文隔离或持久化。
        session_key=spec.session_key,
        # 传入 metadata。
        # dict(spec.metadata or {}) 的含义：
        # - spec.metadata or {}：如果 spec.metadata 为 None 或假值，则使用空字典 {}。
        # - dict(...)：基于传入字典创建一个新的字典副本。
        # 这样做的目的：
        # 1. 避免 None 导致错误。
        # 2. 避免多个上下文共享同一个字典导致意外修改。
        metadata=dict(spec.metadata or {}),
        # 传入 attributes。
        # 和 metadata 类似，同样使用 or {} 兜底，并用 dict(...) 复制一份。
        attributes=dict(spec.attributes or {}),
        # 传入 ephemeral 标志。
        # 让上下文也知道当前是否是临时运行。
        ephemeral=spec.ephemeral,
    )

    # 初始化钩子链列表。
    # 类型注解 list[AgentHook] 表示这个列表只能存放 AgentHook 类型的对象。
    # 初始列表中先放入 progress_hook，因为进度钩子几乎总是需要存在。
    hook_chain: list[AgentHook] = [progress_hook]

    # 遍历所有“已注册的钩子工厂”。
    # 每个 factory 都是一个可调用对象，接收 turn_context，返回 AgentHook 或 None。
    for factory in spec.registered_hook_factories:
        # 使用 try 包裹工厂调用。
        # 目的是防止某一个钩子工厂抛出异常后，导致整个 Agent 回合无法启动。
        try:
            # 调用工厂函数，把当前回合上下文传进去。
            # created_hook 是工厂返回的结果。
            # 它可能是一个 AgentHook，也可能是 None。
            created_hook = factory(turn_context)
        # 捕获所有 Exception 及其子类异常。
        # 这里不捕获 BaseException，因为 KeyboardInterrupt、SystemExit 等通常不应被吞掉。
        except Exception:
            # logger.exception 会记录错误日志，并自动附带当前异常的堆栈信息。
            # loguru 的字符串中使用 {} 作为占位符。
            # 这里的 {} 会被 factory 的 repr 或 str 形式替换。
            # 原日志文本保持不变，仅在此处添加中文注释说明。
            logger.exception("Agent turn hook factory failed: {}", factory)
            # continue 表示跳过当前工厂，继续处理下一个工厂。
            # 即使某个工厂失败，也不影响后续钩子构建。
            continue

        # 只有当工厂返回非 None 时，才把创建出来的钩子加入钩子链。
        if created_hook is not None:
            # append 会把 created_hook 添加到 hook_chain 列表末尾。
            hook_chain.append(created_hook)

    # 把 spec.registered_hooks 中的所有现成钩子一次性追加到 hook_chain。
    # extend 的作用是把一个可迭代对象中的所有元素逐个添加到列表末尾。
    # 这里的顺序是：先工厂创建的钩子，再直接注册的现成钩子。
    hook_chain.extend(spec.registered_hooks)

    # 遍历所有“当前回合专用的钩子工厂”。
    # 这些工厂可能来自本次调用临时传入的配置，而不是全局注册。
    for factory in spec.turn_hook_factories:
        # 同样使用 try 防止单个工厂异常影响整体流程。
        try:
            # 调用当前回合钩子工厂。
            # created_hook 保存工厂返回的钩子对象或 None。
            created_hook = factory(turn_context)
        # 捕获工厂执行过程中抛出的异常。
        except Exception:
            # 记录异常日志，并附带工厂信息。
            # 原日志文本保持不变。
            logger.exception("Agent turn hook factory failed: {}", factory)
            # 跳过当前失败工厂，继续处理下一个。
            continue

        # 如果工厂成功返回了钩子对象，则加入钩子链。
        if created_hook is not None:
            # 将当前回合工厂创建的钩子追加到列表末尾。
            hook_chain.append(created_hook)

    # 把 spec.turn_hooks 中的所有现成回合钩子追加到 hook_chain。
    # 这是钩子链构建过程中的最后一批钩子。
    hook_chain.extend(spec.turn_hooks)

    # 返回最终钩子。
    # 这里使用了 Python 的条件表达式：
    # CompositeHook(hook_chain) if len(hook_chain) > 1 else progress_hook
    # 等价于：
    # if len(hook_chain) > 1:
    #     return CompositeHook(hook_chain)
    # else:
    #     return progress_hook
    #
    # 如果 hook_chain 长度大于 1，说明除了 progress_hook 之外还有其他钩子。
    # 此时使用 CompositeHook 把多个钩子组合成一个钩子对象返回。
    # 如果 hook_chain 长度等于 1，说明只有 progress_hook。
    # 此时直接返回 progress_hook，避免不必要的包装。
    return CompositeHook(hook_chain) if len(hook_chain) > 1 else progress_hook