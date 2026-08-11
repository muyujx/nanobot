"""Shared lifecycle hook primitives for agent runs.

[中文翻译] 用于 agent 运行生命周期的共享钩子（hook）基础组件。
"""

# 导入 __future__ 中的 annotations 特性。
# 作用：让类型注解延迟求值，支持更新的类型写法，例如内置泛型、联合类型等。
from __future__ import annotations

# 从 collections.abc 导入类型注解工具：
# Awaitable：表示一个可以被 await 的对象，通常用于异步函数返回值。
# Callable：表示可调用对象，例如函数、方法、lambda、实现 __call__ 的对象。
from collections.abc import Awaitable, Callable

# 从 dataclasses 导入 dataclass 和 field。
# dataclass：装饰器，用于把类变成“数据类”，自动生成 __init__、__repr__、__eq__ 等。
# field：用于更精细地控制 dataclass 字段，例如给可变默认值指定 default_factory。
from dataclasses import dataclass, field

# 从 pathlib 导入 Path。
# Path 用于表示文件系统路径，比传统字符串路径更面向对象。
from pathlib import Path

# 从 typing 导入 Any。
# Any 表示“任意类型”，用于无法或不想限定具体类型的场景。
from typing import Any

# 从 loguru 导入 logger。
# loguru 是一个日志库，这里用 logger 记录钩子执行异常。
from loguru import logger

# 从 nanobot.providers.base 导入两个类型：
# LLMResponse：大模型响应对象，通常包含模型返回内容、停止原因、usage 等。
# ToolCallRequest：工具调用请求对象，通常包含工具名、参数、调用 ID 等。
from nanobot.providers.base import LLMResponse, ToolCallRequest


# @dataclass 会自动根据字段生成构造方法等。
# slots=True 表示让 dataclass 生成 __slots__，用于限制实例属性、降低内存占用。
@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state exposed to runner hooks.

    [中文翻译] 暴露给运行器钩子的、每次迭代可变的上下文状态。
    """

    # iteration：当前 agent 运行中的迭代序号。
    # agent 通常会经历多轮“模型响应 -> 工具调用 -> 再次模型响应”的循环，
    # 这个字段用于标识当前处于第几次迭代。
    iteration: int

    # messages：当前消息历史。
    # 类型是 list[dict[str, Any]]，也就是一个由消息字典组成的列表。
    # 每个 dict 通常包含 role、content、tool_call 相关字段等。
    messages: list[dict[str, Any]]

    # response：当前迭代中模型返回的响应对象。
    # 类型是 LLMResponse | None，表示可能有响应，也可能尚未产生响应。
    # 默认值是 None。
    response: LLMResponse | None = None

    # usage：模型调用消耗的资源统计，例如 token 数量。
    # 类型是 dict[str, int]，例如 {"input_tokens": 10, "output_tokens": 20}。
    # 使用 field(default_factory=dict) 而不是 default={}，
    # 是为了避免多个实例共享同一个可变字典对象。
    usage: dict[str, int] = field(default_factory=dict)

    # tool_calls：模型在当前迭代中请求执行的工具调用列表。
    # 每个元素都是 ToolCallRequest。
    # default_factory=list 表示每个实例都会得到一个新的空列表。
    tool_calls: list[ToolCallRequest] = field(default_factory=list)

    # tool_results：工具执行结果列表。
    # 类型是 list[Any]，因为不同工具返回值类型可能完全不同。
    tool_results: list[Any] = field(default_factory=list)

    # tool_events：工具生命周期事件列表。
    # 每个事件是一个 dict[str, str]，例如记录事件名、工具名、状态等。
    tool_events: list[dict[str, str]] = field(default_factory=list)

    # streamed_content：是否已经流式输出过正文内容。
    # True 表示钩子或运行器已经通过流式方式发送过 content。
    streamed_content: bool = False

    # streamed_reasoning：是否已经流式输出过推理/思考内容。
    # True 表示 reasoning 内容已经被流式发送过。
    streamed_reasoning: bool = False

    # stream_continues_current_message：流式内容是否继续当前消息。
    # True 表示新的流式片段应追加到当前消息，而不是开启新消息。
    stream_continues_current_message: bool = False

    # final_content：最终文本内容。
    # 如果当前迭代已经能确定最终回复内容，则放在这里；否则为 None。
    final_content: str | None = None

    # stop_reason：模型停止原因。
    # 例如 end_turn、tool_calls、length 等，具体取决于 provider。
    stop_reason: str | None = None

    # error：当前迭代中出现的错误信息。
    # 如果没有错误则为 None。
    error: str | None = None

    # session_key：会话标识。
    # 用于标记当前上下文属于哪个会话；如果不可用则为 None。
    session_key: str | None = None


# 另一个 dataclass，表示“一次完整 run”级别的钩子上下文。
# slots=True 同样用于限制实例属性、降低内存占用。
@dataclass(slots=True)
class AgentRunHookContext:
    """Run-level state snapshot exposed to runner hooks.

    [中文翻译] 暴露给运行器钩子的 run 级别状态快照。
    """

    # messages：run 级别的消息历史。
    # 与 AgentHookContext.messages 类似，但这里表示整个 run 的快照。
    messages: list[dict[str, Any]]

    # final_content：整个 run 的最终文本内容。
    # 如果 run 成功产生最终回复，则通常会被填充。
    final_content: str | None = None

    # tools_used：本次 run 中使用过的工具名列表。
    # default_factory=list 保证每个实例有独立空列表。
    tools_used: list[str] = field(default_factory=list)

    # usage：本次 run 的资源消耗统计。
    # 例如累计 token 使用量。
    usage: dict[str, int] = field(default_factory=dict)

    # stop_reason：run 结束原因。
    stop_reason: str | None = None

    # error：run 级别错误信息。
    error: str | None = None

    # tool_events：run 级别工具事件列表。
    tool_events: list[dict[str, str]] = field(default_factory=list)

    # had_injections：本次 run 中是否发生过消息注入。
    # 例如外部事件、系统消息、用户追加输入等可能会被视作 injection。
    had_injections: bool = False

    # exception：run 中捕获到的异常对象。
    # 类型是 BaseException | None。
    # 使用 BaseException 而不是 Exception，表示理论上可以包含更底层的异常，
    # 例如 KeyboardInterrupt、SystemExit 等。
    exception: BaseException | None = None


# 另一个 dataclass，表示“单个 turn”级别、用于构造 per-turn hooks 的输入。
@dataclass(slots=True)
class AgentTurnHookContext:
    """Turn-local inputs available when constructing per-turn hooks.

    [中文翻译] 构造每回合钩子时可用的回合级输入。
    """

    # on_progress：可选的异步进度回调。
    # 类型 Callable[..., Awaitable[None]] | None 表示：
    #   - 它可以接收任意参数，... 表示不限制参数签名；
    #   - 它返回一个可 await 的对象；
    #   - await 后返回 None；
    #   - 整个字段也可以是 None。
    on_progress: Callable[..., Awaitable[None]] | None = None

    # workspace：当前 turn 关联的工作目录。
    # Path | None 表示可能没有工作目录。
    workspace: Path | None = None

    # channel：消息来源渠道。
    # 默认值是 "cli"，表示命令行。
    channel: str = "cli"

    # chat_id：聊天/会话 ID。
    # 默认值是 "direct"，表示直接对话或本地 CLI 场景。
    chat_id: str = "direct"

    # message_id：单条消息 ID。
    # 如果当前 turn 有明确消息 ID，则填充；否则为 None。
    message_id: str | None = None

    # session_key：会话 key。
    # 用于在多个 turn 之间标识同一个会话。
    session_key: str | None = None

    # metadata：turn 级别元数据。
    # 可保存任意附加信息，例如 cron trigger、请求来源、追踪 ID 等。
    metadata: dict[str, Any] = field(default_factory=dict)

    # ephemeral：是否是短暂/临时 turn。
    # True 通常表示不应持久化、不进入长期历史，或只用于临时执行。
    ephemeral: bool = False

    # attributes：额外属性字典。
    # 用于传递自定义扩展字段。
    attributes: dict[str, Any] = field(default_factory=dict)


# AgentHook 是所有 agent 生命周期钩子的最小基类。
# 子类可以覆盖其中任意异步方法，在 agent 运行过程中的不同阶段插入逻辑。
class AgentHook:
    """Minimal lifecycle surface for shared runner customization.

    [中文翻译] 用于共享运行器定制的最小生命周期接口。
    """

    # 构造方法。
    def __init__(self, reraise: bool = False) -> None:
        # self._reraise：私有实例属性。
        # 用于标记该 hook 是否希望异常被重新抛出，而不是被 CompositeHook 吞掉。
        # 默认 False，表示允许组合钩子捕获并记录异常。
        self._reraise = reraise

    # 同步方法：声明该 hook 是否希望接收流式内容。
    def wants_streaming(self) -> bool:
        # 默认返回 False，表示不需要 on_stream 流式增量。
        return False

    # run 开始前的异步钩子。
    async def before_run(self, context: AgentRunHookContext) -> None:
        # pass 表示默认什么都不做。
        # 子类可覆盖此方法，实现 run 前初始化逻辑。
        pass

    # run 正常结束后的异步钩子。
    async def after_run(self, context: AgentRunHookContext) -> None:
        # 默认空实现。
        pass

    # run 发生错误时的异步钩子。
    async def on_error(self, context: AgentRunHookContext) -> None:
        # 默认空实现。
        pass

    # run 最终清理阶段的异步钩子。
    # 无论成功、失败，通常都可以用来做 finally 式清理。
    async def on_finally(self, context: AgentRunHookContext) -> None:
        # 默认空实现。
        pass

    # 每次迭代开始前的异步钩子。
    async def before_iteration(self, context: AgentHookContext) -> None:
        # 默认空实现。
        pass

    # 收到流式文本增量时的异步钩子。
    # delta 是本次流式新增的一小段文本。
    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        # 默认空实现。
        pass

    # 流式结束时的异步钩子。
    # 参数列表中的 * 表示 resuming 必须用关键字传参。
    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        # resuming：是否只是阶段性结束、后面还会继续流式输出。
        # 默认空实现。
        pass

    # provider 侧工具生命周期事件钩子。
    async def on_provider_tool_event(
        # self：当前钩子实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # event：provider 上报的工具事件字典。
        event: dict[str, Any],
    # 返回 None，表示该异步方法不返回业务值。
    ) -> None:
        """Observe a provider-hosted tool lifecycle event.

        [中文翻译] 观察由 provider 托管的工具生命周期事件。
        """
        # 默认空实现。
        pass

    # 在执行一批工具调用之前的异步钩子。
    async def before_execute_tools(self, context: AgentHookContext) -> None:
        # 默认空实现。
        pass

    # 在执行单个工具调用之前的异步钩子。
    async def before_execute_tool(
        # self：当前钩子实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # tool_call：本次要执行的工具调用请求。
        tool_call: ToolCallRequest,
        # tool：实际工具对象或工具实现。
        # 使用 Any 是因为工具对象类型可能各不相同。
        tool: Any,
        # params：传给工具的参数。
        # 使用 Any 是因为参数结构可能因工具而异。
        params: Any,
    # 返回 None。
    ) -> None:
        # 默认空实现。
        pass

    # 在单个工具调用成功执行后的异步钩子。
    async def after_execute_tool(
        # self：当前钩子实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # tool_call：本次执行的工具调用请求。
        tool_call: ToolCallRequest,
        # tool：实际工具对象或工具实现。
        tool: Any,
        # params：传给工具的参数。
        params: Any,
        # result：工具执行结果。
        result: Any,
    # 返回 None。
    ) -> None:
        # 默认空实现。
        pass

    # 在单个工具调用执行出错时的异步钩子。
    async def on_execute_tool_error(
        # self：当前钩子实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # tool_call：本次执行的工具调用请求。
        tool_call: ToolCallRequest,
        # tool：实际工具对象或工具实现。
        tool: Any,
        # params：传给工具的参数。
        params: Any,
        # error：错误对象或错误信息。
        # 使用 Any 是因为异常、字符串、结构化错误都可能出现。
        error: Any,
    # 返回 None。
    ) -> None:
        # 默认空实现。
        pass

    # 输出 reasoning/思考内容的异步钩子。
    # reasoning_content 可能是一段增量文本，也可能为 None。
    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        # 默认空实现。
        pass

    # reasoning 流结束时的异步钩子。
    async def emit_reasoning_end(self) -> None:
        """Mark the end of an in-flight reasoning stream.

        Hooks that buffer ``emit_reasoning`` chunks (for in-place UI updates)
        flush and freeze the rendered group here. One-shot hooks ignore.
        """
        """
        [中文翻译] 标记一个正在进行中的 reasoning 流结束。

        那些会缓冲 ``emit_reasoning`` 片段（用于原地更新 UI）的钩子，
        会在这里刷新并冻结已经渲染的分组。一次性钩子可以忽略该事件。
        """
        # 默认空实现。
        pass

    # 每次迭代结束后的异步钩子。
    async def after_iteration(self, context: AgentHookContext) -> None:
        # 默认空实现。
        pass

    # 同步方法：最终内容定稿钩子。
    # 它允许钩子在最终内容返回前修改 content。
    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        # 默认原样返回 content。
        # 子类可以覆盖它，实现过滤、格式化、追加说明等逻辑。
        return content


# 这是一个类型别名。
# AgentTurnHookFactory 表示一个函数类型：
#   - 接收一个 AgentTurnHookContext；
#   - 返回 AgentHook 或 None。
# 通常用于根据 turn 上下文动态创建钩子。
AgentTurnHookFactory = Callable[[AgentTurnHookContext], AgentHook | None]


# CompositeHook：组合钩子。
# 它继承 AgentHook，并把多个子钩子组合成一个钩子对外暴露。
class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks.

    Error isolation: async methods catch and log per-hook exceptions
    so a faulty custom hook cannot crash the agent loop.
    ``finalize_content`` is a pipeline (no isolation — bugs should surface).
    """

    """
    [中文翻译] 扇出钩子：把调用委托给一个有序的钩子列表。

    错误隔离：异步方法会捕获并记录每个子钩子抛出的异常，
    这样某个有问题的自定义钩子就不会导致 agent 主循环崩溃。
    ``finalize_content`` 是一个管道（不做隔离——bug 应该暴露出来）。
    """

    # __slots__ 用于显式声明实例属性。
    # 这里只允许/声明 _hooks 这一个实例属性，有助于节省内存并限制动态属性。
    __slots__ = ("_hooks",)

    # 构造方法。
    def __init__(self, hooks: list[AgentHook]) -> None:
        # 调用父类 AgentHook.__init__。
        # 没有传 reraise，因此默认 reraise=False。
        super().__init__()

        # 把传入的 hooks 复制成一个新 list。
        # 这样外部后续修改原 list，不会直接影响 CompositeHook 内部列表。
        # 注意这是浅拷贝：列表是新的，但列表里的 hook 对象仍是同一批对象。
        self._hooks = list(hooks)

    # 重写 wants_streaming。
    def wants_streaming(self) -> bool:
        # any(...) 表示只要任意子 hook 需要流式输出，就返回 True。
        # 括号内是生成器表达式，惰性地依次调用每个 h.wants_streaming()。
        # any 具有短路特性：一旦发现 True，就不会继续检查后面的 hook。
        return any(h.wants_streaming() for h in self._hooks)

    # 私有异步辅助方法：安全地调用每个 hook 的指定方法。
    async def _for_each_hook_safe(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        # method_name：要调用的方法名字符串。
        # *args：收集任意数量的位置参数。
        # **kwargs：收集任意数量的关键字参数。
        # 通过这种方式，可以统一处理 before_run、after_run、on_stream 等各种钩子。

        # 遍历所有子 hook。
        for h in self._hooks:
            # getattr(h, "_reraise", False)：
            # 尝试读取 hook 的 _reraise 属性；如果不存在，默认返回 False。
            # 如果 _reraise 为 True，表示这个 hook 希望异常不要被吞掉。
            if getattr(h, "_reraise", False):
                # getattr(h, method_name)：根据方法名动态获取 bound method。
                # 然后传入 *args 和 **kwargs 调用它。
                # 因为钩子方法大多是 async def，所以这里需要 await。
                await getattr(h, method_name)(*args, **kwargs)

                # continue 跳过下面的 try/except 错误隔离逻辑。
                continue

            # 对普通 hook，捕获并记录异常，避免单个 hook 崩溃影响整个 agent。
            try:
                # 动态获取方法并异步调用。
                await getattr(h, method_name)(*args, **kwargs)
            # 只捕获 Exception，不捕获 BaseException。
            # 这样 KeyboardInterrupt、SystemExit 等更严重异常仍可能向上传播。
            except Exception:
                # logger.exception 会记录当前异常堆栈。
                # loguru 的格式化占位符是 {}。
                # 第一个 {} 会被 method_name 替换。
                # 第二个 {} 会被 type(h).__name__ 替换，也就是 hook 类名。
                logger.exception("AgentHook.{} error in {}", method_name, type(h).__name__)

    # 重写 before_iteration：委托给所有子 hook。
    async def before_iteration(self, context: AgentHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 before_iteration。
        await self._for_each_hook_safe("before_iteration", context)

    # 重写 before_run：委托给所有子 hook。
    async def before_run(self, context: AgentRunHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 before_run。
        await self._for_each_hook_safe("before_run", context)

    # 重写 after_run：委托给所有子 hook。
    async def after_run(self, context: AgentRunHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 after_run。
        await self._for_each_hook_safe("after_run", context)

    # 重写 on_error：委托给所有子 hook。
    async def on_error(self, context: AgentRunHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 on_error。
        await self._for_each_hook_safe("on_error", context)

    # 重写 on_finally：委托给所有子 hook。
    async def on_finally(self, context: AgentRunHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 on_finally。
        await self._for_each_hook_safe("on_finally", context)

    # 重写 on_stream：委托给所有子 hook。
    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 on_stream。
        await self._for_each_hook_safe("on_stream", context, delta)

    # 重写 on_stream_end：委托给所有子 hook。
    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        # 注意：resuming 是关键字参数，因此这里显式用 resuming=resuming 传递。
        await self._for_each_hook_safe("on_stream_end", context, resuming=resuming)

    # 重写 on_provider_tool_event：委托给所有子 hook。
    async def on_provider_tool_event(
        # self：当前 CompositeHook 实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # event：provider 工具事件。
        event: dict[str, Any],
    # 返回 None。
    ) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 on_provider_tool_event。
        await self._for_each_hook_safe("on_provider_tool_event", context, event)

    # 重写 before_execute_tools：委托给所有子 hook。
    async def before_execute_tools(self, context: AgentHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 before_execute_tools。
        await self._for_each_hook_safe("before_execute_tools", context)

    # 重写 before_execute_tool：委托给所有子 hook。
    async def before_execute_tool(
        # self：当前 CompositeHook 实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # tool_call：工具调用请求。
        tool_call: ToolCallRequest,
        # tool：工具对象。
        tool: Any,
        # params：工具参数。
        params: Any,
    # 返回 None。
    ) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 before_execute_tool。
        await self._for_each_hook_safe("before_execute_tool", context, tool_call, tool, params)

    # 重写 after_execute_tool：委托给所有子 hook。
    async def after_execute_tool(
        # self：当前 CompositeHook 实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # tool_call：工具调用请求。
        tool_call: ToolCallRequest,
        # tool：工具对象。
        tool: Any,
        # params：工具参数。
        params: Any,
        # result：工具执行结果。
        result: Any,
    # 返回 None。
    ) -> None:
        # 调用内部辅助方法。
        # 因为参数较多，这里分行传递，增强可读性。
        await self._for_each_hook_safe(
            # 第一个参数是方法名。
            "after_execute_tool",
            # 下面依次是位置参数。
            context,
            tool_call,
            tool,
            params,
            result,
        # 结束 _for_each_hook_safe 调用。
        )

    # 重写 on_execute_tool_error：委托给所有子 hook。
    async def on_execute_tool_error(
        # self：当前 CompositeHook 实例。
        self,
        # context：当前迭代上下文。
        context: AgentHookContext,
        # tool_call：工具调用请求。
        tool_call: ToolCallRequest,
        # tool：工具对象。
        tool: Any,
        # params：工具参数。
        params: Any,
        # error：错误对象或错误信息。
        error: Any,
    # 返回 None。
    ) -> None:
        # 调用内部辅助方法。
        await self._for_each_hook_safe(
            # 第一个参数是方法名。
            "on_execute_tool_error",
            # 下面依次是位置参数。
            context,
            tool_call,
            tool,
            params,
            error,
        # 结束 _for_each_hook_safe 调用。
        )

    # 重写 emit_reasoning：委托给所有子 hook。
    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 emit_reasoning。
        await self._for_each_hook_safe("emit_reasoning", reasoning_content)

    # 重写 emit_reasoning_end：委托给所有子 hook。
    async def emit_reasoning_end(self) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 emit_reasoning_end。
        await self._for_each_hook_safe("emit_reasoning_end")

    # 重写 after_iteration：委托给所有子 hook。
    async def after_iteration(self, context: AgentHookContext) -> None:
        # 调用内部辅助方法，执行每个子 hook 的 after_iteration。
        await self._for_each_hook_safe("after_iteration", context)

    # 重写 finalize_content。
    # 这里不做异常隔离，因为它被设计成内容处理管道；
    # 如果某个 hook 在这里出错，通常说明内容处理逻辑有 bug，应该暴露出来。
    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        # 按顺序遍历所有子 hook。
        for h in self._hooks:
            # 每个 hook 都可以基于上一个 hook 的输出继续修改 content。
            # 因此这是一个管道：
            # initial_content -> hook1 -> hook2 -> hook3 -> final_content
            content = h.finalize_content(context, content)

        # 返回经过所有 hook 处理后的最终 content。
        return content


# SDKCaptureHook：用于捕获 SDK/调用方关心的运行结果。
# 它继承 AgentHook，在 iteration 和 run 结束时记录状态快照。
class SDKCaptureHook(AgentHook):
    """Record tool names and the final message list for ``RunResult``.

    The runner mutates ``context.messages`` in place across iterations, so the
    snapshot is refreshed on every ``after_iteration`` call; the last call
    reflects the end-of-turn state the SDK caller cares about.  The run-level
    snapshot is authoritative when available and covers paths without a final
    per-iteration callback.
    """

    """
    [中文翻译] 为 ``RunResult`` 记录工具名和最终消息列表。

    runner 会在多次迭代过程中原地修改 ``context.messages``，因此快照会在
    每次 ``after_iteration`` 调用时刷新；最后一次调用反映的是 SDK 调用方
    关心的回合结束状态。当存在 run 级快照时，它以 run 级快照为准，
    并能覆盖某些没有最终 per-iteration 回调的路径。
    """

    # 构造方法。
    def __init__(self) -> None:
        # 调用父类 AgentHook 的构造方法。
        # 未传 reraise，因此默认 False。
        super().__init__()

        # tools_used：记录使用过的工具名。
        # 例如 ["search", "read_file", "write_file"]。
        self.tools_used: list[str] = []

        # messages：记录最终消息列表快照。
        self.messages: list[dict[str, Any]] = []

        # usage：记录资源消耗快照。
        self.usage: dict[str, int] = {}

        # stop_reason：记录停止原因。
        self.stop_reason: str | None = None

        # error：记录错误信息。
        self.error: str | None = None

        # tool_events：记录工具事件快照。
        self.tool_events: list[dict[str, str]] = []

        # had_injections：记录是否发生过注入。
        self.had_injections: bool = False

    # 重写 after_iteration。
    # 每次迭代结束后刷新快照。
    async def after_iteration(self, context: AgentHookContext) -> None:
        # 遍历当前迭代中的工具调用请求。
        for call in context.tool_calls:
            # 把工具名追加到 tools_used。
            # call.name 是 ToolCallRequest 中的工具名字段。
            self.tools_used.append(call.name)

        # 复制当前消息列表。
        # list(context.messages) 创建一个新的 list，避免后续原地修改影响快照。
        # 注意这是浅拷贝：消息 dict 对象本身仍被共享。
        self.messages = list(context.messages)

        # 复制 usage 字典。
        # dict(context.usage) 创建新的 dict，避免后续修改影响快照。
        # 同样是浅拷贝。
        self.usage = dict(context.usage)

        # 记录当前停止原因。
        self.stop_reason = context.stop_reason

        # 记录当前错误信息。
        self.error = context.error

        # 复制工具事件列表。
        self.tool_events = list(context.tool_events)

    # 重写 after_run。
    # run 结束后，用 run 级别上下文覆盖/刷新快照。
    async def after_run(self, context: AgentRunHookContext) -> None:
        # 复制 run 级 tools_used。
        self.tools_used = list(context.tools_used)

        # 复制 run 级 messages。
        self.messages = list(context.messages)

        # 复制 run 级 usage。
        self.usage = dict(context.usage)

        # 复制 run 级 stop_reason。
        self.stop_reason = context.stop_reason

        # 复制 run 级 error。
        self.error = context.error

        # 复制 run 级 tool_events。
        self.tool_events = list(context.tool_events)

        # 复制 run 级 had_injections。
        self.had_injections = context.had_injections