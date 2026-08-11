"""Agent hook that adapts runner events into channel progress UI."""
# 翻译：将 runner 事件适配为 channel 进度 UI 的 Agent 钩子（Hook）。

from __future__ import annotations
# Python 3.7+ 特性：允许在类型提示中使用尚未定义的类名（前向引用），并将所有类型注解作为字符串处理（延迟求值）。

import inspect
# 导入 inspect 模块，用于在运行时检查对象（如获取函数的参数签名）。
import json
# 导入 json 模块，用于数据的序列化和反序列化。
from typing import Any, Awaitable, Callable, cast
# 导入类型提示工具：
# Any: 表示任意类型。
# Awaitable: 表示可等待对象（如异步协程）。
# Callable: 表示可调用对象（如函数、方法）。
# cast: 用于类型转换提示，仅在静态类型检查（如 mypy）时生效，运行时不做任何操作。

from loguru import logger
# 导入 loguru 的日志记录器，它比 Python 标准库的 logging 更加易用且美观。

from nanobot.agent.hook import AgentHook, AgentHookContext
# 导入 Agent 钩子基类和钩子执行时的上下文对象。
from nanobot.providers.base import ToolCallRequest
# 导入工具调用的请求数据结构类。
from nanobot.utils.helpers import IncrementalThinkExtractor, strip_think
# 导入增量提取思考过程（<think> 标签）的提取器和剥离思考标签的工具函数。
from nanobot.utils.progress_events import (
    build_tool_event_finish_payloads,
    build_tool_event_start_payload,
    invoke_on_progress,
    on_progress_accepts_tool_events,
)
# 导入处理进度事件的辅助函数：构建工具开始/结束的 payload、触发进度回调、检查回调是否接受工具事件等。
from nanobot.utils.tool_hints import format_tool_hints
# 导入格式化工具调用提示文本的函数。


class AgentProgressHook(AgentHook):
    """Translate runner lifecycle events into user-visible progress signals."""
    # 翻译：将 runner 生命周期事件转换为用户可见的进度信号。

    def __init__(
        self,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        # 进度回调函数：接收任意参数 (...)，返回一个异步对象 (Awaitable)，最终结果为 None。
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        # 流式文本回调函数：接收一个字符串参数 (str)，返回异步对象。
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        # 流式结束回调函数：接收任意参数，返回异步对象。
        *,
        # 【Python 特殊用法】单独的星号 `*` 表示其后的所有参数必须作为关键字参数（Keyword-only）传入，防止位置参数传错。
        session_key: str | None = None,
        # 会话标识键，用于日志追踪。`str | None` 是 Python 3.10+ 的联合类型语法，等同于旧版的 Optional[str]。
        tool_hint_max_length: int = 40,
        # 工具提示文本的最大截断长度。
        on_iteration: Callable[[int], None] | None = None,
        # 每次循环迭代的回调函数：接收一个整数（迭代次数），同步返回 None。
    ) -> None:
        # 调用父类 AgentHook 的初始化方法。`reraise=True` 表示如果在钩子中发生异常，将其重新抛出，不让系统静默吞掉错误。
        super().__init__(reraise=True)
        
        # 将传入的回调和配置保存为实例变量
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._session_key = session_key
        self._tool_hint_max_length = tool_hint_max_length
        self._on_iteration = on_iteration
        
        # 初始化流式文本缓冲区，用于累积接收到的文本 delta，以便处理大模型的思考过程标签（如 <think>）。
        self._stream_buf = ""
        
        # 实例化增量思考提取器，用于从流式文本中分离出 "内部思考过程" 和 "最终回答"。
        self._think_extractor = IncrementalThinkExtractor()
        
        # 状态标记：当前是否正处于 "推理/思考" 状态的输出流中。
        self._reasoning_open = False

    def wants_streaming(self) -> bool:
        # 返回是否需要流式输出。如果设置了 on_stream 回调，则返回 True。
        return self._on_stream is not None

    @staticmethod
    # 【Python 特殊用法】@staticmethod 装饰器表示这是一个静态方法，不需要传入 self（实例）或 cls（类）参数。
    def _strip_think(text: str | None) -> str | None:
        # 如果传入的文本为空（None 或 空字符串），直接返回 None。
        if not text:
            return None
        # 调用 strip_think 剥离思考标签。如果剥离后为空字符串（falsy），则使用 `or None` 将其转换为 None。
        return strip_think(text) or None

    def _tool_hint(self, tool_calls: list[Any]) -> str:
        # 调用 format_tool_hints 生成工具调用的提示文本（例如："正在调用 search..."），并限制最大长度。
        return format_tool_hints(tool_calls, max_length=self._tool_hint_max_length)

    @staticmethod
    def _on_progress_accepts(cb: Callable[..., Any], name: str) -> bool:
        # 此方法用于反射检查：给定的回调函数 `cb` 是否接受名为 `name` 的关键字参数。
        try:
            # inspect.signature 获取函数的签名对象，包含其参数信息。
            sig = inspect.signature(cb)
        except (TypeError, ValueError):
            # 如果获取签名失败（例如传入了内置函数等无法解析的对象），保守地返回 False。
            return False
            
        # 遍历函数的所有参数。`p.kind == inspect.Parameter.VAR_KEYWORD` 用于检查是否包含 `**kwargs`。
        # 如果函数支持接收任意关键字参数（**kwargs），则它必然接受 `name` 参数，直接返回 True。
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return True
            
        # 否则，检查 `name` 是否明确存在于函数的参数列表中。
        return name in sig.parameters

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        # 异步方法：处理流式文本的增量更新 `delta`。
        
        # 在追加新文本前，先获取当前缓冲区剥离 <think> 标签后的纯净文本（作为基准）。
        prev_clean = strip_think(self._stream_buf)
        
        # 将新的文本增量追加到总缓冲区中。
        self._stream_buf += delta
        
        # 追加后，再次获取剥离 <think> 标签后的纯净文本。
        new_clean = strip_think(self._stream_buf)
        
        # 【Python 切片用法】计算两次纯净文本的差异。`[len(prev_clean):]` 表示从 prev_clean 的长度位置截取到末尾，
        # 从而得到本次增量中真正属于 "回答内容"（非思考过程）的新增文本。
        incremental = new_clean[len(prev_clean) :]

        # 将当前的完整缓冲区送入思考提取器。如果提取到了思考内容，则调用 self.emit_reasoning 发送思考事件，
        # 并将上下文标记为已流式输出推理过程。
        if await self._think_extractor.feed(self._stream_buf, self.emit_reasoning):
            context.streamed_reasoning = True

        # 如果产生了新的回答文本增量，说明思考阶段结束或被打断。
        if incremental:
            # Answer text has started; close the reasoning segment so the UI can
            # lock the bubble before the answer renders below it.
            # 翻译：回答文本已经开始；关闭推理片段，以便 UI 可以在下方渲染回答之前锁定气泡（UI 组件）。
            await self.emit_reasoning_end()
            
            # 如果存在流式输出回调，则将新的回答文本增量发送出去。
            if self._on_stream:
                await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        # 流式输出结束的回调。`*` 表示 resuming 必须作为关键字参数传入。`resuming` 表示当前是否是恢复之前的会话。
        
        # 确保结束任何未关闭的推理流。
        await self.emit_reasoning_end()
        
        if self._on_stream_end:
            # 构建传递给回调的关键字参数字典，初始包含 resuming。
            kwargs: dict[str, bool] = {"resuming": resuming}
            
            # 如果上下文指示当前流继续了当前消息（如分段回复），并且 on_stream_end 回调接受 merge_next 参数，
            # 则将其加入 kwargs，指示前端将下一段合并到当前消息中。
            if (
                context.stream_continues_current_message
                and self._on_progress_accepts(self._on_stream_end, "merge_next")
            ):
                kwargs["merge_next"] = True
                
            # 【Python 字典解包】使用 `**kwargs` 将字典解包为关键字参数传递给函数。
            await self._on_stream_end(**kwargs)
            
        # 清空流式缓冲区，并重置思考提取器，为下一次输出做准备。
        self._stream_buf = ""
        self._think_extractor.reset()

    async def before_iteration(self, context: AgentHookContext) -> None:
        # 在每次 Agent 循环迭代前调用的钩子。
        
        # 如果有设置迭代回调，则传入当前的迭代次数（同步调用）。
        if self._on_iteration:
            self._on_iteration(context.iteration)
            
        # 记录调试日志，输出当前迭代次数和会话键。`{}` 是 loguru 的格式化占位符。
        logger.debug(
            "Starting agent loop iteration {} for session {}",
            context.iteration,
            self._session_key,
        )

    async def on_provider_tool_event(
        self,
        context: AgentHookContext,
        event: dict[str, Any],
    ) -> None:
        # 处理由 Provider（如 LLM 提供商自身托管的工具调用，如 OpenAI 的 Code Interpreter）发出的工具事件。
        
        # 如果没有进度回调，直接返回，无需处理 UI 更新。
        if not self._on_progress:
            return
            
        # 从事件字典中安全地提取阶段（start/end/error）、工具名称和调用 ID。
        phase = event.get("phase")
        name = event.get("name")
        call_id = event.get("call_id")
        
        # 验证提取的数据是否合法。`{"start", "end", "error"}` 是集合字面量，`in` 用于成员测试，查找效率为 O(1)。
        if (
            phase not in {"start", "end", "error"}
            or not isinstance(name, str)
            or not name
            or not call_id
        ):
            return
            
        # 提取并验证参数，确保它是一个字典。
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
            
        # 构建发送给前端的事件 payload 字典。
        payload: dict[str, Any] = {
            "version": 1,
            "phase": phase,
            "call_id": str(call_id),
            "name": name,
            "arguments": arguments,
            # 【Python 三元运算符/条件表达式】如果 phase 为 "end" 则返回 result，否则返回 None。
            "result": event.get("result") if phase == "end" else None,
            "error": event.get("error") if phase == "error" else None,
            "files": [],
            "embeds": [],
        }
        
        # 如果是开始阶段：
        if phase == "start":
            await self.emit_reasoning_end()
            # 构建一个内部使用的工具调用请求对象。
            tool_call = ToolCallRequest(id=str(call_id), name=name, arguments=arguments)
            # 生成工具的提示文本，如果生成失败（为空）则回退到工具名称（利用 `or` 的短路特性）。
            tool_hint = self._strip_think(self._tool_hint([tool_call])) or name
            
            # 触发进度回调，将工具提示和事件 payload 传给 UI。
            await invoke_on_progress(
                self._on_progress,
                tool_hint,
                tool_hint=True,
                tool_events=[payload],
            )
            
            # 记录日志，打印工具名称和前 200 个字符的参数。`ensure_ascii=False` 允许打印中文等非 ASCII 字符。
            logger.info(
                "Provider-hosted tool call: {}({})",
                name,
                json.dumps(arguments, ensure_ascii=False)[:200],
            )
            return
            
        # 如果是结束或错误阶段，检查回调是否接受工具事件，如果接受，则发送空提示和 payload。
        if on_progress_accepts_tool_events(self._on_progress):
            await invoke_on_progress(
                self._on_progress,
                "",
                tool_hint=False,
                tool_events=[payload],
            )

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        # 在 Agent 执行本地工具调用之前触发的钩子。
        
        if self._on_progress:
            # 如果存在进度回调，并且没有开启流式输出，且当前没有流式输出过内容：
            if not self._on_stream and not context.streamed_content:
                # 提取 LLM 响应中的内容并剥离思考标签，作为非流式模式下的前置输出。
                thought = self._strip_think(context.response.content if context.response else None)
                if thought:
                    await self._on_progress(thought)
                    
            # 生成工具提示文本。
            tool_hint = self._strip_think(self._tool_hint(context.tool_calls))
            # 【Python 列表推导式】遍历 context.tool_calls，为每个工具调用构建 start 事件的 payload，并生成一个新列表。
            tool_events = [build_tool_event_start_payload(tc) for tc in context.tool_calls]
            
            # 触发进度回调。`cast(str, tool_hint)` 告诉静态类型检查器 tool_hint 此时一定是 str 类型，消除警告。
            await invoke_on_progress(
                self._on_progress,
                cast(str, tool_hint),
                tool_hint=True,
                tool_events=tool_events,
            )
            
        # 遍历所有工具调用，记录日志。
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        """Publish a reasoning chunk; channel plugins decide whether to render."""
        # 翻译：发布一个推理（思考）片段；由 channel 插件决定是否渲染。
        
        # 检查进度回调是否存在、内容是否非空，并且回调是否支持 reasoning 参数。
        if (
            self._on_progress
            and reasoning_content
            and self._on_progress_accepts(self._on_progress, "reasoning")
        ):
            # 标记推理流已打开，并发送推理内容。
            self._reasoning_open = True
            await self._on_progress(reasoning_content, reasoning=True)

    async def emit_reasoning_end(self) -> None:
        """Close the current reasoning stream segment, if any was open."""
        # 翻译：关闭当前的推理流片段（如果之前有打开的话）。
        
        # 如果处于打开状态且有回调，则发送一个空的结束信号，并将标记设为 False。
        if self._reasoning_open and self._on_progress:
            self._reasoning_open = False
            await self._on_progress("", reasoning_end=True)
        else:
            # 否则，仅确保标记被重置为 False。
            self._reasoning_open = False

    async def after_iteration(self, context: AgentHookContext) -> None:
        # 在每次迭代完成后触发的钩子。
        
        # 检查是否需要发送工具调用的结束事件。
        if (
            self._on_progress
            and context.tool_calls
            and context.tool_events
            and on_progress_accepts_tool_events(self._on_progress)
        ):
            # 构建工具完成的 payloads。
            tool_events = build_tool_event_finish_payloads(context)
            if tool_events:
                await invoke_on_progress(
                    self._on_progress,
                    "",
                    tool_hint=False,
                    tool_events=tool_events,
                )
                
        # 【Python `or` 提供默认值】获取 token 使用量统计，如果为 None 则使用空字典。
        u = context.usage or {}
        
        # 记录大模型 Token 消耗的调试日志。`dict.get(key, default)` 用于安全获取字典值，防止 KeyError。
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        # 最终内容处理方法，用于在输出前清理内容，返回剥离了 <think> 标签后的纯净文本。
        return self._strip_think(content)