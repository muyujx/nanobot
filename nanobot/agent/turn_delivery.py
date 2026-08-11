# 模块文档字符串（docstring）：Python 中放在文件开头的字符串会成为模块说明文档。
# 原始注释保留，并追加中文翻译。
"""Route and publish the user-visible lifecycle of an agent turn.

中文翻译：路由并发布一次 agent turn（代理回合）中对用户可见的生命周期。
"""

# 从 __future__ 导入 annotations。
# 作用：让类型注解延迟求值（PEP 563），类型注解不会在运行时被立即解析成真实对象。
# 好处：可以使用较新的类型写法，也可以避免某些循环导入问题。
from __future__ import annotations

# 导入标准库 dataclasses 模块。
# 后面会使用 dataclasses.replace() 来基于已有 dataclass 实例创建新实例。
import dataclasses

# 导入标准库 time 模块。
# 后面会使用 time.time_ns() 生成纳秒级时间戳，用于构造流式消息 ID。
import time

# 从 collections.abc 导入 Awaitable 和 Callable。
# Awaitable：表示可等待对象，通常用于 async 函数的返回类型。
# Callable：表示可调用对象，可用于定义函数/回调类型别名。
from collections.abc import Awaitable, Callable

# 从 dataclasses 导入 dataclass 和 field。
# @dataclass：装饰器，用于自动生成 __init__、__repr__、__eq__ 等方法。
# field：用于给 dataclass 字段配置默认值、初始化行为等。
from dataclasses import dataclass, field

# 从 typing 导入类型工具。
# TYPE_CHECKING：只有在类型检查工具（如 mypy）运行时才为 True，运行时为 False。
# Any：表示任意类型。
# cast：用于类型检查层面的类型转换，运行时不做任何转换。
from typing import TYPE_CHECKING, Any, cast

# 从项目内部模块导入入站/出站消息类型。
# InboundMessage：进入 agent 的消息。
# OutboundMessage：agent 向外发送的消息。
from nanobot.bus.events import InboundMessage, OutboundMessage

# 从 outbound_events 导入各种出站事件以及事件转消息的工具函数。
from nanobot.bus.outbound_events import (
    # RetryWaitEvent：表示“重试等待”事件，通常用于告知用户正在等待重试。
    RetryWaitEvent,
    # StreamDeltaEvent：表示流式输出中的一段增量文本。
    StreamDeltaEvent,
    # StreamedResponseEvent：表示一次流式响应已经发生/完成的事件。
    StreamedResponseEvent,
    # StreamEndEvent：表示流式输出结束。
    StreamEndEvent,
    # outbound_message_for_event：将事件封装为 OutboundMessage 的工具函数。
    outbound_message_for_event,
)

# 导入用于构建进度回调的函数。
# build_bus_progress_callback：根据消息总线和消息对象创建一个进度回调。
from nanobot.bus.progress import build_bus_progress_callback

# 导入消息总线类型 MessageBus。
# MessageBus：负责发布/传递消息的核心总线对象。
from nanobot.bus.queue import MessageBus

# 导入运行时事件总线和运行时事件发布器。
# RuntimeEventBus：运行时事件总线。
# RuntimeEventPublisher：封装发布运行时事件逻辑的发布器。
from nanobot.bus.runtime_events import RuntimeEventBus, RuntimeEventPublisher

# TYPE_CHECKING 在运行时是 False，因此下面的 import 只会在类型检查时生效。
# 这样做的常见目的：避免运行时循环导入，同时让类型检查器知道 LLMRuntime 类型。
if TYPE_CHECKING:
    # 仅在类型检查时导入 LLMRuntime 类型，用于方法注解。
    from nanobot.utils.llm_runtime import LLMRuntime


# @dataclass(frozen=True) 表示 TurnRoute 是一个不可变 dataclass。
# frozen=True 会让实例创建后不能修改字段值，相当于“只读对象”。
@dataclass(frozen=True)
class TurnRoute:
    """Turn delivery destination and lifecycle policy, separate from execution input.

    中文翻译：一次 turn（回合）的投递目的地和生命周期策略，与执行输入分离。
    """

    # channel：目标渠道，例如 "cli"、"slack"、"system" 等。
    channel: str

    # chat_id：目标会话/聊天 ID，用于定位消息发送到哪个会话。
    chat_id: str

    # metadata：附加元数据字典。
    # field(default_factory=dict) 表示默认值不是共享的同一个 dict，
    # 而是每次创建实例时调用 dict() 生成一个新的空字典。
    metadata: dict[str, Any] = field(default_factory=dict)

    # publish_lifecycle：是否发布该 turn 的生命周期事件。
    # 默认 False，表示不发布。
    publish_lifecycle: bool = False


# 类型别名：TurnRoutePolicy。
# 它表示一个可调用对象，参数为 (InboundMessage, str, TurnRoute)，返回 TurnRoute。
# 也就是：根据入站消息、会话 key、默认路由，决定最终路由。
TurnRoutePolicy = Callable[[InboundMessage, str, TurnRoute], TurnRoute]

# 类型别名：ProgressCallback。
# Callable[..., Awaitable[None]] 表示任意参数、返回可等待对象的回调。
# 返回的 Awaitable[None] 通常意味着这是一个 async 函数，最终返回 None。
ProgressCallback = Callable[..., Awaitable[None]]

# 类型别名：StreamCallback。
# 接收一个 str 类型增量内容，返回 Awaitable[None] 的异步回调。
StreamCallback = Callable[[str], Awaitable[None]]

# 类型别名：StreamEndCallback。
# 流结束回调，参数不固定，返回 Awaitable[None]。
StreamEndCallback = Callable[..., Awaitable[None]]

# 类型别名：RetryWaitCallback。
# 重试等待回调，接收一个 str 内容，返回 Awaitable[None]。
RetryWaitCallback = Callable[[str], Awaitable[None]]


# TurnDeliveryFactory：工厂类，用于为每个 turn 创建 TurnDelivery 对象。
class TurnDeliveryFactory:
    """Create per-turn delivery objects from an optional edge-owned route policy.

    中文翻译：根据可选的边缘侧路由策略，为每个 turn 创建投递对象。
    """

    # 初始化方法：创建 TurnDeliveryFactory 实例。
    def __init__(
        self,
        # bus：消息总线，用于后续发布出站消息。
        bus: MessageBus,
        # runtime_events：运行时事件总线，用于发布运行时生命周期事件。
        runtime_events: RuntimeEventBus,
        # route_policy：可选路由策略，默认 None。
        # 如果提供，则可以在默认路由基础上修改最终路由。
        route_policy: TurnRoutePolicy | None = None,
    ) -> None:
        # 将传入的消息总线保存到实例属性 self.bus。
        self.bus = bus

        # 将运行时事件总线保存到实例属性 self.runtime_events。
        self.runtime_events = runtime_events

        # 基于 runtime_events 创建一个 RuntimeEventPublisher。
        # 后续通过 runtime_event_publisher 统一发布运行时事件。
        self.runtime_event_publisher = RuntimeEventPublisher(runtime_events)

        # 保存可选路由策略。
        self.route_policy = route_policy

    # create 方法：根据入站消息和会话 key 创建一个 TurnDelivery。
    def create(
        self,
        # msg：当前 turn 的入站消息。
        msg: InboundMessage,
        # session_key：会话唯一标识，用于关联同一个会话/线程。
        session_key: str,
        # *：Python 特殊语法，表示后面的参数只能以关键字参数形式传入。
        # 例如必须写 enable_stream=True，不能按位置传参。
        *,
        # enable_stream：是否允许流式输出，默认 False。
        enable_stream: bool = False,
    ) -> TurnDelivery:
        # 先计算默认路由。
        # _default_route 会根据 msg.channel、msg.chat_id、session_key 等生成 TurnRoute。
        route = self._default_route(msg, session_key)

        # 如果外部提供了路由策略，则调用该策略修改或替换默认路由。
        if self.route_policy is not None:
            # 调用路由策略函数，传入入站消息、session_key 和默认路由。
            # 返回值应是一个新的 TurnRoute。
            route = self.route_policy(msg, session_key, route)

            # 检查路由策略返回值是否真的是 TurnRoute 类型。
            # cast(object, route) 只在类型检查层面把 route 视为 object，
            # 运行时不会改变 route 的值，只是为了满足 isinstance 的类型检查场景。
            if not isinstance(cast(object, route), TurnRoute):
                # 如果返回值不是 TurnRoute，则抛出类型错误。
                raise TypeError("turn route policy must return TurnRoute")

        # 根据最终路由和相关依赖创建 TurnDelivery 对象。
        return TurnDelivery(
            # 传入消息总线。
            bus=self.bus,
            # 传入运行时事件发布器。
            runtime_event_publisher=self.runtime_event_publisher,
            # 传入原始入站消息。
            input_message=msg,
            # 传入会话 key。
            session_key=session_key,
            # 传入最终路由。
            route=route,
            # 传入是否启用流式输出的开关。
            enable_stream=enable_stream,
        )

    # unrouted 方法：创建一个不经过边缘路由策略的 TurnDelivery。
    def unrouted(self, msg: InboundMessage, session_key: str) -> TurnDelivery:
        """Create a lifecycle fallback without invoking edge routing policy.

        中文翻译：创建一个生命周期回退投递对象，不调用边缘路由策略。
        """
        # 直接创建 TurnDelivery。
        return TurnDelivery(
            # 传入消息总线。
            bus=self.bus,
            # 传入运行时事件发布器。
            runtime_event_publisher=self.runtime_event_publisher,
            # 传入原始入站消息。
            input_message=msg,
            # 传入会话 key。
            session_key=session_key,
            # 手动构造一个最基础的 TurnRoute。
            route=TurnRoute(
                # 直接使用入站消息的 channel。
                channel=msg.channel,
                # 直接使用入站消息的 chat_id。
                chat_id=msg.chat_id,
                # 复制入站消息 metadata。
                # msg.metadata or {}：如果 msg.metadata 为 None 或假值，则使用空字典。
                # dict(...)：创建浅拷贝，避免后续修改影响原始 metadata。
                metadata=dict(msg.metadata or {}),
            ),
        )

    # @staticmethod：静态方法，不需要访问实例 self，也不接收类 cls。
    # 它作为纯函数挂在类下面，用于生成默认路由。
    @staticmethod
    def _default_route(msg: InboundMessage, session_key: str) -> TurnRoute:
        # 如果消息不是来自 "system" 渠道，则默认原样返回到原渠道/会话。
        if msg.channel != "system":
            # 创建默认路由。
            return TurnRoute(
                # 目标 channel 使用原始消息 channel。
                channel=msg.channel,
                # 目标 chat_id 使用原始消息 chat_id。
                chat_id=msg.chat_id,
                # 复制原始 metadata，防止共享引用。
                metadata=dict(msg.metadata or {}),
                # 非 system 渠道默认发布生命周期事件。
                publish_lifecycle=True,
            )

        # 下面处理 channel == "system" 的情况。
        # 这是一个 Python 条件表达式（三元表达式）：
        # 如果 msg.chat_id 中包含 ":"，则按 ":" 最多分割一次，取两部分；
        # 否则默认 channel 为 "cli"，chat_id 为原始 msg.chat_id。
        channel, chat_id = (
            # msg.chat_id.split(":", 1)：按冒号分割，最多分割一次。
            # 例如 "slack:C123" 会变成 ["slack", "C123"]。
            msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
        )

        # 为 system 路由创建新的 metadata 字典。
        metadata: dict[str, Any] = {}

        # 如果目标 channel 是 slack，
        # 并且 session_key 以 "slack:" 开头，
        # 并且 session_key 中至少有两个冒号，
        # 则认为 session_key 中携带了 Slack thread_ts。
        if (
            channel == "slack"
            and session_key.startswith("slack:")
            and session_key.count(":") >= 2
        ):
            # session_key.split(":", 2)：最多分割两次，得到最多三段。
            # [2] 表示取第三段，即 Slack 的 thread_ts。
            # 例如 "slack:team:thread_ts" 会取到 "thread_ts"。
            metadata["slack"] = {"thread_ts": session_key.split(":", 2)[2]}

        # Python 海象运算符 :=。
        # 作用：在 if 条件内部同时赋值并判断真假。
        # 这里从 msg.metadata 获取 origin_message_id，如果非空则赋值给 origin_message_id。
        if origin_message_id := msg.metadata.get("origin_message_id"):
            # 如果存在来源消息 ID，则写入新的 metadata。
            metadata["origin_message_id"] = origin_message_id

        # 返回 system 场景下解析出来的 TurnRoute。
        # 注意这里没有显式设置 publish_lifecycle，因此使用默认值 False。
        return TurnRoute(channel=channel, chat_id=chat_id, metadata=metadata)


# @dataclass：将 TurnDelivery 声明为 dataclass。
# 与 frozen=True 不同，这里默认可变，因为 TurnDelivery 内部需要维护状态。
@dataclass
class TurnDelivery:
    """Own routing, callbacks, and lifecycle publication for one turn.

    中文翻译：负责一个 turn（回合）的路由、回调和生命周期发布。
    """

    # bus：消息总线，用于发布出站消息。
    bus: MessageBus

    # runtime_event_publisher：运行时事件发布器，用于发布生命周期/运行时事件。
    runtime_event_publisher: RuntimeEventPublisher

    # input_message：原始入站消息。
    input_message: InboundMessage

    # session_key：当前会话 key。
    session_key: str

    # route：当前 turn 的目标路由。
    route: TurnRoute

    # enable_stream：是否允许流式输出，默认 False。
    enable_stream: bool = False

    # delivery_message：用于实际投递的消息对象。
    # field(init=False) 表示这个字段不会出现在自动生成的 __init__ 参数中。
    # 它会在 __post_init__ 中赋值。
    delivery_message: InboundMessage = field(init=False)

    # lifecycle_message：用于生命周期事件发布的消息对象。
    # 同样不由 __init__ 参数初始化。
    lifecycle_message: InboundMessage = field(init=False)

    # _stream_base_id：流式输出基础 ID。
    # 下划线开头表示这是内部/私有字段。
    # init=False：不在构造函数参数中出现。
    # default=None：默认值为 None。
    _stream_base_id: str | None = field(init=False, default=None)

    # _stream_segment：流式输出分段编号。
    # 每结束一个不合并的流段，就自增一次。
    _stream_segment: int = field(init=False, default=0)

    # _stream_open：当前流是否处于打开状态。
    _stream_open: bool = field(init=False, default=False)

    # __post_init__ 是 dataclass 的特殊方法。
    # 在自动生成的 __init__ 执行完后自动调用，常用于初始化后处理。
    def __post_init__(self) -> None:
        # 使用 dataclasses.replace 基于 input_message 创建一个新的 InboundMessage。
        # dataclasses.replace(obj, **changes)：
        # 它会复制原 dataclass 实例，并替换指定字段，其他字段保持不变。
        self.delivery_message = dataclasses.replace(
            # 原始入站消息作为复制基础。
            self.input_message,
            # 替换 channel 为路由中的目标 channel。
            channel=self.route.channel,
            # 替换 chat_id 为路由中的目标 chat_id。
            chat_id=self.route.chat_id,
            # 替换 metadata 为路由 metadata 的浅拷贝。
            metadata=dict(self.route.metadata),
        )

        # 根据是否发布生命周期，选择 lifecycle_message：
        # 如果 route.publish_lifecycle 为 True，则生命周期事件跟随最终投递消息；
        # 否则生命周期事件仍使用原始入站消息。
        self.lifecycle_message = (
            self.delivery_message if self.route.publish_lifecycle else self.input_message
        )

        # 如果启用了流式输出，并且投递消息 metadata 中显式标记了 _wants_stream，
        # 则生成一个流式输出的基础 ID。
        if self.enable_stream and self.delivery_message.metadata.get("_wants_stream"):
            # f-string：格式化字符串。
            # time.time_ns()：当前纳秒时间戳。
            # 使用 session_key + 纳秒时间戳生成相对唯一的流 ID。
            self._stream_base_id = f"{self.session_key}:{time.time_ns()}"

    # @property：把方法包装成属性访问。
    # 调用方可以写 delivery.on_stream，而不是 delivery.on_stream()。
    @property
    def on_stream(self) -> StreamCallback | None:
        # 如果存在流基础 ID，说明流式输出可用，返回绑定的 _publish_stream 方法；
        # 否则返回 None。
        # self._publish_stream 是绑定方法，本身可调用，并且是 async 函数。
        return self._publish_stream if self._stream_base_id is not None else None

    # @property：把方法包装成属性访问。
    @property
    def on_stream_end(self) -> StreamEndCallback | None:
        # 如果存在流基础 ID，则返回绑定的 _publish_stream_end 方法；
        # 否则返回 None。
        return self._publish_stream_end if self._stream_base_id is not None else None

    # progress_callback 方法：返回进度回调，或者 None。
    def progress_callback(self) -> ProgressCallback | None:
        # 如果当前路由不发布生命周期，则不提供进度回调。
        if not self.route.publish_lifecycle:
            return None

        # 使用消息总线和投递消息构建进度回调。
        # 返回值通常是一个 async callable，用于发布进度事件。
        return build_bus_progress_callback(self.bus, self.delivery_message)

    # retry_wait_callback 方法：返回重试等待回调，或者 None。
    def retry_wait_callback(self) -> RetryWaitCallback | None:
        # 如果当前路由不发布生命周期，则不提供重试等待回调。
        if not self.route.publish_lifecycle:
            return None

        # 定义内部异步函数 _on_retry_wait。
        # 它会捕获外层 self，因此是一个闭包。
        async def _on_retry_wait(content: str) -> None:
            # 发布出站消息。
            await self.bus.publish_outbound(
                # 将 RetryWaitEvent 包装成 OutboundMessage。
                outbound_message_for_event(
                    # 目标 channel 来自投递消息。
                    channel=self.delivery_message.channel,
                    # 目标 chat_id 来自投递消息。
                    chat_id=self.delivery_message.chat_id,
                    # 事件内容：重试等待，携带提示内容 content。
                    event=RetryWaitEvent(content=content),
                    # 使用投递消息 metadata。
                    metadata=self.delivery_message.metadata,
                )
            )

        # 返回内部闭包函数作为回调。
        return _on_retry_wait

    # started 方法：发布 turn 开始事件。
    # async def 表示这是异步方法，调用时需要 await。
    async def started(self) -> None:
        # 只有路由允许发布生命周期时才发布。
        if self.route.publish_lifecycle:
            # 发布 session turn 开始事件。
            await self.runtime_event_publisher.session_turn_started(
                # 使用最终投递消息作为事件上下文。
                self.delivery_message,
                # 当前会话 key。
                self.session_key,
            )

    # running 方法：发布运行状态变化事件，状态为 running。
    async def running(self, *, started_at: float) -> None:
        # 只有路由允许发布生命周期时才发布。
        if self.route.publish_lifecycle:
            # 发布运行状态变化事件。
            await self.runtime_event_publisher.run_status_changed(
                # 使用最终投递消息作为事件上下文。
                self.delivery_message,
                # 当前会话 key。
                self.session_key,
                # 状态字符串："running"。
                "running",
                # started_at：开始时间戳，作为关键字参数传入。
                started_at=started_at,
            )

    # record_runtime 方法：记录一次 turn 的 LLM 运行时信息。
    def record_runtime(self, runtime: LLMRuntime) -> None:
        # 调用 runtime_event_publisher 记录 runtime。
        # LLMRuntime 只在类型检查时导入，运行时该注解不求值。
        self.runtime_event_publisher.record_turn_runtime(self.session_key, runtime)

    # record_latency 方法：记录一次 turn 的耗时。
    def record_latency(self, latency_ms: int | None) -> None:
        # 调用 runtime_event_publisher 记录毫秒级延迟。
        self.runtime_event_publisher.record_turn_latency(self.session_key, latency_ms)

    # background_response 方法：构造后台任务完成时的 OutboundMessage。
    def background_response(
        self,
        # content：后台任务结果文本，可能为 None。
        content: str | None,
        # *：后面的参数必须用关键字方式传入。
        *,
        # stop_reason：停止原因，例如 "end"、"error"、"tool_error" 等。
        stop_reason: str,
        # streamed：是否已经通过流式方式输出过。
        streamed: bool,
        # latency_ms：耗时毫秒数，可能为 None。
        latency_ms: int | None,
    ) -> OutboundMessage:
        # 复制路由 metadata，避免修改原始路由对象中的字典。
        metadata = dict(self.route.metadata)

        # 如果发布生命周期，并且存在耗时信息，则把耗时写入 metadata。
        if self.route.publish_lifecycle and latency_ms is not None:
            # int(latency_ms)：确保 latency_ms 是整数。
            metadata["latency_ms"] = int(latency_ms)

        # 根据条件决定是否附带 StreamedResponseEvent。
        # 这是一个条件表达式：
        # 满足条件时为 StreamedResponseEvent()，否则为 None。
        event = (
            StreamedResponseEvent()
            # 条件 1：允许发布生命周期。
            if self.route.publish_lifecycle
            # 条件 2：该响应已经流式输出过。
            and streamed
            # 条件 3：停止原因不是 error 或 tool_error。
            # {"error", "tool_error"} 是集合字面量，用于成员判断。
            and stop_reason not in {"error", "tool_error"}
            # 不满足条件时返回 None。
            else None
        )

        # 构造并返回出站消息。
        return OutboundMessage(
            # 目标 channel 来自路由。
            channel=self.route.channel,
            # 目标 chat_id 来自路由。
            chat_id=self.route.chat_id,
            # content or "Background task completed."：
            # 如果 content 为 None 或空字符串/假值，则使用默认完成文案。
            content=content or "Background task completed.",
            # 附带处理后的 metadata。
            metadata=metadata,
            # 附带可能为 None 的事件对象。
            event=event,
        )

    # complete 方法：完成一个 turn，并根据情况发布最终消息/完成事件。
    async def complete(
        self,
        # response：最终出站响应消息，可能为 None。
        response: OutboundMessage | None,
        # *：后面的参数必须用关键字方式传入。
        *,
        # publish_completion：是否发布 turn_completed 事件。
        publish_completion: bool,
    ) -> None:
        # 默认完成事件使用 lifecycle_message 中的 channel。
        completed_channel = self.lifecycle_message.channel

        # 默认完成事件使用 lifecycle_message 中的 chat_id。
        completed_chat_id = self.lifecycle_message.chat_id

        # 如果存在最终响应消息，则先发布该响应。
        if response is not None:
            # 通过消息总线发布最终响应。
            await self.bus.publish_outbound(response)

            # 完成事件中的 channel 改为最终响应实际使用的 channel。
            completed_channel = response.channel

            # 完成事件中的 chat_id 改为最终响应实际使用的 chat_id。
            completed_chat_id = response.chat_id

        # 如果没有最终响应消息，但生命周期消息来自 CLI，
        # 则发布一个空内容消息，可能用于 CLI 端结束当前输出块。
        elif self.lifecycle_message.channel == "cli":
            # 发布空内容的出站消息。
            await self.bus.publish_outbound(
                # 构造一个空 OutboundMessage。
                OutboundMessage(
                    # channel 使用生命周期消息 channel。
                    channel=self.lifecycle_message.channel,
                    # chat_id 使用生命周期消息 chat_id。
                    chat_id=self.lifecycle_message.chat_id,
                    # 内容为空字符串。
                    content="",
                    # 复制 lifecycle_message.metadata。
                    # lifecycle_message.metadata or {}：如果 metadata 为 None 或假值则用空字典。
                    metadata=dict(self.lifecycle_message.metadata or {}),
                )
            )

        # 如果调用方要求发布完成事件，则发布 turn_completed。
        if publish_completion:
            # 发布 turn 完成事件。
            await self.runtime_event_publisher.turn_completed(
                # 完成事件 channel。
                channel=completed_channel,
                # 完成事件 chat_id。
                chat_id=completed_chat_id,
                # 当前会话 key。
                session_key=self.session_key,
                # 使用 lifecycle_message 的 metadata。
                metadata=self.lifecycle_message.metadata,
            )

    # fail 方法：turn 失败时发布错误消息，并可选地发布完成事件。
    async def fail(self, *, publish_completion: bool) -> None:
        # 发布一条用户可见的错误提示。
        await self.bus.publish_outbound(
            # 构造错误出站消息。
            OutboundMessage(
                # channel 使用生命周期消息 channel。
                channel=self.lifecycle_message.channel,
                # chat_id 使用生命周期消息 chat_id。
                chat_id=self.lifecycle_message.chat_id,
                # 错误提示文本。
                content="Sorry, I encountered an error.",
                # 复制 lifecycle_message.metadata。
                metadata=dict(self.lifecycle_message.metadata or {}),
            )
        )

        # 如果需要发布完成事件，则失败也视为 turn 完成。
        if publish_completion:
            # 发布 turn_completed 事件。
            await self.runtime_event_publisher.turn_completed(
                # channel 使用生命周期消息 channel。
                channel=self.lifecycle_message.channel,
                # chat_id 使用生命周期消息 chat_id。
                chat_id=self.lifecycle_message.chat_id,
                # 当前会话 key。
                session_key=self.session_key,
                # 使用 lifecycle_message.metadata。
                metadata=self.lifecycle_message.metadata,
            )

    # idle 方法：将运行状态置为 idle，并清理当前 turn 的运行时状态。
    async def idle(self) -> None:
        # 发布运行状态变化事件，状态为 "idle"。
        await self.runtime_event_publisher.run_status_changed(
            # 使用生命周期消息作为事件上下文。
            self.lifecycle_message,
            # 当前会话 key。
            self.session_key,
            # 状态字符串："idle"。
            "idle",
        )

        # 清理当前 session_key 对应的 turn 运行时状态。
        self.runtime_event_publisher.clear_turn(self.session_key)

    # _stream_id 方法：生成当前流段的完整 ID。
    def _stream_id(self) -> str:
        # assert：断言 _stream_base_id 不为 None。
        # 如果为 None，说明在流未启用时错误调用了该方法，会抛 AssertionError。
        assert self._stream_base_id is not None

        # 用基础 ID 和当前分段编号拼出完整 stream_id。
        return f"{self._stream_base_id}:{self._stream_segment}"

    # _publish_stream 方法：发布一段流式增量内容。
    async def _publish_stream(self, delta: str) -> None:
        # 通过消息总线发布出站消息。
        await self.bus.publish_outbound(
            # 将 StreamDeltaEvent 包装为 OutboundMessage。
            outbound_message_for_event(
                # 目标 channel 来自投递消息。
                channel=self.delivery_message.channel,
                # 目标 chat_id 来自投递消息。
                chat_id=self.delivery_message.chat_id,
                # 流增量事件：携带增量文本 delta 和当前 stream_id。
                event=StreamDeltaEvent(content=delta, stream_id=self._stream_id()),
                # 使用投递消息 metadata。
                metadata=self.delivery_message.metadata,
            )
        )

        # 标记流已经打开。
        self._stream_open = True

    # _publish_stream_end 方法：发布流结束事件。
    async def _publish_stream_end(
        self,
        # *：后面的参数必须使用关键字方式传入。
        *,
        # resuming：是否表示后续还会继续恢复输出。
        resuming: bool = False,
        # merge_next：是否将下一段流合并到当前显示流中。
        merge_next: bool = False,
    ) -> None:
        # 通过消息总线发布出站消息。
        await self.bus.publish_outbound(
            # 将 StreamEndEvent 包装为 OutboundMessage。
            outbound_message_for_event(
                # 目标 channel 来自投递消息。
                channel=self.delivery_message.channel,
                # 目标 chat_id 来自投递消息。
                chat_id=self.delivery_message.chat_id,
                # 流结束事件。
                event=StreamEndEvent(
                    # 当前流段 ID。
                    stream_id=self._stream_id(),
                    # 是否恢复。
                    resuming=resuming,
                    # 是否合并下一段。
                    merge_next=merge_next,
                ),
                # 使用投递消息 metadata。
                metadata=self.delivery_message.metadata,
            )
        )

        # 如果 merge_next 为 True，则认为流仍然保持打开，下一段会合并进来。
        # 否则流关闭。
        self._stream_open = merge_next

        # 如果不合并下一段，则分段编号加一，下一次会开启新的流段。
        if not merge_next:
            self._stream_segment += 1

    # abort_stream 方法：中断/中止流时关闭流。
    async def abort_stream(self) -> None:
        """Close an interrupted stream so stateful channels can release its buffer.

        中文翻译：关闭一个被中断的流，以便有状态的渠道可以释放它的缓冲区。
        """
        # 只有当前流确实打开时才需要发布结束事件。
        if self._stream_open:
            # 发布普通流结束事件，不恢复、不合并。
            await self._publish_stream_end()