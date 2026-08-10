"""Shared coordination for session-bound automation turns.

中文翻译：用于“与会话绑定的自动化轮次（automation turns）”的共享协调逻辑。

这个模块主要负责：
1. 管理某些自动化消息（InboundMessage）是否应该立即执行；
2. 如果目标会话当前正忙，则把自动化消息延后排队；
3. 提交自动化轮次，并等待对应会话产生响应；
4. 在轮次完成、出错或被取消时，唤醒等待方。
"""

# 导入 __future__ 中的 annotations 特性。
# 作用：让类型注解延迟求值（不在运行时立即解析类型注解）。
# 这样可以使用一些较新的类型注解写法，同时也减少类型注解在运行时的开销。
from __future__ import annotations

# 导入 asyncio 标准库。
# asyncio 用于异步编程，这里会用到事件循环、Future、异步函数、取消等能力。
import asyncio

# 导入 dataclasses 标准库。
# 这里主要使用 dataclasses.replace()，用于基于已有 dataclass 实例创建一个“修改部分字段后的新实例”。
import dataclasses

# 从 collections.abc 导入几个类型注解工具：
# Awaitable：表示“可被 await 的对象”，通常是协程对象或实现了 __await__ 的对象。
# Callable：表示“可调用对象”，例如函数、方法、实现了 __call__ 的对象。
# Iterable：表示“可迭代对象”，例如 list、tuple、set、generator 等。
from collections.abc import Awaitable, Callable, Iterable

# 从项目内部模块 nanobot.bus.events 导入两个消息类型。
# InboundMessage：入站消息，通常表示进入 agent / 系统的消息。
# OutboundMessage：出站消息，通常表示 agent / 系统返回给外部的消息。
from nanobot.bus.events import InboundMessage, OutboundMessage


class AutomationTurnError(RuntimeError):
    """Raised when an automation turn reaches the agent and finishes with an error.

    中文翻译：当一个自动化轮次到达 agent，并最终以错误结束时抛出。
    """

    # 这个类本身没有新增字段或方法。
    # 它继承 RuntimeError，目的是定义一种更具体的运行时错误类型，
    # 方便上层代码专门捕获“自动化轮次失败”这类错误。
    pass


async def publish_next_deferred_turn(
    *,  # 这里的星号 * 表示：后面所有参数都必须使用“关键字参数”方式传入，例如 deferred_queues=...。
    deferred_queues: dict[str, list[InboundMessage]],  # 延迟队列字典：key 是会话标识，value 是该会话待处理的入站消息列表。
    publish_inbound: Callable[[InboundMessage], Awaitable[None]],  # 一个异步回调，用于把入站消息发布到消息总线或 agent 输入流。
    session_key: str,  # 目标会话的唯一标识，用来找到该会话对应的延迟队列。
) -> bool:  # 返回 bool：True 表示成功发布了至少一条延迟消息；False 表示没有可发布的消息。
    """Publish the next deferred automation turn for a session.

    中文翻译：为某个会话发布下一条被延迟执行的自动化轮次。
    """

    # 从 deferred_queues 字典中取出当前 session_key 对应的延迟队列。
    # 如果不存在该 key，dict.get() 会返回 None，而不是抛出 KeyError。
    queue = deferred_queues.get(session_key)

    # 如果 queue 为 None，或者 queue 是空列表，则没有消息可发布。
    # Python 中 None、空列表 []、空字符串 ""、数字 0 等在布尔判断中都视为 False。
    if not queue:
        # 返回 False，表示没有发布任何延迟轮次。
        return False

    # 从队列头部弹出一条消息。
    # list.pop(0) 会移除并返回列表的第一个元素，实现类似 FIFO 队列的行为。
    # 注意：list.pop(0) 的时间复杂度是 O(n)，因为后面的元素要整体前移；
    # 如果队列很大，通常会考虑 collections.deque，但这里可能假设队列较短。
    msg = queue.pop(0)

    # 如果弹出这条消息后队列已经空了，就把这个 session_key 从 deferred_queues 中删除。
    # 这样可以避免字典中残留空列表，保持状态干净。
    if not queue:
        # dict.pop(key, default)：如果 key 存在则删除并返回对应值；
        # 如果 key 不存在则返回 default，这里 default 是 None，不会抛异常。
        deferred_queues.pop(session_key, None)

    # 调用异步发布函数，把这条入站消息发布出去。
    # publish_inbound 的类型是 Callable[[InboundMessage], Awaitable[None]]，
    # 因此它接受一个 InboundMessage，并返回一个可 await 的对象。
    # await 会等待发布动作真正完成。
    await publish_inbound(msg)

    # 返回 True，表示成功发布了一条延迟消息。
    return True


class AutomationTurnCoordinator:
    """Manage automation turns without mixing them into live injections.

    中文翻译：管理自动化轮次，避免把它们混入实时注入（live injections）流程中。

    这个协调器的核心职责：
    1. 提交自动化消息，并等待对应的出站响应；
    2. 如果目标会话当前活跃，则可以把自动化消息延后；
    3. 维护每个 turn_id 对应的等待 Future；
    4. 在轮次完成或失败时，通过 complete() 通知等待者。
    """

    def __init__(
        self,
        *,  # 星号 * 表示：__init__ 后面所有参数都必须以关键字方式传入。
        publish_inbound: Callable[[InboundMessage], Awaitable[None]],  # 异步回调：把消息发布到入站消息流。
        dispatch: Callable[[InboundMessage], Awaitable[object]],  # 异步回调：直接分派/处理一条消息；返回值类型这里只标注为 object。
        is_running: Callable[[], bool],  # 无参回调：返回当前系统/agent 是否处于运行状态。
        turn_id: Callable[[InboundMessage], str | None],  # 回调：从入站消息中提取自动化轮次 ID；可能返回 None。
        pending_id: Callable[[InboundMessage], str | None],  # 回调：从入站消息中提取“待处理 ID”；可能返回 None。
        should_defer_turn: Callable[[InboundMessage, str, Iterable[str]], bool],  # 回调：判断某条消息是否应该被延迟。
        missing_id_error: str,  # 当消息缺少 turn_id 时使用的错误信息。
        duplicate_id_error: Callable[[str], str],  # 当 turn_id 重复时，根据 turn_id 生成错误信息。
        deferred_queues: dict[str, list[InboundMessage]] | None = None,  # 可选的延迟队列字典；如果外部没传，则内部新建一个空 dict。
    ) -> None:  # __init__ 返回 None。
        # 保存“发布入站消息”的异步回调。
        # 后续在 submit() 中，如果系统处于运行状态，就会通过它把消息发布出去。
        self._publish_inbound = publish_inbound

        # 保存“直接分派消息”的异步回调。
        # 后续在 submit() 中，如果系统未运行，可能直接调用 dispatch 处理消息。
        self._dispatch = dispatch

        # 保存“是否正在运行”的回调。
        # self._is_running() 会返回 bool，用于决定 publish 还是 dispatch。
        self._is_running = is_running

        # 保存“提取 turn_id”的回调。
        # turn_id 用来唯一标识一次自动化轮次，也是等待响应的 key。
        self._turn_id = turn_id

        # 保存“提取 pending_id”的回调。
        # pending_id 可能用于对外暴露哪些自动化任务正在等待或执行。
        self._pending_id = pending_id

        # 保存“是否应该延迟该轮次”的回调。
        # 它通常会根据消息、目标 session_key、当前活跃会话列表来判断。
        self._should_defer_turn = should_defer_turn

        # 保存缺少 turn_id 时的错误文案。
        # 当 self._turn_id(msg) 返回 None 或空字符串时，会抛出 ValueError(self._missing_id_error)。
        self._missing_id_error = missing_id_error

        # 保存“重复 turn_id 错误信息生成器”。
        # 它是一个函数，输入重复的 turn_id，输出错误消息字符串。
        self._duplicate_id_error = duplicate_id_error

        # 初始化延迟队列字典。
        # 如果调用方传入了 deferred_queues，就复用外部字典；
        # 如果传入 None，就创建一个新的空字典。
        # 这样既允许外部共享状态，也支持独立实例。
        self.deferred_queues = deferred_queues if deferred_queues is not None else {}

        # _waiters 保存正在等待结果的 Future。
        # key 是 turn_id；
        # value 是 asyncio.Future，最终会被设置为 OutboundMessage、None，或异常。
        # Future 可以理解为“未来才会有结果”的占位对象。
        self._waiters: dict[str, asyncio.Future[OutboundMessage | None]] = {}

        # _pending_messages_by_turn_id 保存当前已经提交、但尚未完成的入站消息。
        # key 是 turn_id；
        # value 是对应的 InboundMessage。
        # 它用于查询某个会话当前有哪些正在执行/等待的自动化消息。
        self._pending_messages_by_turn_id: dict[str, InboundMessage] = {}

    async def submit(self, msg: InboundMessage) -> OutboundMessage | None:
        """Submit an automation turn and wait for its session response.

        中文翻译：提交一个自动化轮次，并等待该会话产生的响应。
        """

        # 调用外部传入的 turn_id 回调，从消息中提取本次自动化轮次的 ID。
        # 如果消息中没有 ID，可能返回 None 或空字符串。
        turn_id = self._turn_id(msg)

        # 如果没有 turn_id，则无法跟踪这次自动化轮次，因此直接抛出 ValueError。
        # Python 中 None、"" 等在 if not turn_id 条件下都会被视为 False。
        if not turn_id:
            raise ValueError(self._missing_id_error)

        # 如果同一个 turn_id 已经存在等待者，说明重复提交了同一个自动化轮次。
        # 这可能会导致结果不知道应该唤醒谁，因此直接拒绝。
        if turn_id in self._waiters:
            # duplicate_id_error 是一个函数，传入重复的 turn_id，返回错误文本。
            raise RuntimeError(self._duplicate_id_error(turn_id))

        # 获取当前正在运行的 asyncio 事件循环。
        # 在 async 函数中，asyncio.get_running_loop() 会返回当前事件循环对象。
        # 如果没有正在运行的事件循环，它会抛出 RuntimeError。
        loop = asyncio.get_running_loop()

        # 创建一个新的 Future。
        # Future 是 asyncio 中的底层“等待对象”。
        # 这里未来会被 complete() 方法填入结果或异常。
        # 类型注解：这个 Future 最终可能得到 OutboundMessage，也可能得到 None。
        future: asyncio.Future[OutboundMessage | None] = loop.create_future()

        # 把 Future 保存到 _waiters 字典中。
        # 后续 complete() 会根据 turn_id 找到这个 Future，并设置结果。
        self._waiters[turn_id] = future

        # 把当前提交的入站消息保存到 pending 字典中。
        # 这样后面可以查询“某个 session 当前有哪些正在等待/执行的自动化消息”。
        self._pending_messages_by_turn_id[turn_id] = msg

        # try/finally 用于保证无论成功、失败还是取消，
        # 最终都会清理 _waiters 和 _pending_messages_by_turn_id 中的状态。
        try:
            # 判断当前系统/agent 是否处于运行状态。
            # self._is_running 是一个回调，调用后返回 bool。
            if self._is_running():
                # 如果正在运行，则通过 publish_inbound 把消息发布到正常入站流程。
                # 这通常意味着消息会经过消息总线、队列、agent 输入流等正式路径。
                await self._publish_inbound(msg)
            else:
                # 如果未运行，则直接调用 dispatch 处理消息。
                # 这可能用于离线处理、直接执行、测试环境或无总线模式。
                await self._dispatch(msg)

            # 这里进入第二层 try，用于等待 Future 的结果。
            # 注意：上面的 publish/dispatch 只是“提交消息”，
            # 真正的响应通常要等 agent 异步处理完成后，由 complete() 写入 Future。
            try:
                # await future 会挂起当前协程，直到 Future 被设置结果或异常。
                # 如果 complete() 调用 future.set_result(response)，这里返回 response。
                # 如果 complete() 调用 future.set_exception(error)，这里抛出 error。
                return await future

            except asyncio.CancelledError:
                # 如果等待过程被取消，保持取消异常原样抛出。
                # 在 asyncio 中，CancelledError 表示任务被主动取消，
                # 通常不应该吞掉，而应该继续向上传播。
                raise

            except AutomationTurnError:
                # 如果已经是 AutomationTurnError，也原样抛出。
                # 这样可以避免把自动化错误再次包装成另一个 AutomationTurnError。
                raise

            except Exception as exc:
                # 其他所有普通异常都会被包装成 AutomationTurnError。
                # str(exc) 获取异常文本；如果为空，则使用异常类名 exc.__class__.__name__。
                # from exc 表示保留原始异常链，便于调试时看到真正的异常来源。
                raise AutomationTurnError(str(exc) or exc.__class__.__name__) from exc

        finally:
            # 无论上面发生了什么，都清理等待者记录。
            # dict.pop(key, None)：删除 turn_id 对应的 Future；如果不存在也不会报错。
            self._waiters.pop(turn_id, None)

            # 同时清理 pending 消息记录。
            # 表示这个 turn_id 不再处于“已提交但未完成”的状态。
            self._pending_messages_by_turn_id.pop(turn_id, None)

    def defer_if_active(
        self,
        msg: InboundMessage,  # 要检查并可能延迟执行的入站消息。
        *,  # 后面的参数必须使用关键字参数传入。
        session_key: str,  # 目标会话 key，即这条自动化消息希望进入哪个会话。
        active_session_keys: Iterable[str],  # 当前活跃会话 key 集合/可迭代对象。
    ) -> bool:  # 返回 bool：True 表示消息被成功延迟；False 表示没有延迟。
        """Defer an automation turn when its target session is already active.

        中文翻译：当目标会话已经处于活跃状态时，延迟执行一个自动化轮次。
        """

        # 调用外部传入的 should_defer_turn 回调，判断是否应该延迟这条消息。
        # 参数含义：
        # msg：当前入站消息；
        # session_key：目标会话；
        # active_session_keys：当前活跃会话列表/集合。
        # 如果回调返回 False，表示不需要延迟，直接返回 False。
        if not self._should_defer_turn(msg, session_key, active_session_keys):
            return False

        # pending_msg 默认指向原始消息。
        # 如果目标 session_key 与消息自身携带的 session_key 一致，就直接复用原消息。
        pending_msg = msg

        # 如果目标 session_key 和消息自带的 session_key 不同，
        # 说明这条消息需要被“重定向/覆盖”到另一个会话中执行。
        if session_key != msg.session_key:
            # dataclasses.replace(msg, session_key_override=session_key) 的作用：
            # 1. 以 msg 为基础；
            # 2. 复制出一个新的 dataclass 实例；
            # 3. 将新实例中的 session_key_override 字段设置为 session_key；
            # 4. 其他字段保持不变。
            # 这样可以避免直接修改原始 msg，保持原消息不可变或不被污染。
            pending_msg = dataclasses.replace(
                msg,
                session_key_override=session_key,
            )

        # 将 pending_msg 加入目标 session_key 的延迟队列。
        # dict.setdefault(key, default)：
        # 如果 key 不存在，就把 default 写入字典并返回；
        # 如果 key 已存在，就返回已有值。
        # 因此这里等价于：
        # if session_key not in self.deferred_queues:
        #     self.deferred_queues[session_key] = []
        # self.deferred_queues[session_key].append(pending_msg)
        self.deferred_queues.setdefault(session_key, []).append(pending_msg)

        # 返回 True，表示该消息已被成功延迟。
        return True

    def complete(
        self,
        msg: InboundMessage,  # 触发完成事件的入站消息，通常和之前 submit 的消息对应。
        *,  # 后面的参数必须使用关键字参数传入。
        response: OutboundMessage | None = None,  # 成功完成时的响应；可能为空。
        error: BaseException | None = None,  # 失败完成时的异常；可能为空。
    ) -> None:  # 该方法不返回有意义的值。
        # 从完成消息中提取 turn_id。
        # 这个 turn_id 用于找到 submit() 中创建的 Future。
        turn_id = self._turn_id(msg)

        # 如果没有 turn_id，则无法找到等待者，直接忽略。
        if not turn_id:
            return

        # 根据 turn_id 查找对应的 Future。
        # 如果没有找到，说明可能已经完成过、被清理了，或者不是由本协调器提交的。
        future = self._waiters.get(turn_id)

        # 如果 future 不存在，或者 future 已经完成（已经有结果、异常或被取消），
        # 则不再重复设置结果。
        # future.done() 返回 True 表示 Future 不再接受新的结果。
        if future is None or future.done():
            return

        # 如果外部传入了 error，说明该自动化轮次失败。
        if error is not None:
            # 如果错误是 asyncio.CancelledError，这里会转换成 AutomationTurnError。
            # 这样调用方看到的是“自动化轮次错误”，而不是裸的取消异常。
            # 注意：submit() 中等待 future 被取消时仍会单独处理 CancelledError；
            # 这里主要处理“执行过程被取消”的场景。
            if isinstance(error, asyncio.CancelledError):
                error = AutomationTurnError(str(error) or error.__class__.__name__)

            # 将异常设置到 Future 中。
            # 之后在 submit() 里 await future 时，会抛出这个异常。
            future.set_exception(error)
        else:
            # 如果没有 error，说明成功完成。
            # 将响应结果设置到 Future 中。
            # 之后在 submit() 里 await future 时，会得到这个 response。
            future.set_result(response)

    def pending_ids_for_session(self, session_key: str) -> set[str]:
        """Return automation IDs that are waiting for or running in *session_key*.

        中文翻译：返回当前在指定会话 *session_key* 中等待或正在运行的自动化 ID 集合。
        """

        # 创建一个空集合，用于收集 pending_id。
        # set 会自动去重，因此同一个 pending_id 不会重复出现。
        pending_ids: set[str] = set()

        # 遍历该 session_key 的延迟队列。
        # deferred_queues.get(session_key, [])：
        # 如果该会话没有延迟队列，则返回空列表 []，不会报错。
        for msg in self.deferred_queues.get(session_key, []):
            # 调用 pending_id 回调，从消息中提取 pending_id。
            # pending_id 可能是自动化任务 ID、请求 ID 或其他可用于去重/查询的 ID。
            pending_id = self._pending_id(msg)

            # 如果 pending_id 不是 None、空字符串等假值，就加入集合。
            if pending_id:
                pending_ids.add(pending_id)

        # 遍历所有“已提交但尚未完成”的消息。
        # self._pending_messages_by_turn_id.values() 返回所有正在等待/执行中的 InboundMessage。
        for msg in self._pending_messages_by_turn_id.values():
            # 如果这条消息不属于目标 session_key，则跳过。
            # 这里只关心当前 session 的 pending IDs。
            if msg.session_key != session_key:
                continue

            # 从消息中提取 pending_id。
            pending_id = self._pending_id(msg)

            # 如果 pending_id 有效，则加入结果集合。
            if pending_id:
                pending_ids.add(pending_id)

        # 返回收集到的 pending_id 集合。
        # 返回值类型是 set[str]，即字符串集合。
        return pending_ids