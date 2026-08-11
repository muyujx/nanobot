"""Coordination for scheduled cron turns.

[中文翻译] 定时 cron 回合（turn）的协调模块。
"""

# 导入 __future__ 模块中的 annotations 特性。
# 作用：让类型注解变成“延迟求值”，也就是在运行时不会立即解析类型注解。
# 好处：可以使用较新的类型写法，例如：
#   - dict[str, list[InboundMessage]]
#   - str | None
# 即使某些类型在当前作用域中还没完全定义，也不会在导入注解时报错。
from __future__ import annotations

# 从 collections.abc 导入用于类型注解的抽象类型：
# Awaitable：表示一个“可被 await 的对象”，通常用于异步函数返回值类型。
# Callable：表示“可调用对象”，例如函数、方法、lambda、实现 __call__ 的对象。
# Iterable：表示“可迭代对象”，例如 list、tuple、set、generator 等。
from collections.abc import Awaitable, Callable, Iterable

# 导入 AutomationTurnCoordinator。
# 这是一个通用的“自动化回合协调器”基类，负责管理自动化消息回合的调度、去重、延迟等逻辑。
from nanobot.agent.automation_turns import AutomationTurnCoordinator

# 导入 InboundMessage。
# 它表示一条进入系统内部的消息事件，通常包含消息内容和元数据 metadata。
from nanobot.bus.events import InboundMessage

# 从 nanobot.cron.session_turns 导入三个和 cron 会话回合相关的辅助函数。
from nanobot.cron.session_turns import (
    # cron_run_id：从消息的 metadata 中提取当前 cron 运行的唯一运行 ID（run_id）。
    cron_run_id,
    # cron_trigger：从消息的 metadata 中提取 cron 触发器信息，通常是一个 dict。
    cron_trigger,
    # defer_cron_until_session_idle：判断 metadata 是否要求“等到会话空闲后再执行 cron”。
    defer_cron_until_session_idle,
)


# 定义 CronTurnCoordinator 类。
# 它继承自 AutomationTurnCoordinator，表示这是一个专门用于 cron 定时任务回合的协调器。
class CronTurnCoordinator(AutomationTurnCoordinator):
    """Manage scheduled cron turns without mixing them into live injections.

    [中文翻译] 管理计划任务触发的 cron 回合，避免把它们混入实时注入的消息流中。
    """

    # 构造方法，用于创建 CronTurnCoordinator 实例。
    def __init__(
        # self 表示当前类实例本身。
        self,
        # 单独的星号 * 是 Python 的“关键字参数分隔符”。
        # 它后面的所有参数都必须使用关键字传参，例如：
        # CronTurnCoordinator(publish_inbound=..., dispatch=..., ...)
        # 不能按位置传参。
        *,
        # publish_inbound 是一个可调用对象。
        # 类型 Callable[[InboundMessage], Awaitable[None]] 表示：
        #   - 它接收一个 InboundMessage 参数；
        #   - 它返回一个可 await 的对象；
        #   - await 之后的结果是 None。
        # 通常这是一个异步函数，用于把入站消息发布到消息总线或内部队列。
        publish_inbound: Callable[[InboundMessage], Awaitable[None]],
        # dispatch 也是一个可调用对象。
        # 类型 Callable[[InboundMessage], Awaitable[object]] 表示：
        #   - 它接收一个 InboundMessage 参数；
        #   - 它返回一个可 await 的对象；
        #   - await 之后的结果可以是任意 Python 对象，所以用 object 表示。
        # 它通常负责真正把消息分发给 agent 或后续处理流程。
        dispatch: Callable[[InboundMessage], Awaitable[object]],
        # is_running 是一个无参数可调用对象。
        # 类型 Callable[[], bool] 表示：
        #   - 不接收参数；
        #   - 返回 bool。
        # 它通常用于判断当前系统、agent 或事件循环是否仍在运行。
        is_running: Callable[[], bool],
        # deferred_queues 是一个可选参数，默认值是 None。
        # 类型 dict[str, list[InboundMessage]] | None 表示：
        #   - 要么是 None；
        #   - 要么是一个 dict。
        # 这个 dict 的 key 是 str，通常是 session_key；
        # value 是 list[InboundMessage]，也就是被延迟等待处理的消息列表。
        # 使用 | None 是 Python 3.10+ 的联合类型写法，等价于 Optional[dict[...]]。
        deferred_queues: dict[str, list[InboundMessage]] | None = None,
    # -> None 表示构造方法没有返回值。
    # Python 中 __init__ 按约定返回 None。
    ) -> None:
        # 调用父类 AutomationTurnCoordinator 的构造方法。
        # super() 返回当前对象的父类代理，通过它可以调用父类方法。
        super().__init__(
            # 把当前对象收到的 publish_inbound 原样传给父类。
            # 父类会用它来发布入站消息。
            publish_inbound=publish_inbound,
            # 把 dispatch 原样传给父类。
            # 父类会用它来实际执行消息分发。
            dispatch=dispatch,
            # 把 is_running 原样传给父类。
            # 父类可能用它判断系统是否还在运行，从而决定是否继续处理回合。
            is_running=is_running,
            # turn_id 是父类需要的回调函数，用于从消息中提取“回合 ID”。
            # 这里传入一个 lambda 匿名函数：
            #   lambda msg: cron_run_id(msg.metadata)
            # 含义是：当父类调用 turn_id(msg) 时，
            # 实际上会调用 cron_run_id(msg.metadata)，从消息元数据中取出 cron run_id。
            turn_id=lambda msg: cron_run_id(msg.metadata),
            # pending_id 是父类需要的回调函数，用于从消息中提取“待处理任务 ID”。
            # 这里直接传入函数对象 _cron_job_id。
            # 注意：这里没有写成 _cron_job_id()，因为没有立即调用它。
            # 父类会在需要时自行调用 _cron_job_id(msg)。
            pending_id=_cron_job_id,
            # should_defer_turn 是父类需要的策略函数。
            # 父类会调用它来判断某个回合是否应该被延迟。
            # 这里传入模块级私有函数 _should_defer_cron_turn。
            should_defer_turn=_should_defer_cron_turn,
            # 当 turn_id 缺失时，父类会使用这个错误信息。
            # 这里传入普通字符串。
            missing_id_error="cron turn metadata must include a run_id",
            # 当发现重复 turn_id 时，父类会调用这个函数生成错误信息。
            # 这里传入 lambda：
            #   lambda run_id: f"cron run {run_id!r} is already pending"
            # 其中：
            #   run_id 是重复的运行 ID；
            #   f"..." 是 f-string，即格式化字符串；
            #   {run_id!r} 表示用 repr(run_id) 格式化，便于显示引号和特殊字符。
            duplicate_id_error=lambda run_id: f"cron run {run_id!r} is already pending",
            # 把可选的延迟队列传给父类。
            # 如果外部没有传，则这里是 None，父类可能会自己创建内部队列。
            deferred_queues=deferred_queues,
        # 结束父类构造方法调用。
        )

    # 定义一个公开方法，用于查询某个会话中正在等待或运行的 cron job ID。
    def pending_job_ids_for_session(self, session_key: str) -> set[str]:
        """Return cron jobs that are waiting for or running in *session_key*.

        [中文翻译] 返回当前正在 *session_key* 中等待或执行的 cron 任务 ID。
        """
        # self.pending_ids_for_session(session_key) 是父类提供的方法。
        # 它会返回指定 session_key 当前所有待处理/运行中的 ID 集合。
        # 这里只是把父类方法包装成更具业务语义的名字：
        # “pending_job_ids_for_session”。
        return self.pending_ids_for_session(session_key)


# 定义模块级私有函数，用于判断某个 cron 回合是否应该被延迟。
# 函数名前的下划线 _ 表示它预期只在模块内部使用，不建议外部直接调用。
def _should_defer_cron_turn(
    # msg 是入站消息对象，包含 metadata。
    msg: InboundMessage,
    # session_key 是当前消息所属会话的标识。
    session_key: str,
    # active_session_keys 是当前处于活跃状态的会话 key 集合或可迭代对象。
    # Iterable[str] 表示里面每个元素都应该是 str。
    active_session_keys: Iterable[str],
# -> bool 表示这个函数返回布尔值：
# True：应该延迟该 cron 回合；
# False：不需要延迟。
) -> bool:
    # 这行代码返回两个条件同时成立的结果。
    # Python 的 and 具有短路特性：
    #   1. 先计算 defer_cron_until_session_idle(msg.metadata)；
    #   2. 如果它为 False，则整个表达式直接返回 False，不再判断后半部分；
    #   3. 如果它为 True，再继续判断 session_key 是否在 active_session_keys 中。
    #
    # 逻辑含义：
    #   - 只有当消息元数据要求“等到会话空闲后再执行 cron”；
    #   - 并且当前 session_key 正处于活跃状态；
    # 才需要延迟这个 cron 回合。
    return defer_cron_until_session_idle(msg.metadata) and session_key in active_session_keys


# 定义模块级私有函数，用于从入站消息中提取 cron job_id。
# 返回类型 str | None 表示：
#   - 如果成功提取到有效 job_id，返回字符串；
#   - 如果没有 trigger、没有 job_id，或 job_id 不是非空字符串，返回 None。
def _cron_job_id(msg: InboundMessage) -> str | None:
    # 调用 cron_trigger，从消息元数据中提取 cron 触发器数据。
    # 返回值 trigger 可能是一个 dict，也可能是 None、空 dict 等“假值”。
    trigger = cron_trigger(msg.metadata)

    # if not trigger 会判断 trigger 是否为“假值”。
    # 例如 None、{}、空列表、空字符串等都会让 not trigger 为 True。
    # 如果根本没有 cron trigger 数据，则无法提取 job_id，返回 None。
    if not trigger:
        return None

    # 从 trigger 字典中读取 job_id 字段。
    # dict.get("job_id") 的特点是：
    #   - 如果 key 存在，返回对应值；
    #   - 如果 key 不存在，返回 None；
    #   - 不会抛出 KeyError。
    value = trigger.get("job_id")

    # 这是一个条件表达式，等价于：
    #   if isinstance(value, str) and value:
    #       return value
    #   else:
    #       return None
    #
    # isinstance(value, str) 判断 value 是否是字符串。
    # and value 判断字符串是否非空；空字符串 "" 是假值。
    # 因此只有当 value 同时满足：
    #   1. 是字符串；
    #   2. 不是空字符串；
    # 才返回 value，否则返回 None。
    return value if isinstance(value, str) and value else None