"""Auto compact: proactive compression of idle sessions to reduce token cost and latency."""
# 自动压缩：主动压缩空闲会话，以降低 token 成本和延迟。

from __future__ import annotations
# Python 特殊用法：从 __future__ 导入 annotations。
# 这会开启类型提示的“延迟计算”（PEP 563），允许在类型提示中使用尚未定义的类（前向引用），
# 并能提升模块加载时的性能，避免在运行时解析复杂的类型表达式。

from collections.abc import Collection
# 导入 Collection 抽象基类，用于类型提示，表示任何集合类型（如 list, tuple, set 等）。
from datetime import datetime
# 导入 datetime 类，用于处理日期和时间。
from typing import TYPE_CHECKING, Any, Callable, Coroutine, cast
# 从 typing 模块导入类型提示相关的工具：
# - TYPE_CHECKING: 一个特殊的常量，在运行时常量值为 False，但在静态类型检查工具（如 mypy）中为 True。
# - Any: 表示任意类型。
# - Callable: 用于定义函数或方法的类型签名。
# - Coroutine: 用于定义异步协程的类型。
# - cast: 类型转换提示工具，仅在静态检查时有效，运行时不执行任何操作，用于告诉类型检查器变量的确切类型。

from loguru import logger
# 导入 loguru 的日志记录器。

from nanobot.session.manager import MIN_COMPACTED_REPLAY_MESSAGES, Session, SessionManager
# 从 nanobot 框架导入会话相关的组件：
# - MIN_COMPACTED_REPLAY_MESSAGES: 常量，压缩后需要保留的最近消息的最小数量。
# - Session: 会话数据模型。
# - SessionManager: 负责管理会话生命周期的管理器。

if TYPE_CHECKING:
    # Python 特殊用法：条件导入块。
    # 这里的代码只在类型检查工具（如 mypy）运行代码时才会被执行。
    # 在实际运行 Python 脚本时，这段代码会被完全忽略。这主要用于解决循环导入问题，
    # 并避免在运行时导入仅用于类型提示的重型模块，从而提升启动速度。
    from nanobot.agent.memory import Consolidator
    # 导入 Consolidator 类，负责大模型记忆/上下文的整合与压缩。
    from nanobot.utils.llm_runtime import LLMRuntime
    # 导入 LLMRuntime 类，表示大语言模型的运行时实例。


class AutoCompact:
    # 定义 AutoCompact 类，核心职责是自动扫描并压缩长时间不活跃的会话，以节省 Token。
    
    _RECENT_SUFFIX_MESSAGES = MIN_COMPACTED_REPLAY_MESSAGES
    # 类变量：保留在压缩后的上下文中的“最近消息后缀”数量。直接引用外部定义的常量。
    _INTERNAL_SESSION_PREFIXES = ("dream:",)
    # 类变量：内部系统会话的前缀元组。带有这些前缀的会话（如 "dream:..."）会被排除在自动压缩之外。

    def __init__(self, sessions: SessionManager, consolidator: Consolidator,
                 session_ttl_minutes: int = 0):
        # 初始化方法：
        # - sessions: 会话管理器实例。
        # - consolidator: 记忆整合器实例（用于调用大模型生成摘要）。
        # - session_ttl_minutes: 会话存活时间（Time To Live），单位为分钟。超过此时间未活动的会话将被视为空闲。默认为 0（不自动压缩）。
        
        self.sessions = sessions
        # 保存会话管理器实例。
        self.consolidator = consolidator
        # 保存记忆整合器实例。
        self._ttl = session_ttl_minutes
        # 保存 TTL 设置，作为判断会话是否过期的阈值。
        self._archiving: set[str] = set()
        # 初始化一个集合，用于记录当前正在后台异步进行归档（压缩）操作的会话 key，防止重复触发。
        self._summaries: dict[str, tuple[str, datetime]] = {}
        # 初始化一个字典，用于在内存中缓存会话的摘要。
        # 键是会话的 key，值是包含 (摘要文本, 最后活跃时间) 的元组。

    def _is_expired(self, ts: datetime | str | None,
                    now: datetime | None = None) -> bool:
        # 内部方法：判断给定的时间戳是否已经超过了设定的 TTL（即是否已过期/空闲）。
        # 参数 ts: 时间戳，可以是 datetime 对象、ISO 8601 格式的字符串，或者 None。
        #          Python 特殊用法：`datetime | str | None` 是 PEP 604 引入的新语法，等同于旧版的 `Union[datetime, str, None]`。
        # 参数 now: 当前时间，用于计算差值。如果不传，则自动获取系统当前时间。
        
        if self._ttl <= 0 or not ts:
            # 如果 TTL 设置为 0（或负数，即禁用自动压缩），或者根本没有传入时间戳，则直接返回 False（未过期）。
            return False
        try:
            if isinstance(ts, str):
                # 如果传入的时间戳是字符串，尝试将其解析为 datetime 对象。
                ts = datetime.fromisoformat(ts)
            current = now or datetime.now()
            # 确定“当前时间”：优先使用传入的 now 参数，否则使用系统的当前时间。
            if getattr(ts, "tzinfo", None) is not None or current.tzinfo is not None:
                # Python 特殊用法：getattr(obj, name, default) 安全获取属性。
                # 这里检查 ts 或 current 是否包含时区信息（tzinfo）。
                # 如果两者中有任何一个包含时区信息，为了防止时区计算错误，统一转换为 Unix 时间戳（浮点数秒）进行相减。
                idle_seconds = current.timestamp() - ts.timestamp()
            else:
                # 如果两个时间对象都是“无时区信息”（naive datetime），则直接相减，并调用 total_seconds() 获取总秒数。
                idle_seconds = (current - ts).total_seconds()
        except (OSError, OverflowError, TypeError, ValueError):
            # 捕获所有可能的时间解析或计算异常。
            
            # list_sessions() forwards raw persisted metadata; an unusable value
            # must not escape the idle scan and stop the agent loop.
            # list_sessions() 会直接透传底层持久化的原始元数据；一个不可用的脏数据值
            # 绝不能逃逸出空闲扫描过程，进而导致整个 agent 主循环崩溃停止。
            
            return False
        return idle_seconds >= self._ttl * 60
        # 最终判断：计算出的空闲秒数是否大于等于设定的 TTL 分钟数转换成的秒数（TTL * 60）。

    def _has_unarchived_messages(self, key: str) -> bool:
        # 内部方法：检查指定 key 的会话中，是否还有未被压缩归档的“新消息”。
        session = self.sessions.get_or_create(key)
        # 通过 key 获取或创建对应的 Session 对象。
        return session.last_consolidated < len(session.messages)
        # 比较逻辑：将“上一次压缩时记录的消息索引/数量”与“当前会话实际包含的消息总数”进行对比。
        # 如果当前消息数更多，说明在两次压缩之间产生了新消息，返回 True。

    @staticmethod
    def _format_summary(text: str, last_active: datetime) -> str:
        # Python 特殊用法：@staticmethod 装饰器。
        # 定义静态方法，意味着该方法既不需要访问实例属性（没有 self 参数），也不需要访问类属性（没有 cls 参数）。
        # 它本质上就是一个放在类命名空间里的普通函数。
        
        return f"Previous conversation summary (last active {last_active.isoformat()}):\n{text}"
        # 格式化摘要文本：生成一段包含最后活跃时间（ISO 8601 格式）和实际摘要内容的字符串，供大模型作为上下文阅读。

    @classmethod
    def _is_internal_session(cls, key: str) -> bool:
        # Python 特殊用法：@classmethod 装饰器。
        # 定义类方法，第一个参数必须是 cls（代表类本身，而不是实例 self）。
        # 类方法可以通过类名直接调用，并且能够访问和修改类变量。
        
        return key.startswith(cls._INTERNAL_SESSION_PREFIXES)
        # 检查传入的会话 key 是否以类变量 `_INTERNAL_SESSION_PREFIXES`（元组）中的任意一个字符串开头。
        # 例如前缀是 ("dream:",)，那么 "dream:123" 会返回 True。

    def check_expired(
        self,
        schedule_background: Callable[[Coroutine[Any, Any, None]], None],
        resolve_runtime: Callable[[Session], LLMRuntime],
        active_session_keys: Collection[str] = (),
    ) -> None:
        # 公共方法：扫描所有会话，找出过期的空闲会话，并将压缩任务调度到后台执行。
        # 会跳过当前正在被用户或 agent 活跃使用的会话。
        # 参数：
        # - schedule_background: 回调函数，用于将一个异步协程提交到后台事件循环中执行。
        # - resolve_runtime: 回调函数，用于根据 Session 解析出对应的 LLM 运行时配置（用哪个模型来压缩）。
        # - active_session_keys: 当前处于活跃状态的会话 key 集合，默认为空元组。
        
        """Schedule archival for idle sessions, skipping those with in-flight agent tasks."""
        # 为空闲会话调度归档任务，跳过那些有正在执行的 agent 任务的会话。
        
        now = datetime.now()
        # 获取当前系统时间，作为后续时间对比的基准。
        for info in self.sessions.list_sessions():
            # 遍历会话管理器中提供的所有会话元数据字典。
            key = info.get("key", "")
            # 安全获取会话的 key，如果字典中没有 "key" 键，则默认为空字符串。
            if not key or self._is_internal_session(key) or key in self._archiving:
                # 过滤条件 1：如果 key 为空，或者是内部系统会话，或者该会话已经在后台正在被压缩（防止重复调度），则跳过。
                continue
            if key in active_session_keys:
                # 过滤条件 2：如果该会话在当前活跃会话列表中（说明用户正在跟它聊天），则跳过。
                continue
            updated_at = info.get("updated_at")
            # 获取会话的最后更新时间。
            if self._is_expired(updated_at, now) and self._has_unarchived_messages(key):
                # 核心判断：如果会话已经“过期”（空闲时间达标） 并且 还有“未压缩的新消息”：
                session = self.sessions.get_or_create(key)
                # 获取完整的 Session 实例。
                try:
                    runtime = resolve_runtime(session)
                    # 尝试调用回调函数，获取该会话指定的 LLM 运行时（例如确定使用哪个大模型 API）。
                except (KeyError, ValueError):
                    # Invalid session selections remain recoverable through /model.
                    # 无效的会话选择（例如找不到对应的模型配置）仍然是可以通过 /model 命令来恢复的。
                    continue
                self._archiving.add(key)
                # 将该会话 key 加入 `_archiving` 集合，标记为“正在处理”。
                schedule_background(self._archive(key, runtime=runtime))
                # 调用后台调度函数，将 `_archive` 异步方法包装成协程任务并放入后台执行。

    async def _archive(self, key: str, *, runtime: LLMRuntime) -> None:
        # 异步内部方法：实际执行会话压缩和摘要生成的核心逻辑。
        # Python 特殊用法：参数列表中的 `*` 表示其后的所有参数（这里是 runtime）都必须作为关键字参数传入。
        # 例如必须写成 `self._archive("key", runtime=rt)`，不能写成 `self._archive("key", rt)`。
        
        if self._is_internal_session(key):
            # 防御性检查：再次确认是否为内部会话，如果是则移出归档集合并直接返回。
            self._archiving.discard(key)
            return
        try:
            summary = await self.consolidator.compact_idle_session(
                key,
                runtime=runtime,
                max_suffix=self._RECENT_SUFFIX_MESSAGES,
            )
            # 调用 Consolidator 的异步压缩方法。传入会话 key、模型运行时、以及需要保留的最近消息数量。
            # 压缩完成后，返回大模型生成的摘要文本。
            
            if summary and summary != "(nothing)":
                # 如果摘要生成成功，且内容不是代表无内容的占位符 "(nothing)"：
                session = self.sessions.get_or_create(key)
                # 获取 Session 对象以更新其元数据。
                meta = session.metadata.get("_last_summary")
                # 从会话的 metadata（元数据字典）中尝试获取键为 "_last_summary" 的数据。
                if isinstance(meta, dict):
                    # 如果获取到的数据确实是一个字典：
                    self._summaries[key] = (
                        cast(str, meta["text"]),
                        datetime.fromisoformat(cast(str, meta["last_active"])),
                    )
                    # Python 特殊用法：cast(str, ...) 仅在静态类型检查时起作用。
                    # 告诉 mypy 等工具 meta["text"] 肯定是字符串，然后将其和解析后的时间存入内存缓存 `_summaries` 中。
        except Exception:
            # 捕获压缩过程中发生的任何未知异常。
            logger.exception("Auto-compact: failed for {}", key)
            # 使用 loguru 记录详细的异常堆栈信息，并指明是哪个会话 key 失败了。
        finally:
            # Python 特殊用法：finally 块无论 try 块中是否发生异常，都必定会执行。
            # 用于执行资源清理工作。
            self._archiving.discard(key)
            # 将该 key 从 `_archiving` 集合中移除，表示后台任务已结束，后续可以再次处理。

    def prepare_session(self, session: Session, key: str) -> tuple[Session, str | None]:
        # 公共方法：在 agent 处理新回合（turn）之前，准备会话上下文。
        # 主要任务是检查会话状态，并提取出最新的摘要作为上下文提示注入。
        # 返回一个元组：(准备好的 Session 对象, 摘要文本或 None)。
        
        if self._is_internal_session(key):
            # 如果是内部系统会话：
            self._archiving.discard(key)
            # 确保它不在归档集合中。
            self._summaries.pop(key, None)
            # 从内存缓存中清除它的摘要（内部会话不需要注入摘要）。
            return session, None
            # 直接返回原会话和 None。
            
        if key in self._archiving or self._is_expired(session.updated_at):
            # 如果该会话目前正在后台压缩，或者它在内存中的状态显示已过期：
            logger.info("Auto-compact: reloading session {} (archiving={})", key, key in self._archiving)
            # 记录信息日志。
            session = self.sessions.get_or_create(key)
            # 强制从持久化存储中重新加载该会话，因为后台的压缩任务可能刚刚修改了它的底层数据。
            
        # Hot path: summary from in-memory dict (process hasn't restarted).
        # 热路径：从内存字典中获取摘要（这通常发生在进程没有重启，内存缓存依然有效的情况下，速度最快）。
        entry = self._summaries.pop(key, None)
        # 尝试从内存缓存 `_summaries` 中获取并移除（pop）该会话的摘要。
        if entry:
            # 如果内存中存在缓存：
            return session, self._format_summary(entry[0], entry[1])
            # 将缓存的文本（entry[0]）和时间（entry[1]）格式化后返回。

        # Cold path: summary persisted in session metadata (process restarted).
        # 冷路径：从会话的持久化元数据中获取摘要（通常是因为服务进程刚刚重启，导致内存缓存 `_summaries` 丢失）。
        # Persisted metadata may outlive schema changes; a malformed summary must
        # not abort turn preparation.
        # 持久化的元数据的生命周期可能比代码的结构 (schema) 变更更长；因此，遇到格式损坏的摘要
        # 绝不能导致当前回合 (turn) 的准备工作抛出异常而中断。
        
        meta = session.metadata.get("_last_summary")
        # 从 session 的 metadata 中读取持久化的 "_last_summary"。
        if isinstance(meta, dict):
            # 确认它是一个字典：
            summary_meta = cast(dict[str, object], meta)
            # 使用 cast 进行静态类型断言，将其视为字典。
            text = summary_meta.get("text")
            # 提取 "text" 字段。
            if isinstance(text, str) and text:
                # 确保提取到的文本是合法的字符串且不为空：
                raw_last_active = summary_meta.get("last_active")
                # 提取 "last_active" 字段。
                try:
                    # Python 特殊用法：三元条件表达式 `A if condition else B`。
                    last_active = (
                        datetime.fromisoformat(raw_last_active)
                        if isinstance(raw_last_active, str)
                        else session.updated_at
                    )
                    # 尝试解析时间字符串。如果它是字符串则解析；如果不是，则退回到使用会话的更新时间作为兜底。
                except ValueError:
                    # 如果时间解析失败（例如格式错误）：
                    last_active = session.updated_at
                    # 降级处理，使用 session.updated_at 作为最后活跃时间。
                return session, self._format_summary(text, last_active)
                # 格式化并返回最终的上下文提示字符串。
                
        return session, None
        # 兜底返回：如果内存和持久化数据中都没有找到合法的摘要，则返回原 session 和 None（即不注入任何摘要）。