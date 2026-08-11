"""Memory system: pure file I/O store and lightweight Consolidator."""
# 记忆系统：纯文件 I/O 存储和轻量级的上下文压缩器（Consolidator）。

# Tool schemas are installed by the ``@tool_parameters`` class decorator at
# runtime; static analyzers cannot observe that it clears ``parameters`` from
# ``__abstractmethods__`` before these classes are instantiated.
# pyright: reportAbstractUsage=false, reportPrivateUsage=false
# 【原有注释翻译】工具 schema（模式）是在运行时由 `@tool_parameters` 类装饰器安装的；
# 静态分析器无法观察到它在这些类被实例化之前从 `__abstractmethods__` 中清除了 `parameters`。
# 禁用 pyright 的抽象类使用报错和私有属性使用报错。

# 导入 future 的 annotations，允许在类型提示中使用前向引用（即在类完全定义前使用该类名作为类型）
from __future__ import annotations

import asyncio          # 异步 I/O 库，用于异步锁和异步任务
import json             # JSON 序列化/反序列化库
import os               # 操作系统接口，用于文件描述符和底层 fsync 操作
import re               # 正则表达式库，用于解析旧版历史记录
import threading        # 线程库，用于多线程环境下的同步锁
import weakref          # 弱引用库，用于防止内存泄漏（如在 Consolidator 中缓存锁）
from contextlib import suppress # 上下文管理器，用于静默忽略指定的异常（相当于 try-except-pass）
from datetime import datetime   # 日期时间库，用于生成时间戳
from pathlib import Path        # 面向对象的路径操作库
# 类型提示相关导入
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

from loguru import logger       # 第三方日志库，提供比 logging 更友好的 API

# 导入 Agent 运行时上下文中的公共历史消息过滤函数
from nanobot.runtime_context import public_history_messages
# 导入 Session（会话）管理相关的类和常量
from nanobot.session.manager import MIN_COMPACTED_REPLAY_MESSAGES, Session, SessionManager
from nanobot.utils.gitstore import GitStore # 导入 Git 存储工具，用于追踪文件变更生成 diff
# 导入各种辅助工具函数
from nanobot.utils.helpers import (
    content_with_media_breadcrumbs,  # 处理带有媒体面包屑（占位符）的内容
    ensure_dir,                      # 确保目录存在，不存在则创建
    estimate_message_tokens,         # 估算单条消息的 Token 数量
    estimate_prompt_tokens_chain,    # 估算整个 Prompt 链的 Token 数量
    find_legal_message_start,        # 查找合法的消息起始位置（如确保以 user 角色开始）
    recent_message_start_index,      # 获取最近消息的起始索引
    strip_think,                     # 剥离 LLM 的 "<think>" 思考过程标签，防止泄漏
    truncate_text,                   # 按字符数截断文本
    truncate_text_to_tokens,         # 按 Token 数截断文本
)
from nanobot.utils.prompt_templates import render_template # 导入 Jinja2 或类似模板渲染工具
# 导入 Workspace（工作区）提示词相关的配置和工具
from nanobot.utils.workspace_prompts import (
    WORKSPACE_PROMPT_MAX_CHARS,          # 工作区提示词的最大字符数限制
    has_workspace_prompt_override,       # 检查是否存在自定义的提示词覆盖
    load_workspace_prompt_override,      # 加载自定义的提示词覆盖
    workspace_prompt_file,               # 获取工作区提示词的文件路径
)

# TYPE_CHECKING 在运行时为 False，但在 mypy/pyright 等静态类型检查时为 True。
# 用于避免循环导入，同时保留类型提示。
if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry # 工具注册表
    from nanobot.utils.llm_runtime import LLMRuntime      # LLM 运行时环境

# ---------------------------------------------------------------------------
# MemoryStore — pure file I/O layer (内存存储层 - 纯文件 I/O)
# ---------------------------------------------------------------------------


class DreamRunProgress:
    """Track tool failures that make a nominally completed Dream run unsafe to advance."""
    # 【原有注释翻译】追踪工具调用失败，这些失败会导致名义上已完成的 Dream（梦境/后台思考）运行在推进时变得不安全。

    def __init__(self) -> None:
        # 初始化一个布尔标志，记录是否发生过工具错误
        self.had_tool_errors = False

    # 使该类的实例可以像函数一样被调用（实现 __call__ 魔术方法）
    # 这是一个异步回调函数，用于监听工具事件
    async def __call__(
        self,
        *_args: Any, # 接收任意数量的位置参数（使用 * 收集为元组，这里用 _ 表示忽略）
        tool_events: list[dict[str, Any]] | None = None, # 工具事件列表
        **_kwargs: Any, # 接收任意数量的关键字参数（忽略）
    ) -> None:
        # 检查 tool_events 中是否有任何事件的 "phase" 字段为 "error"
        if any(
            # cast(object, event) 告诉类型检查器 event 是 object 类型，避免 dict 检查报错
            isinstance(cast(object, event), dict) and event.get("phase") == "error"
            for event in tool_events or () # 如果 tool_events 为 None，则迭代空元组
        ):
            # 如果发现错误事件，将标志位设为 True
            self.had_tool_errors = True


class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history.jsonl, SOUL.md, USER.md."""
    # 【原有注释翻译】记忆文件的纯文件 I/O 操作：MEMORY.md（长期记忆）, history.jsonl（历史记录）, SOUL.md（AI人设）, USER.md（用户信息）。

    # 默认最大历史记录条目数
    _DEFAULT_MAX_HISTORY = 1000
    
    # Durable files whose real working-tree delta grounds Dream commit messages.
    # Deliberately excludes memory/.dream_cursor so progress bookkeeping never
    # appears as a durable-memory edit in the audit record.
    # 【原有注释翻译】持久化文件，其真实的工作树差异（delta）构成了 Dream 提交信息的基础。
    # 故意排除了 memory/.dream_cursor，这样进度记录就不会作为持久化记忆编辑出现在审计记录中。
    _DREAM_CONTENT_PATHS = ("SOUL.md", "USER.md", "memory/MEMORY.md")
    
    # Per-file cap when embedding current contents into the Dream prompt. The
    # durable files are tiny in practice (~5 KB total), but a runaway file must
    # not unbounded the prompt.
    # 【原有注释翻译】将当前内容嵌入到 Dream 提示词时，每个文件的字符数上限。
    # 实际上持久化文件很小（总共约 5 KB），但失控的文件不能无限制地撑爆提示词。
    _DREAM_FILE_EMBED_CAP = 8000
    
    # 内部历史会话的前缀和键（用于过滤掉系统内部的 cron 任务或 dream 运行记录）
    _INTERNAL_HISTORY_SESSION_PREFIXES = ("cron:", "dream:")
    _INTERNAL_HISTORY_SESSION_KEYS = {"heartbeat"}
    
    # 用于解析旧版 HISTORY.md 的正则表达式
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*") # 匹配旧版条目起始的日期
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*") # 匹配精确到分钟的时间戳
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    ) # 匹配原始消息格式

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        """
        初始化 MemoryStore。
        :param workspace: 工作区根目录路径
        :param max_history_entries: 历史记录文件保留的最大条目数
        """
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        # 确保 memory 目录存在，并获取其 Path 对象
        self.memory_dir = ensure_dir(workspace / "memory")
        # 定义各个核心记忆文件的路径
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md" # 旧版历史文件
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        # 游标文件，用于记录当前处理到的历史条目 ID，防止重复处理
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor" # Dream 模式专用的游标
        
        # 各种限流日志标志位，防止同样的警告日志刷屏
        self._corruption_logged = False  # rate-limit invalid cursor warning (限制无效游标警告)
        self._malformed_entry_logged = False  # rate-limit bad history shape warning (限制格式错误警告)
        self._oversize_logged = False  # rate-limit oversized-entry warning (限制超大条目警告)
        self._dream_prompt_oversize_logged = False
        
        # 线程锁：序列化游标分配和文件追加操作，保证多线程下的原子性
        self._append_lock = threading.Lock()  
        
        # 初始化 GitStore，用于追踪指定文件的变更，生成 Dream 模式的 diff 提交信息
        self._git = GitStore(workspace, tracked_files=[
            "SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor",
        ])
        # 尝试执行一次性的旧版历史记录迁移
        self._maybe_migrate_legacy_history()

    # @property 装饰器将方法转换为属性，调用时不需要加括号：store.git
    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers (通用辅助方法) -----------------------------------------------------

    @staticmethod # 静态方法，不需要实例化即可调用，且不依赖 self
    def read_file(path: Path) -> str:
        """安全地读取文件内容，如果文件不存在则返回空字符串。"""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl.

        The migration is best-effort and prioritizes preserving as much content
        as possible over perfect parsing.
        """
        # 【原有注释翻译】从旧版 HISTORY.md 升级到 history.jsonl 的一次性操作。
        # 迁移是尽最大努力的，优先保留尽可能多的内容，而不是追求完美的解析。
        
        # 如果旧文件不存在，直接返回
        if not self.legacy_history_file.exists():
            return
        # 如果新文件已经存在且不为空，说明已经迁移过，直接返回
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            # 读取旧文件，使用 errors="replace" 处理可能的编码错误
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        # 解析旧版文本为条目列表
        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                # 将解析后的条目写入新的 JSONL 文件
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                # 更新游标文件
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                # Default to "already processed" so upgrades do not replay the
                # user's entire historical archive into Dream on first start.
                # 【原有注释翻译】默认为“已处理”，这样升级后不会在首次启动时将用户的整个历史档案重放到 Dream 中。
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            # 将旧文件重命名为备份文件
            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        """解析旧版 Markdown 格式的历史文本。"""
        # 统一换行符并去除首尾空白
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        # 获取后备时间戳（当解析不到时间戳时使用）
        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        # 将文本分割成一个个独立的 chunk（条目）
        chunks = self._split_legacy_history_chunks(normalized)

        # enumerate 带有 start=1，使 cursor 从 1 开始递增
        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            # 尝试使用正则匹配提取时间戳
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                # 去除匹配到的时间戳部分，剩下的就是内容
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        """根据空行或特定格式将长文本分割成多个独立的条目块。"""
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = [] # 当前正在收集的块
        saw_blank_separator = False # 是否刚遇到过空行分隔符

        for line in lines:
            # 如果之前遇到了空行，且当前行不为空，且 current 中有内容，说明上一个块结束了
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            # 如果当前行符合新条目的起始特征，也说明上一个块结束了
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            # 否则将当前行加入当前块
            current.append(line)
            # 记录当前行是否为空行
            saw_blank_separator = not line.strip()

        # 循环结束后，把最后一个块加入列表
        if current:
            chunks.append("\n".join(current).strip())
        # 过滤掉空块
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        """判断当前行是否应该作为一个新条目的开始。"""
        if not current:
            return False
        # 如果不符合旧版条目起始的正则（如日期格式），则不是新条目
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        # 如果当前块是 RAW（原始消息）块，且当前行符合 RAW 消息格式，则不切分（属于同一个块）
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        """判断当前块是否是原始消息（RAW）块。"""
        # 获取块中第一个非空行
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        # 检查时间戳后面的内容是否以 "[RAW]" 开头
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        """获取后备时间戳：优先使用旧文件的修改时间，否则使用当前时间。"""
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        """生成下一个可用的旧文件备份路径（如 HISTORY.md.bak, HISTORY.md.bak.2 等）。"""
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- MEMORY.md (long-term facts 长期事实) -----------------------------------------

    def read_memory(self) -> str:
        """读取长期记忆文件。"""
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        """写入长期记忆文件。"""
        self.memory_file.write_text(content, encoding="utf-8")

    # -- SOUL.md (AI 人设) -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md (用户信息) -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # -- context injection (used by context.py 上下文注入) ------------------------------

    def get_memory_context(self) -> str:
        """获取用于注入到 Prompt 中的长期记忆上下文。"""
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- history.jsonl — append-only, JSONL format (仅追加的 JSONL 历史记录) ---------------------------

    def append_history(
        self,
        entry: str,
        *, # * 后面的参数必须作为关键字参数传递
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> int:
        """Append *entry* to history.jsonl and return its auto-incrementing cursor.
        ... (省略部分原有英文注释) ...
        """
        # 【原有注释翻译】将 *entry* 追加到 history.jsonl 并返回其自动递增的游标。
        # 条目在持久化前会通过 `strip_think` 过滤掉模板级别的泄漏（如未闭合的 `<think` 前缀）。
        # 如果清理后的内容为空但原始条目不为空，则记录空字符串而不是回退到原始泄漏内容。
        # 应用防御性的字符数上限（max_chars）作为最后的安全网。
        
        # 如果未指定 max_chars，则使用硬编码的紧急上限（稍后在文件底部定义）
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        ts = datetime.now().strftime("%Y-%m-%d %H:%M") # 生成当前时间戳
        raw = entry.rstrip() # 去除右侧空白
        
        # 检查是否超长
        if len(raw) > limit:
            if not self._oversize_logged:
                self._oversize_logged = True # 标记已记录，防止重复打印
                logger.warning(
                    "history entry exceeds {} chars ({}); truncating. "
                    "Usually means a caller forgot its own cap; "
                    "further occurrences suppressed.",
                    limit, len(raw),
                )
            raw = truncate_text(raw, limit) # 截断文本
            
        content = strip_think(raw) # 剥离思考过程标签
        
        # Cursor allocation and the append must be atomic: concurrent writers
        # could otherwise read the same current cursor and emit duplicates.
        # 【原有注释翻译】游标分配和追加必须是原子的：否则并发写入者可能会读取相同的当前游标并发出重复项。
        with self._append_lock: # 获取线程锁
            cursor = self._next_cursor() # 获取下一个可用的游标 ID
            if raw and not content:
                logger.debug(
                    "history entry {} stripped to empty (likely template leak); "
                    "persisting empty content to avoid re-polluting context",
                    cursor,
                )
            # 构建 JSONL 记录字典
            record = {"cursor": cursor, "timestamp": ts, "content": content}
            if session_key:
                record["session_key"] = session_key # 如果提供了 session_key，则加入记录
                
            # 以追加模式 ("a") 打开文件并写入 JSON 字符串
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 更新游标文件，记录最新分配的 cursor
            self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        """Non-negative int cursors only; reject bool (``isinstance(True, int)`` is True)."""
        # 【原有注释翻译】仅允许非负整数游标；拒绝布尔值（因为在 Python 中 ``isinstance(True, int)`` 为 True）。
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        """Yield ``(entry, cursor)`` for well-formed entries; warn once on corruption."""
        # 【原有注释翻译】为格式良好的条目生成 ``(entry, cursor)``；在遇到损坏时仅警告一次。
        # 这是一个生成器函数（使用 yield），可以惰性返回数据，节省内存。
        poisoned: Any = None # 记录导致中毒（损坏）的游标值
        malformed_cursor: int | None = None # 记录格式错误的游标
        
        for entry in self._read_entries():
            raw = entry.get("cursor")
            if raw is None:
                continue
            cursor = self._valid_cursor(raw)
            if cursor is None:
                poisoned = raw
                continue
            if not self._valid_history_payload(entry):
                malformed_cursor = cursor
                continue
            # 产出合法的条目和游标
            yield entry, cursor
            
        # 如果发现了损坏的游标且尚未记录过警告，则打印警告
        if poisoned is not None and not self._corruption_logged:
            self._corruption_logged = True
            logger.warning(
                "history.jsonl contains an invalid cursor ({!r}); dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                poisoned,
            )
        if malformed_cursor is not None and not self._malformed_entry_logged:
            self._malformed_entry_logged = True
            logger.warning(
                "history.jsonl contains a malformed entry at cursor {}; dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                malformed_cursor,
            )

    @staticmethod
    def _valid_history_payload(entry: dict[str, Any]) -> bool:
        """校验历史条目 payload 的必需字段类型是否正确。"""
        if not isinstance(entry.get("timestamp"), str):
            return False
        if not isinstance(entry.get("content"), str):
            return False
        session_key = entry.get("session_key")
        return session_key is None or isinstance(session_key, str)

    def _read_cursor_counter(self) -> int | None:
        """Return the persisted cursor counter when it is usable."""
        # 【原有注释翻译】在可用时返回持久化的游标计数器。
        if not self._cursor_file.exists():
            return None
        # 使用 suppress 忽略 ValueError（转换失败）和 OSError（读取失败）
        with suppress(ValueError, OSError):
            cursor = int(self._cursor_file.read_text(encoding="utf-8").strip())
            if cursor >= 0:
                return cursor
        return None

    def _next_cursor(self) -> int:
        """Read the current cursor counter and return the next value."""
        # 【原有注释翻译】读取当前游标计数器并返回下一个值。
        cursor_counter = self._read_cursor_counter()
        last = self._read_last_entry() or {}
        last_cursor = self._valid_cursor(last.get("cursor"))
        
        if cursor_counter is not None:
            if last_cursor is not None:
                return max(cursor_counter, last_cursor) + 1
            max_history_cursor = max((c for _, c in self._iter_valid_entries()), default=0)
            return max(cursor_counter, max_history_cursor) + 1

        # Fast path: trust the tail when intact.  Otherwise scan the whole
        # file and take ``max`` — that stays correct even if the monotonic
        # invariant was broken by external writes.
        # 【原有注释翻译】快速路径：如果尾部完好则信任尾部。否则扫描整个文件并取 ``max`` —— 
        # 即使单调递增的不变量被外部写入破坏，这也能保持正确。
        if last_cursor is not None:
            return last_cursor + 1
        return max((c for _, c in self._iter_valid_entries()), default=0) + 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with a valid cursor > *since_cursor*."""
        # 【原有注释翻译】返回游标大于 *since_cursor* 的有效历史条目（即尚未被处理的新条目）。
        return [e for e, c in self._iter_valid_entries() if c > since_cursor]

    @classmethod # 类方法，第一个参数是 cls（类本身），而不是 self（实例）
    def _is_internal_history_session(cls, session_key: str | None) -> bool:
        """判断给定的 session_key 是否属于系统内部的会话（如 cron, dream, heartbeat）。"""
        if not session_key:
            return False
        return (
            session_key in cls._INTERNAL_HISTORY_SESSION_KEYS
            or session_key.startswith(cls._INTERNAL_HISTORY_SESSION_PREFIXES)
        )

    def read_recent_history_for_prompt(
        self,
        since_cursor: int,
        *,
        session_key: str | None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """Return unprocessed history entries safe to inject into a turn prompt."""
        # 【原有注释翻译】返回可以安全注入到回合提示词中的未处理历史条目。
        entries = self.read_unprocessed_history(since_cursor=since_cursor)
        if session_key is None:
            return entries
        if not unified_session:
            # 如果不是统一会话模式，只返回属于当前 session_key 的条目
            return [e for e in entries if e.get("session_key") == session_key]

        # 海象运算符 (:=) 用于在表达式中赋值并复用变量 entry_session
        return [
            entry
            for entry in entries
            if (entry_session := entry.get("session_key")) == session_key
            or not self._is_internal_history_session(entry_session)
        ]

    def compact_history(self) -> None:
        """Drop oldest processed entries without discarding pending Dream input."""
        # 【原有注释翻译】丢弃最旧的已处理条目，但不丢弃待处理的 Dream 输入。
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
            
        last_dream_cursor = self.get_last_dream_cursor()
        # 寻找第一个未被 Dream 处理的条目的索引
        first_unprocessed = next(
            (
                index
                for index, entry in enumerate(entries)
                if (
                    (cursor := self._valid_cursor(entry.get("cursor"))) is not None
                    and cursor > last_dream_cursor
                )
            ),
            len(entries), # 如果找不到，则默认返回 len(entries)
        )
        # 计算需要保留的起始索引：既要满足最大条目数限制，又不能截断未处理的 Dream 输入
        keep_from = min(len(entries) - self.max_history_entries, first_unprocessed)
        kept = entries[keep_from:]
        
        if len(kept) > self.max_history_entries:
            logger.warning(
                "History compaction retained {} unprocessed entries beyond the configured "
                "limit of {}",
                len(kept),
                self.max_history_entries,
            )
        # 将保留的条目重新写入文件（覆盖原文件）
        self._write_entries(kept)
    # -- JSONL helpers (JSONL 文件操作辅助方法) -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history.jsonl."""
        # 【原有注释翻译】从 history.jsonl 读取所有条目。
        entries: list[dict[str, Any]] = []
        # 使用 suppress 忽略 FileNotFoundError，如果文件不存在则直接返回空列表
        with suppress(FileNotFoundError):
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            # 解析每一行为 JSON 对象
                            parsed: object = json.loads(line)
                        except json.JSONDecodeError:
                            # 如果某行 JSON 格式损坏，则跳过该行，保证整个文件读取的鲁棒性
                            continue
                        if isinstance(parsed, dict):
                            # cast 用于告诉类型检查器 parsed 确实是 dict 类型
                            entries.append(cast(dict[str, Any], parsed))

        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """Read the last entry from the JSONL file efficiently."""
        # 【原有注释翻译】高效地从 JSONL 文件中读取最后一个条目。
        try:
            # 以二进制读模式 ("rb") 打开文件，以便使用底层的 seek 操作
            with open(self.history_file, "rb") as f:
                # f.seek(offset, whence): whence=2 表示 os.SEEK_END，即从文件末尾开始偏移
                f.seek(0, 2) 
                size = f.tell() # tell() 返回当前文件指针位置，此时即为文件总大小
                if size == 0:
                    return None
                # 为了高效，不读取整个文件，只读取文件末尾的 4096 字节（通常足够包含最后一行）
                read_size = min(size, 4096)
                f.seek(size - read_size) # 指针回退到距离末尾 4096 字节的位置
                data = f.read().decode("utf-8")
                # 按换行符分割，并过滤掉空行
                lines = [line for line in data.split("\n") if line.strip()]
                if not lines:
                    return None
                # 取最后一行进行 JSON 解析
                parsed: object = json.loads(lines[-1])
                return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history.jsonl with the given entries (atomic write)."""
        # 【原有注释翻译】用给定的条目覆盖 history.jsonl（原子写入）。
        # 生成一个临时文件路径，后缀加上 .tmp
        tmp_path = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush() # 将 Python 内部的缓冲区刷新到操作系统缓冲区
                # os.fsync 强制将操作系统缓冲区的数据写入物理磁盘，防止断电丢失数据
                os.fsync(f.fileno()) 
            # os.replace 是原子重命名操作，保证文件替换的原子性，不会出现写了一半的情况
            os.replace(tmp_path, self.history_file)

            # fsync the directory so the rename is durable.
            # On Windows, opening a directory with O_RDONLY raises
            # PermissionError — skip the dir sync there (NTFS
            # journals metadata synchronously).
            # 【原有注释翻译】对目录进行 fsync 以使重命名操作持久化。
            # 在 Windows 上，以 O_RDONLY 打开目录会引发 PermissionError —— 在那里跳过目录同步
            # （NTFS 会同步记录元数据日志）。
            with suppress(PermissionError):
                # 以只读模式打开目录的文件描述符
                fd = os.open(str(self.history_file.parent), os.O_RDONLY)
                try:
                    os.fsync(fd) # 同步目录元数据
                finally:
                    os.close(fd) # 确保文件描述符被关闭
        except BaseException:
            # 如果发生任何异常，清理临时文件并向上抛出异常
            tmp_path.unlink(missing_ok=True)
            raise

    # -- dream cursor (Dream 模式游标控制) --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        """获取上一次 Dream 运行处理到的历史游标位置。"""
        if self._dream_cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        """设置 Dream 游标。"""
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    def get_latest_cursor(self) -> int:
        """获取当前历史记录的最新游标值（即最后一条记录的 ID）。"""
        return max(self._next_cursor() - 1, 0)

    @property
    def dream_prompt_file(self) -> Path:
        """获取 Dream 提示词的自定义覆盖文件路径。"""
        return workspace_prompt_file(self.workspace, "dream")

    def has_dream_prompt_override(self) -> bool:
        """检查是否存在自定义的 Dream 提示词。"""
        return has_workspace_prompt_override(self.dream_prompt_file)

    @staticmethod
    def default_dream_prompt() -> str:
        """加载并渲染默认的 Dream 提示词模板。"""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        return render_template(
            "agent/dream.md",
            strip=True,
            # 将技能创建者的 SKILL.md 路径作为变量注入模板
            skill_creator_path=str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"),
        )

    def _dream_template(self) -> str:
        """获取 Dream 提示词文本（优先使用自定义覆盖，否则使用默认）。"""
        text, original_chars = load_workspace_prompt_override(self.dream_prompt_file)
        if text is not None:
            # 如果自定义提示词超长，且尚未记录过警告，则记录警告并截断
            if (
                original_chars > WORKSPACE_PROMPT_MAX_CHARS
                and not self._dream_prompt_oversize_logged
            ):
                self._dream_prompt_oversize_logged = True
                logger.warning(
                    "workspace Dream prompt exceeds {} chars ({}); truncating. "
                    "Further occurrences suppressed.",
                    WORKSPACE_PROMPT_MAX_CHARS, original_chars,
                )
            return text
        return self.default_dream_prompt()

    def build_dream_prompt(self, *, max_entries: int = 20) -> tuple[str, int] | None:
        """Build the Dream prompt with unprocessed history context.
        Returns ``(prompt, last_cursor)`` or ``None`` if nothing to process.
        ...
        """
        # 【原有注释翻译】构建带有未处理历史上下文的 Dream 提示词。
        # 返回 ``(prompt, last_cursor)``，如果没有需要处理的内容则返回 ``None``。
        # 嵌入持久化记忆文件的当前内容，以便模型编辑真实文件而不是过时的心理模型。
        
        last_cursor = self.get_last_dream_cursor()
        # 读取自上次 Dream 运行以来的新历史条目
        entries = self.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return None

        # 限制每次处理的最大条目数
        batch = entries[:max_entries]
        # 将历史条目格式化为文本，并限制每条内容的长度
        history_text = "\n".join(
            f"[{e['timestamp']}] {truncate_text(e['content'], 1000)}"
            for e in batch
        )
        template = self._dream_template()
        # 获取当前记忆文件的内容块
        files_section = self._render_current_memory_files()
        # 拼接最终的 Prompt
        prompt = (
            f"{template}\n\n{files_section}\n\n"
            f"## Conversation History\n{history_text}"
        )
        return (prompt, batch[-1]["cursor"])

    def _render_current_memory_files(self) -> str:
        """Render the durable memory files' current contents for the Dream prompt.
        Missing files render as ``(empty)``; oversized files are capped. ...
        """
        # 【原有注释翻译】渲染持久化记忆文件的当前内容以供 Dream 提示词使用。
        # 缺失的文件渲染为 ``(empty)``；超大的文件会被截断。
        files = [
            ("SOUL.md", self.soul_file),
            ("USER.md", self.user_file),
            ("memory/MEMORY.md", self.memory_file),
        ]
        blocks: list[str] = []
        for label, path in files:
            try:
                content = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError:
                content = ""
            # 如果文件内容超过嵌入上限，则截断并添加提示
            if len(content) > self._DREAM_FILE_EMBED_CAP:
                content = truncate_text(content, self._DREAM_FILE_EMBED_CAP) + "\n...[truncated]"
            # 如果内容为空，显示 (empty)，否则显示内容
            blocks.append(f"### {label}\n{content}" if content.strip() else f"### {label}\n(empty)")
        return "## Current Memory Files\n" + "\n\n".join(blocks)

    def dream_content_diff(self) -> str:
        """Structured summary of uncommitted changes to the durable memory files.
        Returns "" when git is unavailable or no content file changed. ...
        """
        # 【原有注释翻译】持久化记忆文件未提交更改的结构化摘要。
        # 当 git 不可用或没有内容文件更改时返回 ""。
        if not self._git.is_initialized():
            return ""
        # 调用 GitStore 总结工作树的变更
        return self._git.summarize_working_tree(list(self._DREAM_CONTENT_PATHS))

    def build_dream_tools(self) -> ToolRegistry:
        """Build the restricted tool registry used by Dream runs."""
        # 【原有注释翻译】构建 Dream 运行使用的受限工具注册表。
        # 导入必要的工具类
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.apply_patch import ApplyPatchTool
        from nanobot.agent.tools.file_state import FileStates
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
        from nanobot.agent.tools.registry import ToolRegistry

        tools = ToolRegistry()
        file_states = FileStates() # 用于跟踪文件状态（如是否被修改）
        workspace = self.workspace
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True) # 确保 skills 目录存在

        # 允许读取内置技能目录
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        # 定义 Dream 模式下允许编辑的核心记忆文件
        editable_files = [self.memory_file, self.soul_file, self.user_file]

        # 注册受限的读文件工具
        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_read_allowed_dirs=extra_read,
            file_states=file_states,
        ))
        # 注册受限的编辑文件工具（仅允许编辑 skills 目录和核心记忆文件）
        tools.register(EditFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        # 注册 ApplyPatch 工具（用于应用 diff 补丁）
        tools.register(ApplyPatchTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        # 注册写文件工具
        tools.register(WriteFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        return tools

    @staticmethod
    def dream_run_completed(
        resp: object | None,
        *,
        had_tool_errors: bool = False,
    ) -> bool:
        """Return True only when a Dream turn completed without tool failures."""
        # 【原有注释翻译】仅当 Dream 回合完成且没有工具失败时返回 True。
        metadata = getattr(resp, "metadata", None) # 安全地获取 resp 的 metadata 属性
        if had_tool_errors or not isinstance(metadata, dict):
            return False
        # 检查停止原因是否为正常完成
        return cast(dict[str, Any], metadata).get("_stop_reason") == "completed"

    # -- message formatting utility (消息格式化工具) ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        """将消息列表格式化为带时间戳、角色和工具使用情况的纯文本。"""
        lines: list[str] = []
        for message in messages:
            # 处理可能包含媒体占位符的内容
            content = content_with_media_breadcrumbs(
                message.get("role"),
                message.get("content", ""),
                message.get("media"),
            )
            if not content:
                continue
            tools_used = message.get("tools_used")
            # 如果使用了工具，则格式化工具列表
            tools = (
                f" [tools: {', '.join(cast(list[str], tools_used))}]"
                if tools_used
                else ""
            )
            raw_timestamp = message.get("timestamp")
            timestamp = str(raw_timestamp) if raw_timestamp is not None else "?"
            role = str(message.get("role") or "unknown")
            # 拼接单行日志格式：[时间] 角色 [工具]: 内容
            lines.append(f"[{timestamp[:16]}] {role.upper()}{tools}: {content}")
        return "\n".join(lines)

    def raw_archive(
        self,
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> None:
        """Fallback: dump raw messages to history.jsonl without LLM summarization."""
        # 【原有注释翻译】后备方案：在不进行 LLM 总结的情况下将原始消息转储到 history.jsonl。
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        # 格式化消息并截断，防止单条记录过大
        formatted = truncate_text(
            self._format_messages(public_history_messages(messages)),
            limit,
        )
        # 以 "[RAW]" 前缀追加到历史记录中
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{formatted}",
            session_key=session_key,
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )

    # ------------------------------------------------------------------
    # Dream helpers (Dream 辅助方法)
    # ------------------------------------------------------------------

    @staticmethod
    def dream_session_key() -> str:
        """Return a unique session key for a Dream run, e.g. ``dream:20260528-100000``."""
        # 【原有注释翻译】返回 Dream 运行的唯一会话键，例如 ``dream:20260528-100000``。
        return f"dream:{datetime.now():%Y%m%d-%H%M%S}"

    @staticmethod
    def build_dream_commit_message(prefix: str, diff_body: str) -> str:
        """Build a Dream commit message grounded in the real working-tree diff. ..."""
        # 【原有注释翻译】构建基于真实工作树 diff 的 Dream 提交信息。
        # *diff_body* 是实际文件更改的结构化、机器派生的摘要。故意排除 LLM 叙述，
        # 以便审计记录反映文件系统的真实情况，而不是模型的自我报告。
        diff_body = (diff_body or "").strip()
        if not diff_body:
            return prefix
        return f"{prefix}\n\n{diff_body}"

    @staticmethod
    def prune_dream_sessions(sessions_dir: Path, *, keep: int = 10) -> None:
        """Remove the oldest Dream session files, keeping only the N most recent. ..."""
        # 【原有注释翻译】删除最旧的 Dream 会话文件，仅保留最近的 N 个。
        dream_files: list[Path] = []
        # 遍历 sessions 目录下的所有 .jsonl 文件
        for path in sessions_dir.glob("*.jsonl"):
            # 解码存储键，检查是否为 dream 会话
            decoded_key = SessionManager.decode_storage_key(path.stem)
            if decoded_key is not None and decoded_key.startswith("dream:"):
                dream_files.append(path)
        # 按修改时间排序
        dream_files.sort(key=lambda p: p.stat().st_mtime)
        if len(dream_files) <= keep:
            return

        # 找出需要删除的旧文件
        to_remove = dream_files[: len(dream_files) - keep]
        for path in to_remove:
            try:
                path.unlink() # 删除文件
                logger.debug("Pruned old dream session: {}", path.stem)
            except OSError:
                logger.warning("Failed to prune dream session {}", path)


# ---------------------------------------------------------------------------
# Consolidator — lightweight token-budget triggered consolidation
# (总结器 - 基于轻量级 Token 预算触发的上下文压缩)
# ---------------------------------------------------------------------------

# 定义各种字符数上限常量
_RAW_ARCHIVE_MAX_CHARS = 16_000       # fallback dump (LLM failed) / 后备转储（LLM 失败时）
_ARCHIVE_SUMMARY_MAX_CHARS = 8_000    # LLM-produced consolidation summary / LLM 生成的总结
_HISTORY_ENTRY_HARD_CAP = 64_000      # emergency cap in append_history / append_history 中的紧急上限


class Consolidator:
    """Summarize compacted messages into history.jsonl."""
    # 【原有注释翻译】将压缩的消息总结并写入 history.jsonl。

    _MAX_CONSOLIDATION_ROUNDS = 5 # 单次触发的最大压缩轮数，防止死循环
    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift / 为分词器估算偏差预留的额外空间

    def __init__(
        self,
        store: MemoryStore,
        sessions: SessionManager,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        consolidation_ratio: float = 0.5, # 目标压缩比例（如 0.5 表示将 prompt 压缩到上下文窗口的一半）
        unified_session: bool = False,
    ):
        self.store = store
        self.sessions = sessions
        self.consolidation_ratio = consolidation_ratio
        self.unified_session = unified_session
        self._build_messages = build_messages # 构建 Prompt 消息列表的回调函数
        self._get_tool_definitions = get_tool_definitions # 获取工具定义的回调函数
        
        # Python 特殊用法：WeakValueDictionary (弱引用字典)
        # 这里的值是 asyncio.Lock。如果使用普通字典，字典会持有 Lock 的强引用，
        # 导致即使 Session 被销毁，Lock 也不会被垃圾回收，造成内存泄漏。
        # 使用弱引用字典，当外部不再引用该 Lock 时，它会自动从字典中移除。
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        # 【原有注释翻译】返回单个会话的共享压缩锁。
        # setdefault：如果键存在则返回其值，如果不存在则设置为默认值并返回。
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        # 【原有注释翻译】选择一个用户回合边界，以移除足够多的旧提示词 Token。
        start = session.last_consolidated # 从上次压缩的位置开始
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        # 遍历未压缩的消息
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            # 寻找 user 角色的消息作为安全的截断边界（保证对话的连贯性）
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary # 如果移除的 token 数达标，则返回该边界
            removed_tokens += estimate_message_tokens(message)

        return last_boundary # 返回能找到的最后一个边界

    @staticmethod
    def _full_replay_history(
        session: Session,
    ) -> list[dict[str, Any]]:
        """Return all messages that can reach the next model prompt."""
        # 【原有注释翻译】返回所有可以到达下一个模型提示词的消息。
        if not session.messages:
            return []
        return session.get_history(max_messages=len(session.messages))

    @staticmethod
    def _replay_overflow_boundary(
        session: Session,
        replay_max_messages: int | None,
    ) -> int | None:
        """计算 Replay Window（重放窗口）溢出时的截断边界。"""
        if not replay_max_messages or replay_max_messages <= 0:
            return None
        # 获取自上次压缩以来的消息尾部
        tail = list(enumerate(session.messages[session.last_consolidated:], session.last_consolidated))
        if len(tail) <= replay_max_messages:
            return None # 如果没有溢出，返回 None

        tail_messages = [message for _idx, message in tail]
        # 寻找最近消息的起始索引，确保不超过 max_messages 且以 user 开始
        start_idx = recent_message_start_index(
            tail_messages,
            replay_max_messages,
            extend_to_user=True,
        )
        sliced = tail[start_idx:]
        # 进一步调整，确保包含频道交付标记等上下文
        for i, (_idx, message) in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1][1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        # 查找合法的消息起始点（如系统提示或 user）
        legal_start = find_legal_message_start([message for _idx, message in sliced])
        if legal_start:
            sliced = sliced[legal_start:]
        if not sliced:
            return len(session.messages)

        first_visible_idx = sliced[0][0]
        if first_visible_idx <= session.last_consolidated:
            return None
        return first_visible_idx

    async def _consolidate_replay_overflow(
        self,
        session: Session,
        replay_max_messages: int | None,
        *,
        runtime: LLMRuntime,
    ) -> str | None:
        """Archive messages that would be hidden by the replay message window."""
        # 【原有注释翻译】归档那些会被重放消息窗口隐藏的消息。
        end_idx = self._replay_overflow_boundary(session, replay_max_messages)
        if end_idx is None:
            return None
        chunk = session.messages[session.last_consolidated:end_idx]
        if not chunk:
            return None
        logger.info(
            "Replay-window consolidation for {}: chunk={} msgs, replay_max={}",
            session.key,
            len(chunk),
            replay_max_messages,
        )
        # 调用 archive 方法让 LLM 总结这部分消息
        summary = await self.archive(
            chunk,
            runtime=runtime,
            session_key=session.key,
        )
        # 更新 session 的压缩游标
        session.last_consolidated = end_idx
        session.provider_state = None # 清除 provider 状态，因为上下文已改变
        self.sessions.save(session)
        return summary

    def _persist_last_summary(self, session: Session, summary: str | None) -> None:
        """将最后的总结文本持久化到 session 的 metadata 中。"""
        if summary and summary != "(nothing)":
            session.metadata["_last_summary"] = {
                "text": summary,
                "last_active": session.updated_at.isoformat(),
            }
            self.sessions.save(session)

    def estimate_session_prompt_tokens(
        self,
        session: Session,
        *,
        runtime: LLMRuntime,
    ) -> tuple[int, str]:
        """Estimate prompt size from the full replayable session history."""
        # 【原有注释翻译】从完整的可重放会话历史中估算提示词大小。
        history = self._full_replay_history(session)
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        
        # 将已归档的总结也包含在估算中，以便预算能考虑到它
        meta = session.metadata.get("_last_summary")
        summary = (
            cast(dict[str, Any], meta).get("text")
            if isinstance(meta, dict)
            else meta
            if isinstance(meta, str)
            else None
        )
        # 构建用于探测 token 数量的消息列表，使用 "[token-probe]" 作为假的用户输入
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            session_summary=summary,
            session_key=session.key,
            unified_session=self.unified_session,
        )
        # 调用底层 API 估算 token 数量
        return estimate_prompt_tokens_chain(
            runtime.provider,
            runtime.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    def _input_token_budget(self, runtime: LLMRuntime) -> int:
        """Available input token budget for consolidation LLM."""
        # 【原有注释翻译】用于压缩的 LLM 的可用输入 Token 预算。
        # 总窗口 - 最大生成 token - 安全缓冲
        return (
            runtime.context_window_tokens
            - runtime.generation.max_tokens
            - self._SAFETY_BUFFER
        )

    def _truncate_to_token_budget(self, text: str, *, runtime: LLMRuntime) -> str:
        """Truncate text so it fits within the consolidation LLM's token budget."""
        # 【原有注释翻译】截断文本，使其适合压缩 LLM 的 Token 预算。
        budget = self._input_token_budget(runtime)
        if budget <= 0:
            return truncate_text(text, _RAW_ARCHIVE_MAX_CHARS)
        return truncate_text_to_tokens(text, budget)

    async def archive(
        self,
        messages: list[dict[str, Any]],
        *,
        runtime: LLMRuntime,
        session_key: str | None = None,
        summary_messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Summarize messages and append the result to history.jsonl.
        ``summary_messages`` adds context but is excluded from raw fallback.
        """
        # 【原有注释翻译】总结消息并将结果追加到 history.jsonl。
        if not messages:
            return None
        # 获取需要总结的消息（过滤掉内部消息）
        messages_to_summarize = public_history_messages(
            summary_messages if summary_messages is not None else messages
        )
        # 格式化并截断以适应 Token 预算
        formatted = MemoryStore._format_messages(messages_to_summarize)
        formatted = self._truncate_to_token_budget(formatted, runtime=runtime)
        
        # 渲染压缩专用的系统提示词
        system_prompt = render_template(
            "agent/consolidator_archive.md",
            strip=True,
        )
        try:
            # 调用 LLM 进行总结
            response = await runtime.provider.chat_with_retry(
                model=runtime.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted},
                ],
                tools=None, # 压缩时不需要工具
                tool_choice=None,
                temperature=runtime.generation.temperature,
                max_tokens=runtime.generation.max_tokens,
                reasoning_effort=runtime.generation.reasoning_effort,
            )
        except Exception:
            logger.warning("Consolidation provider call failed, raw-dumping to history")
            # LLM 调用失败，降级为原始消息转储
            self.store.raw_archive(messages, session_key=session_key)
            return None
            
        if response.finish_reason == "error":
            logger.warning("Consolidation provider returned an error, raw-dumping to history")
            self.store.raw_archive(messages, session_key=session_key)
            return None
            
        summary = response.content or "[no summary]"
        # 将 LLM 生成的总结追加到历史记录中
        self.store.append_history(
            summary,
            max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
            session_key=session_key,
        )
        return summary

    async def maybe_consolidate_by_tokens(
        self,
        session: Session,
        *,
        runtime: LLMRuntime,
        replay_max_messages: int | None = None,
    ) -> None:
        """Loop: archive old messages until prompt fits within safe budget. ..."""
        # 【原有注释翻译】循环：归档旧消息，直到提示词适合安全预算。
        if runtime.context_window_tokens <= 0:
            return

        # 获取当前 session 的异步锁，防止并发压缩同一个 session
        lock = self.get_lock(session.key)
        async with lock: # 异步上下文管理器，获取锁
            # 刷新 session 引用：AutoCompact 可能已经替换了它
            fresh = self.sessions.get_or_create(session.key)
            if fresh is not session:
                session = fresh
            if not session.messages:
                return

            budget = self._input_token_budget(runtime)
            target = int(budget * self.consolidation_ratio) # 计算目标 Token 数
            
            # 首先处理 Replay Window 溢出的部分
            last_summary = await self._consolidate_replay_overflow(
                session,
                replay_max_messages,
                runtime=runtime,
            )
            # 估算当前的 Prompt Token 数
            estimated, source = self.estimate_session_prompt_tokens(
                session,
                runtime=runtime,
            )
            if estimated <= 0:
                self._persist_last_summary(session, last_summary)
                return
            if estimated < budget:
                # 如果当前 Token 数在预算内，则无需压缩
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    runtime.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                self._persist_last_summary(session, last_summary)
                return

            # 开始多轮压缩循环
            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    break # 达到目标，退出循环

                # 寻找安全的截断边界
                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    break

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    runtime.context_window_tokens,
                    source,
                    len(chunk),
                )
                # 调用 LLM 归档这部分消息
                summary = await self.archive(
                    chunk,
                    runtime=runtime,
                    session_key=session.key,
                )
                
                # 无论成功与否，都推进游标。如果失败，archive() 已经将其作为 [RAW] 归档，
                # 下次再压缩同一段会导致重复的 [RAW] 条目。
                if summary:
                    last_summary = summary
                session.last_consolidated = end_idx
                session.provider_state = None
                self.sessions.save(session)
                
                if not summary:
                    # LLM 降级 - 停止本次调用中的重试；下一次调用可以重试新的块。
                    break

                # 重新估算 Token 数
                estimated, source = self.estimate_session_prompt_tokens(
                    session,
                    runtime=runtime,
                )
                if estimated <= 0:
                    break

            # 将最后的总结持久化到 session 元数据中
            self._persist_last_summary(session, last_summary)

    async def compact_idle_session(
        self,
        session_key: str,
        *,
        runtime: LLMRuntime,
        max_suffix: int = MIN_COMPACTED_REPLAY_MESSAGES,
    ) -> str | None:
        """Archive the full idle tail while keeping recent messages replayable. ..."""
        # 【原有注释翻译】归档完整的空闲尾部，同时保持最近的消息可重放。
        if max_suffix != MIN_COMPACTED_REPLAY_MESSAGES:
            logger.debug(
                "Idle-session compact for {} uses the fixed replay window ({}, requested {})",
                session_key,
                MIN_COMPACTED_REPLAY_MESSAGES,
                max_suffix,
            )
        lock = self.get_lock(session_key)
        async with lock:
            self.sessions.invalidate(session_key) # 使缓存失效，强制重新加载
            session = self.sessions.get_or_create(session_key)

            archive_start = session.last_consolidated
            # 获取所有需要归档的消息
            messages_to_archive = list(session.messages[archive_start:])
            if not messages_to_archive:
                return ""

            last_active = session.updated_at
            archive_end = archive_start + len(messages_to_archive)
            
            # 调用 LLM 进行总结
            summary = await self.archive(
                messages_to_archive,
                runtime=runtime,
                session_key=session_key,
            )

            if summary and summary != "(nothing)":
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                }

            # 推进压缩游标。注意：在 provider 调用期间可能会有新消息追加，
            # 所以只推进到捕获的批次末尾，新消息下次依然有资格被处理。
            session.last_consolidated = archive_end
            session.provider_state = None
            self.sessions.save(session)

            # 获取压缩后仍然可见（保留在上下文中）的消息
            visible = session.get_history(
                max_messages=MIN_COMPACTED_REPLAY_MESSAGES,
                extend_to_user=True,
            )

            logger.info(
                "Idle-session compact for {}: archived={}, visible={}, retained={}, summary={}",
                session_key,
                len(messages_to_archive),
                len(visible),
                len(session.messages),
                bool(summary),
            )

            return summary
