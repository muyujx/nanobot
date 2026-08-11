# ======================================================================
# 模块文档字符串（module docstring）说明：
# 原始英文注释：Context builder for assembling agent prompts.
# 中文翻译：上下文构建器，用于组装 Agent 的提示词（prompt）。
#
# 这个文件的核心职责：
# 1. 组装 system prompt（系统提示词）。
# 2. 组装发送给 LLM 的完整 messages 列表。
# 3. 处理 workspace、memory、skills、runtime context、图片附件等上下文。
# ======================================================================
"""Context builder for assembling agent prompts."""

# 导入 base64 模块。
# 作用：把二进制数据编码成 Base64 字符串。
# 这里主要用于把图片文件内容编码成 data URL，方便传给支持多模态输入的 LLM。
import base64

# 导入 mimetypes 模块。
# 作用：根据文件扩展名猜测 MIME 类型，例如 image/png、image/jpeg。
# MIME 类型用于告诉模型当前图片是什么格式。
import mimetypes

# 导入 platform 模块。
# 作用：获取当前操作系统、CPU 架构、Python 版本等运行时信息。
# 这些信息可能会写进 Agent 的 identity / system prompt。
import platform

# 从 pathlib 导入 Path 类。
# Path 是 Python 中面向对象的文件路径处理工具。
# 例如：Path("a") / "b" 会得到路径 "a/b"。
from pathlib import Path

# 从 typing 导入类型注解相关工具：
# Any：表示任意类型。
# Mapping：映射类型，只读字典风格对象，例如 dict。
# Sequence：序列类型，例如 list、tuple。
# cast：用于类型检查器，运行时不做任何转换，只告诉静态类型检查器“把它当成某个类型”。
from typing import Any, Mapping, Sequence, cast

# 从 nanobot.agent.memory 导入 MemoryStore。
# MemoryStore 很可能负责读取和存储 Agent 的记忆数据，
# 包括长期记忆、最近对话历史、dream cursor 等。
from nanobot.agent.memory import MemoryStore

# 从 nanobot.agent.skills 导入 SkillsLoader。
# SkillsLoader 很可能负责加载 Agent 的技能（skills）。
# 技能可以理解为“可插入 system prompt 的能力说明或指令集合”。
from nanobot.agent.skills import SkillsLoader

# 导入图片生成工具模块，并起别名 image_generation_tools。
# 该模块可能包含图片生成相关的 runtime control 处理逻辑。
from nanobot.agent.tools import image_generation as image_generation_tools

# 导入 MCP 工具模块，并起别名 mcp_tools。
# MCP 通常指 Model Context Protocol 或类似外部工具协议。
# 该模块可能负责连接、关闭 MCP server，以及处理 MCP 相关会话附加信息。
from nanobot.agent.tools import mcp as mcp_tools

# 导入 sessions 工具模块，并起别名 session_tools。
# 该模块可能负责会话相关的附加能力或持久化参数。
from nanobot.agent.tools import sessions as session_tools

# 导入 ToolRegistry。
# ToolRegistry 是工具注册表，通常用于保存当前可用工具的集合。
from nanobot.agent.tools.registry import ToolRegistry

# 导入 CLI 应用工具函数模块，并起别名 cli_app_utils。
# 这里可能包含 CLI 场景下会话附加信息的提取逻辑。
from nanobot.apps.cli import utils as cli_app_utils

# 从 nanobot.bus.events 导入事件相关常量和类型：
# INBOUND_META_RUNTIME_CONTROL：入站消息 metadata 中运行时控制字段名。
# RUNTIME_CONTROL_SESSION_DISCARD：表示“丢弃会话”的运行时控制值。
# InboundMessage：入站消息对象类型，通常包含 session_key、metadata 等。
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_SESSION_DISCARD,
    InboundMessage,
)

# 从 nanobot.runtime_context 导入运行时上下文相关常量和函数：
# RUNTIME_CONTEXT_END：运行时上下文结束标记。
# RUNTIME_CONTEXT_MESSAGE_META：消息元数据中保存运行时上下文的键名。
# RUNTIME_CONTEXT_TAG：运行时上下文标签。
# RuntimeContextBlock：运行时上下文块类型。
# append_runtime_context：将运行时上下文块附加到用户消息内容中的函数。
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_END,
    RUNTIME_CONTEXT_MESSAGE_META,
    RUNTIME_CONTEXT_TAG,
    RuntimeContextBlock,
    append_runtime_context,
)

# 从 nanobot.utils.helpers 导入辅助函数：
# detect_image_mime：根据图片二进制内容检测 MIME 类型。
# load_bundled_template：加载内置模板文件。
# truncate_text_to_tokens：按 token 数截断文本，避免上下文过长。
from nanobot.utils.helpers import (
    detect_image_mime,
    load_bundled_template,
    truncate_text_to_tokens,
)

# 从 nanobot.utils.prompt_templates 导入 render_template。
# render_template 很可能用于渲染 Markdown / Jinja 风格模板。
# 例如 render_template("agent/identity.md", workspace_path=...)。
from nanobot.utils.prompt_templates import render_template


# ======================================================================
# 函数：session_extra
# 作用：从消息 metadata 中提取“本轮会话附加能力”需要持久化保存的 kwargs。
# ======================================================================
def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted kwargs for turn-attached capabilities."""
    # 原始 docstring 中文翻译：
    # 返回用于“跟随本轮对话的能力”的持久化关键字参数。

    # metadata 参数说明：
    # 类型是 Mapping[str, Any] | None。
    # 这表示它可能是一个字典风格对象，也可能是 None。
    # Mapping 表示“只读映射”，不要求一定是 dict，只要支持键值访问即可。

    # 返回值说明：
    # 返回一个 dict[str, Any]，也就是字符串键、任意值的字典。
    # 这个字典会被后续流程保存或合并到会话状态中。

    # 这里返回三个 session_extra 结果的字典合并：
    # 1. cli_app_utils.session_extra(metadata)
    #    从 CLI 应用工具中提取会话附加信息。
    # 2. mcp_tools.session_extra(metadata)
    #    从 MCP 工具中提取会话附加信息。
    # 3. session_tools.session_extra(metadata)
    #    从 session 工具中提取会话附加信息。
    #
    # Python 特殊用法：
    # dict_a | dict_b 是 Python 3.9+ 的字典合并语法。
    # 它会生成一个新字典。
    # 如果左右字典有相同键，右侧字典的值会覆盖左侧字典的值。
    return (
        cli_app_utils.session_extra(metadata)
        | mcp_tools.session_extra(metadata)
        | session_tools.session_extra(metadata)
    )


# ======================================================================
# 异步函数：connect_mcp
# 作用：连接尚未连接的 MCP servers。
# ======================================================================
async def connect_mcp(state: Any, tools: ToolRegistry) -> None:
    # state 参数说明：
    # 类型是 Any，表示这里不限制具体类型。
    # 它通常代表当前 Agent / runtime 的状态对象。
    #
    # tools 参数说明：
    # 类型是 ToolRegistry，即工具注册表。
    # MCP server 连接后可能会把对应工具注册进去。
    #
    # 返回值说明：
    # -> None 表示这个函数不返回有意义的值。

    # Python 特殊用法：
    # async def 定义异步函数。
    # 异步函数内部可以使用 await 等待另一个协程完成。
    #
    # 这里调用 mcp_tools.connect_missing_servers(state, tools)。
    # 从函数名看，它会连接那些“还没连接”的 MCP servers。
    # await 会等待连接过程结束。
    await mcp_tools.connect_missing_servers(state, tools)


# ======================================================================
# 异步函数：close_mcp
# 作用：关闭已经打开的 MCP servers。
# ======================================================================
async def close_mcp(state: Any) -> None:
    # state 参数说明：
    # 当前运行时状态对象，里面可能保存了 MCP server 连接信息。
    #
    # 返回值说明：
    # -> None 表示不返回值。

    # 调用 mcp_tools.close_mcp_servers(state)，关闭 state 中记录的 MCP servers。
    # await 表示等待关闭操作完成。
    await mcp_tools.close_mcp_servers(state)


# ======================================================================
# 异步函数：handle_runtime_control
# 作用：处理入站消息中的运行时控制指令。
# 返回值：
# - 如果该消息被运行时控制逻辑消费掉了，返回 True。
# - 如果没有匹配任何运行时控制逻辑，返回 False。
# ======================================================================
async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    # state 参数说明：
    # 当前运行时状态对象。
    #
    # msg 参数说明：
    # InboundMessage，入站消息对象。
    # 通常包含：
    # - session_key：会话标识。
    # - metadata：消息元数据。
    # - 可能还有文本、媒体等内容。
    #
    # tools 参数说明：
    # 当前工具注册表。
    #
    # 返回值说明：
    # bool，表示这条消息是否是运行时控制消息并已经被处理。

    # 判断 msg.metadata 中是否存在运行时控制字段，
    # 并且该字段的值是否等于 RUNTIME_CONTROL_SESSION_DISCARD。
    #
    # msg.metadata.get(INBOUND_META_RUNTIME_CONTROL)：
    # 从 metadata 字典中取值。
    # 如果键不存在，不会抛 KeyError，而是返回 None。
    #
    # 如果等于 RUNTIME_CONTROL_SESSION_DISCARD，
    # 表示当前消息要求丢弃当前会话。
    if msg.metadata.get(INBOUND_META_RUNTIME_CONTROL) == RUNTIME_CONTROL_SESSION_DISCARD:
        # 调用 state.discard_session(msg.session_key)，丢弃指定会话。
        # await 表示这是一个异步操作，需要等待完成。
        await state.discard_session(msg.session_key)

        # 因为消息已经被处理，所以返回 True。
        return True

    # 如果前面的“丢弃会话”控制没有命中，
    # 则继续尝试交给其他 handler 处理。
    #
    # 这里创建一个元组，里面放了两个异步处理函数：
    # 1. image_generation_tools.handle_runtime_control
    # 2. mcp_tools.handle_runtime_control
    #
    # Python 特殊用法：
    # 函数也是对象，可以放进 tuple / list 中，然后循环调用。
    for handler in (
        image_generation_tools.handle_runtime_control,
        mcp_tools.handle_runtime_control,
    ):
        # 逐个调用 handler，并等待其结果。
        # handler(state, msg, tools) 返回一个协程，await 等待协程执行完成。
        #
        # 如果某个 handler 返回 True，表示它已经处理了该运行时控制消息。
        if await handler(state, msg, tools):
            # 一旦有一个处理器处理成功，就立即返回 True。
            return True

    # 所有运行时控制处理器都没有处理该消息，返回 False。
    return False


# ======================================================================
# 类：ContextBuilder
# 作用：构建 Agent 的上下文。
# 上下文包括：
# - system prompt
# - 历史消息
# - 当前用户消息
# - memory
# - skills
# - runtime context
# - 图片等媒体内容
# ======================================================================
class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""
    # 原始 docstring 中文翻译：
    # 为 Agent 构建上下文（system prompt + messages）。

    # 类常量：BOOTSTRAP_FILES
    # 这是一个列表，列出默认引导文件名。
    # 这些文件可能用于提供 Agent 的初始规则、人格、用户信息等。
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]

    # 类常量：_SKIPPABLE_DEFAULTS
    # 这是一个集合（set），包含“如果是默认模板内容就可以跳过”的文件名。
    # 前缀下划线通常表示这是类内部使用的常量。
    #
    # Python 特殊用法：
    # {"AGENTS.md", "USER.md"} 是 set，不是 dict。
    # set 适合做成员判断，例如 filename in self._SKIPPABLE_DEFAULTS。
    _SKIPPABLE_DEFAULTS = {"AGENTS.md", "USER.md"}

    # 类常量：_RUNTIME_CONTEXT_TAG
    # 直接引用模块级常量 RUNTIME_CONTEXT_TAG。
    # 这可能是运行时上下文的标签名。
    _RUNTIME_CONTEXT_TAG = RUNTIME_CONTEXT_TAG

    # 类常量：_MAX_RECENT_HISTORY
    # 最近历史最多保留多少条。
    # 这里是 50 条。
    _MAX_RECENT_HISTORY = 50

    # 类常量：_MAX_HISTORY_TOKENS
    # 原始英文注释：hard cap on recent history section size (tokens)
    # 中文翻译：最近历史部分大小的硬上限（单位：token）。
    #
    # Python 特殊用法：
    # 8_000 等价于 8000。
    # 下划线只是让数字更易读。
    _MAX_HISTORY_TOKENS = 8_000  # hard cap on recent history section size (tokens)
    # 中文注释补充：
    # 这个值用于限制“最近历史”文本长度，避免 prompt 太长。

    # 类常量：_RUNTIME_CONTEXT_END
    # 引用模块级常量 RUNTIME_CONTEXT_END。
    # 这可能是运行时上下文结束标记。
    _RUNTIME_CONTEXT_END = RUNTIME_CONTEXT_END

    # ==================================================================
    # 构造方法：__init__
    # 作用：初始化 ContextBuilder。
    # ==================================================================
    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None):
        # workspace 参数说明：
        # 类型是 Path，表示 Agent 的工作目录。
        # memory、skills、bootstrap 文件等都可能基于这个目录查找。
        #
        # timezone 参数说明：
        # 类型是 str | None，默认 None。
        # 表示时区字符串，例如 "Asia/Shanghai"。
        # 当前代码里只是保存下来，后续可能由其他模块使用。
        #
        # disabled_skills 参数说明：
        # 类型是 list[str] | None，默认 None。
        # 表示禁用的技能名列表。
        # 如果传入列表，会被转换成 set，方便后续快速判断某个技能是否被禁用。

        # 将传入的 workspace 保存为实例属性 self.workspace。
        # self 表示当前对象实例。
        self.workspace = workspace

        # 将传入的 timezone 保存为实例属性 self.timezone。
        self.timezone = timezone

        # 创建 MemoryStore 实例，并保存为 self.memory。
        # MemoryStore 使用 workspace 作为记忆数据存储位置。
        self.memory = MemoryStore(workspace)

        # 创建 SkillsLoader 实例，并保存为 self.skills。
        #
        # disabled_skills=set(disabled_skills) if disabled_skills else None
        # 这是一个条件表达式：
        # - 如果 disabled_skills 不是 None / 空列表等假值，则转换为 set。
        # - 否则传 None。
        #
        # set 比 list 更适合做“是否包含某项”的判断。
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    # ==================================================================
    # 方法：build_system_prompt
    # 作用：构建 system prompt。
    # system prompt 通常作为 LLM messages 中的第一条 system 消息。
    # ==================================================================
    def build_system_prompt(
        self,
        *,
        active_skill_names: Sequence[str] | None = None,
        channel: str | None = None,
        session_summary: str | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        # 原始 docstring 中文翻译：
        # 根据身份、引导文件、memory 和 skills 构建 system prompt。

        # Python 特殊用法：
        # 方法参数中的单独星号 * 表示：
        # 从 * 后面的所有参数都必须通过关键字传参。
        # 例如必须写 active_skill_names=...，不能只按位置传。
        #
        # 参数说明：
        # active_skill_names：当前明确激活的技能名列表。
        # channel：消息渠道，例如 cli、web、discord 等。
        # session_summary：会话摘要，可能用于归档上下文。
        # workspace：可以临时指定一个 workspace，不传则用 self.workspace。
        # include_memory：是否包含长期 memory。
        # include_memory_recent_history：是否包含最近历史。
        # session_key：当前会话 key。
        # unified_session：是否统一会话模式。

        # root = workspace or self.workspace
        # 如果调用方传入了 workspace，则使用传入值；
        # 否则使用当前对象默认的 self.workspace。
        #
        # Python 特殊用法：
        # a or b 会返回第一个“真值”表达式。
        # 如果 workspace 不是 None，通常就是真值，因此返回 workspace。
        # 如果 workspace 是 None，则返回 self.workspace。
        root = workspace or self.workspace

        # parts 是一个字符串列表，用来收集 system prompt 的各个部分。
        # 最终这些部分会用分隔符 join 起来。
        #
        # 第一部分调用 self._get_identity(...)，获取 Agent 核心身份描述。
        parts = [self._get_identity(channel=channel, workspace=root)]

        # 加载 bootstrap files，例如 AGENTS.md、SOUL.md、USER.md。
        # bootstrap 通常指启动时需要加载的基础配置或基础人格文件。
        bootstrap = self._load_bootstrap_files(root)

        # 如果 bootstrap 内容非空，则追加到 parts。
        if bootstrap:
            parts.append(bootstrap)

        # 渲染并追加 tool_contract.md 模板。
        # 这部分很可能是在告诉模型“你可以如何使用工具 / 工具调用契约”。
        parts.append(render_template("agent/tool_contract.md"))

        # 如果需要包含 memory，则尝试读取长期记忆。
        if include_memory:
            # 调用 self.memory.read_memory() 读取长期记忆内容。
            memory = self.memory.read_memory()

            # 条件：
            # 1. memory 非空。
            # 2. memory 内容不等于默认模板 memory/MEMORY.md。
            #
            # 这样做是为了避免把未自定义的默认模板塞进 prompt。
            if memory and not self._is_template_content(memory, "memory/MEMORY.md"):
                # 使用 f-string 把 memory 包装成 Markdown 标题格式后追加。
                parts.append(f"# Memory\n\n## Long-term Memory\n{memory}")

        # 获取“总是激活”的技能列表。
        # active_skills 是一个列表，后续会被继续扩展。
        active_skills = self.skills.get_always_skills()

        # 将调用方传入的 active_skill_names 中尚未包含的技能名加入 active_skills。
        #
        # active_skills.extend(...)
        # extend 会把可迭代对象中的每个元素追加到列表末尾。
        #
        # 里面使用的是生成器表达式：
        # name
        # for name in (active_skill_names or ())
        # if name not in active_skills
        #
        # (active_skill_names or ())：
        # 如果 active_skill_names 是 None，则用空元组 () 代替，避免 for None 报错。
        #
        # if name not in active_skills：
        # 避免重复添加同名技能。
        active_skills.extend(
            name
            for name in (active_skill_names or ())
            if name not in active_skills
        )

        # 如果存在激活技能，则加载这些技能的上下文内容。
        if active_skills:
            # 根据技能名列表加载对应技能内容。
            active_content = self.skills.load_skills_for_context(active_skills)

            # 如果技能内容非空，则追加到 system prompt。
            if active_content:
                parts.append(f"# Active Skills\n\n{active_content}")

        # 构建技能摘要。
        # exclude=set(active_skills) 表示排除已经完整加载到上下文中的技能。
        # 这样可以避免重复：已激活技能放详细内容，其他技能只放摘要。
        skills_summary = self.skills.build_skills_summary(exclude=set(active_skills))

        # 如果存在技能摘要，则渲染 skills_section.md 模板并追加。
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        # 如果需要包含最近历史，则从 memory 中读取最近历史条目。
        if include_memory_recent_history:
            # read_recent_history_for_prompt 参数说明：
            # since_cursor=self.memory.get_last_dream_cursor()
            #   从某个“梦境指针/游标”之后开始读取最近历史。
            # session_key=session_key
            #   指定当前会话 key。
            # unified_session=unified_session
            #   指定是否统一会话模式。
            entries = self.memory.read_recent_history_for_prompt(
                since_cursor=self.memory.get_last_dream_cursor(),
                session_key=session_key,
                unified_session=unified_session,
            )

            # 如果读到了历史条目，则把它们格式化后加入 prompt。
            if entries:
                # capped = entries[-self._MAX_RECENT_HISTORY:]
                # 这是列表切片。
                # entries[-50:] 表示取最后 50 条。
                # 如果条目不足 50 条，则全部保留。
                capped = entries[-self._MAX_RECENT_HISTORY:]

                # 使用生成器表达式把每条历史格式化成一行：
                # - [timestamp] content
                #
                # "\n".join(...) 会把多行字符串用换行符连接成一个大字符串。
                history_text = "\n".join(
                    f"- [{e['timestamp']}] {e['content']}" for e in capped
                )

                # 按 token 数截断历史文本。
                # 如果 history_text 超过 self._MAX_HISTORY_TOKENS，
                # 则截断到允许的最大长度。
                history_text = truncate_text_to_tokens(history_text, self._MAX_HISTORY_TOKENS)

                # 加上 Markdown 标题后追加到 parts。
                parts.append("# Recent History\n\n" + history_text)

        # 如果存在 session_summary，则追加归档上下文摘要。
        if session_summary:
            parts.append(f"[Archived Context Summary]\n\n{session_summary}")

        # 最终把所有 parts 用 "\n\n---\n\n" 连接。
        # 也就是每个部分之间插入：
        # 空行、水平分割线、空行。
        return "\n\n---\n\n".join(parts)

    # ==================================================================
    # 方法：_get_identity
    # 作用：获取 system prompt 中的核心身份部分。
    # 前缀下划线表示这是类内部方法。
    # ==================================================================
    def _get_identity(self, channel: str | None = None, workspace: Path | None = None) -> str:
        """Get the core identity section."""
        # 原始 docstring 中文翻译：
        # 获取核心身份部分。

        # 参数说明：
        # channel：消息渠道，可能影响身份模板内容。
        # workspace：当前会话使用的 workspace。
        #
        # 返回值：
        # 渲染后的 identity Markdown 文本。

        # 如果调用时传了 workspace，就使用传入的 workspace；
        # 否则使用 self.workspace。
        root = workspace or self.workspace

        # 得到当前 workspace 的绝对路径字符串。
        #
        # root.expanduser()：
        # 将路径中的 ~ 展开成用户主目录。
        # 例如 "~/project" 变成 "/home/user/project"。
        #
        # resolve()：
        # 将路径解析为绝对路径，并尽量解析符号链接。
        #
        # str(...)：
        # 将 Path 对象转成普通字符串，方便模板渲染。
        workspace_path = str(root.expanduser().resolve())

        # 得到 Agent 默认 workspace 的绝对路径字符串。
        # 注意这里用的是 self.workspace，而不是 root。
        # 这意味着即使当前请求使用其他 workspace，也会保留 Agent 自身默认工作目录。
        agent_workspace_path = str(self.workspace.expanduser().resolve())

        # platform.system() 返回操作系统名称。
        # 常见返回值：
        # - "Darwin" 表示 macOS。
        # - "Linux" 表示 Linux。
        # - "Windows" 表示 Windows。
        system = platform.system()

        # 构建运行时描述字符串。
        #
        # f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        #
        # Python 特殊用法：
        # f-string 中可以嵌入表达式。
        # 'macOS' if system == 'Darwin' else system 是条件表达式。
        # 如果 system 是 "Darwin"，显示为 macOS，否则原样显示。
        #
        # platform.machine()：CPU 架构，例如 arm64、x86_64。
        # platform.python_version()：Python 版本，例如 3.12.4。
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        # 渲染 identity 模板。
        # 传入模板变量：
        # workspace_path：当前 workspace 路径。
        # agent_workspace_path：Agent 默认 workspace 路径。
        # runtime：运行时信息。
        # platform_policy：平台策略模板渲染结果。
        # channel：渠道，如果是 None 则传空字符串。
        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            agent_workspace_path=agent_workspace_path,
            runtime=runtime,
            # 这里嵌套调用了 render_template。
            # 先渲染 platform_policy.md，再把结果作为变量传入 identity.md。
            platform_policy=render_template("agent/platform_policy.md", system=system),
            # channel or ""：
            # 如果 channel 是 None 或空字符串，则使用空字符串。
            channel=channel or "",
        )

    # ==================================================================
    # 静态方法：_merge_message_content
    # 作用：合并两个 message content。
    # content 可能是纯字符串，也可能是多模态 block 列表。
    # ==================================================================
    @staticmethod
    # Python 特殊用法：
    # @staticmethod 表示这是一个静态方法。
    # 静态方法不会自动接收 self 或 cls 参数。
    # 它更像普通函数，只是逻辑上属于这个类。
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        # 参数说明：
        # left：左侧 content，可能是字符串、列表、None 等。
        # right：右侧 content，可能是字符串、列表、None 等。
        #
        # 返回值：
        # 如果两边都是字符串，则返回合并后的字符串。
        # 否则返回多模态 block 列表。

        # 如果 left 和 right 都是字符串，则走纯文本合并逻辑。
        if isinstance(left, str) and isinstance(right, str):
            # 如果 left 是空字符串，则直接返回 right。
            # 避免产生不必要的换行。
            if not left:
                return right

            # 如果 right 是空字符串，则直接返回 left。
            if not right:
                return left

            # 两边都非空时，用两个换行连接。
            return f"{left}\n\n{right}"

        # 如果至少有一边不是字符串，则需要按“多模态内容块”处理。
        # 多模态内容块通常是形如：
        # {"type": "text", "text": "..."}
        # 或：
        # {"type": "image_url", "image_url": {...}}
        # 的字典列表。

        # 内部函数：_to_blocks
        # 作用：把任意 content 值规范化成 block 列表。
        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            # 如果 value 本身已经是列表，则认为它是 block 列表。
            if isinstance(value, list):
                # 返回列表推导式结果：
                # 对 value 中每个 item：
                # - 如果 item 是 dict，则用 cast 标记为 dict[str, Any]。
                # - 否则把它包装成 text block。
                #
                # Python 特殊用法：
                # [expr for item in iterable] 是列表推导式。
                # 这里使用了条件表达式：
                # cast(...) if isinstance(item, dict) else {...}
                return [
                    cast(dict[str, Any], item)
                    if isinstance(item, dict)
                    else {"type": "text", "text": str(item)}
                    for item in cast(list[Any], value)
                ]

            # 如果 value 是 None，则返回空列表。
            if value is None:
                return []

            # 其他情况统一转成单个 text block。
            return [{"type": "text", "text": str(value)}]

        # 将 left 规范化为 block 列表，将 right 规范化为 block 列表，然后拼接。
        # Python 特殊用法：
        # list_a + list_b 会生成一个新列表，包含两个列表的所有元素。
        return _to_blocks(left) + _to_blocks(right)

    # ==================================================================
    # 方法：_load_bootstrap_files
    # 作用：加载项目说明文件和 Agent 全局 profile 文件。
    # 包括：
    # - AGENTS.md：项目级 Agent 指令。
    # - SOUL.md：Agent 灵魂 / 人格文件。
    # - USER.md：用户信息文件。
    # ==================================================================
    def _load_bootstrap_files(self, workspace: Path | None = None) -> str:
        """Load project instructions plus the agent's global profile files."""
        # 原始 docstring 中文翻译：
        # 加载项目指令，以及 Agent 的全局 profile 文件。

        # parts 用于保存每个文件渲染后的 Markdown 片段。
        # 类型注解 list[str] 表示这是字符串列表。
        parts: list[str] = []

        # 如果调用时传了 workspace，则使用传入值；
        # 否则使用 self.workspace。
        project_root = workspace or self.workspace

        # sources 是一个列表，每个元素是二元组：
        # (文件名, 根目录)
        #
        # 含义：
        # AGENTS.md 从 project_root 读取。
        # SOUL.md 从 self.workspace 读取。
        # USER.md 从 self.workspace 读取。
        sources = [
            ("AGENTS.md", project_root),
            ("SOUL.md", self.workspace),
            ("USER.md", self.workspace),
        ]

        # 遍历 sources。
        # Python 特殊用法：
        # for filename, root in sources:
        # 这是元组解包。
        # 每次循环把 ("AGENTS.md", project_root) 中的两个元素分别赋给 filename 和 root。
        for filename, root in sources:
            # Path 对象支持 / 运算符。
            # root / filename 表示拼接路径。
            # 例如 root 是 Path("/home/user")，filename 是 "SOUL.md"，
            # 结果是 Path("/home/user/SOUL.md")。
            file_path = root / filename

            # 如果文件存在，才继续读取。
            if file_path.exists():
                # 以 UTF-8 编码读取文件全部文本内容。
                content = file_path.read_text(encoding="utf-8")

                # 特殊处理 SOUL.md：
                # 如果当前 SOUL.md 内容等于旧版模板 legacy/SOUL.md，
                # 说明用户可能没有自定义它。
                if filename == "SOUL.md" and self._is_template_content(
                    content,
                    "legacy/SOUL.md",
                ):
                    # 尝试加载内置的新版 SOUL.md 模板。
                    # load_bundled_template("SOUL.md") 可能返回字符串或 None。
                    #
                    # Python 特殊用法：
                    # a or b：
                    # 如果 a 是真值，返回 a；否则返回 b。
                    # 这里如果新版模板存在且非空，就用新版模板；
                    # 否则继续用原 content。
                    content = load_bundled_template("SOUL.md") or content

                # 如果内容 strip 后为空，则跳过该文件。
                # strip() 会移除字符串开头和结尾的空白字符。
                if not content.strip():
                    continue

                # 如果文件属于“可跳过的默认文件”，
                # 并且内容完全等于内置默认模板，
                # 则跳过，不把它塞进 system prompt。
                #
                # 这样做可以避免默认模板污染 prompt。
                if filename in self._SKIPPABLE_DEFAULTS and self._is_template_content(
                    content, filename
                ):
                    continue

                # 将有效内容包装成 Markdown 二级标题后追加到 parts。
                parts.append(f"## {filename}\n\n{content}")

        # 如果 parts 非空，用两个换行连接；否则返回空字符串。
        # Python 特殊用法：
        # expression_if_true if condition else expression_if_false
        # 这是条件表达式。
        return "\n\n".join(parts) if parts else ""

    # ==================================================================
    # 静态方法：_is_template_content
    # 作用：判断给定内容是否与内置模板完全一致。
    # 如果一致，说明用户可能没有自定义该文件。
    # ==================================================================
    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        # 原始 docstring 中文翻译：
        # 检查 content 是否与内置模板完全一致（即用户尚未自定义）。

        # 参数说明：
        # content：实际读取到的文件内容。
        # template_path：内置模板路径。
        #
        # 返回值：
        # True 表示 content 与内置模板一致。
        # False 表示不一致，或者模板不存在。

        # 加载内置模板。
        # 如果模板不存在，可能返回 None。
        tpl = load_bundled_template(template_path)

        # 如果模板加载成功，则比较两者 strip 后的内容是否相同。
        # strip 是为了忽略首尾空白差异。
        if tpl is not None:
            return content.strip() == tpl.strip()

        # 如果模板不存在，则不能判定为模板内容，返回 False。
        return False

    # ==================================================================
    # 方法：build_messages
    # 作用：构建一次 LLM 调用所需的完整 messages 列表。
    # 返回格式通常是：
    # [
    #   {"role": "system", "content": "..."},
    #   {"role": "user", "content": "..."},
    #   {"role": "assistant", "content": "..."},
    #   ...
    # ]
    # ==================================================================
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        *,
        media: list[str] | None = None,
        channel: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        # 原始 docstring 中文翻译：
        # 为一次 LLM 调用构建完整消息列表。

        # 参数说明：
        # history：历史消息列表。
        # current_message：当前消息文本。
        # media：当前消息附带的媒体文件路径列表，通常是图片。
        # channel：消息渠道。
        # current_role：当前消息角色，默认 user。
        # session_summary：会话摘要。
        # runtime_context_blocks：运行时上下文块。
        # workspace：当前 workspace。
        # include_memory：是否包含 memory。
        # include_memory_recent_history：是否包含最近历史。
        # session_key：会话 key。
        # unified_session：是否统一会话。

        # 如果传入 workspace，则使用传入值；否则使用 self.workspace。
        root = workspace or self.workspace

        # 判断当前消息是否会显式触发技能。
        #
        # 只有当前角色是 user 时，才从当前消息中解析显式调用的技能。
        # 否则传空列表。
        #
        # Python 特殊用法：
        # (
        #   expression_if_true
        #   if condition
        #   else expression_if_false
        # )
        # 这是跨行的条件表达式。
        active_skill_names = (
            self.skills.get_explicitly_invoked_skills(current_message)
            if current_role == "user"
            else []
        )

        # messages 是最终要返回的消息列表。
        #
        # 列表中包含：
        # 1. system 消息。
        # 2. 历史消息。
        #
        # Python 特殊用法：
        # *history 是解包操作。
        # 如果 history = [a, b]，那么：
        # [system_message, *history]
        # 等价于：
        # [system_message, a, b]
        messages: list[dict[str, Any]] = [
            {
                # system 消息，角色固定为 "system"。
                "role": "system",
                # content 由 build_system_prompt 构建。
                "content": self.build_system_prompt(
                    active_skill_names=active_skill_names,
                    channel=channel,
                    session_summary=session_summary,
                    workspace=root,
                    include_memory=include_memory,
                    include_memory_recent_history=include_memory_recent_history,
                    session_key=session_key,
                    unified_session=unified_session,
                ),
            },
            # 将历史消息展开插入到列表中。
            *history,
        ]

        # 构建当前轮次消息对象。
        # 该对象通常形如：
        # {"role": "user", "content": ...}
        # 如果带图片，content 可能是 block 列表。
        current = self.build_current_message(
            current_message,
            media=media,
            current_role=current_role,
            runtime_context_blocks=runtime_context_blocks,
        )

        # 如果 messages 中最后一条消息的角色与当前消息角色相同，
        # 则考虑合并它们，而不是连续追加两条同角色消息。
        #
        # 例如最后一条已经是 user，当前也是 user，
        # 那么可以合并成一条 user 消息。
        if messages[-1].get("role") == current_role:
            # dict(messages[-1])：
            # 对最后一条消息字典做浅拷贝，避免直接修改原字典。
            last = dict(messages[-1])

            # 合并旧 content 和当前 content。
            # _merge_message_content 会处理字符串或多模态 block 列表。
            last["content"] = self._merge_message_content(
                last.get("content"),
                current.get("content"),
            )

            # 获取当前消息中的 _meta。
            # _meta 通常是内部元数据字段。
            current_meta = current.get("_meta")

            # 只有当前角色是 user，并且 current_meta 是字典时，才合并 _meta。
            if current_role == "user" and isinstance(current_meta, dict):
                # last.get("_meta") or {}：
                # 如果 last 中没有 _meta，或者 _meta 为 None，则使用空字典。
                internal_meta = dict(last.get("_meta") or {})

                # 将 current_meta 合并进 internal_meta。
                # cast 用于告诉类型检查器 current_meta 是 dict[str, Any]。
                # 运行时 cast 不改变值。
                internal_meta.update(cast(dict[str, Any], current_meta))

                # 把合并后的 _meta 写回 last。
                last["_meta"] = internal_meta

            # 用合并后的 last 替换原最后一条消息。
            messages[-1] = last

            # 返回合并后的完整消息列表。
            return messages

        # 如果最后一条消息角色与当前消息角色不同，
        # 则直接把当前消息追加到 messages。
        messages.append(current)

        # 返回最终消息列表。
        return messages

    # ==================================================================
    # 方法：build_current_message
    # 作用：只构建当前这一轮的新消息对象。
    # 不会主动把它合并进 history。
    # ==================================================================
    def build_current_message(
        self,
        current_message: str,
        *,
        media: list[str] | None = None,
        current_role: str = "user",
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
    ) -> dict[str, Any]:
        """Build only the fresh turn message without merging it into history."""
        # 原始 docstring 中文翻译：
        # 只构建当前轮次的新消息，不将其合并进历史。

        # 参数说明：
        # current_message：当前消息文本。
        # media：当前消息的图片路径列表。
        # current_role：当前消息角色，默认 user。
        # runtime_context_blocks：运行时上下文块。
        #
        # 返回值：
        # 一个消息字典，例如：
        # {"role": "user", "content": "..."}。

        # 构建用户内容。
        # 如果没有图片，content 是字符串。
        # 如果有有效图片，content 是 block 列表。
        content = self.build_user_content(current_message, image_paths=media)

        # 如果当前角色是 user，则使用 runtime_context_blocks；
        # 否则 runtime context 不加入消息。
        #
        # runtime_context_blocks or ()：
        # 如果 runtime_context_blocks 是 None，则用空元组代替。
        #
        # list(...)：
        # 将 Sequence 转成 list，方便后续函数处理。
        #
        # 如果不是 user，则给空列表。
        blocks = list(runtime_context_blocks or ()) if current_role == "user" else []

        # 调用 append_runtime_context，把运行时上下文块附加到 content。
        #
        # 返回值有两个：
        # merged：附加 runtime context 后的内容。
        # runtime_context_meta：运行时上下文元数据，可能为 None。
        #
        # Python 特殊用法：
        # merged, runtime_context_meta = ...
        # 这是元组解包。
        merged, runtime_context_meta = append_runtime_context(content, blocks)

        # 构造当前消息字典。
        current: dict[str, Any] = {"role": current_role, "content": merged}

        # 如果当前角色是 user，并且存在 runtime context 元数据，
        # 则把元数据放入 current["_meta"]。
        if current_role == "user" and runtime_context_meta is not None:
            current["_meta"] = {
                # RUNTIME_CONTEXT_MESSAGE_META 是元数据键名。
                # runtime_context_meta 是实际元数据值。
                RUNTIME_CONTEXT_MESSAGE_META: runtime_context_meta,
            }

        # 返回当前消息字典。
        return current

    # ==================================================================
    # 方法：build_user_content
    # 作用：根据文本和图片路径构建用户消息 content。
    # 返回值可能是：
    # - 纯文本字符串。
    # - 多模态 block 列表。
    # ==================================================================
    def build_user_content(
        self,
        text: str,
        image_paths: list[str] | None,
    ) -> str | list[dict[str, Any]]:
        """Build user message content from prefiltered image paths."""
        # 原始 docstring 中文翻译：
        # 根据预过滤过的图片路径构建用户消息内容。

        # 参数说明：
        # text：用户文本。
        # image_paths：图片路径列表，可能为 None。
        #
        # 返回值说明：
        # 如果没有图片，返回 text。
        # 如果有有效图片，返回 block 列表：
        # [image_block, image_block, ..., text_block]

        # 如果没有图片路径，直接返回纯文本。
        # 注意：
        # not image_paths 在以下情况为 True：
        # - image_paths 是 None。
        # - image_paths 是空列表 []。
        if not image_paths:
            return text

        # image_blocks 用于保存成功处理的图片 block。
        image_blocks: list[dict[str, Any]] = []

        # 遍历每个图片路径。
        for path in image_paths:
            # 把字符串路径转换成 Path 对象。
            p = Path(path)

            # 如果路径不是普通文件，则跳过。
            # 例如目录、不存在的路径都会被跳过。
            if not p.is_file():
                continue

            # 读取文件的原始二进制内容。
            raw = p.read_bytes()

            # 原始英文注释：
            # Re-detect from the bytes used for the request: the file may have
            # changed since attachment routing, and the data URL needs its MIME.
            #
            # 中文翻译：
            # 根据本次请求实际使用的字节重新检测 MIME 类型：
            # 文件可能在附件路由之后发生了变化，
            # 而 data URL 需要正确的 MIME 类型。

            # 检测 MIME 类型。
            #
            # detect_image_mime(raw)：
            # 优先根据图片二进制内容判断 MIME。
            #
            # mimetypes.guess_type(path)：
            # 如果二进制检测失败，则根据文件扩展名猜测 MIME。
            # guess_type 返回一个元组：(mime_type, encoding)。
            # 所以这里取 [0] 获取第一个元素，即 MIME 类型。
            #
            # Python 特殊用法：
            # a or b：
            # 如果 a 为真，则返回 a；否则返回 b。
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]

            # 如果没有检测到 MIME，或者 MIME 不以 "image/" 开头，
            # 说明不是有效图片，跳过。
            if not mime or not mime.startswith("image/"):
                continue

            # 将图片二进制内容进行 Base64 编码。
            # base64.b64encode(raw) 返回 bytes。
            # .decode() 将 bytes 解码为普通字符串。
            b64 = base64.b64encode(raw).decode()

            # 构造一个 image_url 类型的多模态 block。
            # 这与一些 LLM API 的图片输入格式类似。
            image_blocks.append({
                # block 类型为 image_url。
                "type": "image_url",
                # image_url 字段中包含 data URL。
                # data URL 格式：
                # data:<mime>;base64,<encoded-data>
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                # _meta 是内部元数据，记录原始文件路径。
                # 这个字段通常不会直接发给 LLM，可能供内部追踪使用。
                "_meta": {"path": str(p)},
            })

        # 如果最终没有有效图片 block，则返回纯文本。
        if not image_blocks:
            return text

        # 如果有图片，则返回：
        # 所有图片 block + 一个文本 block。
        #
        # Python 特殊用法：
        # list_a + list_b 会拼接两个列表，生成新列表。
        #
        # 这里最后追加文本 block，确保图片之后仍然包含用户文本。
        return image_blocks + [{"type": "text", "text": text}]