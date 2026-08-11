"""Subagent manager for background task execution."""
"""用于后台任务执行的子智能体（Subagent）管理器。"""

import asyncio # 异步I/O库，用于并发执行子智能体任务
import json # JSON序列化和反序列化库
import time # 时间相关库，用于记录单调时间（monotonic time）
import uuid # 通用唯一识别码生成库，用于生成任务ID
import warnings # 警告处理库，用于发出弃用警告等
from collections.abc import Mapping # 抽象基类，用于类型提示（映射/字典类型）
from dataclasses import dataclass, field # dataclass装饰器用于创建数据类，field用于自定义字段行为
from pathlib import Path # 路径操作库，提供面向对象的路径操作
from typing import Any, Callable, TypedDict # 类型提示工具：Any表示任意类型，Callable表示可调用对象，TypedDict用于创建字典的类型提示

from loguru import logger # 强大的日志库

# 导入 nanobot 项目内部的相关模块和类
from nanobot.agent.hook import AgentHook, AgentHookContext # 智能体钩子（Hook）基类和上下文
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec # 智能体执行器、执行结果和执行规格
from nanobot.agent.tools.base import ToolResult # 工具执行结果基础类
from nanobot.agent.tools.context import (
    RequestContext, # 请求上下文
    ToolContext, # 工具上下文
    bind_request_context, # 绑定请求上下文到上下文变量（ContextVar）
    reset_request_context, # 重置请求上下文
)
from nanobot.agent.tools.exec_session import ExecSessionManager # 执行会话管理器
from nanobot.agent.tools.file_state import FileStates # 文件状态管理
from nanobot.agent.tools.loader import ToolLoader # 工具加载器
from nanobot.agent.tools.registry import ToolRegistry # 工具注册表
from nanobot.bus.events import InboundMessage # 消息总线中的入站消息事件
from nanobot.bus.queue import MessageBus # 消息总线队列
from nanobot.config.schema import AgentDefaults, ToolsConfig # 智能体默认配置和工具配置模式
from nanobot.providers.base import LLMProvider # 大型语言模型提供者基类
from nanobot.security.workspace_access import (
    WorkspaceScope, # 工作空间访问范围
    bind_workspace_scope, # 绑定工作空间范围
    reset_workspace_scope, # 重置工作空间范围
    workspace_sandbox_status, # 工作空间沙箱状态
)
from nanobot.utils.llm_runtime import LLMRuntime # LLM 运行时封装
from nanobot.utils.prompt_templates import render_template # 提示词模板渲染工具


# 定义一个类型字典（TypedDict），用于在类型检查时规范子智能体来源信息的结构
# TypedDict 允许为字典的每个键指定特定的值类型，但在运行时它仍然是普通的 Python 字典
class _SubagentOrigin(TypedDict):
    channel: str         # 来源渠道（例如 "cli", "web" 等）
    chat_id: str         # 聊天ID，标识具体的对话或会话
    session_key: str | None # 会话键，用于关联统一会话，可能为空（使用 PEP 604 的联合类型语法，等同于 Optional[str]）


# 使用 dataclass 装饰器创建数据类，自动生成 __init__、__repr__ 等方法
# slots=True 是 Python 3.10 引入的特性，它使用 __slots__ 替代 __dict__ 存储实例属性
# 这样可以显著减少内存占用，并加快属性访问速度，非常适合需要创建大量实例的状态类
@dataclass(slots=True)
class SubagentStatus:
    """正在运行的子智能体的实时状态。""" # 翻译自原注释

    task_id: str             # 任务的唯一标识符
    label: str               # 任务的显示标签（简短描述）
    task_description: str    # 任务的详细描述
    started_at: float        # 任务开始的时间戳，使用 time.monotonic() 获取（不受系统时钟回拨影响）
    # 任务所处的阶段，默认值为 "initializing"（初始化中）
    # 可能的状态包括：initializing | awaiting_tools | tools_completed | final_response | done | error
    phase: str = "initializing"  
    iteration: int = 0       # 当前智能体思考/工具调用的迭代轮数
    # 工具调用事件列表，使用 field(default_factory=list) 确保每个实例都有独立的空列表，避免多个实例共享同一个可变对象
    tool_events: list[dict[str, str]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict) # Token 使用情况统计（如 prompt_tokens, completion_tokens）
    stop_reason: str | None = None # 任务停止的原因（如 "done", "error", "tool_error"）
    error: str | None = None       # 如果发生错误，记录错误信息


# 继承自 AgentHook 的钩子类，用于在子智能体执行生命周期中插入自定义逻辑
class _SubagentHook(AgentHook):
    """子智能体执行的钩子（Hook） —— 记录工具调用日志并更新状态。""" # 翻译自原注释

    # 初始化钩子，接收任务ID和状态对象引用
    def __init__(self, task_id: str, status: SubagentStatus | None = None) -> None:
        # 调用父类的初始化方法
        super().__init__()
        self._task_id = task_id   # 保存任务ID，用于日志记录
        self._status = status     # 保存状态对象的引用，用于实时更新状态

    # 异步钩子方法：在智能体准备执行工具之前调用
    async def before_execute_tools(self, context: AgentHookContext) -> None:
        # 遍历当前轮次中智能体决定调用的所有工具
        for tool_call in context.tool_calls:
            # 将工具参数转换为 JSON 字符串，ensure_ascii=False 保证中文字符正常显示
            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
            # 记录调试级别的日志，输出任务ID、工具名称和参数
            logger.debug(
                "Subagent [{}] executing: {} with arguments: {}",
                self._task_id, tool_call.name, args_str,
            )

    # 异步钩子方法：在每一轮（迭代）结束后调用
    async def after_iteration(self, context: AgentHookContext) -> None:
        # 如果没有传入状态对象，则直接返回，不更新状态
        if self._status is None:
            return
        # 更新状态对象中的各项指标，这些状态会被外部实时读取（例如用于前端进度显示）
        self._status.iteration = context.iteration          # 更新当前迭代轮数
        self._status.tool_events = list(context.tool_events) # 更新工具执行事件记录
        self._status.usage = dict(context.usage)            # 更新 Token 消耗统计
        # 如果上下文中包含错误信息，将其转换为字符串并记录到状态中
        if context.error:
            self._status.error = str(context.error)


class SubagentManager:
    """管理后台子智能体（subagent）的执行。""" # 翻译自原注释

    def __init__(
        self,
        provider: LLMProvider | None = None, # LLM提供者实例（已弃用，推荐使用runtime）
        workspace: Path | None = None,       # 智能体工作空间路径对象
        bus: MessageBus | None = None,       # 消息总线实例，用于跨智能体通信
        max_tool_result_chars: int | None = None, # 工具返回结果的最大字符数限制
        model: str | None = None,            # LLM模型名称（已弃用）
        tools_config: ToolsConfig | None = None, # 工具配置对象
        restrict_to_workspace: bool = False, # 是否将文件/执行操作限制在工作空间内
        disabled_skills: list[str] | None = None, # 禁用的技能列表
        max_iterations: int | None = None,   # 最大迭代次数（思考+工具调用轮数）
        max_concurrent_subagents: int | None = None, # 最大并发子智能体数量
        fail_on_tool_error: bool | None = None, # 工具执行出错时是否直接让智能体失败
        # 一个回调函数，根据会话键获取 LLM 的墙钟超时时间（wall timeout）
        llm_wall_timeout_for_session: Callable[[str | None], float | None] | None = None,
    ):
        # 强制校验几个核心必填参数，如果为 None 则抛出类型错误（TypeError）
        if workspace is None:
            raise TypeError("SubagentManager.__init__() missing required argument: 'workspace'")
        if bus is None:
            raise TypeError("SubagentManager.__init__() missing required argument: 'bus'")
        if max_tool_result_chars is None:
            raise TypeError(
                "SubagentManager.__init__() missing required argument: 'max_tool_result_chars'"
            )
        # 如果指定了 model 但没有指定 provider，则报错，因为模型依赖于提供者
        if model is not None and provider is None:
            raise TypeError("SubagentManager model compatibility argument requires provider")

        # 获取智能体的默认配置
        defaults = AgentDefaults()
        self._compat_runtime: LLMRuntime | None = None # 兼容旧版 API 的运行时对象
        
        # 如果传入了 provider（旧版 API 用法），则发出弃用警告并构建兼容的运行时
        if provider is not None:
            # 发出 DeprecationWarning（弃用警告），stacklevel=2 表示警告指向调用 __init__ 的代码行，而不是 warnings.warn 这一行
            warnings.warn(
                "SubagentManager provider/model constructor arguments are deprecated; "
                "pass runtime=... to spawn() instead",
                DeprecationWarning,
                stacklevel=2,
            )
            # 捕获/创建 LLMRuntime 实例
            self._compat_runtime = LLMRuntime.capture(
                provider,
                model or provider.get_default_model(), # 如果未指定 model，则使用 provider 的默认模型
                context_window_tokens=defaults.context_window_tokens, # 上下文窗口大小
            )
        
        # 保存核心实例变量
        self.workspace = workspace
        self.bus = bus
        # 如果未提供工具配置，则实例化一个默认配置
        self.tools_config = tools_config or ToolsConfig()
        self.max_tool_result_chars = max_tool_result_chars
        self.restrict_to_workspace = restrict_to_workspace
        # 将禁用技能列表转换为集合（set），以便进行 O(1) 时间复杂度的查找
        self.disabled_skills = set(disabled_skills or [])
        
        # 如果未指定最大迭代次数，则使用默认配置中的值
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else defaults.max_tool_iterations
        )
        # 如果未指定最大并发数，则使用默认配置中的值
        self.max_concurrent_subagents = (
            max_concurrent_subagents
            if max_concurrent_subagents is not None
            else defaults.max_concurrent_subagents
        )
        # 如果未指定工具报错策略，则使用默认配置中的值
        self.fail_on_tool_error = (
            fail_on_tool_error
            if fail_on_tool_error is not None
            else defaults.fail_on_tool_error
        )
        
        # 实例化智能体执行器、执行会话管理器等内部组件
        self.runner = AgentRunner()
        self._exec_session_manager = ExecSessionManager()
        self._llm_wall_timeout_for_session = llm_wall_timeout_for_session
        
        # 用于跟踪正在运行的异步任务及其状态的字典
        self._running_tasks: dict[str, asyncio.Task[str]] = {} # task_id -> asyncio.Task
        self._task_statuses: dict[str, SubagentStatus] = {}    # task_id -> SubagentStatus
        # 维护 session_key 到 task_id 集合的映射，方便按会话批量管理任务
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    def runtime_statuses(self) -> Mapping[str, SubagentStatus]:
        """返回用于运行时控制快照的可观察任务状态。""" # 翻译自原注释
        # 返回状态字典，使用 collections.abc.Mapping 作为返回类型，暗示调用者不应直接修改此字典
        return self._task_statuses

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        """更新旧的 ``spawn`` 调用所使用的已弃用的运行时源。""" # 翻译自原注释
        # 发出弃用警告
        warnings.warn(
            "SubagentManager.set_provider() is deprecated; pass runtime=... to spawn() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # 获取当前的上下文窗口大小，如果存在旧的运行时则复用其配置，否则使用默认配置
        context_window_tokens = (
            self._compat_runtime.context_window_tokens
            if self._compat_runtime is not None
            else AgentDefaults().context_window_tokens
        )
        # 重新捕获/创建 LLMRuntime 实例
        self._compat_runtime = LLMRuntime.capture(
            provider,
            model,
            context_window_tokens=context_window_tokens,
        )

    def _compat_spawn_runtime(self) -> LLMRuntime:
        # 获取兼容的运行时对象
        runtime = self._compat_runtime
        # 如果没有运行时对象，说明调用方使用了新版 API 但忘记传入 runtime 参数
        if runtime is None:
            raise TypeError(
                "SubagentManager.spawn() missing required keyword-only argument: 'runtime'"
            )
        # 发出警告，提示调用方应该显式传入 runtime 参数
        warnings.warn(
            "SubagentManager.spawn() without runtime is deprecated; pass runtime=... explicitly",
            DeprecationWarning,
            stacklevel=3, # stacklevel=3 指向调用 spawn() 的代码，跳过 _compat_spawn_runtime 和 spawn 本身
        )
        # 重新生成并返回一个 LLMRuntime 实例
        return LLMRuntime.capture(
            runtime.provider,
            runtime.model,
            context_window_tokens=runtime.context_window_tokens,
        )

    def _subagent_tools_config(self) -> ToolsConfig:
        """构建一个专门用于子智能体作用域的 ToolsConfig（工具配置）。""" # 翻译自原注释
        # 根据当前的配置生成一个新的配置对象，并将 restrict_to_workspace 强制设为当前管理器的设置
        return ToolsConfig(
            exec=self.tools_config.exec,   # 执行工具配置
            web=self.tools_config.web,     # 网页抓取工具配置
            file=self.tools_config.file,   # 文件操作工具配置
            restrict_to_workspace=self.restrict_to_workspace, # 沙箱限制开关
        )

    def _build_tools(
        self,
        workspace: Path | None = None,
        tools_config: ToolsConfig | None = None,
    ) -> ToolRegistry:
        """通过 ToolLoader 构建一个隔离的子智能体工具注册表。""" # 翻译自原注释
        # 确定工作空间根目录：如果参数未提供，则使用管理器默认的 workspace
        root = self.workspace if workspace is None else workspace
        # 实例化一个空的工具注册表
        registry = ToolRegistry()
        # 确定工具配置：如果参数未提供，则使用专门构建的子智能体配置
        cfg = tools_config if tools_config is not None else self._subagent_tools_config()
        
        # 构建工具上下文（ToolContext），包含工具运行所需的所有依赖和环境信息
        ctx = ToolContext(
            config=cfg,
            workspace=str(root.resolve()), # resolve() 将路径转换为绝对路径并解析符号链接
            exec_session_manager=self._exec_session_manager, # 注入执行会话管理器
            file_state_store=FileStates(), # 注入文件状态存储
            # 获取当前工作空间的沙箱状态，用于限制文件访问
            workspace_sandbox=workspace_sandbox_status(
                restrict_to_workspace=cfg.restrict_to_workspace,
                workspace=root,
            ),
        )
        # 使用工具加载器（ToolLoader）加载指定作用域（scope="subagent"）的工具到注册表中
        ToolLoader().load(ctx, registry, scope="subagent")
        return registry

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *, # 星号（*）后的参数（如 runtime）必须作为关键字参数传递（keyword-only）
        runtime: LLMRuntime | None = None,
    ) -> str:
        """生成（Spawn）一个子智能体在后台执行任务。""" # 翻译自原注释
        # 如果未传入运行时对象，则尝试使用兼容旧版 API 的方法获取
        if runtime is None:
            runtime = self._compat_spawn_runtime()
        # 如果指定了温度参数（temperature），则生成一个覆盖该参数的新运行时对象
        if temperature is not None:
            runtime = runtime.with_generation_overrides(temperature=temperature)
            
        # 生成一个 8 位的短 UUID 作为任务 ID，方便日志显示和追踪
        task_id = str(uuid.uuid4())[:8]
        # 生成显示标签：如果未提供 label，则截取 task 前 30 个字符，超出部分用 "..." 表示
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        
        # 组装来源信息字典
        origin: _SubagentOrigin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "session_key": session_key,
        }

        # 初始化子智能体状态对象，记录当前时间和阶段
        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
        )
        # 将状态存入管理器的状态字典中
        self._task_statuses[task_id] = status

        # 使用 asyncio.create_task 将子智能体执行协程包装为后台任务，实现非阻塞并发执行
        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                status,
                runtime,
                origin_message_id,
                workspace_scope,
            )
        )
        # 将任务对象保存在运行任务字典中
        self._running_tasks[task_id] = bg_task
        # 如果存在 session_key，则将 task_id 加入到对应的会话任务集合中
        # setdefault 确保如果键不存在，会先创建一个空集合
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        # 定义一个内部回调函数，用于在后台任务完成（无论成功或失败）时进行清理
        def _cleanup(_: asyncio.Task[str]) -> None:
            # 从运行任务字典中移除该任务，pop 的第二个参数是默认值，防止 KeyError
            self._running_tasks.pop(task_id, None)
            # 从状态字典中移除该任务的状态
            self._task_statuses.pop(task_id, None)
            # 海象运算符 (:=) 在获取会话任务集合并赋值给 ids 的同时，判断其是否为真
            if session_key and (ids := self._session_tasks.get(session_key)):
                # 从集合中丢弃该任务 ID（discard 即使元素不存在也不会报错）
                ids.discard(task_id)
                # 如果该会话下已经没有其他运行的任务，则从字典中删除该会话键，释放内存
                if not ids:
                    del self._session_tasks[session_key]

        # 将清理函数注册为任务完成时的回调
        bg_task.add_done_callback(_cleanup)

        # 记录任务生成日志
        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        # 返回给调用者的提示信息
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def run_inline(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime | None = None,
    ) -> str:
        """同步运行子智能体，并将其结果返回给调用者。""" # 翻译自原注释
        if runtime is None:
            runtime = self._compat_spawn_runtime()
        if temperature is not None:
            runtime = runtime.with_generation_overrides(temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin: _SubagentOrigin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "session_key": session_key,
        }
        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
        )
        self._task_statuses[task_id] = status
        logger.info("Running inline subagent [{}]: {}", task_id, display_label)
        
        # 同样创建异步任务，但通过 announce=False 参数告诉子智能体不要自动通过消息总线广播结果
        inline_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                status,
                runtime,
                origin_message_id,
                workspace_scope,
                announce=False,
            )
        )
        self._running_tasks[task_id] = inline_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)
            
        try:
            # await 等待任务执行完成并获取结果
            result = await inline_task
            # 检查状态或停止原因，判断任务是否出错
            if status.phase == "error" or status.stop_reason in {"error", "tool_error"}:
                # 如果出错，将结果包装为 ToolResult 错误对象返回
                return ToolResult.error(result)
            # 否则正常返回字符串结果
            return result
        finally:
            # finally 块确保无论发生什么（包括抛出异常），都会执行清理逻辑
            self._running_tasks.pop(task_id, None)
            self._task_statuses.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: _SubagentOrigin,
        status: SubagentStatus,
        runtime: LLMRuntime,
        origin_message_id: str | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        announce: bool = True, # 是否在执行完成后自动广播结果
    ) -> str:
        """执行子智能体任务并广播结果。""" # 翻译自原注释
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        # 定义一个内部的异步检查点回调函数，用于更新状态对象
        async def _on_checkpoint(payload: dict[str, Any]) -> None:
            status.phase = payload.get("phase", status.phase)
            status.iteration = payload.get("iteration", status.iteration)

        try:
            # 确定工作空间根目录，优先使用传入的作用域路径
            root = workspace_scope.project_path if workspace_scope is not None else self.workspace
            cfg = None
            if workspace_scope is not None:
                cfg = self._subagent_tools_config()
                # 强制使用作用域中定义的沙箱限制配置
                cfg.restrict_to_workspace = workspace_scope.restrict_to_workspace
            
            # 从智能体工作空间构建工具；下方绑定的作用域（scope）会提供项目当前的工作目录（cwd）。
            tools = self._build_tools(tools_config=cfg)
            # 构建子智能体的系统提示词
            system_prompt = self._build_subagent_prompt(workspace=root)
            
            # 构建初始的消息列表，包含系统提示词和用户的具体任务指令
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # 获取会话键
            sess_key = origin.get("session_key")
            # 根据会话键获取 LLM 墙钟超时时间（如果设置了回调函数）
            llm_timeout = (
                self._llm_wall_timeout_for_session(sess_key)
                if self._llm_wall_timeout_for_session
                else None
            )
            
            # 绑定请求上下文到 ContextVar（上下文变量），以便在异步调用栈的深处也能获取到当前请求的信息
            # bind_request_context 会返回一个 token，用于后续重置上下文
            request_token = bind_request_context(RequestContext(
                channel=origin["channel"],
                chat_id=origin["chat_id"],
                message_id=origin_message_id,
                session_key=sess_key,
                runtime=runtime,
            ))
            # 如果提供了工作空间作用域，则绑定它，同样获取 token
            token = bind_workspace_scope(workspace_scope) if workspace_scope is not None else None
            
            try:
                # 使用 AgentRunner 启动智能体执行循环
                result = await self.runner.run(AgentRunSpec(
                    initial_messages=messages,         # 初始消息历史
                    tools=tools,                       # 可用的工具注册表
                    runtime=runtime,                   # LLM 运行时配置
                    max_iterations=self.max_iterations, # 最大思考/调用轮数
                    max_tool_result_chars=self.max_tool_result_chars, # 工具结果截断长度
                    hook=_SubagentHook(task_id, status), # 注入日志和状态更新钩子
                    max_iterations_message="Task completed but no final response was generated.", # 达到最大轮数时的提示
                    finalize_on_max_iterations=False,  # 达到最大轮数时不自动强制结束生成最终回复
                    error_message=None,                # 自定义错误消息
                    fail_on_tool_error=self.fail_on_tool_error, # 工具报错是否直接终止
                    checkpoint_callback=_on_checkpoint, # 状态检查点回调
                    session_key=sess_key,              # 会话键
                    workspace=root,                    # 工作空间路径
                    llm_timeout_s=llm_timeout,         # LLM 请求超时时间
                ))
            finally:
                # 无论 run() 是否发生异常，都必须重置上下文变量，防止上下文泄漏到其他异步任务中
                if token is not None:
                    reset_workspace_scope(token)
                reset_request_context(request_token)
                
            # 执行完成，更新状态为 "done" 并记录停止原因
            status.phase = "done"
            status.stop_reason = result.stop_reason

            # 根据停止原因分类处理结果
            if result.stop_reason == "tool_error":
                status.tool_events = list(result.tool_events)
                # 格式化工具报错时的部分进度信息
                final_result = self._format_partial_progress(result)
                final_status = "error"
            elif result.stop_reason == "error":
                # 智能体自身发生错误（如 LLM API 报错）
                final_result = result.error or "Error: subagent execution failed."
                final_status = "error"
            else:
                # 正常完成，获取最终生成的文本内容
                final_result = result.final_content or "Task completed but no final response was generated."
                final_status = "ok"
                logger.info("Subagent [{}] completed successfully", task_id)
                
            # 如果允许广播，则将结果发送到消息总线
            if announce:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    final_status,
                    origin_message_id,
                )
            return final_result

        except Exception as e:
            # 捕获任何未预料的异常（如代码 Bug 等）
            status.phase = "error"
            status.error = str(e)
            # 记录完整的异常堆栈信息
            logger.exception("Subagent [{}] failed", task_id)
            final_result = f"Error: {e}"
            if announce:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    "error",
                    origin_message_id,
                )
            return final_result

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: _SubagentOrigin,
        status: str,
        origin_message_id: str | None = None,
    ) -> None:
        """通过消息总线将子智能体的结果广播给主智能体。""" # 翻译自原注释
        # 根据状态字符串生成可读的状态文本
        status_text = "completed successfully" if status == "ok" else "failed"

        # 使用 Jinja2 或其他模板引擎渲染通知消息内容
        announce_content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=task,
            result=result,
        )

        # 作为系统消息注入，以触发主智能体。
        # 使用 session_key_override（会话键覆盖）来与主智能体的有效会话键（考虑了统一会话）对齐，
        # 以便将结果路由到正确的待处理队列（回合中注入），而不是作为竞争性的独立任务被分发。
        override = origin.get("session_key") or f"{origin['channel']}:{origin['chat_id']}"
        
        # 构建消息元数据
        metadata: dict[str, Any] = {
            "injected_event": "subagent_result", # 标记为子智能体结果事件
            "subagent_task_id": task_id,
        }
        if origin_message_id:
            metadata["origin_message_id"] = origin_message_id
            
        # 构建入站消息对象
        msg = InboundMessage(
            channel="system", # 标记为系统渠道
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            session_key_override=override, # 关键：指定路由的会话键
            metadata=metadata,
        )

        # 将消息发布到消息总线的入站队列中
        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    @staticmethod
    def _format_partial_progress(result: AgentRunResult) -> str:
        # 列表推导式：筛选出所有状态为 "ok" 的工具执行事件
        completed = [e for e in result.tool_events if e["status"] == "ok"]
        # 生成器表达式结合 next() 和 reversed()：从后向前查找第一个状态为 "error" 的事件
        # 这样可以找到导致任务失败的那个最直接的工具错误
        failure = next((e for e in reversed(result.tool_events) if e["status"] == "error"), None)
        lines: list[str] = [] # 用于拼接最终输出的文本行
        
        if completed:
            lines.append("Completed steps:")
            # 只展示最后完成的 3 个步骤，避免信息过长
            for event in completed[-3:]:
                lines.append(f"- {event['name']}: {event['detail']}")
        if failure:
            if lines:
                lines.append("") # 添加空行分隔
            lines.append("Failure:")
            lines.append(f"- {failure['name']}: {failure['detail']}")
        # 如果存在全局错误信息且没有找到具体的工具失败事件
        if result.error and not failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {result.error}")
            
        # 将行列表用换行符连接，如果列表为空，则回退到默认的错误信息
        return "\n".join(lines) or (result.error or "Error: subagent execution failed.")

    def _build_subagent_prompt(self, workspace: Path | None = None) -> str:
        """为子智能体构建一个专注的系统提示词（system prompt）。""" # 翻译自原注释
        # 局部导入，避免循环引用或减少模块初始化时的开销
        from nanobot.agent.skills import SkillsLoader

        # expanduser() 将 ~ 替换为用户主目录，resolve() 获取绝对路径并解析符号链接
        agent_workspace = self.workspace.expanduser().resolve()
        # 如果提供了项目工作空间则使用它，否则默认使用智能体工作空间
        project_workspace = workspace.expanduser().resolve() if workspace else agent_workspace
        
        # 使用 SkillsLoader 加载并构建可用技能的摘要文本
        skills_summary = SkillsLoader(
            self.workspace,
            disabled_skills=self.disabled_skills, # 传入禁用列表
        ).build_skills_summary()
        
        # 渲染提示词模板，注入相关变量
        return render_template(
            "agent/subagent_system.md",
            workspace=str(project_workspace),
            agent_workspace=str(agent_workspace),
            history_log=str(agent_workspace / "memory" / "history.jsonl"), # Path 对象的 / 运算符用于路径拼接
            skills_summary=skills_summary or "",
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """取消给定会话的所有子智能体。返回被取消的数量。""" # 翻译自原注释
        # 列表推导式：获取属于该 session_key 且尚未完成的所有 asyncio.Task 对象
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        
        # 遍历并调用 cancel() 请求取消这些任务
        for t in tasks:
            t.cancel()
            
        if tasks:
            # 使用 asyncio.gather 并发等待所有取消操作完成
            # return_exceptions=True 确保即使某个任务在取消时抛出 CancelledError，也不会中断 gather 的执行
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # 终止该会话拥有的所有底层执行会话（如终端 shell 进程）
        await self._exec_session_manager.terminate_by_owner(session_key)
        return len(tasks)

    async def close(self) -> None:
        """取消正在运行的子智能体，并关闭它们共享的执行会话。""" # 翻译自原注释
        # 获取所有尚未完成的运行任务
        tasks = [task for task in self._running_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # 关闭执行会话管理器中的所有活跃会话
        await self._exec_session_manager.close_all()

    def get_running_count(self) -> int:
        """返回当前正在运行的子智能体数量。""" # 翻译自原注释
        # 直接返回运行任务字典的长度
        return len(self._running_tasks)

    def get_running_count_by_session(self, session_key: str) -> int:
        """返回某个会话当前正在运行的子智能体数量。""" # 翻译自原注释
        # 获取该会话关联的任务 ID 集合
        tids = self._session_tasks.get(session_key, set())
        # 使用生成器表达式结合 sum() 统计其中仍然在运行任务字典中且未完成的任务数量
        return sum(
            1 for tid in tids
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        )