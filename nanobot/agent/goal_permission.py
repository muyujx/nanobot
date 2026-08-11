"""Turn-local permission for explicit sustained-goal mutations.

[中文翻译] 回合局部权限：用于显式修改持续目标（sustained-goal）的权限控制。
"""

# 导入 __future__ 中的 annotations 特性。
# 作用：让类型注解延迟求值，允许使用较新的类型写法，也避免某些类型在定义时尚未解析的问题。
from __future__ import annotations

# 从 contextlib 导入 contextmanager 装饰器。
# @contextmanager 用于把一个生成器函数包装成上下文管理器，
# 使其可以配合 with 语句使用：
#     with goal_mutation_permission(True):
#         ...
from contextlib import contextmanager

# 从 contextvars 导入 ContextVar。
# ContextVar 是 Python 的“上下文局部变量”。
# 它常用于在异步任务、线程或不同执行上下文中保存彼此独立的状态。
# 在这里，它用于保存“当前回合/当前执行作用域是否允许修改持续目标”的权限。
from contextvars import ContextVar


# 定义一个模块级私有上下文变量 _GOAL_MUTATION_ALLOWED。
# 变量名前面的下划线 _ 表示这是模块内部使用的私有对象，不建议外部直接访问。
#
# 类型注解 ContextVar[bool] 表示：
#   这个 ContextVar 里保存的值类型是 bool。
# 也就是说，它保存的是 True / False：
#   True 表示当前上下文允许目标变更；
#   False 表示当前上下文不允许目标变更。
_GOAL_MUTATION_ALLOWED: ContextVar[bool] = ContextVar(
    # 这是 ContextVar 的名称字符串。
    # 它主要用于调试、日志或内部标识，并不是 Python 变量名本身。
    # 真正在代码里访问这个变量使用的是 _GOAL_MUTATION_ALLOWED。
    "nanobot_goal_mutation_allowed",

    # default=False 表示：
    # 如果当前上下文从未调用过 _GOAL_MUTATION_ALLOWED.set(...)，
    # 那么调用 _GOAL_MUTATION_ALLOWED.get() 时会返回默认值 False。
    # 也就是默认不允许修改持续目标。
    default=False,
# 结束 ContextVar(...) 的调用。
)


# 定义一个公开函数，用于查询当前上下文是否允许目标变更。
def goal_mutation_allowed() -> bool:
    # 返回当前上下文中的权限值。
    #
    # _GOAL_MUTATION_ALLOWED.get() 的作用：
    #   1. 如果当前上下文曾经设置过该 ContextVar，则返回最近设置的值；
    #   2. 如果当前上下文没有设置过，则返回创建 ContextVar 时指定的 default=False。
    #
    # 因此这个函数返回值含义是：
    #   True  -> 当前回合/作用域允许显式修改持续目标；
    #   False -> 当前回合/作用域不允许显式修改持续目标。
    return _GOAL_MUTATION_ALLOWED.get()


# 定义一个公开函数，用于撤销当前上下文中的目标变更权限。
def revoke_goal_mutation_permission() -> None:
    # 将当前上下文中的 _GOAL_MUTATION_ALLOWED 设置为 False。
    #
    # 注意：
    # ContextVar.set(...) 通常只影响“当前上下文”，
    # 不会自动污染其他上下文、其他任务或其他回合。
    # 这对 agent 的 turn-local 权限控制非常重要。
    _GOAL_MUTATION_ALLOWED.set(False)


# @contextmanager 装饰器会把下面的生成器函数包装成上下文管理器。
#
# 被装饰后的函数可以这样使用：
#     with goal_mutation_permission(True):
#         do_something()
#
# 在进入 with 块时，会执行 yield 之前的代码；
# 在退出 with 块时，会执行 yield 之后的代码；
# 如果 yield 之后的逻辑放在 finally 中，则无论是否发生异常都会执行清理逻辑。
@contextmanager
# 定义目标权限上下文管理器。
# 这个函数本身是一个生成器函数，因为函数体内部使用了 yield。
def goal_mutation_permission(
    # allowed 参数是一个布尔值：
    #   True  -> 在该作用域内允许目标变更；
    #   False -> 在该作用域内禁止目标变更。
    allowed: bool,
# 原代码没有显式写返回类型注解。
# 被 @contextmanager 装饰后，它对外表现为一个上下文管理器对象。
):
    """Bind goal permission for one agent-run or direct tool execution scope.

    [中文翻译] 为一次 agent 运行或直接工具执行作用域绑定目标权限。
    """

    # 进入 with 块时首先执行这一行。
    #
    # _GOAL_MUTATION_ALLOWED.set(allowed) 的作用：
    #   把当前上下文中的“目标变更权限”设置为参数 allowed 指定的值。
    #
    # set(...) 会返回一个 token。
    # 这个 token 记录了设置之前的上下文状态，后面可以用它恢复原状态。
    # 这是 ContextVar 的标准用法：
    #   token = var.set(new_value)
    #   ...
    #   var.reset(token)
    token = _GOAL_MUTATION_ALLOWED.set(allowed)

    # try / finally 用于确保无论 with 块内部是否发生异常，
    # 最后都会恢复 ContextVar 原来的状态。
    try:
        # yield 是 Python 生成器的核心关键字。
        #
        # 在 @contextmanager 的语境下：
        #   yield 之前的代码相当于 with 块的“进入逻辑”（__enter__）；
        #   yield 之后的代码相当于 with 块的“退出逻辑”（__exit__）。
        #
        # 因此当用户写：
        #     with goal_mutation_permission(True):
        #         some_code()
        #
        # 执行顺序大致是：
        #   1. token = _GOAL_MUTATION_ALLOWED.set(True)
        #   2. 执行 some_code()
        #   3. finally 中恢复原来的权限状态
        #
        # 这里 yield 后面没有表达式，表示该上下文管理器不向 with ... as x 提供值。
        yield
    finally:
        # finally 块无论是否发生异常都会执行。
        #
        # _GOAL_MUTATION_ALLOWED.reset(token) 的作用：
        #   根据前面 set(allowed) 返回的 token，把当前上下文恢复到设置前的状态。
        #
        # 这一步非常关键：
        #   它保证权限只是临时绑定在当前 agent-run 或工具执行作用域内，
        #   作用域结束后不会泄漏到后续逻辑中。
        _GOAL_MUTATION_ALLOWED.reset(token)