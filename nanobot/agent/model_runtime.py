"""Public resolution boundary for default and overridden LLM runtimes.

中文：默认 LLM 运行时（runtime）以及被覆盖（override）的 LLM 运行时的公开解析边界。
这个模块主要负责：
1. 管理当前默认使用哪个 LLM 运行时；
2. 根据 model / model_preset / provider snapshot 解析出不可变运行时；
3. 给 Agent、SDK、命令层等提供统一的运行时解析入口；
4. 避免外部直接修改内部 AgentLoop 状态。
"""

# 导入 __future__ 中的 annotations 特性。
# 作用：让类型注解变成“延迟求值”的字符串形式。
# 好处：
# 1. 可以在类型注解中使用尚未定义的类；
# 2. 可以使用较新的类型写法，例如 `str | None`；
# 3. 减少运行时解析类型注解的开销。
from __future__ import annotations

# 从 collections.abc 导入两个抽象类型：
# Callable：表示“可调用对象”，例如函数、lambda、实现了 __call__ 的对象。
# Mapping：表示“只读映射”接口，dict 实现了该接口，但 Mapping 更强调只读取，不承诺可写入。
from collections.abc import Callable, Mapping

# 从 dataclasses 导入 replace 函数。
# replace 用于基于一个 dataclass 实例创建一个“新实例”，并替换其中某些字段。
# 例如：replace(old_runtime, model="gpt-x") 会返回新的 runtime，而不会修改 old_runtime。
# 这对保持 LLMRuntime 的不可变性非常重要。
from dataclasses import replace

# 从 types 导入 MappingProxyType。
# MappingProxyType 可以把一个普通 dict 包装成只读映射视图。
# 外部可以读取，但不能通过该视图修改内部 dict。
from types import MappingProxyType

# 从 typing 导入 cast。
# cast 不会在运行时做任何类型转换，它只是给静态类型检查器（如 mypy）看的。
# 例如 cast(object, model) 表示“请在类型检查时把 model 当成 object 看待”。
from typing import cast

# 导入 nanobot.agent.model_presets 模块，并取别名为 preset_helpers。
# 该模块提供和模型预设（preset）相关的辅助函数。
# 例如：规范化 preset 名称、构建 preset snapshot、计算默认选择签名等。
from nanobot.agent import model_presets as preset_helpers

# 导入配置相关类型：
# Config：完整配置对象，通常包含 provider、model、preset 等信息。
# ModelPresetConfig：单个模型预设的配置结构。
from nanobot.config.schema import Config, ModelPresetConfig

# 导入 provider 工厂相关类型和函数：
# ProviderSnapshot：表示某个 provider 在某一时刻的快照，例如当前 provider 配置、模型、签名、generation 等。
# build_provider_snapshot：根据 Config 和 preset 构建 ProviderSnapshot。
from nanobot.providers.factory import ProviderSnapshot, build_provider_snapshot

# 导入 LLM 运行时相关类型和函数：
# LLMRuntime：最终可执行的 LLM 运行时对象，通常包含 provider、model、generation、context window 等。
# runtime_from_provider_snapshot：把 ProviderSnapshot 转换成 LLMRuntime。
from nanobot.utils.llm_runtime import LLMRuntime, runtime_from_provider_snapshot


class ModelRuntimeResolver:
    """Own model selection and resolve it to immutable execution values.

    The resolver is deliberately independent of ``AgentLoop``.  Command, SDK,
    and tool admission layers can depend on this public service without reading
    or mutating private loop state.

    中文：拥有模型选择，并将其解析为不可变的执行值。

    该解析器刻意独立于 ``AgentLoop``。命令层、SDK 层和工具准入层可以依赖这个公共服务，
    而不需要读取或修改 AgentLoop 的私有内部状态。
    """

    def __init__(
        self,
        # initial_runtime：初始化时使用的默认 LLM 运行时。
        # 这个 resolver 一开始会把它作为当前默认运行时。
        initial_runtime: LLMRuntime,
        # 单独的 `*` 是 Python 的特殊语法：
        # 它后面的所有参数都必须使用“关键字参数”方式传入。
        # 例如必须写 model_presets=xxx，而不能按位置传入。
        *,
        # model_presets：初始模型预设集合。
        # key 是 preset 名称，value 是 ModelPresetConfig 配置。
        # 如果为 None，则内部会初始化为空 dict。
        model_presets: Mapping[str, ModelPresetConfig] | None = None,
        # preset_catalog_loader：预设目录加载器。
        # 它应该是一个可调用对象，调用后返回最新的 preset 映射。
        # 用于在需要时重新加载 preset catalog。
        preset_catalog_loader: preset_helpers.PresetCatalogLoader | None = None,
        # configured_default_preset：配置中指定的默认 preset 名称。
        # 如果系统配置里写了“默认使用某个 preset”，这里会传入该名称。
        configured_default_preset: str | None = None,
        # provider_snapshot_loader：provider 快照加载器。
        # 调用它会返回最新的 ProviderSnapshot。
        # 用于 refresh 时重新读取 provider 配置。
        provider_snapshot_loader: Callable[[], ProviderSnapshot] | None = None,
        # preset_snapshot_loader：preset 快照加载器。
        # 用于根据 preset 名称和配置生成 ProviderSnapshot。
        preset_snapshot_loader: preset_helpers.PresetSnapshotLoader | None = None,
    ) -> None:
        # 将传入的初始运行时保存到实例变量 _runtime。
        # _runtime 表示“当前默认运行时”。
        self._runtime = initial_runtime

        # 初始化模型预设字典。
        # `model_presets or {}` 表示：
        # 如果 model_presets 为 None，则使用空 dict；
        # 否则使用传入的 model_presets。
        # dict(...) 会复制一份新的字典，避免外部修改影响内部状态。
        self._model_presets = dict(model_presets or {})

        # 保存 preset catalog 加载器。
        # 后续如果需要刷新 preset 列表，会调用这个 loader。
        self._preset_catalog_loader = preset_catalog_loader

        # 标记：是否需要刷新 preset catalog。
        # False 表示当前 _model_presets 还是有效的。
        self._preset_catalog_refresh_required = False

        # 保存 provider snapshot 加载器。
        # refresh() 时会调用它获取最新 provider 快照。
        self._provider_snapshot_loader = provider_snapshot_loader

        # 保存 preset snapshot 加载器。
        # resolve_preset() 时可能用它生成 preset 对应的 provider snapshot。
        self._preset_snapshot_loader = preset_snapshot_loader

        # 标记：是否需要刷新当前默认运行时。
        # invalidate() 会把它设为 True。
        self._refresh_required = False

        # 已解析 preset 的缓存。
        # key 是规范化后的 preset 名称，value 是解析出的 LLMRuntime。
        # 使用缓存可以避免重复解析同一个 preset。
        self._resolved_presets: dict[str, LLMRuntime] = {}

        # 是否跟踪 provider 自身的 generation 变化。
        # 如果当前 runtime 不是来自 preset（model_preset is None），
        # 说明它是直接由 provider 默认配置驱动的，因此需要跟踪 provider generation。
        # 如果当前 runtime 来自 preset，则通常不再跟踪 provider generation。
        self._tracks_provider_generation = initial_runtime.model_preset is None

        # 计算当前默认选择的“签名”。
        # 这个签名用于后续 refresh 时判断：
        # 当前的默认选择是否仍然是配置中的默认选择。
        # snapshot_signature 表示 provider 快照的身份/状态签名；
        # configured_default_preset 表示配置中的默认 preset。
        self._default_selection_signature = preset_helpers.default_selection_signature(
            initial_runtime.snapshot_signature,
            configured_default_preset,
        )

    # @property 把方法包装成只读属性。
    # 外部使用 resolver.runtime 访问，而不是 resolver.runtime()。
    @property
    def runtime(self) -> LLMRuntime:
        """Return the current immutable default without refreshing configuration.

        中文：返回当前不可变的默认运行时，但不会刷新配置。
        """
        # 直接返回当前内部保存的默认运行时。
        # 注意：这里不会触发 refresh，也不会重新加载 provider 或 preset。
        return self._runtime

    # 将 model_presets 暴露为只读属性。
    @property
    def model_presets(self) -> Mapping[str, ModelPresetConfig]:
        # 在返回 presets 前，先检查是否需要刷新 preset catalog。
        # 如果 _preset_catalog_refresh_required 为 True，
        # _refresh_preset_catalog() 会重新调用 loader 加载 presets。
        self._refresh_preset_catalog()

        # 返回一个只读映射，防止外部修改内部 preset 字典。
        # MappingProxyType(...) 会把内部 dict 包装成只读视图。
        return MappingProxyType({
            # 字典推导式：遍历 self._model_presets 中的每个 preset。
            # name 是 preset 名称，preset 是 ModelPresetConfig 对象。
            name: preset.model_copy(deep=True)
            # preset.model_copy(deep=True)：
            # 复制 preset 配置对象，避免外部拿到内部对象后直接修改。
            # deep=True 表示深拷贝，尽可能复制内部嵌套字段。
            for name, preset in self._model_presets.items()
        })

    # 当前默认运行时所使用的 model preset 名称。
    @property
    def model_preset(self) -> str | None:
        # 如果当前 runtime 来自某个 preset，则返回 preset 名称；
        # 如果不是 preset 驱动，则通常为 None。
        return self._runtime.model_preset

    # 当前 provider 快照签名。
    @property
    def provider_signature(self) -> tuple[object, ...] | None:
        # snapshot_signature 通常用于判断 provider 配置是否发生变化。
        # 它可能是一个 tuple，包含 provider、model、配置版本等信息。
        return self._runtime.snapshot_signature

    def current(self, *, refresh: bool = False) -> LLMRuntime:
        """Return the selected runtime, optionally refreshing the default source.

        中文：返回当前选中的运行时；如果需要，可以先刷新默认来源。
        """
        # `*,` 表示 refresh 必须通过关键字传入，例如 current(refresh=True)。
        # 如果调用方要求刷新，则执行刷新逻辑。
        if refresh:
            # 调用 refresh()，尝试从 provider_snapshot_loader 重新加载默认运行时。
            # 如果没有变化，refresh() 返回 None；如果有变化，返回新的 runtime。
            self.refresh()

            # 再检查 provider generation 是否发生变化。
            # 这主要用于跟踪 provider 自身的更新，例如 provider 配置版本变更。
            self._refresh_provider_generation()

        # 返回当前默认运行时。
        return self._runtime

    def admit(self) -> LLMRuntime:
        """Resolve the immutable runtime for the next turn admission.

        中文：为下一轮“准入”解析出不可变的运行时。
        """
        # 如果之前调用过 invalidate()，则 _refresh_required 为 True。
        # 此时在真正使用前先刷新默认运行时。
        if self._refresh_required:
            self.refresh()

        # 即使没有显式 refresh，也检查 provider generation 是否需要更新。
        # 如果 provider 的 generation 变了，会生成一个新的 runtime。
        self._refresh_provider_generation()

        # 返回最终用于本轮执行的不可变运行时。
        return self._runtime

    def invalidate(self) -> None:
        """Refresh configured runtime state on the next admission.

        中文：在下一次准入时刷新已配置的运行时状态。
        """
        # 标记下次 admit/current 时需要刷新 runtime。
        self._refresh_required = True

        # 标记下次访问 preset 时需要重新加载 preset catalog。
        self._preset_catalog_refresh_required = True

        # 清空已解析 preset 缓存。
        # 因为配置可能已经变化，旧缓存可能失效。
        self._resolved_presets.clear()

    def _refresh_preset_catalog(self) -> None:
        # 内部方法：必要时刷新模型预设目录。
        # 如果刷新标志为 False，说明当前缓存仍然有效，直接返回。
        if not self._preset_catalog_refresh_required:
            return

        # 如果配置了 preset catalog loader，则调用它加载最新 presets。
        # loader 返回值通常是一个 Mapping[str, ModelPresetConfig]。
        if self._preset_catalog_loader is not None:
            # dict(...) 复制一份新的字典，避免外部直接持有内部对象。
            self._model_presets = dict(self._preset_catalog_loader())

        # 刷新完成后，清除刷新标志。
        self._preset_catalog_refresh_required = False

    def resolve_snapshot(
        self,
        snapshot: ProviderSnapshot,
    ) -> LLMRuntime:
        """Resolve a factory snapshot without changing the selected default.

        中文：解析一个 provider 工厂快照，但不会改变当前选中的默认运行时。
        """
        # runtime_from_provider_snapshot 会根据 ProviderSnapshot 创建 LLMRuntime。
        # 这里只是“解析”，不修改 self._runtime。
        return runtime_from_provider_snapshot(snapshot)

    def adopt_snapshot(
        self,
        snapshot: ProviderSnapshot,
    ) -> LLMRuntime:
        """Select a snapshot as the default for future turns.

        中文：把一个 provider 快照选为未来轮次的默认运行时。
        """
        # 先把 snapshot 解析成 LLMRuntime。
        runtime = self.resolve_snapshot(snapshot)

        # 将当前默认运行时替换为新的 runtime。
        self._runtime = runtime

        # 如果新的 runtime 没有关联 model preset，
        # 说明它是 provider 直接驱动的默认值，因此需要跟踪 provider generation。
        # 如果它来自 preset，则不跟踪 provider generation。
        self._tracks_provider_generation = runtime.model_preset is None

        # 更新“默认选择签名”。
        # 后续 refresh 时会用它判断当前默认选择是否仍然是原始配置默认。
        self._default_selection_signature = preset_helpers.default_selection_signature(
            runtime.snapshot_signature,
            runtime.model_preset,
        )

        # 返回新的默认运行时。
        return runtime

    def resolve_preset(self, name: str | None) -> LLMRuntime:
        """Resolve a named preset without changing the selected default.

        中文：解析一个具名 preset，但不会改变当前选中的默认运行时。
        """
        # 在解析 preset 前，先确保 preset catalog 是最新的。
        self._refresh_preset_catalog()

        # 规范化 preset 名称。
        # 例如处理 None、大小写、别名、默认值等情况。
        # 返回值 normalized 是内部统一使用的 preset 名称。
        normalized = preset_helpers.normalize_preset_name(name, self._model_presets)

        # 先查缓存，看该 preset 是否已经解析过。
        cached = self._resolved_presets.get(normalized)

        # 如果缓存命中，则直接返回缓存结果，避免重复构建。
        if cached is not None:
            return cached

        # 如果缓存未命中，则构建该 preset 对应的 ProviderSnapshot。
        snapshot = preset_helpers.build_runtime_preset_snapshot(
            # 使用规范化后的 preset 名称。
            name=normalized,
            # 当前可用的全部 preset 配置。
            presets=self._model_presets,
            # 当前 runtime 的 provider。
            # 如果 preset 没有显式指定 provider，可能会使用该 provider 作为基础。
            provider=self._runtime.provider,
            # 可选的 preset snapshot loader，用于根据 preset 生成 snapshot。
            loader=self._preset_snapshot_loader,
        )

        # 将 ProviderSnapshot 解析为 LLMRuntime。
        runtime = self.resolve_snapshot(snapshot)

        # 把解析结果写入缓存，供后续调用使用。
        self._resolved_presets[normalized] = runtime

        # 返回解析出的 runtime。
        return runtime

    def select_preset(self, name: str | None) -> LLMRuntime:
        """Select a named preset as the default for future turns.

        中文：选择一个具名 preset 作为未来轮次的默认运行时。
        """
        # 先解析 preset，得到对应的 LLMRuntime。
        runtime = self.resolve_preset(name)

        # 将当前默认运行时切换为该 preset 对应的 runtime。
        self._runtime = runtime

        # 因为当前默认值现在来自 preset，而不是 provider 直接默认值，
        # 所以不再跟踪 provider generation。
        self._tracks_provider_generation = False

        # 返回新的默认运行时。
        return runtime

    def select_model(self, model: str) -> LLMRuntime:
        """Change the default model without reconstructing downstream consumers.

        中文：修改默认模型，但不重建下游消费者。
        """
        # 运行时类型检查：确保传入的 model 是非空字符串。
        #
        # 这里使用 cast(object, model) 的原因：
        # 从静态类型看，model 已经被标注为 str；
        # 但运行时可能有人传入错误类型。
        # cast(object, model) 告诉类型检查器“把 model 当 object 看”，
        # 这样 isinstance(...) 检查在类型层面也更合理。
        #
        # `not model.strip()` 用于判断字符串是否只包含空白字符。
        # 如果 model 不是 str，或者 strip 后为空，就抛出 ValueError。
        if not isinstance(cast(object, model), str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        # 使用 dataclasses.replace 基于当前 runtime 创建新 runtime。
        # 这样不会修改原对象，而是返回新的不可变对象。
        self._runtime = replace(
            # 被替换的原始对象。
            self._runtime,
            # 将 model 字段替换为去除首尾空白后的新模型名。
            model=model.strip(),
            # 因为现在是直接指定 model，而不是通过 preset 选择，
            # 所以清除 model_preset。
            model_preset=None,
        )

        # 返回更新后的默认运行时。
        return self._runtime

    def select_context_window(self, context_window_tokens: int) -> LLMRuntime:
        """Change the default context limit for future admissions.

        中文：修改未来准入时使用的默认上下文窗口限制。
        """
        # 同样使用 cast(object, ...) 方便做运行时类型检查。
        # 静态类型标注是 int，但运行时可能传入错误类型。
        raw_context_window = cast(object, context_window_tokens)

        # 检查 context_window_tokens 是否是整数。
        #
        # 注意：在 Python 中 bool 是 int 的子类，
        # 所以 isinstance(True, int) 也会返回 True。
        # 但业务上 True/False 不应该被当成 context window token 数量，
        # 因此需要额外排除 bool。
        if not isinstance(raw_context_window, int) or isinstance(
            raw_context_window,
            bool,
        ):
            raise TypeError("context_window_tokens must be an integer")

        # 使用 replace 创建新的 runtime，并更新 context_window_tokens 字段。
        self._runtime = replace(
            self._runtime,
            context_window_tokens=context_window_tokens,
        )

        # 返回更新后的默认运行时。
        return self._runtime

    def _refresh_provider_generation(self) -> LLMRuntime | None:
        """Adopt direct provider-default changes only for provider-backed defaults.

        中文：仅当默认值由 provider 直接驱动时，才采纳 provider 默认值的 generation 变化。
        """
        # 如果当前默认值不是 provider 直接驱动，例如来自 preset，
        # 则不跟踪 provider generation，直接返回 None。
        if not self._tracks_provider_generation:
            return None

        # 取出当前 runtime，便于后续读取字段。
        runtime = self._runtime

        # 调用 LLMRuntime.capture 重新捕获 provider 的当前状态。
        # 这里传入当前 runtime 的 provider、model、context window、preset、signature，
        # 以便 capture 根据这些上下文判断当前 provider generation。
        captured = LLMRuntime.capture(
            runtime.provider,
            runtime.model,
            context_window_tokens=runtime.context_window_tokens,
            model_preset=runtime.model_preset,
            snapshot_signature=runtime.snapshot_signature,
        )

        # 如果捕获到的 generation 和当前 runtime 的 generation 相同，
        # 说明 provider 没有发生代际变化，无需更新。
        if captured.generation == runtime.generation:
            return None

        # 如果 generation 发生变化，则基于当前 runtime 创建新 runtime，
        # 只更新 generation 字段。
        self._runtime = replace(runtime, generation=captured.generation)

        # 返回更新后的 runtime。
        return self._runtime

    def refresh(self) -> LLMRuntime | None:
        """Refresh configured defaults and return the replacement when changed.

        中文：刷新已配置的默认值；如果默认值发生变化，则返回新的运行时。
        """
        # 如果没有 provider snapshot loader，则无法从外部重新加载 provider 配置。
        # 此时直接把刷新标志清掉，并返回 None 表示没有变化。
        if self._provider_snapshot_loader is None:
            self._refresh_required = False
            return None

        # 刷新 provider 默认值时，preset 缓存也可能失效，因此清空。
        self._resolved_presets.clear()

        # 调用 loader 获取最新的 ProviderSnapshot。
        # snapshot 表示当前 provider 配置/状态的最新视图。
        snapshot = self._provider_snapshot_loader()

        # 根据最新 snapshot 计算“配置默认选择签名”。
        # 这个签名用于判断配置层面的默认选择是否变化。
        default_selection = preset_helpers.default_selection_signature(
            snapshot.signature,
            snapshot.model_preset,
        )

        # 当前 runtime 正在使用的 preset 名称。
        # 如果当前 runtime 是直接 model/provider 驱动，则 active_preset 可能为 None。
        active_preset = self._runtime.model_preset

        # 判断是否应继续解析当前活动 preset：
        # 条件一：active_preset 非空，说明当前默认值来自 preset；
        # 条件二：self._default_selection_signature in (None, default_selection)
        #     表示之前记录的默认选择签名要么是未知（None），要么和最新默认选择一致。
        # 如果两个条件都满足，说明当前 preset 仍然应该被视为默认选择，
        # 因此重新解析该 preset。
        if active_preset and self._default_selection_signature in (None, default_selection):
            runtime = self.resolve_preset(active_preset)
        else:
            # 否则，直接采用最新 provider snapshot 解析出的 runtime。
            runtime = self.resolve_snapshot(snapshot)

        # 判断刷新后的 runtime 是否和当前 runtime 实质相同。
        # 这里比较两个关键字段：
        # 1. snapshot_signature：provider 快照签名是否相同；
        # 2. model_preset：是否来自同一个 preset。
        unchanged = (
            runtime.snapshot_signature == self._runtime.snapshot_signature
            and runtime.model_preset == self._runtime.model_preset
        )

        # 无论是否变化，本次 refresh 都已完成，因此清除刷新标志。
        self._refresh_required = False

        # 如果没有变化，则更新默认选择签名，但不替换当前 runtime。
        if unchanged:
            self._default_selection_signature = default_selection
            # 返回 None 表示没有产生新的默认运行时。
            return None

        # 如果发生变化，则使用元组解包赋值，同时更新多个状态：
        # self._runtime：新的默认运行时；
        # self._tracks_provider_generation：是否跟踪 provider generation；
        # self._default_selection_signature：新的默认选择签名。
        (
            self._runtime,
            self._tracks_provider_generation,
            self._default_selection_signature,
        ) = (
            # 新的运行时对象。
            runtime,
            # 如果新 runtime 没有 model_preset，说明它是 provider 直接驱动，
            # 因此继续跟踪 provider generation。
            runtime.model_preset is None,
            # 记录最新默认选择签名。
            default_selection,
        )

        # 返回新的默认运行时，表示 refresh 导致了变化。
        return runtime

    def resolve_override(
        self,
        # `*,` 后面的参数都必须使用关键字传参。
        *,
        # model：单次运行覆盖使用的模型名。
        model: str | None,
        # model_preset：单次运行覆盖使用的 preset 名称。
        model_preset: str | None,
        # config：可选配置对象。
        # 如果提供 config，则可以通过完整配置构建 provider snapshot。
        config: Config | None = None,
    ) -> LLMRuntime | None:
        """Resolve an SDK-style per-run override without mutating the default.

        中文：解析 SDK 风格的“单次运行覆盖”，但不会修改默认运行时。
        """
        # model 和 model_preset 互斥。
        # 一次覆盖不能同时指定“具体模型”和“预设”。
        if model is not None and model_preset is not None:
            raise ValueError("model and model_preset are mutually exclusive")

        # 如果指定了 model_preset，则直接解析该 preset。
        # 注意：这里调用 resolve_preset，不会修改默认 runtime。
        if model_preset is not None:
            return self.resolve_preset(model_preset)

        # 如果既没有指定 model，也没有指定 model_preset，
        # 则没有覆盖可解析，返回 None。
        if model is None:
            return None

        # 如果没有提供完整 Config，则只能基于当前 runtime 构造一个轻量覆盖。
        if config is None:
            # 直接创建一个新的 LLMRuntime：
            # provider、generation、context_window_tokens 沿用当前默认值；
            # model 使用覆盖传入的 model；
            # snapshot_signature 使用一个特殊元组 ("model_override", model)，
            # 表示这是一个仅由 model 覆盖产生的运行时。
            return LLMRuntime(
                provider=self._runtime.provider,
                model=model,
                generation=self._runtime.generation,
                context_window_tokens=self._runtime.context_window_tokens,
                snapshot_signature=("model_override", model),
            )

        # 如果提供了 Config，则走完整配置解析路径。
        #
        # self.model_preset 是 property，返回当前默认 runtime 对应的 preset 名称。
        # config.resolve_preset(...) 会根据该 preset 名称取出基础 preset 配置。
        base = config.resolve_preset(self.model_preset)

        # 基于 base preset 配置复制一份，并更新两个字段：
        # model：覆盖为调用方指定的模型；
        # provider：设为 "auto"，表示让 provider 解析逻辑自动选择 provider。
        #
        # model_copy(update=...) 常见于 pydantic 模型，
        # 表示复制模型并应用字段更新。
        preset = base.model_copy(update={"model": model, "provider": "auto"})

        # 根据完整 Config 和更新后的 preset 构建 ProviderSnapshot，
        # 然后解析为 LLMRuntime 返回。
        # 该结果只用于本次覆盖，不会改变 resolver 的默认 runtime。
        return self.resolve_snapshot(build_provider_snapshot(config, preset=preset))