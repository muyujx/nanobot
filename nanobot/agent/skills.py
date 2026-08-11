"""Skills loader for agent capabilities.

中文翻译：用于加载 Agent 能力（skills）的加载器。
"""

import json
# 中文：导入 json 模块，用于解析 JSON 字符串，例如把 metadata 里的 JSON 字符串转成 Python dict。

import os
# 中文：导入 os 模块，用于访问环境变量，例如检查某个 env 是否存在。

import re
# 中文：导入 re 模块，用于正则表达式匹配，例如解析 frontmatter 和 $skill 引用。

import shutil
# 中文：导入 shutil 模块，用于查找系统命令行工具是否存在，例如 shutil.which。

from pathlib import Path
# 中文：从 pathlib 导入 Path 类，用于以面向对象方式处理文件路径。

from typing import Any, cast
# 中文：从 typing 导入 Any 和 cast。
# Any 表示任意类型；cast 用于给类型检查器“声明类型”，运行时不做实际转换。

import yaml
# 中文：导入 yaml 模块，用于解析 SKILL.md frontmatter 里的 YAML 内容。

# Default builtin skills directory (relative to this file)
# 中文翻译：默认内置 skills 目录（相对于当前这个文件）。
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"
# 中文：BUILTIN_SKILLS_DIR 是默认内置技能目录。
# __file__ 是当前 Python 文件路径。
# Path(__file__) 把字符串路径转换成 Path 对象。
# .parent 获取当前文件所在目录。
# 再一个 .parent 获取上一级目录。
# / "skills" 是 Path 重载的路径拼接运算符，等价于拼接出上一级目录下的 skills 路径。

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
# 中文翻译：匹配开头的 ---，中间的 YAML 正文（第 1 个捕获组），以及单独一行结束的 ---；支持 CRLF 换行。
_STRIP_SKILL_FRONTMATTER = re.compile(
    # 中文：正则表达式主体：
    # ^--- 匹配文档开头的三个短横线。
    # \s* 匹配任意空白字符，例如空格或制表符。
    # \r?\n 匹配换行，兼容 Windows 的 CRLF（\r\n）和 Unix 的 LF（\n）。
    # (.*?) 非贪婪捕获任意内容，作为 YAML frontmatter 正文，捕获组编号为 1。
    # \r?\n--- 匹配 YAML 正文后的换行和结束分隔符 ---。
    # \s*\r?\n? 匹配结束后可能存在的空白和换行。
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    # 中文：re.DOTALL 让 . 也能匹配换行符，否则 . 默认不匹配换行。
    re.DOTALL,
)
# 中文：_STRIP_SKILL_FRONTMATTER 用于从 Markdown 文件顶部剥离 YAML frontmatter。

_SKILL_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]+)")
# 中文：_SKILL_REFERENCE 用于匹配文本中的显式技能引用，例如 $skill-name。
# (?<![\w$]) 是负向后顾断言，表示 $ 前面不能是字母、数字、下划线或 $。
# \$ 匹配字面量美元符号 $。
# ([A-Za-z0-9_-]+) 捕获技能名，允许字母、数字、下划线和短横线。


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.

    中文翻译：Agent 技能加载器。

    Skills 是 Markdown 文件（SKILL.md），用于教 Agent 如何使用特定工具或完成特定任务。
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None, disabled_skills: set[str] | None = None):
        # 中文：初始化 SkillsLoader。
        # workspace：当前工作区路径。
        # builtin_skills_dir：可选的内置技能目录，如果为 None 则使用默认目录。
        # disabled_skills：可选的禁用技能名集合，如果为 None 则使用空集合。
        self.workspace = workspace
        # 中文：保存当前工作区根目录，后续用于定位 workspace 下的 skills。

        self.workspace_skills = workspace / "skills"
        # 中文：工作区技能目录，位于 workspace/skills。
        # 这里使用 Path 的 / 运算符拼接路径。

        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        # 中文：内置技能目录。
        # `builtin_skills_dir or BUILTIN_SKILLS_DIR` 表示如果传入了非 None/非空值就使用传入值，否则使用默认值。

        self.disabled_skills = disabled_skills or set()
        # 中文：禁用技能集合。
        # 使用 `or set()` 避免把 None 当作集合使用，也避免默认参数使用可变对象 set() 的常见坑。

    def _skill_entries_from_dir(self, base: Path, source: str, *, skip_names: set[str] | None = None) -> list[dict[str, str]]:
        # 中文：从指定目录扫描技能条目。
        # base：技能根目录，例如 workspace/skills 或内置 skills 目录。
        # source：来源标识，例如 "workspace" 或 "builtin"。
        # skip_names：关键字参数，表示需要跳过的技能名集合。
        # `*` 后面的参数必须以关键字方式传递，例如必须写 skip_names=...。
        if not base.exists():
            # 中文：如果目录不存在，直接返回空列表。
            return []

        entries: list[dict[str, str]] = []
        # 中文：entries 用于保存扫描到的技能信息。
        # 每个技能是一个 dict，包含 name、path、source 三个字段。

        for skill_dir in base.iterdir():
            # 中文：遍历 base 目录下的所有条目。
            # base.iterdir() 返回一个生成器，生成当前目录中的 Path 对象。

            if not skill_dir.is_dir():
                # 中文：如果当前条目不是目录，跳过。
                # skill 目录通常应该是一个文件夹。
                continue

            skill_file = skill_dir / "SKILL.md"
            # 中文：技能定义文件固定为目录下的 SKILL.md。
            # 例如 skills/demo/SKILL.md。

            if not skill_file.exists():
                # 中文：如果 SKILL.md 不存在，则这个目录不是有效 skill，跳过。
                continue

            name = skill_dir.name
            # 中文：技能名使用目录名。
            # 例如 skills/foo/SKILL.md 的技能名是 foo。

            if skip_names is not None and name in skip_names:
                # 中文：如果指定了 skip_names，并且当前技能名在跳过集合中，则跳过。
                # 这里使用短路逻辑：如果 skip_names 是 None，就不会执行 name in skip_names。
                continue

            entries.append({"name": name, "path": str(skill_file), "source": source})
            # 中文：把当前技能信息加入结果列表。
            # name：技能名。
            # path：SKILL.md 文件路径，转成字符串方便序列化或展示。
            # source：来源，workspace 或 builtin。

        return entries
        # 中文：返回当前目录扫描到的所有有效技能条目。

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.

        中文翻译：列出所有可用技能。

        参数：
            filter_unavailable：如果为 True，则过滤掉依赖要求未满足的技能。

        返回：
            技能信息字典列表，每个字典包含 'name'、'path'、'source'。
        """
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        # 中文：先扫描工作区 skills 目录。
        # workspace 技能优先级更高，因此先加载。

        workspace_names = {entry["name"] for entry in skills}
        # 中文：收集工作区中已经存在的技能名集合。
        # 这是集合推导式，用于快速判断某个内置技能是否被工作区技能覆盖。

        if self.builtin_skills and self.builtin_skills.exists():
            # 中文：如果内置技能目录存在，则继续扫描内置技能。
            # `self.builtin_skills` 非 None/非空，并且目录真实存在。
            skills.extend(
                self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=workspace_names)
            )
            # 中文：扫描内置技能目录，并跳过与工作区同名的技能。
            # skills.extend(...) 会把另一个列表中的元素追加到 skills 末尾。
            # 这样实现“工作区技能覆盖内置技能”的效果。

        if self.disabled_skills:
            # 中文：如果配置了禁用技能集合，则过滤掉被禁用的技能。
            skills = [s for s in skills if s["name"] not in self.disabled_skills]
            # 中文：列表推导式，只保留名字不在 disabled_skills 中的技能。

        if filter_unavailable:
            # 中文：如果要求过滤不可用技能，则检查每个技能的运行要求。
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
            # 中文：只保留满足依赖要求的技能。
            # skill["name"] 获取技能名。
            # self._get_skill_meta(...) 获取技能元数据。
            # self._check_requirements(...) 判断 bins/env 要求是否满足。

        return skills
        # 中文：如果不需要过滤不可用技能，直接返回全部技能列表。

    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.

        中文翻译：根据名称加载技能。

        参数：
            name：技能名，也就是技能目录名。

        返回：
            技能文件内容；如果找不到则返回 None。
        """
        roots = [self.workspace_skills]
        # 中文：roots 是查找技能文件的根目录列表。
        # 优先查找 workspace skills。

        if self.builtin_skills:
            # 中文：如果存在内置技能目录，也加入查找路径。
            roots.append(self.builtin_skills)
            # 中文：append 把内置 skills 目录追加到 roots 列表末尾。

        for root in roots:
            # 中文：依次在每个根目录中查找同名技能。
            path = root / name / "SKILL.md"
            # 中文：拼接技能文件路径：root/技能名/SKILL.md。

            if path.exists():
                # 中文：如果技能文件存在，则读取并返回。
                return path.read_text(encoding="utf-8")
                # 中文：Path.read_text 读取文本文件内容。
                # encoding="utf-8" 指定使用 UTF-8 编码，避免系统默认编码造成问题。

        return None
        # 中文：所有目录都没找到时返回 None。

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.

        中文翻译：加载指定技能，以便放入 Agent 上下文中。

        参数：
            skill_names：需要加载的技能名列表。

        返回：
            格式化后的技能内容。
        """
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        # 中文：parts 是一个列表推导式生成的 Markdown 片段列表。
        # for name in skill_names：遍历要加载的技能名。
        # `markdown := self.load_skill(name)` 是 Python 海象运算符（assignment expression）。
        # 它的作用是在表达式内部调用 load_skill，并把返回值赋给 markdown。
        # 如果 load_skill 返回 None，则条件为 False，该技能被跳过。
        # f-string 用于格式化字符串：
        # - 插入技能标题 "### Skill: {name}"
        # - 插入去除 frontmatter 后的 Markdown 正文。
        # self._strip_frontmatter(markdown) 用于去掉 YAML frontmatter。

        return "\n\n---\n\n".join(parts)
        # 中文：把多个技能片段用 Markdown 分隔线连接成一个完整字符串。
        # join 前面的字符串是分隔符。

    def get_explicitly_invoked_skills(self, text: str) -> list[str]:
        """Resolve ``$skill-name`` references to enabled, available skills.

        中文翻译：解析文本中的 ``$skill-name`` 引用，并返回已启用且可用的技能名。
        """
        if not text:
            # 中文：如果输入文本为空字符串、None 或其他假值，直接返回空列表。
            return []

        available = {
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
        }
        # 中文：available 是当前可用技能名集合。
        # list_skills(filter_unavailable=True) 会过滤掉禁用技能和不满足依赖的技能。
        # 集合推导式只保留技能名字段。

        invoked: list[str] = []
        # 中文：invoked 保存文本中显式引用且可用的技能名。
        # 使用列表而不是集合，是为了保留首次出现顺序。

        for match in _SKILL_REFERENCE.finditer(text):
            # 中文：使用正则表达式在文本中查找所有 $skill-name 引用。
            # finditer 返回一个迭代器，逐个产生 Match 对象。

            name = match.group(1)
            # 中文：match.group(1) 获取正则第 1 个捕获组，也就是技能名本身。

            if name in available and name not in invoked:
                # 中文：只保留可用技能，并避免重复添加。
                invoked.append(name)
                # 中文：把技能名加入结果列表。

        return invoked
        # 中文：返回解析出的技能引用列表。

    def build_skills_summary(self, exclude: set[str] | None = None) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.

        Args:
            exclude: Set of skill names to omit from the summary.

        Returns:
            Markdown-formatted skills summary.

        中文翻译：构建所有技能的摘要（名称、描述、路径、可用性）。

        这用于渐进式加载——Agent 可以在需要时使用 read_file 读取完整技能内容。

        参数：
            exclude：需要从摘要中排除的技能名集合。

        返回：
            Markdown 格式的技能摘要。
        """
        all_skills = self.list_skills(filter_unavailable=False)
        # 中文：获取所有技能，包括不可用技能。
        # 摘要需要展示 unavailable 原因，所以这里不过滤不可用技能。

        if not all_skills:
            # 中文：如果没有技能，返回空字符串。
            return ""

        sections: list[str] = []
        # 中文：sections 保存最终要拼接的 Markdown 分组段落。

        groups = (
            ("Workspace skills", "workspace", self.workspace_skills),
            ("Built-in skills", "builtin", self.builtin_skills),
        )
        # 中文：groups 是一个元组，包含两个分组：
        # 第一个元素：分组标题。
        # 第二个元素：技能来源标识，与 entry["source"] 对应。
        # 第三个元素：技能根目录，用于展示路径和计算相对路径。

        for label, source, root in groups:
            # 中文：遍历每个分组。
            # 这里使用元组解包：label、source、root 分别对应 groups 中每个三元组的三个值。

            entries = [
                entry
                for entry in all_skills
                if entry["source"] == source and (not exclude or entry["name"] not in exclude)
            ]
            # 中文：筛选当前分组的技能条目。
            # entry["source"] == source：只保留当前来源。
            # `(not exclude or entry["name"] not in exclude)`：
            # - 如果 exclude 为 None 或空集合，则不排除任何技能。
            # - 否则只保留不在 exclude 中的技能。

            if not entries:
                # 中文：如果当前分组没有技能，跳过该分组。
                continue

            lines = [f"### {label} (`{root.expanduser().resolve()}`)"]
            # 中文：lines 保存当前分组的 Markdown 行。
            # 第一行是分组标题，并显示技能根目录绝对路径。
            # root.expanduser() 展开路径中的 ~，例如 /home/user。
            # root.resolve() 解析成绝对路径并规范化。
            # f-string 中的反引号用于在 Markdown 中显示代码样式。

            for entry in entries:
                # 中文：遍历当前分组中的每个技能。
                skill_name = entry["name"]
                # 中文：技能名。

                meta = self._get_skill_meta(skill_name)
                # 中文：获取技能的 nanobot/openclaw 元数据。
                # 该元数据通常来自 frontmatter 的 metadata 字段。

                available = self._check_requirements(meta)
                # 中文：检查该技能是否满足 bins/env 要求。
                # available 是 bool 类型。

                desc = self.get_skill_description(skill_name)
                # 中文：获取技能描述。
                # 如果 frontmatter 没有 description，会回退为技能名。

                suffix = ""
                # 中文：suffix 用于在技能不可用时追加说明。

                if not available:
                    # 中文：如果技能不可用，则计算缺失依赖。
                    missing = self._get_missing_requirements(meta)
                    # 中文：missing 是缺失依赖的可读描述，例如 "CLI: git, ENV: API_KEY"。

                    suffix = f" (unavailable: {missing})" if missing else " (unavailable)"
                    # 中文：如果 missing 非空，显示缺失内容；否则只显示 unavailable。
                    # 这是 Python 条件表达式：值A if 条件 else 值B。

                relative_path = Path(entry["path"]).relative_to(root).as_posix()
                # 中文：计算技能文件相对于当前分组根目录的路径。
                # Path(entry["path"]) 把字符串路径转成 Path。
                # relative_to(root) 得到相对路径。
                # as_posix() 把路径分隔符统一转换成 /，方便跨平台展示。

                lines.append(f"- **{skill_name}** — {desc}{suffix}  `{relative_path}`")
                # 中文：追加一行 Markdown 列表项。
                # 包含技能名、描述、不可用后缀、相对路径。

            sections.append("\n".join(lines))
            # 中文：把当前分组所有行用换行符连接成一个字符串，并加入 sections。

        return "\n\n".join(sections)
        # 中文：把所有分组之间用空行分隔，返回最终 Markdown 摘要。

    @staticmethod
    def _requirement_lists(skill_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (bins, env) lists from skill metadata, tolerating null/wrong shapes.

        中文翻译：从技能元数据中返回 (bins, env) 列表，并容忍 null 或错误结构。
        """
        requires = cast(dict[str, Any], skill_meta.get("requires") or {})
        # 中文：从 skill_meta 中读取 requires 字段。
        # skill_meta.get("requires") 可能返回 None。
        # `or {}` 保证后续可以按 dict 处理。
        # cast(dict[str, Any], ...) 只是告诉类型检查器这个值可以看作 dict，运行时不会真的转换。

        if not isinstance(skill_meta.get("requires") or {}, dict):
            # 中文：再次检查 requires 是否是 dict。
            # 如果原始 requires 不是 dict，比如是字符串或列表，则返回空要求。
            return [], []

        bins_raw: object = requires.get("bins") or []
        # 中文：读取 requires.bins，表示需要的命令行工具列表。
        # 类型标注为 object，表示暂时不知道具体类型，需要后续检查。
        # 如果不存在或为 None，则使用空列表。

        env_raw: object = requires.get("env") or []
        # 中文：读取 requires.env，表示需要的环境变量列表。
        # 同样先作为未知类型 object 处理。

        bins = [value for value in cast(list[object], bins_raw) if isinstance(value, str) and value.strip()] if isinstance(bins_raw, list) else []
        # 中文：把 bins_raw 规范化为字符串列表。
        # 如果 bins_raw 不是 list，则返回空列表。
        # 如果是 list：
        # - cast(list[object], bins_raw) 告诉类型检查器可以把它看作列表。
        # - isinstance(value, str) 只保留字符串元素。
        # - value.strip() 判断字符串去除空白后是否非空。
        # 这是“条件表达式 + 列表推导式”的组合。

        env = [value for value in cast(list[object], env_raw) if isinstance(value, str) and value.strip()] if isinstance(env_raw, list) else []
        # 中文：把 env_raw 规范化为字符串列表，逻辑与 bins 相同。

        return bins, env
        # 中文：返回两个列表：需要的命令列表、需要的环境变量列表。

    def _get_missing_requirements(self, skill_meta: dict[str, Any]) -> str:
        """Get a description of missing requirements.

        中文翻译：获取缺失依赖要求的描述。
        """
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        # 中文：从技能元数据中解析出需要的命令和环境变量。
        # 这里使用元组解包。

        return ", ".join(
            [f"CLI: {command_name}" for command_name in required_bins if not shutil.which(command_name)]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )
        # 中文：返回缺失依赖的拼接字符串。
        # 第一个列表推导式：检查每个命令是否存在。
        # shutil.which(command_name) 会在 PATH 中查找可执行文件，找不到返回 None。
        # 第二个列表推导式：检查每个环境变量是否存在。
        # os.environ.get(env_name) 如果不存在返回 None；如果为空字符串也被视为未设置/不可用。
        # 两个列表用 + 合并，然后用 ", ".join(...) 拼接成一段文本。

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """Return whether a skill can run and why not when it cannot.

        中文翻译：返回某个技能是否可运行；如果不可运行，则返回原因。
        """
        meta = self._get_skill_meta(name)
        # 中文：获取技能元数据。

        available = self._check_requirements(meta)
        # 中文：检查技能依赖是否满足。

        return available, "" if available else self._get_missing_requirements(meta)
        # 中文：返回二元组。
        # 第一个值是 bool，表示是否可用。
        # 第二个值是字符串，如果可用则为空；如果不可用则返回缺失依赖说明。

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        """Return explicit command/env requirements and currently missing entries.

        中文翻译：返回显式声明的命令/环境依赖，以及当前缺失的条目。
        """
        bins, env = self._requirement_lists(self._get_skill_meta(name))
        # 中文：获取该技能声明的 bins 和 env 列表。

        return {
            "bins": bins,
            # 中文：技能声明需要的全部命令。
            "env": env,
            # 中文：技能声明需要的全部环境变量。
            "missing_bins": [value for value in bins if not shutil.which(value)],
            # 中文：当前缺失的命令列表。
            # shutil.which(value) 找不到命令时返回 None，因此会被保留。
            "missing_env": [value for value in env if not os.environ.get(value)],
            # 中文：当前缺失的环境变量列表。
            # os.environ.get(value) 不存在或为空时被视为缺失。
        }

    def get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter.

        中文翻译：从技能的 frontmatter 中获取描述。
        """
        meta = self.get_skill_metadata(name)
        # 中文：读取技能 frontmatter 中的完整 metadata dict。

        description = meta.get("description") if meta else None
        # 中文：如果 meta 存在，则读取 description 字段；否则置为 None。
        # 这是条件表达式，避免对 None 调用 .get。

        if isinstance(description, str) and description:
            # 中文：如果 description 是非空字符串，则返回它。
            return description

        return name  # Fallback to skill name
        # 中文翻译：回退为技能名。
        # 如果没有有效描述，就用技能名作为描述。

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content.

        中文翻译：从 Markdown 内容中移除 YAML frontmatter。
        """
        if not content.startswith("---"):
            # 中文：如果内容不以 --- 开头，说明没有 frontmatter，直接返回原内容。
            return content

        match = _STRIP_SKILL_FRONTMATTER.match(content)
        # 中文：使用预编译正则匹配 frontmatter。
        # match 只从字符串开头匹配。

        if match:
            # 中文：如果匹配成功，则跳过匹配到的 frontmatter 部分。
            return content[match.end():].strip()
            # 中文：match.end() 返回匹配结束位置的下标。
            # content[match.end():] 是字符串切片，取 frontmatter 后面的正文。
            # .strip() 去掉首尾多余空白。

        return content
        # 中文：如果正则没有匹配成功，返回原内容。

    def _parse_nanobot_metadata(self, raw: object) -> dict[str, Any]:
        """Extract nanobot/openclaw metadata from a frontmatter field.

        ``raw`` may be a dict (already parsed by yaml.safe_load) or a JSON str.

        中文翻译：从 frontmatter 字段中提取 nanobot/openclaw 元数据。

        ``raw`` 可能是 dict（已经由 yaml.safe_load 解析），也可能是 JSON 字符串。
        """
        if isinstance(raw, dict):
            # 中文：如果 raw 已经是 dict，直接使用。
            data = cast(dict[str, Any], raw)
            # 中文：cast 用于类型检查，告诉类型检查器 data 是 dict[str, Any]。

        elif isinstance(raw, str):
            # 中文：如果 raw 是字符串，尝试按 JSON 解析。
            try:
                data = json.loads(raw)
                # 中文：json.loads 把 JSON 字符串解析成 Python 对象。
            except (json.JSONDecodeError, TypeError):
                # 中文：如果 JSON 解析失败，或者类型不支持解析，则返回空 dict。
                return {}
        else:
            # 中文：如果 raw 既不是 dict 也不是 str，返回空 dict。
            return {}

        if not isinstance(data, dict):
            # 中文：JSON 解析结果也可能是 list、str、number 等。
            # 这里只接受 dict，否则返回空 dict。
            return {}

        data_object = cast(dict[str, Any], data)
        # 中文：再次用 cast 明确 data_object 类型，方便后续类型检查。

        payload = data_object.get("nanobot", data_object.get("openclaw", {}))
        # 中文：优先读取 nanobot 字段。
        # 如果没有 nanobot，则读取 openclaw 字段。
        # 如果 openclaw 也没有，则默认空 dict。

        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}
        # 中文：如果 payload 是 dict，则返回它；否则返回空 dict。

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        """Check if skill requirements are met (bins, env vars).

        中文翻译：检查技能依赖要求是否满足（命令行工具、环境变量）。
        """
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        # 中文：解析出技能需要的命令列表和环境变量列表。

        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )
        # 中文：判断所有要求是否满足。
        # all(...) 当生成器中所有值都为真时返回 True；空列表也返回 True。
        # shutil.which(cmd) 找到命令返回路径字符串，找不到返回 None。
        # os.environ.get(var) 找到且非空时为真，否则为假。
        # and 表示命令和环境变量都必须满足。

    def _get_skill_meta(self, name: str) -> dict[str, Any]:
        """Get nanobot metadata for a skill (cached in frontmatter).

        中文翻译：获取某个技能的 nanobot 元数据（来自 frontmatter）。
        """
        raw_meta = self.get_skill_metadata(name) or {}
        # 中文：读取完整 frontmatter metadata。
        # 如果返回 None，则使用空 dict。

        return self._parse_nanobot_metadata(raw_meta.get("metadata"))
        # 中文：从 frontmatter 的 metadata 字段中提取 nanobot/openclaw 配置。

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements.

        中文翻译：获取标记为 always=true 且满足依赖要求的技能。
        """
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")
            )
        ]
        # 中文：列表推导式，返回所有 always 技能名。
        # list_skills(filter_unavailable=True) 只取可用技能。
        # `(meta := self.get_skill_metadata(entry["name"]) or {})` 使用海象运算符：
        # - 调用 get_skill_metadata 读取 frontmatter。
        # - 如果结果为 None，则赋值为空 dict。
        # - 同时把结果绑定到 meta，供后续条件判断。
        # 条件第一部分：meta 必须是非空 dict。
        # 条件第二部分：
        # - 优先检查 metadata.nanobot/openclaw 中的 always。
        # - 也兼容 frontmatter 顶层 always 字段。
        # or 表示任一 always 为真即可。

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.

        中文翻译：从技能的 frontmatter 中获取 metadata。

        参数：
            name：技能名。

        返回：
            metadata 字典，如果无法获取则返回 None。
        """
        content = self.load_skill(name)
        # 中文：加载 SKILL.md 全文内容。

        if not content or not content.startswith("---"):
            # 中文：如果内容为空，或者没有以 --- 开头，则没有 frontmatter。
            return None

        match = _STRIP_SKILL_FRONTMATTER.match(content)
        # 中文：尝试匹配 frontmatter。

        if not match:
            # 中文：如果匹配失败，返回 None。
            return None

        try:
            parsed = yaml.safe_load(match.group(1))
            # 中文：解析 frontmatter 中的 YAML 正文。
            # match.group(1) 是正则捕获到的 YAML 内容。
            # yaml.safe_load 会返回 Python 原生对象，例如 dict、list、str、int、bool。
        except yaml.YAMLError:
            # 中文：如果 YAML 解析失败，返回 None。
            return None

        if not isinstance(parsed, dict):
            # 中文：frontmatter 顶层必须是字典，否则无法作为 metadata 使用。
            return None

        # yaml.safe_load returns native types (int, bool, list, etc.);
        # keep values as-is so downstream consumers get correct types.
        # 中文翻译：yaml.safe_load 会返回原生类型（int、bool、list 等）；
        # 这里保持值不变，以便后续使用者拿到正确类型。
        metadata: dict[str, object] = {}
        # 中文：metadata 用于保存标准化 key 为字符串后的结果。
        # value 类型保持 object，因为 YAML 值可能是任意类型。

        for key, value in cast(dict[object, object], parsed).items():
            # 中文：遍历解析后的 YAML dict。
            # cast(dict[object, object], parsed) 只是为了类型检查。
            # .items() 返回 key-value 键值对。

            metadata[str(key)] = value
            # 中文：把 key 强制转换为字符串。
            # value 保持原样，避免破坏 bool、int、list 等类型。

        return metadata
        # 中文：返回解析后的 frontmatter metadata。