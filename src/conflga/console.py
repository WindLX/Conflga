from typing import Any
from pathlib import Path

from rich.console import Console as RichConsole
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule


class Console:
    """
    Conflga 控制台输出管理器
    """

    def __init__(
        self,
        console: RichConsole | None = None,
        prefix: str = "[Conflga]",
        enabled: bool = True,
    ) -> None:
        """
        初始化 Console 实例

        Args:
            console: Rich Console 实例，None 使用默认控制台
            prefix: 输出前缀
            enabled: 是否启用输出
        """
        self._console = console or RichConsole()
        self._prefix = prefix
        self._enabled = enabled

    @property
    def console(self) -> RichConsole:
        """
        获取当前的 Rich Console 实例

        Returns:
            Rich Console 实例
        """
        return self._console

    @console.setter
    def console(self, console: RichConsole | None = None) -> None:
        """
        设置 Rich Console 实例

        Args:
            console: Rich Console 实例，None 使用默认控制台
        """
        self._console = console or RichConsole()

    @property
    def enabled(self) -> bool:
        """
        获取当前输出是否启用

        Returns:
            是否启用输出
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """
        设置输出是否启用

        Args:
            value: 是否启用输出
        """
        self._enabled = value

    def info(self, message: str, **kwargs: Any) -> None:
        """
        输出信息消息

        Args:
            message: 消息内容
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        text = Text(f"{self._prefix} ", style="blue bold")
        text.append(message, style="white")
        self._console.print(text, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        输出调试消息

        Args:
            message: 消息内容
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        text = Text(f"{self._prefix} ", style="dim blue")
        text.append(f"[DEBUG] {message}", style="dim white")
        self._console.print(text, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        输出警告消息

        Args:
            message: 消息内容
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        text = Text(f"{self._prefix} ", style="yellow bold")
        text.append(f"⚠️  {message}", style="yellow")
        self._console.print(text, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        输出错误消息

        Args:
            message: 消息内容
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        text = Text(f"{self._prefix} ", style="red bold")
        text.append(f"❌ {message}", style="red")
        self._console.print(text, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """
        输出成功消息

        Args:
            message: 消息内容
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        text = Text(f"{self._prefix} ", style="green bold")
        text.append(f"✅ {message}", style="green")
        self._console.print(text, **kwargs)

    def panel(
        self, message: str, title: str = "", style: str = "blue", **kwargs: Any
    ) -> None:
        """
        输出面板消息

        Args:
            message: 消息内容
            title: 面板标题
            style: 面板样式
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        panel = Panel(message, title=title or self._prefix, border_style=style)
        self._console.print(panel, **kwargs)

    def rule(self, title: str = "", style: str = "blue", **kwargs: Any) -> None:
        """
        输出分隔线

        Args:
            title: 分隔线标题
            style: 样式
            **kwargs: 额外参数
        """
        if not self._enabled:
            return
        rule = Rule(title, style=style)
        self._console.print(rule, **kwargs)

    def print_config(
        self,
        config_data: Any,
        title: str = "Configuration",
        directory: str | None = None,
        files: list[str] | None = None,
    ) -> None:
        """
        使用 rich 库美观地打印配置内容。

        Args:
            config_data: 配置数据（ConflgaConfig实例或其_data属性）
            title: 配置树的标题
            directory: 配置文件所在的目录路径
            files: 配置文件列表
        """
        console = self._console

        # 如果提供了配置来源信息，先显示来源
        if directory is not None or files is not None:
            self._print_config_source(console, directory, files)
            console.print()  # 添加一个空行分隔

        tree = Tree(Text(title, style="bold blue"))

        # 处理不同类型的配置数据
        if hasattr(config_data, "_data"):
            # ConflgaConfig 实例
            self._build_tree(tree, config_data._data)
        elif isinstance(config_data, dict):
            # 普通字典
            self._build_tree(tree, config_data)
        else:
            # 其他类型
            self._build_tree(tree, config_data)

        console.print(tree)

    def _print_config_source(
        self,
        console: RichConsole,
        directory: str | None = None,
        files: list[str] | None = None,
    ) -> None:
        """
        打印配置来源信息。

        Args:
            console: rich Console 实例
            directory: 配置文件所在的目录路径
            files: 配置文件列表
        """

        # 创建配置来源信息表格
        source_table = Table(show_header=True, header_style="bold magenta")
        source_table.add_column("Property", style="cyan", no_wrap=True)
        source_table.add_column("Value", style="white")

        if directory is not None:
            dir_path = Path(directory).resolve()
            source_table.add_row("Config Directory", str(dir_path))

            if dir_path.exists():
                source_table.add_row("Directory Status", "[green]✓ Exists[/green]")
            else:
                source_table.add_row("Directory Status", "[red]✗ Not Found[/red]")

        if files is not None and len(files) > 0:
            for i, file in enumerate(files):
                file_label = f"Config File {i+1}" if len(files) > 1 else "Config File"
                file_name = f"{file}.toml"

                if directory is not None:
                    file_path = Path(directory) / file_name
                    abs_path = file_path.resolve()
                    source_table.add_row(file_label, file_name)
                    source_table.add_row(f"  └─ Full Path", str(abs_path))

                    if file_path.exists():
                        source_table.add_row(
                            f"  └─ File Status", "[green]✓ Exists[/green]"
                        )
                        try:
                            size = file_path.stat().st_size
                            if size < 1024:
                                size_str = f"{size} B"
                            elif size < 1024 * 1024:
                                size_str = f"{size / 1024:.1f} KB"
                            else:
                                size_str = f"{size / (1024 * 1024):.1f} MB"
                                source_table.add_row(f"  └─ File Size", size_str)
                        except OSError:
                            pass
                    else:
                        source_table.add_row(
                            f"  └─ File Status", "[red]✗ Not Found[/red]"
                        )
                else:
                    file_path = Path(f"{file}.toml").resolve()
                    source_table.add_row(file_label, str(file_path))

                    if file_path.exists():
                        source_table.add_row(
                            f"  └─ File Status", "[green]✓ Exists[/green]"
                        )
                    else:
                        source_table.add_row(
                            f"  └─ File Status", "[red]✗ Not Found[/red]"
                        )

        # 在面板中显示配置来源信息
        panel = Panel(
            source_table,
            title="[bold yellow]📁 Configuration Source[/bold yellow]",
            border_style="yellow",
            padding=(1, 1),
        )
        console.print(panel)

    def _build_tree(self, parent: Tree, data: Any) -> None:
        """
        递归构建配置树结构。

        Args:
            parent: 父级树节点
            data: 要添加到树中的数据
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(value, "_data"):
                    # ConflgaConfig 对象
                    branch = parent.add(Text(f"{key}", style="bold green"))
                    self._build_tree(branch, value._data)
                elif isinstance(value, dict):
                    # 普通字典
                    branch = parent.add(Text(f"{key}", style="bold green"))
                    self._build_tree(branch, value)
                elif isinstance(value, list):
                    # 列表
                    branch = parent.add(Text(f"{key}", style="bold cyan"))
                    self._build_tree(branch, value)
                else:
                    # 简单值
                    value_str = self._format_value(value)
                    parent.add(Text(f"{key}: {value_str}", style="dim"))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)) or hasattr(item, "_data"):
                    branch = parent.add(Text(f"[{i}]", style="bold yellow"))
                    if hasattr(item, "_data"):
                        self._build_tree(branch, getattr(item, "_data"))
                    else:
                        self._build_tree(branch, item)
                else:
                    value_str = self._format_value(item)
                    parent.add(Text(f"[{i}]: {value_str}", style="dim"))

    def _format_value(self, value: Any) -> str:
        """
        格式化单个值的显示。

        Args:
            value: 要格式化的值

        Returns:
            格式化后的字符串
        """
        if value is None:
            return "null"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return str(value)


_default_console = Console()


def get_console() -> Console:
    """
    获取全局 Console 实例

    Returns:
        Console: 全局 Console 实例
    """
    return _default_console


def set_console(console: RichConsole | None = None) -> None:
    """
    设置默认 Console 控制台输出配置（向后兼容）

    Args:
        console: Rich Console 实例，None 使用默认控制台
        prefix: 输出前缀
    """
    _default_console.console = console


def enable_console_output(enabled: bool = True) -> None:
    """
    启用或禁用默认 Console 输出（向后兼容）

    Args:
        enabled: 是否启用输出
    """
    _default_console.enabled = enabled


def info(message: str) -> None:
    """
    打印信息级别的日志消息

    Args:
        message: 要打印的消息
    """
    _default_console.info(message)


def warning(message: str) -> None:
    """
    打印警告级别的日志消息

    Args:
        message: 要打印的消息
    """
    _default_console.warning(message)


def error(message: str) -> None:
    """
    打印错误级别的日志消息

    Args:
        message: 要打印的消息
    """
    _default_console.error(message)


def debug(message: str) -> None:
    """
    打印调试级别的日志消息

    Args:
        message: 要打印的消息
    """
    _default_console.debug(message)
