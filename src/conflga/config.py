from typing import Any, Iterator, Mapping
from collections.abc import MutableMapping
from pathlib import Path

import rtoml as toml
from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


class ConflgaConfig(MutableMapping):
    """
    A simple TOML-based configuration object with dot-access and merging capabilities.
    Inspired by OmegaConf.
    """

    def __init__(self, initial_data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if initial_data:
            for k, v in initial_data.items():
                self._data[k] = self._create_nested_config(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":  # Avoid infinite recursion for _data itself
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self._data[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __repr__(self) -> str:
        return f"ConflgaConfig({self._data})"

    @classmethod
    def load(cls, toml_path: str) -> "ConflgaConfig":
        """
        Loads configuration from a TOML file.
        """
        with open(toml_path, "r", encoding="utf-8") as f:
            data = toml.load(f, none_value="@None")
        return cls._create_nested_config(data)

    @classmethod
    def from_string(cls, toml_string: str) -> "ConflgaConfig":
        """
        Loads configuration from a TOML string.
        """
        data = toml.loads(toml_string, none_value="@None")
        return cls._create_nested_config(data)

    @staticmethod
    def _create_nested_config(data: Any) -> Any:
        """
        Recursively converts dictionaries in data to ConflgaConfig instances
        to enable dot-access for nested configurations.
        """
        if isinstance(data, dict):
            return ConflgaConfig(
                {k: ConflgaConfig._create_nested_config(v) for k, v in data.items()}
            )
        elif isinstance(data, list):
            return [ConflgaConfig._create_nested_config(item) for item in data]
        return data

    def merge_with(self, other_config: "ConflgaConfig") -> "ConflgaConfig":
        """
        Merges this configuration with another ConflgaConfig instance.
        Values from `other_config` will override existing values.
        """
        if not isinstance(other_config, ConflgaConfig):
            raise TypeError("Can only merge with another ConflgaConfig instance.")

        def _recursive_merge(d1: dict[str, Any], d2: dict[str, Any]) -> None:
            for k, v in d2.items():
                if (
                    isinstance(v, ConflgaConfig)
                    and k in d1
                    and isinstance(d1[k], ConflgaConfig)
                ):
                    _recursive_merge(d1[k]._data, v._data)
                else:
                    d1[k] = v

        _recursive_merge(self._data, other_config._data)
        return self  # Return self for chaining

    def pretty_print(
        self,
        title: str = "Configuration",
        console: Console | None = None,
        directory: str | None = None,
        files: list[str] | None = None,
    ) -> None:
        """
        使用 rich 库美观地打印配置内容。

        Args:
            title: 配置树的标题
            console: rich Console 实例，如果为 None 则创建新的实例
            directory: 配置文件所在的目录路径
            files: 配置文件列表
        """
        if console is None:
            console = Console()

        # 如果提供了配置来源信息，先显示来源
        if directory is not None or files is not None:
            self._print_config_source(console, directory, files)
            console.print()  # 添加一个空行分隔

        tree = Tree(Text(title, style="bold blue"))
        self._build_tree(tree, self._data)
        console.print(tree)

    def _print_config_source(
        self,
        console: Console,
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
                if isinstance(value, ConflgaConfig):
                    # 嵌套配置对象
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
                if isinstance(item, (dict, ConflgaConfig, list)):
                    branch = parent.add(Text(f"[{i}]", style="bold yellow"))
                    if isinstance(item, ConflgaConfig):
                        self._build_tree(branch, item._data)
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
