from typing import Any, Iterator, Mapping
from collections.abc import MutableMapping

import rtoml as toml
from rich.console import Console

from .console import get_echoa


class ConflgaConfig(MutableMapping):
    """
    A simple TOML-based configuration object with dot-access and merging capabilities.
    """

    def __init__(self, initial_data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if initial_data:
            for k, v in initial_data.items():
                self._data[k] = self._create_nested_config(v)

    def __getstate__(self) -> dict[str, Any]:
        """
        告诉pickle如何序列化这个对象。
        我们只返回核心的、可序列化的 _data 成员。
        """
        return {"_data": self._data}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        告诉pickle如何根据state重建这个对象。
        """
        self._data = state.get("_data", {})

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
    def load(cls, file_path: str) -> "ConflgaConfig":
        """
        Loads configuration from a TOML file.
        """
        from pathlib import Path

        file_content = Path(file_path).read_text(encoding="utf-8")
        return cls.loads(file_content)

    @classmethod
    def loads(cls, toml_string: str) -> "ConflgaConfig":
        """
        Loads configuration from a TOML string.
        """
        data = toml.loads(toml_string, none_value="@None")
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

    def _convert_value(self, value: Any) -> Any:
        """Recursively convert ConflgaConfig objects to regular dicts."""
        if isinstance(value, ConflgaConfig):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._convert_value(item) for item in value]
        else:
            return value

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the configuration to a regular dictionary.
        """

        return {k: self._convert_value(v) for k, v in self._data.items()}

    def pretty_print(
        self,
        title: str = "Configuration",
        console: Console | None = None,
        directory: str | None = None,
        files: list[str] | None = None,
    ) -> None:
        """
        使用 ConflgaEchoa 美观地打印配置内容。

        Args:
            title: 配置树的标题
            console: rich Console 实例，如果为 None 则使用默认的 echoa 控制台
            directory: 配置文件所在的目录路径
            files: 配置文件列表
        """
        echoa = get_echoa()
        if console is not None:
            echoa.set_console(console)

        echoa.print_config(
            config_data=self,
            title=title,
            directory=directory,
            files=files,
        )
