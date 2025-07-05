import logging
from typing import Any
from pathlib import Path


class ConflgaLogger:
    """
    Conflga 库的日志管理器。
    """

    _logger_name: str = "conflga"
    _logger: logging.Logger | None = None

    @classmethod
    def set_logger_name(cls, name: str) -> None:
        """
        设置库使用的 logger 名称。

        Args:
            name: logger 名称
        """
        cls._logger_name = name
        cls._logger = None  # 重置 logger 实例，下次获取时会使用新名称

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        获取 logger 实例。

        Returns:
            logging.Logger: logger 实例
        """
        if cls._logger is None:
            cls._logger = logging.getLogger(cls._logger_name)
        return cls._logger

    @classmethod
    def print_config(
        cls,
        config_data: dict[str, Any],
        title: str = "Configuration",
        directory: str | None = None,
        files: list[str] | None = None,
        level: int = logging.INFO,
    ) -> None:
        """
        打印配置信息到日志。

        Args:
            config_data: 配置数据字典
            title: 配置的标题
            directory: 配置文件所在目录
            files: 配置文件列表
            level: 日志级别
        """
        logger = cls.get_logger()

        # 构建日志消息
        message_parts = [f"=== {title} ==="]

        # 添加配置来源信息
        if directory is not None or files is not None:
            message_parts.append("Configuration Source:")

            if directory is not None:
                dir_path = Path(directory).resolve()
                message_parts.append(f"  Config Directory: {dir_path}")
                status = "Exists" if dir_path.exists() else "Not Found"
                message_parts.append(f"  Directory Status: {status}")

            if files is not None and len(files) > 0:
                for i, file in enumerate(files):
                    file_label = (
                        f"Config File {i+1}" if len(files) > 1 else "Config File"
                    )
                    file_name = f"{file}.toml"
                    message_parts.append(f"  {file_label}: {file_name}")

                    if directory is not None:
                        file_path = Path(directory) / file_name
                        abs_path = file_path.resolve()
                        message_parts.append(f"    Full Path: {abs_path}")

                        if file_path.exists():
                            message_parts.append(f"    File Status: Exists")
                            try:
                                size = file_path.stat().st_size
                                if size < 1024:
                                    size_str = f"{size} B"
                                elif size < 1024 * 1024:
                                    size_str = f"{size / 1024:.1f} KB"
                                else:
                                    size_str = f"{size / (1024 * 1024):.1f} MB"
                                message_parts.append(f"    File Size: {size_str}")
                            except OSError:
                                pass
                        else:
                            message_parts.append(f"    File Status: Not Found")
                    else:
                        file_path = Path(f"{file}.toml").resolve()
                        message_parts.append(f"    Full Path: {file_path}")
                        status = "Exists" if file_path.exists() else "Not Found"
                        message_parts.append(f"    File Status: {status}")

            message_parts.append("")  # 空行分隔

        # 添加配置内容
        message_parts.append("Configuration Content:")
        config_str = cls._format_config_tree(config_data)
        message_parts.extend(config_str.split("\n"))

        # 发送到日志
        full_message = "\n".join(message_parts)
        logger.log(level, full_message)

    @classmethod
    def _format_config_tree(cls, data: Any, indent: int = 0) -> str:
        """
        将配置数据格式化为树状结构字符串。

        Args:
            data: 要格式化的数据
            indent: 缩进级别

        Returns:
            格式化后的字符串
        """
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                # 检查是否是 ConflgaConfig 对象
                if (
                    hasattr(value, "_data")
                    and hasattr(value, "__class__")
                    and value.__class__.__name__ == "ConflgaConfig"
                ):
                    lines.append(f"{prefix}{key}:")
                    sub_lines = cls._format_config_tree(value._data, indent + 1)
                    if sub_lines.strip():
                        lines.append(sub_lines)
                elif isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    sub_lines = cls._format_config_tree(value, indent + 1)
                    if sub_lines.strip():  # 只有在有内容时才添加
                        lines.append(sub_lines)
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key}:")
                    sub_lines = cls._format_config_tree(value, indent + 1)
                    if sub_lines.strip():
                        lines.append(sub_lines)
                else:
                    formatted_value = cls._format_value(value)
                    lines.append(f"{prefix}{key}: {formatted_value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                # 检查是否是 ConflgaConfig 对象
                if (
                    hasattr(item, "_data")
                    and hasattr(item, "__class__")
                    and item.__class__.__name__ == "ConflgaConfig"
                ):
                    lines.append(f"{prefix}[{i}]:")
                    sub_lines = cls._format_config_tree(item._data, indent + 1)
                    if sub_lines.strip():
                        lines.append(sub_lines)
                elif isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    sub_lines = cls._format_config_tree(item, indent + 1)
                    if sub_lines.strip():
                        lines.append(sub_lines)
                else:
                    formatted_value = cls._format_value(item)
                    lines.append(f"{prefix}[{i}]: {formatted_value}")

        return "\n".join(line for line in lines if line.strip())  # 过滤空行

    @classmethod
    def _format_value(cls, value: Any) -> str:
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


# 便利函数
def set_conflga_logger_name(name: str) -> None:
    """
    设置 Conflga 库使用的 logger 名称。

    Args:
        name: logger 名称

    Example:
        >>> import conflga
        >>> conflga.set_conflga_logger_name("my_app.config")
    """
    ConflgaLogger.set_logger_name(name)


def get_conflga_logger() -> logging.Logger:
    """
    获取 Conflga 库使用的 logger 实例。

    Returns:
        logging.Logger: logger 实例

    Example:
        >>> import conflga
        >>> logger = conflga.get_conflga_logger()
        >>> logger.info("Custom log message")
    """
    return ConflgaLogger.get_logger()
