from functools import wraps
from typing import Callable, Concatenate, ParamSpec, TypeVar
import logging

from .manager import ConflgaManager
from .config import ConflgaConfig
from .cli import ConflgaCLI

P = ParamSpec("P")  # P 代表装饰后函数所接受的参数
T = TypeVar("T")  # T 代表原始函数的返回类型


def conflga_main(
    config_dir: str = "conf",
    default_config: str = "config",
    configs_to_merge: list[str] | None = None,
    enable_cli_override: bool = True,
    auto_print: bool = True,  # 是否自动打印配置
    auto_print_override: bool = True,  # 是否自动打印覆盖的配置
    log_level: int = logging.INFO,  # 日志级别
    use_namespace_prefix: bool = True,  # 是否使用命名空间前缀避免冲突
) -> Callable[
    [Callable[Concatenate[ConflgaConfig, P], T]],  # 接收的函数类型
    Callable[P, T],  # 返回的函数类型
]:
    """
    A decorator to initialize and pass the ConflgaConfig object to a function.
    Supports command line override functionality.

    Args:
        config_dir (str): The base directory for configuration files.
        default_config (str): The name of the default configuration file to load.
        configs_to_merge (Optional[list[str]]): A list of additional configuration files
                                                 to merge, overriding the default.
        enable_cli_override (bool): Whether to enable command line override functionality.
                                   When True, the decorator will parse command line arguments
                                   for configuration overrides.
        auto_print (bool): Whether to automatically print the final configuration.
        auto_print_override (bool): Whether to automatically print override configuration.
        log_level (int): Logging level for configuration printing.
        use_namespace_prefix (bool): Whether to use --conflga-override (True) to avoid conflicts
                                   or -o/--override (False) for backward compatibility.

    Example:
        @conflga_main(config_dir="conf", default_config="config", enable_cli_override=True)
        def main(cfg: ConflgaConfig):
            print(f"Learning rate: {cfg.model.learning_rate}")

        # Can be called with:
        # python script.py --conflga-override model.learning_rate=0.001 --conflga-override dataset.batch_size=32
        # or with use_namespace_prefix=False:
        # python script.py -o model.learning_rate=0.001 -o dataset.batch_size=32
    """

    def decorator(func: Callable[Concatenate[ConflgaConfig, P], T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Initialize configuration manager and load base configuration
            manager = ConflgaManager(config_dir=config_dir)

            # Try to load default config, create empty config if file doesn't exist
            try:
                manager.load_default(default_config)
            except FileNotFoundError:
                # Create empty config if no default config file exists
                manager._config = ConflgaConfig({})

            # Merge additional configuration files if specified
            if configs_to_merge:
                # Only try to merge configs that actually exist
                existing_configs = []
                for config_name in configs_to_merge:
                    config_path = manager.config_dir / f"{config_name}.toml"
                    if config_path.exists():
                        existing_configs.append(config_name)

                if existing_configs:
                    manager.merge_config(*existing_configs)

            # Apply command line overrides if enabled
            if enable_cli_override:
                # Use configurable namespace prefix
                cli = ConflgaCLI(use_namespace_prefix=use_namespace_prefix)
                override_config = cli.parse_overrides()
                # Only apply overrides if there are any
                if override_config._data:  # Check if override config has any data
                    manager.override_config(override_config)
                    if auto_print_override:
                        override_config.pretty_print(
                            title="Command Line Overrides", level=log_level
                        )

            cfg = manager.get_config()
            if auto_print:
                cfg.pretty_print(
                    title=f"Final Configuration",
                    directory=config_dir,
                    files=[default_config] + (configs_to_merge or []),
                    level=log_level,
                )
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
