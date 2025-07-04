from functools import wraps
from typing import Callable, Concatenate, Optional, ParamSpec, TypeVar

from pydantic import BaseModel

from .manager import ConflgaManager
from .config import ConflgaConfig
from .cli import ConflgaCLI

P = ParamSpec("P")  # P 代表装饰后函数所接受的参数
T = TypeVar("T")  # T 代表原始函数的返回类型


def conflga_main(
    config_dir: str = "conf",
    default_config: str = "config",
    configs_to_merge: Optional[list[str]] = None,
    enable_cli_override: bool = True,
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
                                   for configuration overrides using -o/--override flags.

    Example:
        @conflga_main(config_dir="conf", default_config="config", enable_cli_override=True)
        def main(cfg: ConflgaConfig):
            print(f"Learning rate: {cfg.model.learning_rate}")

        # Can be called with: python script.py -o model.learning_rate=0.001 -o dataset.batch_size=32
    """

    def decorator(func: Callable[Concatenate[ConflgaConfig, P], T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Initialize configuration manager and load base configuration
            manager = ConflgaManager(config_dir=config_dir)
            manager.load_default(default_config)

            # Merge additional configuration files if specified
            if configs_to_merge:
                manager.merge_config(*configs_to_merge)

            # Apply command line overrides if enabled
            if enable_cli_override:
                cli = ConflgaCLI()
                override_config = cli.parse_overrides()
                # Only apply overrides if there are any
                if override_config._data:  # Check if override config has any data
                    manager.override_config(override_config)

            cfg = manager.get_config()
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
