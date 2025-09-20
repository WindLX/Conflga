from functools import wraps
from typing import Callable, Concatenate, ParamSpec, TypeVar

from rich.console import Console

from .config import ConflgaConfig
from .api import get_config

ParamType = ParamSpec("ParamType")  # ParamType 代表装饰后函数所接受的参数
ReturnType = TypeVar("ReturnType")  # ReturnType 代表原始函数的返回类型
SelfType = TypeVar("SelfType")  # 用于类方法装饰器，表示实例类型或类类型


def conflga_func(
    config_dir: str = "conf",
    default_config: str = "config",
    configs_to_merge: list[str] | None = None,
    enable_preprocessor: bool = True,
    enable_cli_override: bool = True,
    use_namespace_prefix: bool = True,  # 是否使用命名空间前缀避免冲突
    auto_print: bool = True,  # 是否自动打印配置
    auto_print_override: bool = True,  # 是否自动打印覆盖的配置
    console: Console | None = None,  # 控制台对象，默认为 None
    backup_path: str | None = None,  # 备份目录路径，默认为 None
) -> Callable[
    [Callable[Concatenate[ConflgaConfig, ParamType], ReturnType]],  # 接收的函数类型
    Callable[ParamType, ReturnType],  # 返回的函数类型
]:
    """
    A decorator to initialize and pass the ConflgaConfig object to a function or static method.
    Supports command line override functionality.

    Args:
        config_dir (str): The base directory for configuration files.
        default_config (str): The name of the default configuration file to load.
        configs_to_merge (Optional[list[str]]): A list of additional configuration files
                                                 to merge, overriding the default.
        enable_preprocessor (bool): Whether to enable preprocessor functionality.
                                    When True, the decorator will preprocess configuration files
                                    to handle macros and templates.
        enable_cli_override (bool): Whether to enable command line override functionality.
                                   When True, the decorator will parse command line arguments
                                   for configuration overrides.
        use_namespace_prefix (bool): Whether to use -co/--conflga-override (True) to avoid conflicts
                                   or -o/--override (False) for backward compatibility.
        auto_print (bool): Whether to automatically print the final configuration.
        auto_print_override (bool): Whether to automatically print override configurations.
        console (Optional[Console]): Rich Console instance to use for output.
        backup_path (Optional[str]): Directory path where the final configuration will be backed up
                                   as a TOML file. If None, no backup will be created.

    Example:
        @conflga_func(config_dir="conf", default_config="config", enable_cli_override=True)
        def main(cfg: ConflgaConfig):
            print(f"Learning rate: {cfg.model.learning_rate}")

        # Can be called with:
        # python script.py --conflga-override model.learning_rate=0.001 --conflga-override dataset.batch_size=32
        # or with use_namespace_prefix=False:
        # python script.py -o model.learning_rate=0.001 -o dataset.batch_size=32

        # With backup functionality:
        @conflga_func(config_dir="conf", default_config="config", backup_path="backup")
        def main_with_backup(cfg: ConflgaConfig):
            print(f"Learning rate: {cfg.model.learning_rate}")
    """

    def decorator(
        func: Callable[Concatenate[ConflgaConfig, ParamType], ReturnType],
    ) -> Callable[ParamType, ReturnType]:
        @wraps(func)
        def wrapper(*args: ParamType.args, **kwargs: ParamType.kwargs) -> ReturnType:
            cfg = get_config(
                config_dir=config_dir,
                default_config=default_config,
                configs_to_merge=configs_to_merge,
                enable_preprocessor=enable_preprocessor,
                enable_cli_override=enable_cli_override,
                use_namespace_prefix=use_namespace_prefix,
                auto_print=auto_print,
                auto_print_override=auto_print_override,
                console=console,
                backup_path=backup_path,
            )
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator


def conflga_method(
    config_dir: str = "conf",
    default_config: str = "config",
    configs_to_merge: list[str] | None = None,
    enable_preprocessor: bool = True,
    enable_cli_override: bool = True,
    use_namespace_prefix: bool = True,
    auto_print: bool = True,
    auto_print_override: bool = True,
    console: Console | None = None,
    backup_path: str | None = None,  # 备份目录路径，默认为 None
) -> Callable[
    [Callable[Concatenate[SelfType, ConflgaConfig, ParamType], ReturnType]],
    Callable[Concatenate[SelfType, ParamType], ReturnType],
]:
    """
    A decorator to initialize and pass the ConflgaConfig object to a class method.
    Supports command line override functionality.

    Args:
        config_dir (str): The base directory for configuration files.
        default_config (str): The name of the default configuration file to load.
        configs_to_merge (Optional[list[str]]): A list of additional configuration files
                                                 to merge, overriding the default.
        enable_preprocessor (bool): Whether to enable preprocessor functionality.
                                    When True, the decorator will preprocess configuration files
                                    to handle macros and templates.
        enable_cli_override (bool): Whether to enable command line override functionality.
                                   When True, the decorator will parse command line arguments
                                   for configuration overrides.
        use_namespace_prefix (bool): Whether to use -co/--conflga-override (True) to avoid conflicts
                                   or -o/--override (False) for backward compatibility.
        auto_print (bool): Whether to automatically print the final configuration.
        auto_print_override (bool): Whether to automatically print override configurations.
        console (Optional[Console]): Rich Console instance to use for output.
        backup_path (Optional[str]): Directory path where the final configuration will be backed up
                                   as a TOML file. If None, no backup will be created.

    Example:
        class MyApp:
            @conflga_method(config_dir="conf", default_config="config", enable_cli_override=True)
            def run(self, cfg: ConflgaConfig):
                print(f"Learning rate: {cfg.model.learning_rate}")

            @conflga_method(config_dir="conf", default_config="config", backup_path="backup")
            def run_with_backup(self, cfg: ConflgaConfig):
                print(f"Learning rate: {cfg.model.learning_rate}")
    """

    def decorator(
        func: Callable[Concatenate[SelfType, ConflgaConfig, ParamType], ReturnType],
    ) -> Callable[Concatenate[SelfType, ParamType], ReturnType]:
        @wraps(func)
        def wrapper(
            self: SelfType, *args: ParamType.args, **kwargs: ParamType.kwargs
        ) -> ReturnType:
            cfg = get_config(
                config_dir=config_dir,
                default_config=default_config,
                configs_to_merge=configs_to_merge,
                enable_preprocessor=enable_preprocessor,
                enable_cli_override=enable_cli_override,
                use_namespace_prefix=use_namespace_prefix,
                auto_print=auto_print,
                auto_print_override=auto_print_override,
                console=console,
                backup_path=backup_path,
            )
            return func(self, cfg, *args, **kwargs)

        return wrapper

    return decorator
