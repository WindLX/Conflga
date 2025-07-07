from functools import wraps
from typing import Callable, Concatenate, ParamSpec, TypeVar

from rich.console import Console

from .preprocessor import ConflgaPreprocessor
from .manager import ConflgaManager
from .config import ConflgaConfig
from .cli import ConflgaCLI
from .console import get_echoa, ConflgaEchoa

P = ParamSpec("P")  # P 代表装饰后函数所接受的参数
T = TypeVar("T")  # T 代表原始函数的返回类型


def conflga_entry(
    config_dir: str = "conf",
    default_config: str = "config",
    configs_to_merge: list[str] | None = None,
    enable_preprocessor: bool = True,
    enable_cli_override: bool = True,
    use_namespace_prefix: bool = True,  # 是否使用命名空间前缀避免冲突
    auto_print: bool = True,  # 是否自动打印配置
    auto_print_override: bool = True,  # 是否自动打印覆盖的配置
    console: Console | None = None,  # 控制台对象，默认为 None
    echoa: ConflgaEchoa | None = None,  # 可选的 ConflgaEchoa 实例
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
        echoa (Optional[ConflgaEchoa]): Optional ConflgaEchoa instance for custom output control.

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
            # 获取或创建 echoa 实例
            if echoa is not None:
                output_echoa = echoa
            else:
                output_echoa = get_echoa()

            # 如果提供了 console，设置到 echoa 中
            if console is not None:
                output_echoa.set_console(console)

            # Initialize configuration manager and load base configuration
            manager = ConflgaManager(config_dir=config_dir)
            default_config_str = manager.load_default_file(default_config)
            if configs_to_merge:
                merged_config_strs = (
                    manager.load_merged_file(*configs_to_merge)
                    if configs_to_merge
                    else []
                )
            else:
                merged_config_strs = []

            if enable_preprocessor:
                preprocessor = ConflgaPreprocessor()
                # Preprocess the default configuration
                default_config_str = preprocessor.preprocess_text(default_config_str)
                # Preprocess additional configurations if any, sharing the same preprocessor context
                merged_config_strs = [
                    preprocessor.preprocess_text(config_str)
                    for config_str in merged_config_strs
                ]

            manager.load_default(default_config_str)

            # Merge additional configuration files if specified
            if configs_to_merge:
                manager.merge_config(*merged_config_strs)

            # Apply command line overrides if enabled
            if enable_cli_override:
                # Use configurable namespace prefix
                cli = ConflgaCLI(use_namespace_prefix=use_namespace_prefix)
                override_config = cli.parse_overrides()
                # Only apply overrides if there are any
                if override_config._data:  # Check if override config has any data
                    manager.override_config(override_config)
                    if auto_print_override:
                        output_echoa.print_config(
                            config_data=override_config, title="Command Line Overrides"
                        )

            cfg = manager.get_config()
            if auto_print:
                output_echoa.print_config(
                    config_data=cfg,
                    title="Final Configuration",
                    directory=config_dir,
                    files=[default_config] + (configs_to_merge or []),
                )
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
