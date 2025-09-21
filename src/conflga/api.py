import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console

from .preprocessor import ConflgaPreprocessor
from .manager import ConflgaManager
from .config import ConflgaConfig
from .cli import ConflgaCLI
from .console import get_console


def _sort_key(filename: str) -> int | float:
    m = re.match(r"^(\d+)[-_]", filename)
    return -int(m.group(1)) if m else float("-inf")


def get_config(
    config_dir: str = "conf",
    default_config: str | None = "config",
    configs_to_merge: Optional[list[str]] = None,
    enable_preprocessor: bool = True,
    enable_cli_override: bool = True,
    use_namespace_prefix: bool = True,
    auto_print: bool = True,
    auto_print_override: bool = True,
    console: Optional[Console] = None,
    backup_path: Optional[str] = None,
) -> ConflgaConfig:
    output_console = get_console()
    if console is not None:
        output_console.console = console

    manager = ConflgaManager(config_dir=config_dir)
    if default_config is not None:
        default_config_str = manager.load_default_file(default_config)
    else:
        default_config_str = ""

    if configs_to_merge:
        # Sort files: if filename starts with digits + '_', sort by number descending
        sorted_configs = sorted(configs_to_merge, key=_sort_key)
        merged_config_strs = manager.load_merged_file(*sorted_configs)
    else:
        merged_config_strs = []

    if enable_preprocessor:
        preprocessor = ConflgaPreprocessor()
        default_config_str = preprocessor.preprocess_text(default_config_str)
        merged_config_strs = [
            preprocessor.preprocess_text(config_str)
            for config_str in merged_config_strs
        ]

    manager.load_default(default_config_str)
    if configs_to_merge:
        manager.merge_config(*merged_config_strs)

    if enable_cli_override:
        cli = ConflgaCLI(use_namespace_prefix=use_namespace_prefix)
        override_config = cli.parse_overrides()
        if override_config._data:
            manager.override_config(override_config)
            if auto_print_override:
                output_console.print_config(
                    config_data=override_config, title="Command Line Overrides"
                )

    cfg = manager.get_config()

    if backup_path is not None:
        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        backup_filename = f"config_backup_{timestamp}.toml"
        backup_file_path = backup_dir / backup_filename
        backup_file_path.write_text(cfg.to_toml(), encoding="utf-8")

    if auto_print:
        output_console.print_config(
            config_data=cfg,
            title="Final Configuration",
            directory=config_dir,
            files=([default_config] if default_config is not None else [])
            + (configs_to_merge or []),
        )
    return cfg
