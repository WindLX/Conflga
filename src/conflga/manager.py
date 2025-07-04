from pathlib import Path
from typing import Any

from .config import ConflgaConfig


class ConflgaManager:
    """
    A simple configuration manager.
    Manages loading and merging ConflgaConfig instances.
    """

    def __init__(self, config_dir: str = "conf") -> None:
        """
        Initializes the ConflgaManager.

        Args:
            config_dir (str): The directory where configuration files are located.
        """
        self.config_dir = Path(config_dir)
        self._config: ConflgaConfig | None = None

    def load_default(self, default_config_name: str = "config") -> "ConflgaManager":
        """
        Loads the base configuration. This should be the first config loaded.

        Args:
            default_config_name (str): The name of the default TOML file (without .toml extension).
                                       Expected to be in the `config_dir`.
        """
        default_path = self.config_dir / f"{default_config_name}.toml"
        if not default_path.exists():
            raise FileNotFoundError(f"Default config file not found: {default_path}")
        self._config = ConflgaConfig.load(str(default_path))
        return self

    def merge_config(self, *config_names: str) -> "ConflgaManager":
        """
        Merges additional configurations on top of the current configuration.
        Later arguments override earlier ones.

        Args:
            *config_names (str): Names of the TOML files to merge (without .toml extension).
                                Expected to be in the `config_dir`.
        """
        if self._config is None:
            raise RuntimeError(
                "Load a default configuration first using load_default()."
            )

        for config_name in config_names:
            config_path = self.config_dir / f"{config_name}.toml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            new_config = ConflgaConfig.load(str(config_path))
            self._config.merge_with(new_config)
        return self

    def override_config(self, override_config: ConflgaConfig) -> "ConflgaManager":
        """
        Overrides parts of the current configuration with values from another ConflgaConfig.
        This is useful for implementing command-line argument overrides.

        Args:
            override_config (ConflgaConfig): The ConflgaConfig instance containing
                                           override values to merge with the current config.
        """
        if self._config is None:
            raise RuntimeError(
                "Load a default configuration first using load_default()."
            )

        if not isinstance(override_config, ConflgaConfig):
            raise TypeError("override_config must be a ConflgaConfig instance.")

        self._config.merge_with(override_config)
        return self

    def override_from_dict(self, override_dict: dict[str, Any]) -> "ConflgaManager":
        """
        Overrides parts of the current configuration with values from a dictionary.
        This is a convenience method for override_config() that accepts a dictionary.

        Args:
            override_dict (dict): A dictionary containing override values.
        """
        override_config = ConflgaConfig(override_dict)
        return self.override_config(override_config)

    def get_config(self) -> ConflgaConfig:
        """
        Returns the merged ConflgaConfig object.
        """
        if self._config is None:
            raise RuntimeError("No configuration has been loaded yet.")
        return self._config
