import argparse
import ast
from typing import Any

from .config import ConflgaConfig


class ConflgaCLI:
    """
    Command-line interface for Conflga configuration management.
    Supports override syntax for configuration values.
    """

    def __init__(
        self, use_namespace_prefix: bool = True, custom_arg_name: str | None = None
    ):
        """
        Initialize CLI parser with configurable argument naming.

        Args:
            use_namespace_prefix: If True, use --conflga-override to avoid conflicts.
                                If False, use -o/--override (may conflict with other parsers).
            custom_arg_name: Custom argument name to use instead of defaults.
                           Should start with -- for long options.
        """
        self.use_namespace_prefix = use_namespace_prefix
        self.custom_arg_name = custom_arg_name

        # Use add_help=False to avoid help conflicts in embedded scenarios
        self.parser = argparse.ArgumentParser(
            description="Conflga Configuration Manager",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,
            epilog=self._get_epilog_text(),
        )
        self._setup_arguments()

    def _get_epilog_text(self) -> str:
        """Get epilog text with appropriate argument names."""
        arg_name = self._get_argument_name()
        return f"""
Override Examples:
  {arg_name} model.learning_rate=0.001
  {arg_name} dataset.batch_size=32
  {arg_name} training.epochs=100
  {arg_name} model.dropout=true
  {arg_name} data.paths="['/path1', '/path2']"
  {arg_name} params="{{'key': 'value', 'num': 42}}"
  {arg_name} nested.deep.value="hello world"
        """

    def _get_argument_name(self) -> str:
        """Get the argument name being used."""
        if self.custom_arg_name:
            return self.custom_arg_name
        elif self.use_namespace_prefix:
            return "--conflga-override"
        else:
            return "-o"

    def _setup_arguments(self):
        """Setup command line arguments."""
        if self.custom_arg_name:
            # Use custom argument name
            self.parser.add_argument(
                self.custom_arg_name,
                action="append",
                dest="overrides",
                metavar="KEY=VALUE",
                help="Override configuration values (can be used multiple times). "
                "Supports nested keys with dot notation (e.g., model.lr=0.01). "
                "Values are automatically parsed as Python literals.",
            )
        elif self.use_namespace_prefix:
            # Use namespace prefix to avoid conflicts, with short alias -c
            self.parser.add_argument(
                "-co",
                "--conflga-override",
                action="append",
                dest="overrides",
                metavar="KEY=VALUE",
                help="Override Conflga configuration values (can be used multiple times). "
                "Supports nested keys with dot notation (e.g., model.lr=0.01). "
                "Values are automatically parsed as Python literals.",
            )
        else:
            # Use short options (may conflict)
            self.parser.add_argument(
                "-o",
                "--override",
                action="append",
                dest="overrides",
                metavar="KEY=VALUE",
                help="Override configuration values (can be used multiple times). "
                "Supports nested keys with dot notation (e.g., model.lr=0.01). "
                "Values are automatically parsed as Python literals.",
            )

    def parse_overrides(
        self, override_strings: list[str] | None = None
    ) -> ConflgaConfig:
        """
        Parse override strings and return a ConflgaConfig object.

        Args:
            override_strings: List of override strings in format "key=value"
                            If None, uses command line arguments.

        Returns:
            ConflgaConfig: Configuration object with override values

        Examples:
            >>> cli = ConflgaCLI()
            >>> config = cli.parse_overrides([
            ...     "model.learning_rate=0.001",
            ...     "dataset.batch_size=32",
            ...     "training.use_gpu=true"
            ... ])
        """
        if override_strings is None:
            # Use parse_known_args to avoid conflicts with other argument parsers
            try:
                args, unknown = self.parser.parse_known_args()
                override_strings = args.overrides or []
            except SystemExit:
                # If parsing fails, return empty list to avoid crashes
                override_strings = []

        # Ensure override_strings is not None
        if override_strings is None:
            override_strings = []

        override_dict = {}

        for override_str in override_strings:
            if "=" not in override_str:
                raise ValueError(
                    f"Invalid override format: '{override_str}'. Expected 'key=value'"
                )

            key, value_str = override_str.split("=", 1)
            key = key.strip()
            value_str = value_str.strip()

            # Parse the value
            parsed_value = self._parse_value(value_str)

            # Set nested key in dictionary
            self._set_nested_key(override_dict, key, parsed_value)

        return ConflgaConfig(override_dict)

    def _parse_value(self, value_str: str) -> Any:
        """
        Parse a string value to appropriate Python type.

        Args:
            value_str: String representation of the value

        Returns:
            Parsed value with appropriate type
        """
        value_str = value_str.strip()

        # Handle special cases first
        if value_str.lower() == "true":
            return True
        elif value_str.lower() == "false":
            return False
        elif value_str.lower() == "null" or value_str.lower() == "none":
            return None

        # Try to parse as Python literal (handles int, float, str, list, dict)
        try:
            # First try direct literal evaluation
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If literal_eval fails, try some common patterns
            pass

        # Try parsing as number
        try:
            # Try integer first
            if "." not in value_str and "e" not in value_str.lower():
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            pass

        # Try parsing as list (simple comma-separated values)
        if value_str.startswith("[") and value_str.endswith("]"):
            try:
                return ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Parse as simple comma-separated list
                inner = value_str[1:-1].strip()
                if not inner:
                    return []
                items = [item.strip().strip("\"'") for item in inner.split(",")]
                return [self._parse_simple_value(item) for item in items if item]

        # Try parsing as dict
        if value_str.startswith("{") and value_str.endswith("}"):
            try:
                return ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                pass

        # Default to string (remove quotes if present)
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        return value_str

    def _parse_simple_value(self, value_str: str) -> Any:
        """Parse simple values (used for list items)."""
        value_str = value_str.strip()

        if value_str.lower() == "true":
            return True
        elif value_str.lower() == "false":
            return False
        elif value_str.lower() in ("null", "none"):
            return None

        try:
            if "." not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            return value_str

    def _set_nested_key(self, data: dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a nested key in a dictionary using dot notation.

        Args:
            data: Dictionary to modify
            key_path: Dot-separated key path (e.g., "model.learning_rate")
            value: Value to set
        """
        keys = key_path.split(".")
        current = data

        # Navigate to the parent of the final key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # If there's a conflict, convert to dict
                current[key] = {}
            current = current[key]

        # Set the final value
        final_key = keys[-1]
        current[final_key] = value


def create_override_config_from_args(
    override_strings: list[str] | None = None,
    use_namespace_prefix: bool = True,
    custom_arg_name: str | None = None,
) -> ConflgaConfig:
    """
    Convenience function to create override configuration from command line arguments.

    Args:
        override_strings: List of override strings. If None, parses from command line.
        use_namespace_prefix: If True, use --conflga-override to avoid conflicts.
        custom_arg_name: Custom argument name to use instead of defaults.

    Returns:
        ConflgaConfig: Configuration object with override values

    Example:
        >>> # Use default namespace prefix (recommended)
        >>> override_config = create_override_config_from_args([
        ...     "model.learning_rate=0.001",
        ...     "dataset.batch_size=32"
        ... ])
        >>>
        >>> # Use custom argument name
        >>> override_config = create_override_config_from_args(
        ...     custom_arg_name="--my-override"
        ... )
        >>>
        >>> # Use short options (may conflict)
        >>> override_config = create_override_config_from_args(
        ...     use_namespace_prefix=False
        ... )
    """
    cli = ConflgaCLI(
        use_namespace_prefix=use_namespace_prefix, custom_arg_name=custom_arg_name
    )
    return cli.parse_overrides(override_strings)
