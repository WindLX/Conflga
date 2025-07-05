import pytest
import rtoml as toml
import os
import tempfile
import sys
from unittest.mock import patch, MagicMock

from conflga import ConflgaConfig, conflga_main
from conflga.cli import ConflgaCLI


@pytest.fixture
def temp_config_dir():
    """Creates a temporary directory for config files and cleans up after tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def create_toml_file(temp_config_dir):
    """
    A fixture factory to create TOML files in the temporary config directory.
    Usage: create_toml_file("filename", {"key": "value"})
    """

    def _creator(filename: str, content: dict):
        filepath = os.path.join(temp_config_dir, f"{filename}.toml")
        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(content, f)
        return filepath

    return _creator


# --- Basic Decorator Tests ---


def test_conflga_main_decorator_basic(temp_config_dir, create_toml_file):
    """Test basic decorator functionality with default configuration."""
    create_toml_file("base_cfg", {"test_val": 100})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_main(config_dir=temp_config_dir, default_config="base_cfg")
        def decorated_func(cfg: ConflgaConfig):
            return cfg.test_val

        result = decorated_func()
        assert result == 100


def test_conflga_main_decorator_with_merge(temp_config_dir, create_toml_file):
    """Test decorator with config merging functionality."""
    create_toml_file("main_cfg", {"model": "CNN", "lr": 0.01})
    create_toml_file("exp_cfg", {"lr": 0.001, "epochs": 50})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="main_cfg",
            configs_to_merge=["exp_cfg"],
        )
        def decorated_func_with_merge(cfg: ConflgaConfig):
            return cfg.model, cfg.lr, cfg.epochs

        model, lr, epochs = decorated_func_with_merge()
        assert model == "CNN"
        assert lr == 0.001  # Should be overridden by exp_cfg
        assert epochs == 50


def test_conflga_main_decorator_passes_other_args(temp_config_dir, create_toml_file):
    """Test that decorator properly passes through other function arguments."""
    create_toml_file("arg_cfg", {"val": 1})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_main(config_dir=temp_config_dir, default_config="arg_cfg")
        def decorated_func_with_args(cfg: ConflgaConfig, x: int, y: str = "default"):
            return cfg.val + x, y

        result_val, result_y = decorated_func_with_args(5, y="hello")
        assert result_val == 6
        assert result_y == "hello"


def test_conflga_main_decorator_no_merge_configs(temp_config_dir, create_toml_file):
    """Test decorator with no additional configs to merge."""
    create_toml_file("simple_cfg", {"value": 123})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="simple_cfg",
            configs_to_merge=None,
        )
        def func_no_merge(cfg: ConflgaConfig):
            return cfg.value

        assert func_no_merge() == 123


# --- Error Handling Tests ---


def test_conflga_main_decorator_missing_default_config(temp_config_dir):
    """Test decorator behavior when default config file is missing - should create empty config."""

    @conflga_main(config_dir=temp_config_dir, default_config="non_existent")
    def func_missing_default(cfg: ConflgaConfig):
        assert len(cfg) == 0  # Should have empty config
        return "success"

    result = func_missing_default()
    assert result == "success"


def test_conflga_main_decorator_missing_merge_config(temp_config_dir, create_toml_file):
    """Test decorator behavior when merge config file is missing - should skip missing configs."""
    create_toml_file("base_for_missing", {"key": "value"})

    @conflga_main(
        config_dir=temp_config_dir,
        default_config="base_for_missing",
        configs_to_merge=["missing_one"],
    )
    def func_missing_merge(cfg: ConflgaConfig):
        # Should only have the base config, missing merge configs are ignored
        assert cfg.key == "value"
        assert len(cfg) == 1  # Only the base config key
        return "success"

    result = func_missing_merge()
    assert result == "success"


# --- CLI Override Tests ---


def test_conflga_main_decorator_cli_override_disabled(
    temp_config_dir, create_toml_file
):
    """Test decorator with CLI override disabled."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    @conflga_main(
        config_dir=temp_config_dir, default_config="base_cfg", enable_cli_override=False
    )
    def func_no_cli(cfg: ConflgaConfig):
        return cfg.model.lr, cfg.epochs

    # Even if there were CLI args, they should be ignored
    lr, epochs = func_no_cli()
    assert lr == 0.01
    assert epochs == 10


def test_conflga_main_decorator_cli_override_enabled_no_args(
    temp_config_dir, create_toml_file
):
    """Test decorator with CLI override enabled but no override arguments."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    with patch("sys.argv", ["test_script.py"]):  # No override args

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            enable_cli_override=True,
        )
        def func_with_cli_no_args(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs

        lr, epochs = func_with_cli_no_args()
        assert lr == 0.01
        assert epochs == 10


def test_conflga_main_decorator_cli_override_with_args(
    temp_config_dir, create_toml_file
):
    """Test decorator with CLI override enabled and override arguments."""
    create_toml_file(
        "base_cfg", {"model": {"lr": 0.01}, "epochs": 10, "batch_size": 32}
    )

    # Mock sys.argv to simulate command line arguments
    with patch(
        "sys.argv",
        [
            "test_script.py",
            "-o",
            "model.lr=0.001",
            "-o",
            "epochs=100",
            "-o",
            "model.dropout=0.5",
        ],
    ):

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            enable_cli_override=True,
            use_namespace_prefix=False,  # Use -o for backward compatibility in tests
        )
        def func_with_cli_args(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs, cfg.batch_size, cfg.model.dropout

        lr, epochs, batch_size, dropout = func_with_cli_args()
        assert lr == 0.001  # Overridden
        assert epochs == 100  # Overridden
        assert batch_size == 32  # Not overridden
        assert dropout == 0.5  # New value added


def test_conflga_main_decorator_cli_override_complex_values(
    temp_config_dir, create_toml_file
):
    """Test decorator with CLI override using complex values."""
    create_toml_file(
        "base_cfg", {"model": {"layers": [64, 32]}, "training": {"enabled": True}}
    )

    with patch(
        "sys.argv",
        [
            "test_script.py",
            "-o",
            "model.layers=[128,64,32]",
            "-o",
            "training.enabled=false",
            "-o",
            'data.paths=["path1","path2"]',
        ],
    ):

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            enable_cli_override=True,
            use_namespace_prefix=False,  # Use -o for backward compatibility in tests
        )
        def func_with_complex_overrides(cfg: ConflgaConfig):
            return cfg.model.layers, cfg.training.enabled, cfg.data.paths

        layers, enabled, paths = func_with_complex_overrides()
        assert layers == [128, 64, 32]
        assert enabled is False
        assert paths == ["path1", "path2"]


def test_conflga_main_decorator_cli_override_with_merge(
    temp_config_dir, create_toml_file
):
    """Test decorator with both config merging and CLI overrides."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01}, "epochs": 10})
    create_toml_file("exp_cfg", {"model": {"lr": 0.005}, "batch_size": 64})

    with patch(
        "sys.argv", ["test_script.py", "-o", "model.lr=0.001", "-o", "epochs=200"]
    ):

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            configs_to_merge=["exp_cfg"],
            enable_cli_override=True,
            use_namespace_prefix=False,  # Use -o for backward compatibility in tests
        )
        def func_merge_and_override(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs, cfg.batch_size

        lr, epochs, batch_size = func_merge_and_override()
        assert lr == 0.001  # CLI override takes precedence
        assert epochs == 200  # CLI override
        assert batch_size == 64  # From merged config


def test_conflga_main_decorator_nested_overrides(temp_config_dir, create_toml_file):
    """Test decorator with deeply nested CLI overrides."""
    create_toml_file(
        "nested_cfg",
        {
            "model": {
                "architecture": {
                    "encoder": {"layers": 6, "attention_heads": 8},
                    "decoder": {"layers": 6},
                }
            }
        },
    )

    with patch(
        "sys.argv",
        [
            "test_script.py",
            "-o",
            "model.architecture.encoder.layers=12",
            "-o",
            "model.architecture.encoder.attention_heads=16",
            "-o",
            "model.architecture.decoder.dropout=0.1",
        ],
    ):

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="nested_cfg",
            enable_cli_override=True,
            use_namespace_prefix=False,  # Use -o for backward compatibility in tests
        )
        def func_nested_overrides(cfg: ConflgaConfig):
            return (
                cfg.model.architecture.encoder.layers,
                cfg.model.architecture.encoder.attention_heads,
                cfg.model.architecture.decoder.layers,
                cfg.model.architecture.decoder.dropout,
            )

        enc_layers, attention_heads, dec_layers, dropout = func_nested_overrides()
        assert enc_layers == 12  # Overridden
        assert attention_heads == 16  # Overridden
        assert dec_layers == 6  # Original value
        assert dropout == 0.1  # New value


# --- Function Signature Tests ---


def test_conflga_main_decorator_preserves_function_metadata(
    temp_config_dir, create_toml_file
):
    """Test that decorator preserves function metadata."""
    create_toml_file("meta_cfg", {"value": 42})

    @conflga_main(config_dir=temp_config_dir, default_config="meta_cfg")
    def documented_function(cfg: ConflgaConfig):
        """This is a documented function."""
        return cfg.value

    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a documented function."


def test_conflga_main_decorator_with_kwargs(temp_config_dir, create_toml_file):
    """Test decorator with functions that accept **kwargs."""
    create_toml_file("kwargs_cfg", {"base_value": 10})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_main(config_dir=temp_config_dir, default_config="kwargs_cfg")
        def func_with_kwargs(cfg: ConflgaConfig, multiplier: int = 1, **kwargs):
            return cfg.base_value * multiplier, kwargs

        result_value, kwargs_dict = func_with_kwargs(multiplier=3, extra_param="test")
        assert result_value == 30
        assert kwargs_dict == {"extra_param": "test"}


def test_conflga_main_decorator_with_args_and_kwargs(temp_config_dir, create_toml_file):
    """Test decorator with functions that accept *args and **kwargs."""
    create_toml_file("flexible_cfg", {"multiplier": 2})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_main(config_dir=temp_config_dir, default_config="flexible_cfg")
        def flexible_func(cfg: ConflgaConfig, *args, **kwargs):
            return cfg.multiplier, args, kwargs

        multiplier, args_tuple, kwargs_dict = flexible_func(1, 2, 3, key="value")
        assert multiplier == 2
        assert args_tuple == (1, 2, 3)
        assert kwargs_dict == {"key": "value"}


def test_conflga_main_decorator_namespace_prefix_parameter(
    temp_config_dir, create_toml_file
):
    """Test decorator with namespace prefix parameter."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    # Test with namespace prefix enabled (default)
    with patch(
        "sys.argv",
        [
            "test_script.py",
            "--conflga-override",
            "model.lr=0.001",
            "--conflga-override",
            "epochs=100",
        ],
    ):

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            enable_cli_override=True,
            use_namespace_prefix=True,  # Default behavior
        )
        def func_with_namespace_prefix(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs

        lr, epochs = func_with_namespace_prefix()
        assert lr == 0.001
        assert epochs == 100


def test_conflga_main_decorator_backward_compatibility_parameter(
    temp_config_dir, create_toml_file
):
    """Test decorator with backward compatibility parameter."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    # Test with namespace prefix disabled (backward compatibility)
    with patch(
        "sys.argv",
        ["test_script.py", "-o", "model.lr=0.001", "-o", "epochs=100"],
    ):

        @conflga_main(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            enable_cli_override=True,
            use_namespace_prefix=False,  # Backward compatibility
        )
        def func_with_backward_compatibility(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs

        lr, epochs = func_with_backward_compatibility()
        assert lr == 0.001
        assert epochs == 100
