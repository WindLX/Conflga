import pytest
import rtoml as toml
import os
import tempfile
import sys
from unittest.mock import patch, MagicMock

from conflga import ConflgaConfig, conflga_entry
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

        @conflga_entry(config_dir=temp_config_dir, default_config="base_cfg")
        def decorated_func(cfg: ConflgaConfig):
            return cfg.test_val

        result = decorated_func()
        assert result == 100


def test_conflga_main_decorator_with_merge(temp_config_dir, create_toml_file):
    """Test decorator with config merging functionality."""
    create_toml_file("main_cfg", {"model": "CNN", "lr": 0.01})
    create_toml_file("exp_cfg", {"lr": 0.001, "epochs": 50})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_entry(
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

        @conflga_entry(config_dir=temp_config_dir, default_config="arg_cfg")
        def decorated_func_with_args(cfg: ConflgaConfig, x: int, y: str = "default"):
            return cfg.val + x, y

        result_val, result_y = decorated_func_with_args(5, y="hello")
        assert result_val == 6
        assert result_y == "hello"


def test_conflga_main_decorator_no_merge_configs(temp_config_dir, create_toml_file):
    """Test decorator with no additional configs to merge."""
    create_toml_file("simple_cfg", {"value": 123})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="simple_cfg",
            configs_to_merge=None,
        )
        def func_no_merge(cfg: ConflgaConfig):
            return cfg.value

        assert func_no_merge() == 123


# --- Error Handling Tests ---


def test_conflga_main_decorator_missing_default_config(temp_config_dir):
    """Test decorator behavior when default config file is missing."""

    @conflga_entry(config_dir=temp_config_dir, default_config="non_existent")
    def func_missing_default(cfg: ConflgaConfig):
        pass  # This won't be reached

    with pytest.raises(FileNotFoundError, match="Default config file not found"):
        func_missing_default()


def test_conflga_main_decorator_missing_merge_config(temp_config_dir, create_toml_file):
    """Test decorator behavior when merge config file is missing."""
    create_toml_file("base_for_missing", {"key": "value"})

    @conflga_entry(
        config_dir=temp_config_dir,
        default_config="base_for_missing",
        configs_to_merge=["missing_one"],
    )
    def func_missing_merge(cfg: ConflgaConfig):
        pass  # This won't be reached

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        func_missing_merge()


# --- CLI Override Tests ---


def test_conflga_main_decorator_cli_override_disabled(
    temp_config_dir, create_toml_file
):
    """Test decorator with CLI override disabled."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    @conflga_entry(
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

        @conflga_entry(
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

        @conflga_entry(
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

        @conflga_entry(
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

        @conflga_entry(
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

        @conflga_entry(
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

    @conflga_entry(config_dir=temp_config_dir, default_config="meta_cfg")
    def documented_function(cfg: ConflgaConfig):
        """This is a documented function."""
        return cfg.value

    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a documented function."


def test_conflga_main_decorator_with_kwargs(temp_config_dir, create_toml_file):
    """Test decorator with functions that accept **kwargs."""
    create_toml_file("kwargs_cfg", {"base_value": 10})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_entry(config_dir=temp_config_dir, default_config="kwargs_cfg")
        def func_with_kwargs(cfg: ConflgaConfig, multiplier: int = 1, **kwargs):
            return cfg.base_value * multiplier, kwargs

        result_value, kwargs_dict = func_with_kwargs(multiplier=3, extra_param="test")
        assert result_value == 30
        assert kwargs_dict == {"extra_param": "test"}


def test_conflga_main_decorator_with_args_and_kwargs(temp_config_dir, create_toml_file):
    """Test decorator with functions that accept *args and **kwargs."""
    create_toml_file("flexible_cfg", {"multiplier": 2})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_entry(config_dir=temp_config_dir, default_config="flexible_cfg")
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

        @conflga_entry(
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

        @conflga_entry(
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


def test_conflga_main_decorator_multiple_functions(temp_config_dir, create_toml_file):
    """Test that multiple functions can be decorated and each receives configuration correctly."""
    create_toml_file(
        "multi_cfg",
        {
            "model": {"lr": 0.01, "dropout": 0.2},
            "epochs": 10,
            "batch_size": 32,
            "data": {"path": "/data", "augment": True},
        },
    )

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        # First function using the decorator
        @conflga_entry(config_dir=temp_config_dir, default_config="multi_cfg")
        def train_model(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs, cfg.batch_size

        # Second function using the same decorator
        @conflga_entry(config_dir=temp_config_dir, default_config="multi_cfg")
        def prepare_data(cfg: ConflgaConfig):
            return cfg.data.path, cfg.data.augment, cfg.batch_size

        # Third function with additional arguments
        @conflga_entry(config_dir=temp_config_dir, default_config="multi_cfg")
        def evaluate_model(cfg: ConflgaConfig, model_name: str = "default"):
            return cfg.model.dropout, cfg.epochs, model_name

        # Test that all functions work independently and correctly
        lr, epochs, batch_size = train_model()
        assert lr == 0.01
        assert epochs == 10
        assert batch_size == 32

        data_path, augment, batch_size_data = prepare_data()
        assert data_path == "/data"
        assert augment is True
        assert batch_size_data == 32

        dropout, epochs_eval, model_name = evaluate_model("custom_model")
        assert dropout == 0.2
        assert epochs_eval == 10
        assert model_name == "custom_model"


def test_conflga_main_decorator_multiple_functions_with_cli_override(
    temp_config_dir, create_toml_file
):
    """Test that multiple functions work correctly with CLI overrides."""
    create_toml_file(
        "multi_override_cfg",
        {
            "model": {"lr": 0.01, "layers": [64, 32]},
            "epochs": 10,
            "training": {"optimizer": "adam"},
        },
    )

    # Test with CLI overrides
    with patch(
        "sys.argv",
        [
            "test_script.py",
            "-o",
            "model.lr=0.001",
            "-o",
            "epochs=50",
            "-o",
            "training.optimizer=sgd",
        ],
    ):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="multi_override_cfg",
            enable_cli_override=True,
            use_namespace_prefix=False,
        )
        def function_one(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.epochs

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="multi_override_cfg",
            enable_cli_override=True,
            use_namespace_prefix=False,
        )
        def function_two(cfg: ConflgaConfig):
            return cfg.training.optimizer, cfg.model.layers

        # Both functions should see the CLI overrides
        lr, epochs = function_one()
        assert lr == 0.001  # Overridden by CLI
        assert epochs == 50  # Overridden by CLI

        optimizer, layers = function_two()
        assert optimizer == "sgd"  # Overridden by CLI
        assert layers == [64, 32]  # Original value


def test_conflga_main_decorator_multiple_functions_different_configs(
    temp_config_dir, create_toml_file
):
    """Test that multiple functions can use different base configurations."""
    create_toml_file("config_a", {"model": "CNN", "lr": 0.01})
    create_toml_file("config_b", {"model": "RNN", "lr": 0.001, "hidden_size": 128})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_entry(config_dir=temp_config_dir, default_config="config_a")
        def use_config_a(cfg: ConflgaConfig):
            return cfg.model, cfg.lr

        @conflga_entry(config_dir=temp_config_dir, default_config="config_b")
        def use_config_b(cfg: ConflgaConfig):
            return cfg.model, cfg.lr, cfg.hidden_size

        # Each function should use its own configuration
        model_a, lr_a = use_config_a()
        assert model_a == "CNN"
        assert lr_a == 0.01

        model_b, lr_b, hidden_size = use_config_b()
        assert model_b == "RNN"
        assert lr_b == 0.001
        assert hidden_size == 128


def test_conflga_main_decorator_multiple_functions_concurrent_calls(
    temp_config_dir, create_toml_file
):
    """Test that multiple decorated functions can be called in sequence without interference."""
    create_toml_file(
        "concurrent_cfg",
        {
            "global_setting": "shared",
            "func1": {"value": 100},
            "func2": {"value": 200},
            "func3": {"value": 300},
        },
    )

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_entry(config_dir=temp_config_dir, default_config="concurrent_cfg")
        def first_function(cfg: ConflgaConfig):
            return cfg.global_setting, cfg.func1.value

        @conflga_entry(config_dir=temp_config_dir, default_config="concurrent_cfg")
        def second_function(cfg: ConflgaConfig):
            return cfg.global_setting, cfg.func2.value

        @conflga_entry(config_dir=temp_config_dir, default_config="concurrent_cfg")
        def third_function(cfg: ConflgaConfig):
            return cfg.global_setting, cfg.func3.value

        # Call functions multiple times and in different orders
        # to ensure no state interference between calls

        # First round of calls
        global1, val1 = first_function()
        global2, val2 = second_function()
        global3, val3 = third_function()

        assert global1 == "shared"
        assert val1 == 100
        assert global2 == "shared"
        assert val2 == 200
        assert global3 == "shared"
        assert val3 == 300

        # Second round in different order
        global3_2, val3_2 = third_function()
        global1_2, val1_2 = first_function()
        global2_2, val2_2 = second_function()

        assert global3_2 == "shared"
        assert val3_2 == 300
        assert global1_2 == "shared"
        assert val1_2 == 100
        assert global2_2 == "shared"
        assert val2_2 == 200


# --- Preprocessor Tests ---


def test_conflga_decorator_with_preprocessor_enabled(temp_config_dir, create_toml_file):
    """Test decorator with preprocessor enabled using macros and templates."""
    # Create a config file with macros and templates
    config_content = """
#define APP_NAME = "MyApp"
#define VERSION = "1.0.0"
#define PORT = 8080
#define DEBUG = true

[app]
name = "{{ APP_NAME }}"
version = "{{ VERSION }}"
debug = {{ DEBUG }}

[server]
port = {{ PORT }}
host = "localhost:{{ PORT }}"
"""

    # Write the raw config content (not using create_toml_file since it has macros)
    config_path = os.path.join(temp_config_dir, "macro_config.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="macro_config",
            enable_preprocessor=True,
            auto_print=False,  # Disable auto print for test
        )
        def func_with_preprocessor(cfg: ConflgaConfig):
            return (
                cfg.app.name,
                cfg.app.version,
                cfg.app.debug,
                cfg.server.port,
                cfg.server.host,
            )

        name, version, debug, port, host = func_with_preprocessor()
        assert name == "MyApp"
        assert version == "1.0.0"
        assert debug is True
        assert port == 8080
        assert host == "localhost:8080"


def test_conflga_decorator_with_preprocessor_disabled(temp_config_dir):
    """Test decorator with preprocessor disabled - macros should remain as strings."""
    # Create a config file with macros and templates
    config_content = """
#define APP_NAME = "MyApp"
#define PORT = 8080

[app]
name = "{{ APP_NAME }}"

[server]
port = "{{ PORT }}"
"""

    config_path = os.path.join(temp_config_dir, "raw_config.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="raw_config",
            enable_preprocessor=False,
            auto_print=False,
        )
        def func_without_preprocessor(cfg: ConflgaConfig):
            return cfg.app.name, cfg.server.port

        name, port = func_without_preprocessor()
        # Templates should remain as literal strings when preprocessor is disabled
        assert name == "{{ APP_NAME }}"
        assert port == "{{ PORT }}"


def test_conflga_decorator_preprocessor_with_merge_configs(temp_config_dir):
    """Test decorator with preprocessor enabled and multiple config files."""
    # Base config with macros
    base_content = """
#define BASE_URL = "https://api.example.com"
#define VERSION = "v1"
#define TIMEOUT = 30

[api]
base_url = "{{ BASE_URL }}"
version = "{{ VERSION }}"
timeout = {{ TIMEOUT }}
"""

    # Override config with macros that reference base values
    override_content = """
#define RETRY_COUNT = 3
#define TOTAL_TIMEOUT = TIMEOUT * RETRY_COUNT

[api]
retries = {{ RETRY_COUNT }}
total_timeout = {{ TOTAL_TIMEOUT }}

[features]
enabled = {{ TIMEOUT > 10 }}
"""

    base_path = os.path.join(temp_config_dir, "base.toml")
    override_path = os.path.join(temp_config_dir, "override.toml")

    with open(base_path, "w", encoding="utf-8") as f:
        f.write(base_content)
    with open(override_path, "w", encoding="utf-8") as f:
        f.write(override_content)

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="base",
            configs_to_merge=["override"],
            enable_preprocessor=True,
            auto_print=False,
        )
        def func_with_merged_preprocessor(cfg: ConflgaConfig):
            return (
                cfg.api.base_url,
                cfg.api.version,
                cfg.api.timeout,
                cfg.api.retries,
                cfg.api.total_timeout,
                cfg.features.enabled,
            )

        base_url, version, timeout, retries, total_timeout, enabled = (
            func_with_merged_preprocessor()
        )
        assert base_url == "https://api.example.com"
        assert version == "v1"
        assert timeout == 30
        assert retries == 3
        assert total_timeout == 90
        assert enabled is True


def test_conflga_decorator_preprocessor_with_cli_override(temp_config_dir):
    """Test decorator with preprocessor and CLI overrides combined."""
    config_content = """
#define DEFAULT_PORT = 8080
#define DEBUG_MODE = false

[server]
port = {{ DEFAULT_PORT }}
debug = {{ DEBUG_MODE }}

[database]
port = {{ DEFAULT_PORT + 1 }}
"""

    config_path = os.path.join(temp_config_dir, "cli_test.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    # Test with CLI overrides
    with patch(
        "sys.argv",
        [
            "test_script.py",
            "--conflga-override",
            "server.port=9000",
            "--conflga-override",
            "server.debug=true",
        ],
    ):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="cli_test",
            enable_preprocessor=True,
            enable_cli_override=True,
            use_namespace_prefix=True,
            auto_print=False,
            auto_print_override=False,
        )
        def func_preprocessor_with_cli(cfg: ConflgaConfig):
            return cfg.server.port, cfg.server.debug, cfg.database.port

        server_port, debug, db_port = func_preprocessor_with_cli()
        assert server_port == 9000  # Overridden by CLI
        assert debug is True  # Overridden by CLI
        assert db_port == 8081  # Processed by preprocessor (8080 + 1)


def test_conflga_decorator_preprocessor_complex_expressions(temp_config_dir):
    """Test decorator with preprocessor using complex expressions."""
    config_content = """
#define WORKERS = 4
#define BASE_MEMORY = 512
#define MEMORY_PER_WORKER = BASE_MEMORY * 2
#define TOTAL_MEMORY = MEMORY_PER_WORKER * WORKERS
#define HIGH_MEMORY = TOTAL_MEMORY > 2000

[system]
workers = {{ WORKERS }}
memory_per_worker = {{ MEMORY_PER_WORKER }}
total_memory = {{ TOTAL_MEMORY }}
high_memory_mode = {{ HIGH_MEMORY }}

[scaling]
max_workers = {{ WORKERS * 2 }}
memory_limit = "{{ TOTAL_MEMORY }}MB"
"""

    config_path = os.path.join(temp_config_dir, "complex.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="complex",
            enable_preprocessor=True,
            auto_print=False,
        )
        def func_complex_preprocessor(cfg: ConflgaConfig):
            return (
                cfg.system.workers,
                cfg.system.memory_per_worker,
                cfg.system.total_memory,
                cfg.system.high_memory_mode,
                cfg.scaling.max_workers,
                cfg.scaling.memory_limit,
            )

        workers, mem_per_worker, total_mem, high_mem, max_workers, mem_limit = (
            func_complex_preprocessor()
        )
        assert workers == 4
        assert mem_per_worker == 1024  # 512 * 2
        assert total_mem == 4096  # 1024 * 4
        assert high_mem is True  # 4096 > 2000
        assert max_workers == 8  # 4 * 2
        assert mem_limit == "4096MB"


def test_conflga_decorator_preprocessor_string_operations(temp_config_dir):
    """Test decorator with preprocessor using string operations."""
    config_content = """
#define SERVICE_NAME = "api"
#define ENVIRONMENT = "prod"
#define NAMESPACE = SERVICE_NAME + "-" + ENVIRONMENT
#define IMAGE_TAG = "v1.2.3"

[deployment]
name = "{{ NAMESPACE }}"
image = "myregistry/{{ SERVICE_NAME }}:{{ IMAGE_TAG }}"
full_name = "{{ SERVICE_NAME }}-service-{{ ENVIRONMENT }}"

[labels]
app = "{{ SERVICE_NAME }}"
env = "{{ ENVIRONMENT }}"
version = "{{ IMAGE_TAG }}"
"""

    config_path = os.path.join(temp_config_dir, "strings.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="strings",
            enable_preprocessor=True,
            auto_print=False,
        )
        def func_string_preprocessor(cfg: ConflgaConfig):
            return (
                cfg.deployment.name,
                cfg.deployment.image,
                cfg.deployment.full_name,
                cfg.labels.app,
                cfg.labels.env,
                cfg.labels.version,
            )

        name, image, full_name, app, env, version = func_string_preprocessor()
        assert name == "api-prod"
        assert image == "myregistry/api:v1.2.3"
        assert full_name == "api-service-prod"
        assert app == "api"
        assert env == "prod"
        assert version == "v1.2.3"


def test_conflga_decorator_preprocessor_error_handling(temp_config_dir):
    """Test decorator with preprocessor error handling."""
    # Config with invalid macro
    invalid_config_content = """
#define INVALID = undefined_variable + 1

[test]
value = {{ INVALID }}
"""

    config_path = os.path.join(temp_config_dir, "invalid.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(invalid_config_content)

    @conflga_entry(
        config_dir=temp_config_dir,
        default_config="invalid",
        enable_preprocessor=True,
        auto_print=False,
    )
    def func_with_invalid_preprocessor(cfg: ConflgaConfig):
        return cfg.test.value

    # Should raise an error due to invalid macro
    with pytest.raises(RuntimeError, match="Marcos compute failed"):
        func_with_invalid_preprocessor()


def test_conflga_decorator_preprocessor_empty_config(temp_config_dir):
    """Test decorator with preprocessor on empty config."""
    # Empty config file
    empty_config_path = os.path.join(temp_config_dir, "empty.toml")
    with open(empty_config_path, "w", encoding="utf-8") as f:
        f.write("")

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="empty",
            enable_preprocessor=True,
            auto_print=False,
        )
        def func_empty_preprocessor(cfg: ConflgaConfig):
            return len(cfg._data)  # Should be empty

        result = func_empty_preprocessor()
        assert result == 0


def test_conflga_decorator_preprocessor_only_macros_no_templates(temp_config_dir):
    """Test decorator with preprocessor when config has only macros but no templates."""
    config_content = """
#define PORT = 8080
#define DEBUG = true
#define NAME = "test"

[server]
port = 3000
debug = false
name = "actual"
"""

    config_path = os.path.join(temp_config_dir, "no_templates.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    with patch("sys.argv", ["test_script.py"]):

        @conflga_entry(
            config_dir=temp_config_dir,
            default_config="no_templates",
            enable_preprocessor=True,
            auto_print=False,
        )
        def func_no_templates(cfg: ConflgaConfig):
            return cfg.server.port, cfg.server.debug, cfg.server.name

        port, debug, name = func_no_templates()
        # Values should be from the actual config, not the macros
        assert port == 3000
        assert debug is False
        assert name == "actual"


def test_conflga_decorator_preprocessor_only_templates_no_macros(temp_config_dir):
    """Test decorator with preprocessor when config has templates but references undefined variables."""
    config_content = """
[server]
port = "{{ UNDEFINED_PORT }}"
"""

    config_path = os.path.join(temp_config_dir, "undefined_templates.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    @conflga_entry(
        config_dir=temp_config_dir,
        default_config="undefined_templates",
        enable_preprocessor=True,
        auto_print=False,
    )
    def func_undefined_templates(cfg: ConflgaConfig):
        return cfg.server.port

    # Should raise an error due to undefined variable in template
    with pytest.raises(RuntimeError, match="Expression evaluation failed"):
        func_undefined_templates()
