import pytest
import rtoml as toml
import os
import tempfile
from unittest.mock import patch

from conflga import ConflgaConfig, conflga_method, conflga_func


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


# --- Method Decorator Tests ---


def test_conflga_method_decorator_basic(temp_config_dir, create_toml_file):
    """Test basic method decorator functionality with default configuration."""
    create_toml_file("method_cfg", {"test_val": 100})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class TestClass:
            @conflga_method(config_dir=temp_config_dir, default_config="method_cfg")
            def get_value(self, cfg: ConflgaConfig):
                return cfg.test_val

        test_instance = TestClass()
        result = test_instance.get_value()
        assert result == 100


def test_conflga_method_decorator_with_merge(temp_config_dir, create_toml_file):
    """Test method decorator with config merging functionality."""
    create_toml_file("method_main_cfg", {"model": "CNN", "lr": 0.01})
    create_toml_file("method_exp_cfg", {"lr": 0.001, "epochs": 50})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class TrainingClass:
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="method_main_cfg",
                configs_to_merge=["method_exp_cfg"],
            )
            def get_config(self, cfg: ConflgaConfig):
                return cfg.model, cfg.lr, cfg.epochs

        trainer = TrainingClass()
        model, lr, epochs = trainer.get_config()
        assert model == "CNN"
        assert lr == 0.001  # Should be overridden by exp_cfg
        assert epochs == 50


def test_conflga_method_decorator_passes_other_args(temp_config_dir, create_toml_file):
    """Test that method decorator properly passes through other method arguments."""
    create_toml_file("method_arg_cfg", {"val": 1})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class ArgsClass:
            def __init__(self, multiplier=1):
                self.multiplier = multiplier

            @conflga_method(config_dir=temp_config_dir, default_config="method_arg_cfg")
            def calculate(self, cfg: ConflgaConfig, x: int, y: str = "default"):
                return (cfg.val + x) * self.multiplier, y

        calc = ArgsClass(multiplier=2)
        result_val, result_y = calc.calculate(5, y="hello")
        assert result_val == 12  # (1 + 5) * 2
        assert result_y == "hello"


def test_conflga_method_decorator_cli_override(temp_config_dir, create_toml_file):
    """Test method decorator with CLI override functionality."""
    create_toml_file("method_cli_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    with patch(
        "sys.argv",
        [
            "test_script.py",
            "-o",
            "model.lr=0.001",
            "-o",
            "epochs=100",
        ],
    ):

        class CLIClass:
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="method_cli_cfg",
                enable_cli_override=True,
                use_namespace_prefix=False,  # Use -o for backward compatibility
            )
            def get_params(self, cfg: ConflgaConfig):
                return cfg.model.lr, cfg.epochs

        cli_instance = CLIClass()
        lr, epochs = cli_instance.get_params()
        assert lr == 0.001  # Overridden
        assert epochs == 100  # Overridden


def test_conflga_method_decorator_multiple_methods_same_class(
    temp_config_dir, create_toml_file
):
    """Test multiple methods in the same class using the decorator."""
    create_toml_file(
        "multi_method_cfg",
        {
            "model": {"lr": 0.01, "layers": [64, 32]},
            "data": {"batch_size": 32, "path": "/data"},
            "training": {"epochs": 10, "optimizer": "adam"},
        },
    )

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class MultiMethodClass:
            @conflga_method(
                config_dir=temp_config_dir, default_config="multi_method_cfg"
            )
            def get_model_config(self, cfg: ConflgaConfig):
                return cfg.model.lr, cfg.model.layers

            @conflga_method(
                config_dir=temp_config_dir, default_config="multi_method_cfg"
            )
            def get_data_config(self, cfg: ConflgaConfig):
                return cfg.data.batch_size, cfg.data.path

            @conflga_method(
                config_dir=temp_config_dir, default_config="multi_method_cfg"
            )
            def get_training_config(self, cfg: ConflgaConfig):
                return cfg.training.epochs, cfg.training.optimizer

        instance = MultiMethodClass()

        # Test each method independently
        lr, layers = instance.get_model_config()
        assert lr == 0.01
        assert layers == [64, 32]

        batch_size, path = instance.get_data_config()
        assert batch_size == 32
        assert path == "/data"

        epochs, optimizer = instance.get_training_config()
        assert epochs == 10
        assert optimizer == "adam"


def test_conflga_method_decorator_inheritance(temp_config_dir, create_toml_file):
    """Test method decorator with class inheritance."""
    create_toml_file("inherit_cfg", {"base_value": 100, "derived_value": 200})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class BaseClass:
            @conflga_method(config_dir=temp_config_dir, default_config="inherit_cfg")
            def get_base_value(self, cfg: ConflgaConfig):
                return cfg.base_value

        class DerivedClass(BaseClass):
            @conflga_method(config_dir=temp_config_dir, default_config="inherit_cfg")
            def get_derived_value(self, cfg: ConflgaConfig):
                return cfg.derived_value

            @conflga_method(config_dir=temp_config_dir, default_config="inherit_cfg")
            def get_both_values(self, cfg: ConflgaConfig):
                return cfg.base_value, cfg.derived_value

        derived = DerivedClass()

        # Test inherited method
        base_val = derived.get_base_value()
        assert base_val == 100

        # Test derived methods
        derived_val = derived.get_derived_value()
        assert derived_val == 200

        base_val2, derived_val2 = derived.get_both_values()
        assert base_val2 == 100
        assert derived_val2 == 200


def test_conflga_method_decorator_with_properties(temp_config_dir, create_toml_file):
    """Test method decorator with class properties and state."""
    create_toml_file("state_cfg", {"multiplier": 3, "offset": 10})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class StatefulClass:
            def __init__(self, initial_value=5):
                self.value = initial_value

            @conflga_method(config_dir=temp_config_dir, default_config="state_cfg")
            def calculate_with_config(self, cfg: ConflgaConfig):
                return self.value * cfg.multiplier + cfg.offset

            @conflga_method(config_dir=temp_config_dir, default_config="state_cfg")
            def update_value(self, cfg: ConflgaConfig, new_value: int):
                self.value = new_value * cfg.multiplier
                return self.value

        instance = StatefulClass(initial_value=2)

        # Test calculation with initial state
        result1 = instance.calculate_with_config()
        assert result1 == 16  # 2 * 3 + 10

        # Test state modification
        new_val = instance.update_value(4)
        assert new_val == 12  # 4 * 3
        assert instance.value == 12

        # Test calculation with updated state
        result2 = instance.calculate_with_config()
        assert result2 == 46  # 12 * 3 + 10


def test_conflga_method_decorator_namespace_prefix(temp_config_dir, create_toml_file):
    """Test method decorator with namespace prefix parameter."""
    create_toml_file("namespace_cfg", {"model": {"lr": 0.01}, "epochs": 10})

    # Test with namespace prefix enabled
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

        class NamespaceClass:
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="namespace_cfg",
                enable_cli_override=True,
                use_namespace_prefix=True,  # Default behavior
            )
            def get_config(self, cfg: ConflgaConfig):
                return cfg.model.lr, cfg.epochs

        instance = NamespaceClass()
        lr, epochs = instance.get_config()
        assert lr == 0.001
        assert epochs == 100


def test_conflga_method_decorator_with_preprocessor(temp_config_dir):
    """Test method decorator with preprocessor enabled."""
    config_content = """
#let BASE_LR = 0.01
#let EPOCHS = 100
#let BATCH_SIZE = 32

[training]
learning_rate = {{ BASE_LR }}
epochs = {{ EPOCHS }}
batch_size = {{ BATCH_SIZE }}

[model]
hidden_size = {{ BATCH_SIZE * 4 }}
"""

    config_path = os.path.join(temp_config_dir, "preprocessor_method.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    with patch("sys.argv", ["test_script.py"]):

        class PreprocessorClass:
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="preprocessor_method",
                enable_preprocessor=True,
                auto_print=False,  # Disable auto print for test
            )
            def get_training_params(self, cfg: ConflgaConfig):
                return (
                    cfg.training.learning_rate,
                    cfg.training.epochs,
                    cfg.training.batch_size,
                    cfg.model.hidden_size,
                )

        instance = PreprocessorClass()
        lr, epochs, batch_size, hidden_size = instance.get_training_params()
        assert lr == 0.01
        assert epochs == 100
        assert batch_size == 32
        assert hidden_size == 128  # 32 * 4


def test_conflga_method_decorator_error_handling(temp_config_dir):
    """Test method decorator error handling with missing config files."""

    class ErrorTestClass:
        @conflga_method(config_dir=temp_config_dir, default_config="non_existent")
        def failing_method(self, cfg: ConflgaConfig):
            pass  # This won't be reached

    instance = ErrorTestClass()
    with pytest.raises(FileNotFoundError, match="Default config file not found"):
        instance.failing_method()


def test_conflga_method_decorator_preserves_method_metadata(
    temp_config_dir, create_toml_file
):
    """Test that method decorator preserves method metadata."""
    create_toml_file("meta_method_cfg", {"value": 42})

    class MetadataClass:
        @conflga_method(config_dir=temp_config_dir, default_config="meta_method_cfg")
        def documented_method(self, cfg: ConflgaConfig):
            """This is a documented method."""
            return cfg.value

    instance = MetadataClass()
    assert instance.documented_method.__name__ == "documented_method"
    assert instance.documented_method.__doc__ == "This is a documented method."


def test_conflga_method_decorator_with_kwargs(temp_config_dir, create_toml_file):
    """Test method decorator with methods that accept **kwargs."""
    create_toml_file("kwargs_method_cfg", {"base_value": 10})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class KwargsClass:
            @conflga_method(
                config_dir=temp_config_dir, default_config="kwargs_method_cfg"
            )
            def method_with_kwargs(
                self, cfg: ConflgaConfig, multiplier: int = 1, **kwargs
            ):
                return cfg.base_value * multiplier, kwargs

        instance = KwargsClass()
        result_value, kwargs_dict = instance.method_with_kwargs(
            multiplier=3, extra_param="test"
        )
        assert result_value == 30
        assert kwargs_dict == {"extra_param": "test"}


def test_conflga_method_decorator_static_and_class_methods(
    temp_config_dir, create_toml_file
):
    """Test method decorator with static and class methods."""
    create_toml_file("static_cfg", {"value": 100, "class_value": 200})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class StaticMethodClass:
            class_attribute = "test"

            @staticmethod
            @conflga_func(config_dir=temp_config_dir, default_config="static_cfg")
            def static_method(cfg: ConflgaConfig):
                return cfg.value

            @classmethod
            @conflga_method(config_dir=temp_config_dir, default_config="static_cfg")
            def class_method(cls, cfg: ConflgaConfig):
                return cfg.class_value, cls.class_attribute

        # Test static method
        static_result = StaticMethodClass.static_method()
        assert static_result == 100

        # Test class method
        class_result, attr = StaticMethodClass.class_method()
        assert class_result == 200
        assert attr == "test"


def test_conflga_method_decorator_multiple_instances(temp_config_dir, create_toml_file):
    """Test that method decorator works correctly with multiple instances of the same class."""
    create_toml_file("instance_cfg", {"multiplier": 5})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class InstanceClass:
            def __init__(self, instance_id: int):
                self.instance_id = instance_id

            @conflga_method(config_dir=temp_config_dir, default_config="instance_cfg")
            def calculate(self, cfg: ConflgaConfig):
                return self.instance_id * cfg.multiplier

        instance1 = InstanceClass(1)
        instance2 = InstanceClass(2)
        instance3 = InstanceClass(3)

        # Each instance should work independently
        result1 = instance1.calculate()
        result2 = instance2.calculate()
        result3 = instance3.calculate()

        assert result1 == 5  # 1 * 5
        assert result2 == 10  # 2 * 5
        assert result3 == 15  # 3 * 5

        # Call methods multiple times to ensure no interference
        result1_2 = instance1.calculate()
        result2_2 = instance2.calculate()

        assert result1_2 == 5
        assert result2_2 == 10
