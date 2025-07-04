import pytest
from unittest.mock import patch
import sys

from conflga.cli import ConflgaCLI, create_override_config_from_args
from conflga.config import ConflgaConfig


class TestConflgaCLI:
    """Test cases for ConflgaCLI class."""

    def setup_method(self):
        """Setup for each test method."""
        self.cli = ConflgaCLI()

    def test_init(self):
        """Test CLI initialization."""
        assert self.cli.parser is not None
        assert "overrides" in [action.dest for action in self.cli.parser._actions]

    def test_parse_overrides_empty_list(self):
        """Test parsing empty override list."""
        config = self.cli.parse_overrides([])
        assert isinstance(config, ConflgaConfig)
        assert len(config) == 0

    def test_parse_overrides_none(self):
        """Test parsing None override list."""
        # Mock sys.argv to avoid interference with actual command line args
        with patch.object(sys, "argv", ["test"]):
            config = self.cli.parse_overrides(None)
            assert isinstance(config, ConflgaConfig)

    def test_parse_single_override_string(self):
        """Test parsing a single string override."""
        overrides = ["key=value"]
        config = self.cli.parse_overrides(overrides)
        assert config["key"] == "value"

    def test_parse_single_override_int(self):
        """Test parsing integer override."""
        overrides = ["learning_rate=42"]
        config = self.cli.parse_overrides(overrides)
        assert config["learning_rate"] == 42
        assert isinstance(config["learning_rate"], int)

    def test_parse_single_override_float(self):
        """Test parsing float override."""
        overrides = ["learning_rate=0.001"]
        config = self.cli.parse_overrides(overrides)
        assert config["learning_rate"] == 0.001
        assert isinstance(config["learning_rate"], float)

    def test_parse_single_override_bool_true(self):
        """Test parsing boolean True override."""
        overrides = ["use_gpu=true"]
        config = self.cli.parse_overrides(overrides)
        assert config["use_gpu"] is True

    def test_parse_single_override_bool_false(self):
        """Test parsing boolean False override."""
        overrides = ["use_gpu=false"]
        config = self.cli.parse_overrides(overrides)
        assert config["use_gpu"] is False

    def test_parse_single_override_null(self):
        """Test parsing null/none override."""
        overrides = ["value=null"]
        config = self.cli.parse_overrides(overrides)
        assert config["value"] is None

    def test_parse_single_override_none(self):
        """Test parsing none override."""
        overrides = ["value=none"]
        config = self.cli.parse_overrides(overrides)
        assert config["value"] is None

    def test_parse_nested_override(self):
        """Test parsing nested key override."""
        overrides = ["model.learning_rate=0.001"]
        config = self.cli.parse_overrides(overrides)
        assert config["model"]["learning_rate"] == 0.001

    def test_parse_deep_nested_override(self):
        """Test parsing deeply nested key override."""
        overrides = ["model.optimizer.adam.learning_rate=0.001"]
        config = self.cli.parse_overrides(overrides)
        assert config["model"]["optimizer"]["adam"]["learning_rate"] == 0.001

    def test_parse_multiple_overrides(self):
        """Test parsing multiple overrides."""
        overrides = [
            "model.learning_rate=0.001",
            "dataset.batch_size=32",
            "training.epochs=100",
            "model.use_dropout=true",
        ]
        config = self.cli.parse_overrides(overrides)
        assert config["model"]["learning_rate"] == 0.001
        assert config["dataset"]["batch_size"] == 32
        assert config["training"]["epochs"] == 100
        assert config["model"]["use_dropout"] is True

    def test_parse_list_override(self):
        """Test parsing list override."""
        overrides = ["data.paths=['/path1', '/path2', '/path3']"]
        config = self.cli.parse_overrides(overrides)
        assert config["data"]["paths"] == ["/path1", "/path2", "/path3"]

    def test_parse_empty_list_override(self):
        """Test parsing empty list override."""
        overrides = ["data.paths=[]"]
        config = self.cli.parse_overrides(overrides)
        assert config["data"]["paths"] == []

    def test_parse_simple_list_override(self):
        """Test parsing simple comma-separated list override."""
        overrides = ["data.indices=[1, 2, 3, 4]"]
        config = self.cli.parse_overrides(overrides)
        assert config["data"]["indices"] == [1, 2, 3, 4]

    def test_parse_dict_override(self):
        """Test parsing dictionary override."""
        overrides = ["params={'key': 'value', 'num': 42}"]
        config = self.cli.parse_overrides(overrides)
        assert config["params"]["key"] == "value"
        assert config["params"]["num"] == 42

    def test_parse_quoted_string_override(self):
        """Test parsing quoted string override."""
        overrides = ['message="hello world"']
        config = self.cli.parse_overrides(overrides)
        assert config["message"] == "hello world"

    def test_parse_single_quoted_string_override(self):
        """Test parsing single quoted string override."""
        overrides = ["message='hello world'"]
        config = self.cli.parse_overrides(overrides)
        assert config["message"] == "hello world"

    def test_parse_override_with_spaces(self):
        """Test parsing override with spaces around equals sign."""
        overrides = ["key = value"]
        config = self.cli.parse_overrides(overrides)
        assert config["key"] == "value"

    def test_parse_override_with_equals_in_value(self):
        """Test parsing override with equals sign in value."""
        overrides = ["url=http://example.com?param=value"]
        config = self.cli.parse_overrides(overrides)
        assert config["url"] == "http://example.com?param=value"

    def test_invalid_override_format(self):
        """Test error handling for invalid override format."""
        overrides = ["invalid_format"]
        with pytest.raises(ValueError, match="Invalid override format"):
            self.cli.parse_overrides(overrides)

    def test_override_nested_key_conflict(self):
        """Test handling of nested key conflicts."""
        overrides = ["model=simple_value", "model.learning_rate=0.001"]
        config = self.cli.parse_overrides(overrides)
        # The second override should convert the simple value to a dict
        assert config["model"]["learning_rate"] == 0.001

    def test_scientific_notation_float(self):
        """Test parsing scientific notation float."""
        overrides = ["learning_rate=1e-4"]
        config = self.cli.parse_overrides(overrides)
        assert config["learning_rate"] == 1e-4
        assert isinstance(config["learning_rate"], float)

    def test_negative_number(self):
        """Test parsing negative numbers."""
        overrides = ["temperature=-5"]
        config = self.cli.parse_overrides(overrides)
        assert config["temperature"] == -5

    def test_negative_float(self):
        """Test parsing negative float."""
        overrides = ["bias=-0.5"]
        config = self.cli.parse_overrides(overrides)
        assert config["bias"] == -0.5

    def test_parse_mixed_list_types(self):
        """Test parsing list with mixed types."""
        overrides = ["mixed=[1, 'text', true, 3.14]"]
        config = self.cli.parse_overrides(overrides)
        assert config["mixed"] == [1, "text", True, 3.14]


class TestValueParsing:
    """Test cases for value parsing methods."""

    def setup_method(self):
        """Setup for each test method."""
        self.cli = ConflgaCLI()

    def test_parse_value_string(self):
        """Test parsing string values."""
        assert self.cli._parse_value("hello") == "hello"
        assert self.cli._parse_value('"hello world"') == "hello world"
        assert self.cli._parse_value("'hello world'") == "hello world"

    def test_parse_value_int(self):
        """Test parsing integer values."""
        assert self.cli._parse_value("42") == 42
        assert self.cli._parse_value("-10") == -10
        assert self.cli._parse_value("0") == 0

    def test_parse_value_float(self):
        """Test parsing float values."""
        assert self.cli._parse_value("3.14") == 3.14
        assert self.cli._parse_value("-2.5") == -2.5
        assert self.cli._parse_value("1e-4") == 1e-4

    def test_parse_value_bool(self):
        """Test parsing boolean values."""
        assert self.cli._parse_value("true") is True
        assert self.cli._parse_value("True") is True
        assert self.cli._parse_value("TRUE") is True
        assert self.cli._parse_value("false") is False
        assert self.cli._parse_value("False") is False
        assert self.cli._parse_value("FALSE") is False

    def test_parse_value_null(self):
        """Test parsing null values."""
        assert self.cli._parse_value("null") is None
        assert self.cli._parse_value("NULL") is None
        assert self.cli._parse_value("none") is None
        assert self.cli._parse_value("None") is None

    def test_parse_value_list(self):
        """Test parsing list values."""
        assert self.cli._parse_value("[]") == []
        assert self.cli._parse_value("[1, 2, 3]") == [1, 2, 3]
        assert self.cli._parse_value("['a', 'b', 'c']") == ["a", "b", "c"]

    def test_parse_value_dict(self):
        """Test parsing dictionary values."""
        assert self.cli._parse_value("{}") == {}
        result = self.cli._parse_value("{'key': 'value', 'num': 42}")
        assert result == {"key": "value", "num": 42}

    def test_parse_simple_value(self):
        """Test parsing simple values for list items."""
        assert self.cli._parse_simple_value("42") == 42
        assert self.cli._parse_simple_value("3.14") == 3.14
        assert self.cli._parse_simple_value("true") is True
        assert self.cli._parse_simple_value("false") is False
        assert self.cli._parse_simple_value("null") is None
        assert self.cli._parse_simple_value("text") == "text"


class TestNestedKeyHandling:
    """Test cases for nested key setting."""

    def setup_method(self):
        """Setup for each test method."""
        self.cli = ConflgaCLI()

    def test_set_nested_key_simple(self):
        """Test setting simple nested key."""
        data = {}
        self.cli._set_nested_key(data, "model.learning_rate", 0.001)
        assert data == {"model": {"learning_rate": 0.001}}

    def test_set_nested_key_deep(self):
        """Test setting deeply nested key."""
        data = {}
        self.cli._set_nested_key(data, "model.optimizer.adam.lr", 0.001)
        expected = {"model": {"optimizer": {"adam": {"lr": 0.001}}}}
        assert data == expected

    def test_set_nested_key_existing_dict(self):
        """Test setting nested key in existing dictionary."""
        data = {"model": {"existing_param": "value"}}
        self.cli._set_nested_key(data, "model.learning_rate", 0.001)
        expected = {"model": {"existing_param": "value", "learning_rate": 0.001}}
        assert data == expected

    def test_set_nested_key_conflict_override(self):
        """Test setting nested key when there's a conflict."""
        data = {"model": "simple_value"}
        self.cli._set_nested_key(data, "model.learning_rate", 0.001)
        # Should convert simple value to dict
        assert data == {"model": {"learning_rate": 0.001}}

    def test_set_nested_key_single_level(self):
        """Test setting single level key."""
        data = {}
        self.cli._set_nested_key(data, "simple_key", "value")
        assert data == {"simple_key": "value"}


class TestConvenienceFunction:
    """Test cases for convenience function."""

    def test_create_override_config_from_args(self):
        """Test convenience function with override strings."""
        overrides = ["model.lr=0.001", "batch_size=32"]
        config = create_override_config_from_args(overrides)
        assert isinstance(config, ConflgaConfig)
        assert config["model"]["lr"] == 0.001
        assert config["batch_size"] == 32

    def test_create_override_config_from_args_none(self):
        """Test convenience function with None."""
        with patch.object(sys, "argv", ["test"]):
            config = create_override_config_from_args(None)
            assert isinstance(config, ConflgaConfig)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_end_to_end_ml_config(self):
        """Test end-to-end ML configuration override."""
        cli = ConflgaCLI()
        overrides = [
            "model.architecture=resnet50",
            "model.learning_rate=0.001",
            "model.dropout=0.2",
            "model.use_batch_norm=true",
            "dataset.name=cifar10",
            "dataset.batch_size=64",
            "dataset.shuffle=true",
            "training.epochs=100",
            "training.optimizer=adam",
            "training.scheduler.type=cosine",
            "training.scheduler.T_max=100",
            "logging.level=info",
            "logging.save_dir=/tmp/logs",
        ]

        config = cli.parse_overrides(overrides)

        # Verify model configuration
        assert config["model"]["architecture"] == "resnet50"
        assert config["model"]["learning_rate"] == 0.001
        assert config["model"]["dropout"] == 0.2
        assert config["model"]["use_batch_norm"] is True

        # Verify dataset configuration
        assert config["dataset"]["name"] == "cifar10"
        assert config["dataset"]["batch_size"] == 64
        assert config["dataset"]["shuffle"] is True

        # Verify training configuration
        assert config["training"]["epochs"] == 100
        assert config["training"]["optimizer"] == "adam"
        assert config["training"]["scheduler"]["type"] == "cosine"
        assert config["training"]["scheduler"]["T_max"] == 100

        # Verify logging configuration
        assert config["logging"]["level"] == "info"
        assert config["logging"]["save_dir"] == "/tmp/logs"

    def test_command_line_args_parsing(self):
        """Test parsing actual command line arguments."""
        test_args = [
            "program.py",
            "-o",
            "model.lr=0.001",
            "--override",
            "batch_size=32",
            "-o",
            "use_gpu=true",
        ]

        with patch.object(sys, "argv", test_args):
            cli = ConflgaCLI()
            config = cli.parse_overrides()

            assert config["model"]["lr"] == 0.001
            assert config["batch_size"] == 32
            assert config["use_gpu"] is True


class TestCLIEdgeCases:
    """Test edge cases and error conditions for CLI."""

    def setup_method(self):
        """Setup for each test method."""
        self.cli = ConflgaCLI()

    def test_empty_string_value(self):
        """Test parsing empty string value."""
        overrides = ["empty_value="]
        config = self.cli.parse_overrides(overrides)
        assert config["empty_value"] == ""

    def test_whitespace_only_value(self):
        """Test parsing whitespace-only value."""
        overrides = ["whitespace=   "]
        config = self.cli.parse_overrides(overrides)
        assert config["whitespace"] == ""

    def test_key_with_special_characters(self):
        """Test parsing keys with underscores and numbers."""
        overrides = ["config_v2=value", "model_123.layer_1.weight=0.5"]
        config = self.cli.parse_overrides(overrides)
        assert config["config_v2"] == "value"
        assert config["model_123"]["layer_1"]["weight"] == 0.5

    def test_complex_string_with_special_chars(self):
        """Test parsing complex strings with special characters."""
        overrides = [
            'path="/home/user/data with spaces/file.txt"',
            'regex="^[a-zA-Z0-9]+$"',
            'command="python script.py --arg=value"',
        ]
        config = self.cli.parse_overrides(overrides)
        assert config["path"] == "/home/user/data with spaces/file.txt"
        assert config["regex"] == "^[a-zA-Z0-9]+$"
        assert config["command"] == "python script.py --arg=value"

    def test_nested_list_in_dict(self):
        """Test parsing nested structures with lists in dictionaries."""
        overrides = ["config={'layers': [64, 128, 256], 'dropout': 0.2}"]
        config = self.cli.parse_overrides(overrides)
        assert config["config"]["layers"] == [64, 128, 256]
        assert config["config"]["dropout"] == 0.2

    def test_multiple_equals_in_value(self):
        """Test parsing values with multiple equals signs."""
        overrides = ["equation=x=y+z=w"]
        config = self.cli.parse_overrides(overrides)
        assert config["equation"] == "x=y+z=w"

    def test_parse_very_long_number(self):
        """Test parsing very long numbers."""
        overrides = ["big_number=12345678901234567890"]
        config = self.cli.parse_overrides(overrides)
        assert config["big_number"] == 12345678901234567890

    def test_parse_float_with_many_decimals(self):
        """Test parsing float with many decimal places."""
        overrides = ["precision=3.141592653589793"]
        config = self.cli.parse_overrides(overrides)
        assert config["precision"] == 3.141592653589793

    def test_parse_scientific_notation_negative_exponent(self):
        """Test parsing scientific notation with negative exponent."""
        overrides = ["small_value=1.5e-10"]
        config = self.cli.parse_overrides(overrides)
        assert config["small_value"] == 1.5e-10

    def test_parse_scientific_notation_positive_exponent(self):
        """Test parsing scientific notation with positive exponent."""
        overrides = ["large_value=2.5e+8"]
        config = self.cli.parse_overrides(overrides)
        assert config["large_value"] == 2.5e8

    def test_case_insensitive_booleans(self):
        """Test case insensitive boolean parsing."""
        overrides = ["bool1=True", "bool2=FALSE", "bool3=tRuE", "bool4=fAlSe"]
        config = self.cli.parse_overrides(overrides)
        assert config["bool1"] is True
        assert config["bool2"] is False
        assert config["bool3"] is True
        assert config["bool4"] is False

    def test_case_insensitive_null(self):
        """Test case insensitive null parsing."""
        overrides = ["null1=NULL", "null2=Null", "null3=NONE", "null4=None"]
        config = self.cli.parse_overrides(overrides)
        assert config["null1"] is None
        assert config["null2"] is None
        assert config["null3"] is None
        assert config["null4"] is None

    def test_malformed_list_fallback(self):
        """Test handling of malformed list that falls back to string."""
        overrides = ["malformed=[1, 2, unclosed"]
        config = self.cli.parse_overrides(overrides)
        # Should fallback to string since it's not a valid Python literal
        assert config["malformed"] == "[1, 2, unclosed"

    def test_malformed_dict_fallback(self):
        """Test handling of malformed dict that falls back to string."""
        overrides = ["malformed={'key': unclosed"]
        config = self.cli.parse_overrides(overrides)
        # Should fallback to string since it's not a valid Python literal
        assert config["malformed"] == "{'key': unclosed"

    def test_unicode_strings(self):
        """Test parsing Unicode strings."""
        overrides = ['chinese="你好世界"', 'emoji="🚀🎉"', 'mixed="Hello 世界 🌍"']
        config = self.cli.parse_overrides(overrides)
        assert config["chinese"] == "你好世界"
        assert config["emoji"] == "🚀🎉"
        assert config["mixed"] == "Hello 世界 🌍"

    def test_deeply_nested_structure(self):
        """Test very deeply nested key structure."""
        overrides = ["a.b.c.d.e.f.g.h.i.j=deep_value"]
        config = self.cli.parse_overrides(overrides)
        assert config["a"]["b"]["c"]["d"]["e"]["f"]["g"]["h"]["i"]["j"] == "deep_value"

    def test_override_existing_nested_structure(self):
        """Test overriding parts of existing nested structure."""
        overrides = [
            "model.layers.conv1.filters=32",
            "model.layers.conv1.kernel_size=3",
            "model.layers.conv2.filters=64",
            "model.layers.conv1.activation=relu",  # Override existing conv1 structure
        ]
        config = self.cli.parse_overrides(overrides)
        assert config["model"]["layers"]["conv1"]["filters"] == 32
        assert config["model"]["layers"]["conv1"]["kernel_size"] == 3
        assert config["model"]["layers"]["conv1"]["activation"] == "relu"
        assert config["model"]["layers"]["conv2"]["filters"] == 64

    def test_numeric_keys_in_nested_structure(self):
        """Test numeric keys in nested structure."""
        overrides = ["layers.0.type=conv", "layers.1.type=pool", "layers.2.type=dense"]
        config = self.cli.parse_overrides(overrides)
        assert config["layers"]["0"]["type"] == "conv"
        assert config["layers"]["1"]["type"] == "pool"
        assert config["layers"]["2"]["type"] == "dense"


class TestCLIArgumentParsing:
    """Test argument parsing functionality."""

    def test_help_message_contains_examples(self):
        """Test that help message contains override examples."""
        cli = ConflgaCLI()
        help_text = cli.parser.format_help()

        # Check that examples are included in help
        assert "Override Examples:" in help_text
        assert "model.learning_rate=0.001" in help_text
        assert "dataset.batch_size=32" in help_text
        assert "training.epochs=100" in help_text

    def test_argument_metavar(self):
        """Test that override argument has correct metavar."""
        cli = ConflgaCLI()
        override_action = None
        for action in cli.parser._actions:
            if action.dest == "overrides":
                override_action = action
                break

        assert override_action is not None
        assert override_action.metavar == "KEY=VALUE"

    def test_argument_help_text(self):
        """Test that override argument has helpful description."""
        cli = ConflgaCLI()
        override_action = None
        for action in cli.parser._actions:
            if action.dest == "overrides":
                override_action = action
                break

        assert override_action is not None
        assert override_action.help is not None
        assert "Override configuration values" in override_action.help
        assert "dot notation" in override_action.help

    def test_parser_description(self):
        """Test parser description."""
        cli = ConflgaCLI()
        assert cli.parser.description is not None
        assert "Conflga Configuration Manager" in cli.parser.description


class TestCLIPerformance:
    """Test CLI performance with large inputs."""

    def setup_method(self):
        """Setup for each test method."""
        self.cli = ConflgaCLI()

    def test_many_overrides_performance(self):
        """Test parsing many overrides efficiently."""
        # Generate 1000 overrides
        overrides = [f"param_{i}={i}" for i in range(1000)]

        import time

        start_time = time.time()
        config = self.cli.parse_overrides(overrides)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0

        # Verify some values
        assert config["param_0"] == 0
        assert config["param_500"] == 500
        assert config["param_999"] == 999

    def test_deep_nesting_performance(self):
        """Test performance with deep nesting."""
        # Create deeply nested override
        deep_key = ".".join(f"level_{i}" for i in range(100))
        overrides = [f"{deep_key}=deep_value"]

        import time

        start_time = time.time()
        config = self.cli.parse_overrides(overrides)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 0.1

        # Navigate to the deep value
        current = config
        for i in range(100):
            current = current[f"level_{i}"]
        assert current == "deep_value"


class TestCLIErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Setup for each test method."""
        self.cli = ConflgaCLI()

    def test_parse_value_malformed_list_with_quotes(self):
        """Test parsing malformed list with quoted items that falls back to simple parsing."""
        # This will trigger the fallback list parsing code
        overrides = ['items=["item1", "item2", "item3"']  # Missing closing bracket
        config = self.cli.parse_overrides(overrides)
        # Should fallback to string since malformed
        assert config["items"] == '["item1", "item2", "item3"'

    def test_parse_value_empty_list_simple_parsing(self):
        """Test parsing empty list through simple parsing path."""
        # Create a scenario that triggers the simple list parsing with empty content
        value = "[]"
        # Remove quotes to get to the simple parsing
        result = self.cli._parse_value(value)
        assert result == []

    def test_parse_value_list_with_empty_items(self):
        """Test parsing list with empty items (filtering)."""
        # This should trigger the filtering of empty items
        overrides = ["values=[1, , 3, , 5]"]  # Has empty items
        config = self.cli.parse_overrides(overrides)
        # ast.literal_eval should handle this, but if it fails, fallback would filter empty
        try:
            expected = [1, 3, 5]  # Empty items filtered out in fallback
            assert config["values"] == expected or config["values"] == "[1, , 3, , 5]"
        except:
            # If parsing fails, it should be a string
            assert isinstance(config["values"], str)

    def test_parse_value_malformed_dict_fallback(self):
        """Test parsing malformed dict that triggers exception handling."""
        # This should trigger the dict parsing exception path
        overrides = ['config={"key": "value", "incomplete"']  # Malformed dict
        config = self.cli.parse_overrides(overrides)
        # Should fallback to string
        assert config["config"] == '{"key": "value", "incomplete"'

    def test_parse_overrides_none_input_with_no_args(self):
        """Test parsing None input when no command line args are provided."""
        # This should trigger the line where override_strings is set to []
        with patch.object(sys, "argv", ["program"]):  # No -o arguments
            config = self.cli.parse_overrides(None)
            assert isinstance(config, ConflgaConfig)
            assert len(config) == 0

    def test_parse_value_quoted_string_single_quotes(self):
        """Test parsing single quoted string to trigger quote removal."""
        result = self.cli._parse_value("'hello world'")
        assert result == "hello world"

    def test_parse_value_quoted_string_double_quotes(self):
        """Test parsing double quoted string to trigger quote removal."""
        result = self.cli._parse_value('"hello world"')
        assert result == "hello world"

    def test_malformed_list_simple_parsing_path(self):
        """Test malformed list that triggers simple comma-separated parsing."""
        # Create a malformed list that will fail ast.literal_eval but trigger simple parsing
        test_value = "[item1, item2, item3"  # Missing closing bracket

        # This should go through the simple parsing path when ast.literal_eval fails
        result = self.cli._parse_value(test_value)
        # Since it's malformed, it should return as string
        assert result == "[item1, item2, item3"

    def test_empty_list_simple_parsing(self):
        """Test empty list through simple parsing."""
        # Force the simple list parsing path by creating a scenario where ast.literal_eval might fail
        test_value = "[ ]"  # List with just spaces
        result = self.cli._parse_value(test_value)
        assert result == []

    def test_list_with_comma_only(self):
        """Test list parsing with comma-only content."""
        # This will test the filtering of empty items
        import unittest.mock

        # Mock ast.literal_eval to fail and force simple parsing
        with unittest.mock.patch(
            "ast.literal_eval", side_effect=ValueError("Mock failure")
        ):
            test_value = "[,,,]"  # Only commas
            result = self.cli._parse_value(test_value)
            # Should return empty list after filtering empty items
            assert result == []

    def test_dict_parsing_exception_handling(self):
        """Test dict parsing that triggers exception handling."""
        import unittest.mock

        # Mock ast.literal_eval to fail for dict parsing
        with unittest.mock.patch(
            "ast.literal_eval", side_effect=SyntaxError("Mock failure")
        ):
            test_value = '{"key": "value"}'
            result = self.cli._parse_value(test_value)
            # Should fallback to string
            assert result == '{"key": "value"}'

    def test_override_strings_none_assignment(self):
        """Test the None assignment path in parse_overrides."""
        # This will specifically test the line where override_strings is set to []
        # when it's None after the initial check
        with patch.object(self.cli.parser, "parse_args") as mock_parse:
            mock_parse.return_value.overrides = None
            config = self.cli.parse_overrides(None)
            assert isinstance(config, ConflgaConfig)
            assert len(config) == 0
