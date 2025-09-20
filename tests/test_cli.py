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

    def test_init_default(self):
        """Test CLI initialization with default settings."""
        assert self.cli.parser is not None
        assert self.cli.use_namespace_prefix is True
        assert "overrides" in [action.dest for action in self.cli.parser._actions]

    def test_init_no_namespace_prefix(self):
        """Test CLI initialization without namespace prefix."""
        cli = ConflgaCLI(use_namespace_prefix=False)
        assert cli.use_namespace_prefix is False

        # Check that -o/--override is used
        override_action = None
        for action in cli.parser._actions:
            if action.dest == "overrides":
                override_action = action
                break
        assert override_action is not None
        assert "-o" in override_action.option_strings
        assert "--override" in override_action.option_strings

    def test_init_namespace_prefix(self):
        """Test CLI initialization with namespace prefix."""
        cli = ConflgaCLI(use_namespace_prefix=True)
        assert cli.use_namespace_prefix is True

        # Check that --conflga-override is used
        override_action = None
        for action in cli.parser._actions:
            if action.dest == "overrides":
                override_action = action
                break
        assert override_action is not None
        assert "--conflga-override" in override_action.option_strings

    def test_get_argument_name(self):
        """Test _get_argument_name method."""
        # Default namespace prefix
        cli = ConflgaCLI()
        assert cli._get_argument_name() == ["--conflga-override"]

        # No namespace prefix
        cli = ConflgaCLI(use_namespace_prefix=False)
        assert cli._get_argument_name() == ["-o", "--override"]

    def test_epilog_text_updates(self):
        """Test that epilog text updates based on argument name."""
        # Default
        cli = ConflgaCLI()
        epilog = cli._get_epilog_text()
        assert "--conflga-override" in epilog

        # No namespace prefix
        cli = ConflgaCLI(use_namespace_prefix=False)
        epilog = cli._get_epilog_text()
        assert "-o" in epilog

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

    def test_create_override_config_from_args_default(self):
        """Test convenience function with default settings."""
        overrides = ["model.lr=0.001", "batch_size=32"]
        config = create_override_config_from_args(overrides)
        assert isinstance(config, ConflgaConfig)
        assert config["model"]["lr"] == 0.001
        assert config["batch_size"] == 32

    def test_create_override_config_from_args_no_namespace(self):
        """Test convenience function without namespace prefix."""
        overrides = ["model.lr=0.001", "batch_size=32"]
        config = create_override_config_from_args(overrides, use_namespace_prefix=False)
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
            "--conflga-override",
            "model.lr=0.001",
            "--conflga-override",
            "batch_size=32",
            "--conflga-override",
            "use_gpu=true",
        ]

        with patch.object(sys, "argv", test_args):
            cli = ConflgaCLI()
            config = cli.parse_overrides()

            assert config["model"]["lr"] == 0.001
            assert config["batch_size"] == 32
            assert config["use_gpu"] is True

    def test_command_line_args_parsing_short_options(self):
        """Test parsing with short options when namespace prefix is disabled."""
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
            cli = ConflgaCLI(use_namespace_prefix=False)
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


class TestConflictAvoidance:
    """Test cases for conflict avoidance functionality."""

    def test_namespace_prefix_avoids_conflicts(self):
        """Test that namespace prefix avoids conflicts with other parsers."""
        # Create a CLI with namespace prefix
        cli = ConflgaCLI(use_namespace_prefix=True)

        # Simulate command line args that would conflict with -o
        test_args = [
            "program.py",
            "-o",
            "other_option",  # This should be ignored
            "--conflga-override",
            "model.lr=0.001",
            "--conflga-override",
            "batch_size=32",
        ]

        with patch.object(sys, "argv", test_args):
            config = cli.parse_overrides()
            assert config["model"]["lr"] == 0.001
            assert config["batch_size"] == 32
            # Should not pick up the "other_option" from -o

    def test_parse_known_args_behavior(self):
        """Test that parse_known_args ignores unknown arguments."""
        cli = ConflgaCLI(use_namespace_prefix=True)

        # Simulate args with unknown options
        test_args = [
            "program.py",
            "--unknown-option",
            "value",
            "--conflga-override",
            "model.lr=0.001",
            "--another-unknown",
            "another_value",
            "--conflga-override",
            "batch_size=32",
        ]

        with patch.object(sys, "argv", test_args):
            # This should not raise an exception
            config = cli.parse_overrides()
            assert config["model"]["lr"] == 0.001
            assert config["batch_size"] == 32

    def test_error_handling_in_parse_overrides(self):
        """Test error handling when argument parsing fails."""
        cli = ConflgaCLI()

        # Mock parse_known_args to raise SystemExit
        with patch.object(cli.parser, "parse_known_args", side_effect=SystemExit(1)):
            config = cli.parse_overrides()
            # Should return empty config instead of crashing
            assert isinstance(config, ConflgaConfig)
            assert len(config) == 0

    def test_no_help_conflicts(self):
        """Test that add_help=False prevents help conflicts."""
        cli = ConflgaCLI()

        # Check that help is disabled
        assert cli.parser.add_help is False

        # The parser should not have a help action
        help_actions = [
            action
            for action in cli.parser._actions
            if action.option_strings and "-h" in action.option_strings
        ]
        assert len(help_actions) == 0

    def test_multiple_cli_instances_independence(self):
        """Test that multiple CLI instances don't interfere with each other."""
        cli1 = ConflgaCLI(use_namespace_prefix=True)
        cli2 = ConflgaCLI(use_namespace_prefix=False)

        # Each should have different argument configurations
        assert cli1.use_namespace_prefix is True
        assert cli2.use_namespace_prefix is False

        # Test that they work independently
        overrides = ["test=value"]
        config1 = cli1.parse_overrides(overrides)
        config2 = cli2.parse_overrides(overrides)

        assert config1["test"] == "value"
        assert config2["test"] == "value"


class TestCLIArgumentParsing:
    """Test argument parsing functionality."""

    def test_help_message_contains_examples_default(self):
        """Test that help message contains override examples with default settings."""
        cli = ConflgaCLI()
        help_text = cli.parser.format_help()

        # Check that examples are included in help with namespace prefix
        assert "Override Examples:" in help_text
        assert "--conflga-override model.learning_rate=0.001" in help_text
        assert "--conflga-override dataset.batch_size=32" in help_text
        assert "--conflga-override training.epochs=100" in help_text

    def test_help_message_contains_examples_short_options(self):
        """Test that help message contains override examples with short options."""
        cli = ConflgaCLI(use_namespace_prefix=False)
        help_text = cli.parser.format_help()

        # Check that examples are included in help with short options
        assert "Override Examples:" in help_text
        assert "-o/--override model.learning_rate=0.001" in help_text
        assert "-o/--override dataset.batch_size=32" in help_text
        assert "-o/--override training.epochs=100" in help_text

    def test_argument_metavar_default(self):
        """Test that override argument has correct metavar with default settings."""
        cli = ConflgaCLI()
        override_action = None
        for action in cli.parser._actions:
            if action.dest == "overrides":
                override_action = action
                break

        assert override_action is not None
        assert override_action.metavar == "KEY=VALUE"

    def test_argument_help_text_namespace_prefix(self):
        """Test that override argument has helpful description with namespace prefix."""
        cli = ConflgaCLI(use_namespace_prefix=True)
        override_action = None
        for action in cli.parser._actions:
            if action.dest == "overrides":
                override_action = action
                break

        assert override_action is not None
        assert override_action.help is not None
        assert "Override Conflga configuration values" in override_action.help
        assert "dot notation" in override_action.help

    def test_argument_help_text_short_options(self):
        """Test that override argument has helpful description with short options."""
        cli = ConflgaCLI(use_namespace_prefix=False)
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

    def test_parser_add_help_disabled(self):
        """Test that parser has help disabled to avoid conflicts."""
        cli = ConflgaCLI()
        assert cli.parser.add_help is False


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


class TestCLIFlexibilityIntegration:
    """Integration tests for CLI flexibility features."""

    def test_end_to_end_namespace_prefix(self):
        """Test end-to-end with namespace prefix."""
        cli = ConflgaCLI(use_namespace_prefix=True)
        overrides = [
            "model.architecture=resnet50",
            "model.learning_rate=0.001",
            "dataset.batch_size=64",
            "training.epochs=100",
        ]

        config = cli.parse_overrides(overrides)

        assert config["model"]["architecture"] == "resnet50"
        assert config["model"]["learning_rate"] == 0.001
        assert config["dataset"]["batch_size"] == 64
        assert config["training"]["epochs"] == 100

    def test_end_to_end_short_options(self):
        """Test end-to-end with short options."""
        cli = ConflgaCLI(use_namespace_prefix=False)
        overrides = [
            "model.architecture=resnet50",
            "model.learning_rate=0.001",
            "dataset.batch_size=64",
            "training.epochs=100",
        ]

        config = cli.parse_overrides(overrides)

        assert config["model"]["architecture"] == "resnet50"
        assert config["model"]["learning_rate"] == 0.001
        assert config["dataset"]["batch_size"] == 64
        assert config["training"]["epochs"] == 100

    def test_convenience_function_flexibility(self):
        """Test convenience function with different configurations."""
        overrides = ["model.lr=0.001", "batch_size=32"]

        # Test with default (namespace prefix)
        config1 = create_override_config_from_args(overrides)
        assert config1["model"]["lr"] == 0.001

        # Test without namespace prefix
        config2 = create_override_config_from_args(
            overrides, use_namespace_prefix=False
        )
        assert config2["model"]["lr"] == 0.001

    def test_real_world_integration_scenario(self):
        """Test a real-world scenario where the library is embedded in another project."""
        # Simulate a scenario where another project uses -o for something else
        # but we want to use Conflga without conflicts

        # Create Conflga CLI with namespace prefix
        conflga_cli = ConflgaCLI(use_namespace_prefix=True)

        # Simulate sys.argv that has both conflga and other project args
        test_args = [
            "my_project.py",
            "-o",
            "other_project_output_file.txt",  # Another project's -o option
            "--verbose",
            "--conflga-override",
            "model.lr=0.001",  # Conflga's override
            "--conflga-override",
            "dataset.name=cifar10",
            "--other-project-flag",
        ]

        with patch.object(sys, "argv", test_args):
            # This should work without conflicts
            config = conflga_cli.parse_overrides()

            # Should only get Conflga overrides, not the other -o
            assert config["model"]["lr"] == 0.001
            assert config["dataset"]["name"] == "cifar10"
            assert len(config) == 2  # Only 2 items from Conflga

    def test_backward_compatibility_when_needed(self):
        """Test backward compatibility when user explicitly wants -o."""
        # When user explicitly wants to use -o (knowing there won't be conflicts)
        cli = ConflgaCLI(use_namespace_prefix=False)

        test_args = [
            "program.py",
            "-o",
            "model.lr=0.001",
            "--override",
            "batch_size=32",
        ]

        with patch.object(sys, "argv", test_args):
            config = cli.parse_overrides()
            assert config["model"]["lr"] == 0.001
            assert config["batch_size"] == 32
