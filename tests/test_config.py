import pytest

from conflga import ConflgaConfig

# Define some TOML content for testing
BASIC_TOML_CONTENT = """
[app]
name = "MyApp"
version = 1.0

[database]
host = "localhost"
port = 5432
"""

OVERRIDE_TOML_CONTENT = """
[app]
version = 1.1 # Override
debug = true # New key

[database]
port = 5433 # Override
user = "testuser" # New key
"""

# --- Fixtures ---


@pytest.fixture
def basic_toml_file(tmp_path):
    """
    Creates a temporary TOML file with basic configuration.
    `tmp_path` is a pytest built-in fixture for temporary directories.
    """
    file_path = tmp_path / "config.toml"
    file_path.write_text(BASIC_TOML_CONTENT)
    return str(file_path)


@pytest.fixture
def override_toml_file(tmp_path):
    """
    Creates a temporary TOML file with override configuration.
    """
    file_path = tmp_path / "override_config.toml"
    file_path.write_text(OVERRIDE_TOML_CONTENT)
    return str(file_path)


@pytest.fixture
def nested_toml_content():
    """Provides TOML content with deeper nesting."""
    return """
    [network]
    ip = "192.168.1.1"
    [network.ports]
    http = 80
    https = 443
    """


# --- Test Cases ---


def test_load_from_file(basic_toml_file):
    """Test loading configuration from a TOML file."""
    config = ConflgaConfig.load(basic_toml_file)
    assert config.app.name == "MyApp"
    assert config.database.port == 5432


def test_load_from_string():
    """Test loading configuration from a TOML string."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    assert config.app.name == "MyApp"
    assert config.database.host == "localhost"


def test_dot_access():
    """Test accessing configuration values using dot notation."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    assert config.app.name == "MyApp"
    assert config.app.version == 1.0
    assert config.database.host == "localhost"


def test_nested_dot_access(nested_toml_content):
    """Test dot access for deeply nested configurations."""
    config = ConflgaConfig.from_string(nested_toml_content)
    assert config.network.ip == "192.168.1.1"
    assert config.network.ports.http == 80
    assert config.network.ports.https == 443


def test_missing_key_dot_access():
    """Test that accessing a missing key via dot notation raises AttributeError."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    with pytest.raises(
        AttributeError, match="Configuration key 'non_existent_key' not found."
    ):
        _ = config.non_existent_key

    with pytest.raises(
        AttributeError, match="Configuration key 'non_existent_sub_key' not found."
    ):
        _ = config.app.non_existent_sub_key


def test_dict_like_access():
    """Test accessing configuration values using dictionary-like notation."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    assert config["app"]["name"] == "MyApp"
    assert config["database"]["port"] == 5432


def test_missing_key_dict_access():
    """Test that accessing a missing key via dictionary notation raises KeyError."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    with pytest.raises(KeyError):
        _ = config["non_existent_key"]
    with pytest.raises(KeyError):
        _ = config["app"]["non_existent_sub_key"]


def test_merge_configurations(basic_toml_file, override_toml_file):
    """Test merging two configurations."""
    base_config = ConflgaConfig.load(basic_toml_file)
    override_config = ConflgaConfig.load(override_toml_file)

    base_config.merge_with(override_config)

    # Check overridden values
    assert base_config.app.version == 1.1
    assert base_config.database.port == 5433

    # Check new values
    assert base_config.app.debug is True
    assert base_config.database.user == "testuser"

    # Check retained values
    assert base_config.app.name == "MyApp"
    assert base_config.database.host == "localhost"


def test_merge_with_non_conflgaconfig_raises_typeerror():
    """Test that merging with a non-ConflgaConfig object raises TypeError."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    with pytest.raises(
        TypeError, match="Can only merge with another ConflgaConfig instance."
    ):
        config.merge_with({"key": "value"})  # type: ignore[call-arg]


def test_set_and_del_attributes():
    """Test setting and deleting attributes."""
    config = ConflgaConfig()
    config.new_key = "new_value"
    assert config.new_key == "new_value"

    config.nested = ConflgaConfig({"a": 1})
    assert config.nested.a == 1

    del config.new_key
    with pytest.raises(AttributeError):
        _ = config.new_key


def test_set_and_del_items():
    """Test setting and deleting items."""
    config = ConflgaConfig()
    config["another_key"] = "another_value"
    assert config["another_key"] == "another_value"

    del config["another_key"]
    with pytest.raises(KeyError):
        _ = config["another_key"]


def test_config_length():
    """Test the __len__ method."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    assert len(config) == 2  # 'app' and 'database'


def test_config_iteration():
    """Test the __iter__ method."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    keys = list(config)
    assert "app" in keys
    assert "database" in keys
    assert len(keys) == 2


def test_empty_config():
    """Test handling of an empty configuration."""
    config = ConflgaConfig()
    assert len(config) == 0
    assert not list(config)  # Should be empty iterable

    config_from_string = ConflgaConfig.from_string("")
    assert len(config_from_string) == 0
