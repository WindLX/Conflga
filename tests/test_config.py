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


def test_to_dict_basic():
    """Test basic to_dict functionality."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    result = config.to_dict()

    # Check that result is a regular dict, not ConflgaConfig
    assert isinstance(result, dict)
    assert not isinstance(result, ConflgaConfig)

    # Check values
    assert result["app"]["name"] == "MyApp"
    assert result["app"]["version"] == 1.0
    assert result["database"]["host"] == "localhost"
    assert result["database"]["port"] == 5432

    # Check that nested values are also regular dicts
    assert isinstance(result["app"], dict)
    assert isinstance(result["database"], dict)
    assert not isinstance(result["app"], ConflgaConfig)
    assert not isinstance(result["database"], ConflgaConfig)


def test_to_dict_with_lists():
    """Test to_dict with lists containing various types."""
    toml_content = """
    [app]
    name = "MyApp"
    tags = ["web", "api", "production"]

    [[servers]]
    name = "server1"
    ip = "192.168.1.1"

    [[servers]]
    name = "server2"
    ip = "192.168.1.2"
    """

    config = ConflgaConfig.from_string(toml_content)
    result = config.to_dict()

    # Check list of strings
    assert result["app"]["tags"] == ["web", "api", "production"]
    assert isinstance(result["app"]["tags"], list)

    # Check list of objects (servers)
    assert len(result["servers"]) == 2
    assert isinstance(result["servers"], list)
    assert isinstance(result["servers"][0], dict)
    assert not isinstance(result["servers"][0], ConflgaConfig)

    assert result["servers"][0]["name"] == "server1"
    assert result["servers"][0]["ip"] == "192.168.1.1"
    assert result["servers"][1]["name"] == "server2"
    assert result["servers"][1]["ip"] == "192.168.1.2"


def test_to_dict_deeply_nested():
    """Test to_dict with deeply nested configurations."""
    toml_content = """
    [app]
    name = "MyApp"

    [app.database]
    host = "localhost"
    port = 5432

    [app.database.connection]
    timeout = 30
    retry_count = 3

    [app.database.connection.ssl]
    enabled = true
    cert_path = "/path/to/cert"
    """

    config = ConflgaConfig.from_string(toml_content)
    result = config.to_dict()

    # Check deep nesting is preserved
    assert result["app"]["name"] == "MyApp"
    assert result["app"]["database"]["host"] == "localhost"
    assert result["app"]["database"]["port"] == 5432
    assert result["app"]["database"]["connection"]["timeout"] == 30
    assert result["app"]["database"]["connection"]["retry_count"] == 3
    assert result["app"]["database"]["connection"]["ssl"]["enabled"] is True
    assert (
        result["app"]["database"]["connection"]["ssl"]["cert_path"] == "/path/to/cert"
    )

    # Check all levels are regular dicts
    assert isinstance(result["app"], dict)
    assert isinstance(result["app"]["database"], dict)
    assert isinstance(result["app"]["database"]["connection"], dict)
    assert isinstance(result["app"]["database"]["connection"]["ssl"], dict)

    # Ensure none are ConflgaConfig instances
    assert not isinstance(result["app"], ConflgaConfig)
    assert not isinstance(result["app"]["database"], ConflgaConfig)
    assert not isinstance(result["app"]["database"]["connection"], ConflgaConfig)
    assert not isinstance(result["app"]["database"]["connection"]["ssl"], ConflgaConfig)


def test_to_dict_with_mixed_list_content():
    """Test to_dict with lists containing both ConflgaConfig objects and simple values."""
    # Create a config with manually added list content
    config = ConflgaConfig()
    config.mixed_list = [
        "simple_string",
        42,
        ConflgaConfig({"nested": "value"}),
        ["nested_list", "with_strings"],
        ConflgaConfig({"another": ConflgaConfig({"deep": "nesting"})}),
    ]

    result = config.to_dict()

    assert isinstance(result["mixed_list"], list)
    assert len(result["mixed_list"]) == 5

    # Check simple values are preserved
    assert result["mixed_list"][0] == "simple_string"
    assert result["mixed_list"][1] == 42

    # Check ConflgaConfig objects are converted to dicts
    assert isinstance(result["mixed_list"][2], dict)
    assert result["mixed_list"][2]["nested"] == "value"
    assert not isinstance(result["mixed_list"][2], ConflgaConfig)

    # Check nested lists are preserved
    assert isinstance(result["mixed_list"][3], list)
    assert result["mixed_list"][3] == ["nested_list", "with_strings"]

    # Check deeply nested ConflgaConfig objects
    assert isinstance(result["mixed_list"][4], dict)
    assert isinstance(result["mixed_list"][4]["another"], dict)
    assert result["mixed_list"][4]["another"]["deep"] == "nesting"
    assert not isinstance(result["mixed_list"][4], ConflgaConfig)
    assert not isinstance(result["mixed_list"][4]["another"], ConflgaConfig)


def test_to_dict_empty_config():
    """Test to_dict with empty configuration."""
    config = ConflgaConfig()
    result = config.to_dict()

    assert isinstance(result, dict)
    assert len(result) == 0
    assert result == {}


def test_to_dict_preserves_original():
    """Test that to_dict doesn't modify the original config."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    original_app_name = config.app.name

    result = config.to_dict()

    # Modify the result
    result["app"]["name"] = "ModifiedApp"

    # Original should be unchanged
    assert config.app.name == original_app_name
    assert config.app.name == "MyApp"

    # Original should still be ConflgaConfig instances
    assert isinstance(config.app, ConflgaConfig)
    assert isinstance(config.database, ConflgaConfig)


def test_to_dict_with_none_values():
    """Test to_dict with None values."""
    toml_content = """
    [app]
    name = "MyApp"
    description = "@None"  # Special syntax for None in rtoml

    [database]
    password = "@None"
    port = 5432
    """

    config = ConflgaConfig.from_string(toml_content)
    result = config.to_dict()

    assert result["app"]["name"] == "MyApp"
    assert result["app"]["description"] is None
    assert result["database"]["password"] is None
    assert result["database"]["port"] == 5432
