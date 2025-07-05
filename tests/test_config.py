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
    """Test converting configuration to dictionary with basic structure."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    result = config.to_dict()

    # Should return a regular dict, not ConflgaConfig
    assert isinstance(result, dict)
    assert not isinstance(result, ConflgaConfig)

    # Check the structure and values
    assert result == {
        "app": {"name": "MyApp", "version": 1.0},
        "database": {"host": "localhost", "port": 5432},
    }

    # Nested dictionaries should also be regular dicts
    assert isinstance(result["app"], dict)
    assert isinstance(result["database"], dict)
    assert not isinstance(result["app"], ConflgaConfig)
    assert not isinstance(result["database"], ConflgaConfig)


def test_to_dict_nested():
    """Test converting deeply nested configuration to dictionary."""
    nested_toml = """
    [network]
    ip = "192.168.1.1"
    [network.ports]
    http = 80
    https = 443
    [network.ports.ssl]
    certificate = "/path/to/cert"
    key = "/path/to/key"
    """
    config = ConflgaConfig.from_string(nested_toml)
    result = config.to_dict()

    expected = {
        "network": {
            "ip": "192.168.1.1",
            "ports": {
                "http": 80,
                "https": 443,
                "ssl": {"certificate": "/path/to/cert", "key": "/path/to/key"},
            },
        }
    }

    assert result == expected

    # Verify all nested levels are regular dicts
    assert isinstance(result["network"], dict)
    assert isinstance(result["network"]["ports"], dict)
    assert isinstance(result["network"]["ports"]["ssl"], dict)
    assert not isinstance(result["network"], ConflgaConfig)
    assert not isinstance(result["network"]["ports"], ConflgaConfig)
    assert not isinstance(result["network"]["ports"]["ssl"], ConflgaConfig)


def test_to_dict_with_lists():
    """Test converting configuration with lists to dictionary."""
    toml_with_lists = """
    servers = ["server1", "server2", "server3"]

    [database]
    replicas = [
        {host = "db1", port = 5432},
        {host = "db2", port = 5433}
    ]
    """
    config = ConflgaConfig.from_string(toml_with_lists)
    result = config.to_dict()

    expected = {
        "servers": ["server1", "server2", "server3"],
        "database": {
            "replicas": [
                {"host": "db1", "port": 5432},
                {"host": "db2", "port": 5433},
            ]
        },
    }

    assert result == expected

    # Lists should remain as lists
    assert isinstance(result["servers"], list)
    assert isinstance(result["database"]["replicas"], list)

    # Dict items in lists should be regular dicts
    for replica in result["database"]["replicas"]:
        assert isinstance(replica, dict)
        assert not isinstance(replica, ConflgaConfig)


def test_to_dict_empty_config():
    """Test converting empty configuration to dictionary."""
    config = ConflgaConfig()
    result = config.to_dict()

    assert result == {}
    assert isinstance(result, dict)


def test_to_dict_mixed_types():
    """Test converting configuration with mixed data types."""
    toml_mixed = """
    string_val = "hello"
    int_val = 42
    float_val = 3.14
    bool_val = true

    [section]
    none_val = "@None"  # Special handling for None values
    list_val = [1, 2, 3]
    """
    config = ConflgaConfig.from_string(toml_mixed)
    result = config.to_dict()

    expected = {
        "string_val": "hello",
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "section": {
            "none_val": None,
            "list_val": [1, 2, 3],
        },
    }

    assert result == expected


def test_to_dict_preserves_original_config():
    """Test that to_dict doesn't modify the original configuration."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)
    original_data = config._data.copy()

    result = config.to_dict()

    # Original config should remain unchanged
    assert config._data == original_data
    assert isinstance(config.app, ConflgaConfig)
    assert isinstance(config.database, ConflgaConfig)

    # But result should be regular dicts
    assert isinstance(result["app"], dict)
    assert isinstance(result["database"], dict)


def test_to_dict_after_modifications():
    """Test to_dict after modifying the configuration."""
    config = ConflgaConfig.from_string(BASIC_TOML_CONTENT)

    # Modify the config
    config.app.debug = True
    config.new_section = ConflgaConfig({"key": "value"})

    result = config.to_dict()

    expected = {
        "app": {"name": "MyApp", "version": 1.0, "debug": True},
        "database": {"host": "localhost", "port": 5432},
        "new_section": {"key": "value"},
    }

    assert result == expected


def test_to_dict_deeply_nested_lists():
    """Test converting configuration with deeply nested lists and configs."""
    toml_complex = """
    [[services]]
    name = "web"
    
    [services.config]
    port = 80
    host = "localhost"
    
    [[services]]
    name = "db"
    
    [services.config]
    port = 5432
    host = "db.example.com"
    
    [meta]
    nested_lists = [
        ["item1", "item2"],
        ["item3", "item4"]
    ]
    """
    config = ConflgaConfig.from_string(toml_complex)
    result = config.to_dict()

    # Verify the structure is correctly converted
    assert isinstance(result, dict)
    assert isinstance(result["services"], list)
    assert len(result["services"]) == 2

    # Check each service in the list
    for service in result["services"]:
        assert isinstance(service, dict)
        assert not isinstance(service, ConflgaConfig)
        assert "name" in service
        assert "config" in service
        assert isinstance(service["config"], dict)
        assert not isinstance(service["config"], ConflgaConfig)

    # Check nested lists
    assert isinstance(result["meta"]["nested_lists"], list)
    for nested_list in result["meta"]["nested_lists"]:
        assert isinstance(nested_list, list)
        for item in nested_list:
            assert isinstance(item, str)
