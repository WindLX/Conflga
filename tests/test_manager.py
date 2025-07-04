import pytest
import rtoml as toml
import os
import tempfile

from conflga import ConflgaConfig, ConflgaManager


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


# --- ConflgaConfig Tests ---


def test_conflgaconfig_initialization():
    cfg = ConflgaConfig({"a": 1, "b": {"c": 2}})
    assert cfg.a == 1
    assert cfg.b.c == 2
    assert isinstance(cfg.b, ConflgaConfig)


def test_conflgaconfig_dot_access_and_assignment():
    cfg = ConflgaConfig()
    cfg.name = "test"
    cfg.data = ConflgaConfig({"value": 10})
    assert cfg.name == "test"
    assert cfg.data.value == 10

    cfg.data.value = 20
    assert cfg.data.value == 20


def test_conflgaconfig_getitem_setitem_delitem():
    cfg = ConflgaConfig({"a": 1})
    assert cfg["a"] == 1
    cfg["b"] = 2
    assert cfg.b == 2
    del cfg["a"]
    with pytest.raises(KeyError):
        _ = cfg["a"]


def test_conflgaconfig_len_iter():
    cfg = ConflgaConfig({"a": 1, "b": 2})
    assert len(cfg) == 2
    assert set(iter(cfg)) == {"a", "b"}


def test_conflgaconfig_load_from_file(temp_config_dir, create_toml_file):
    create_toml_file("test_load", {"param": "value", "nested": {"key": 123}})
    cfg = ConflgaConfig.load(os.path.join(temp_config_dir, "test_load.toml"))
    assert cfg.param == "value"
    assert cfg.nested.key == 123
    assert isinstance(cfg.nested, ConflgaConfig)


def test_conflgaconfig_from_string():
    toml_str = """
    [section]
    item = "data"
    number = 42
    """
    cfg = ConflgaConfig.from_string(toml_str)
    assert cfg.section.item == "data"
    assert cfg.section.number == 42


def test_conflgaconfig_merge_with_flat():
    cfg1 = ConflgaConfig({"a": 1, "b": 2})
    cfg2 = ConflgaConfig({"b": 3, "c": 4})
    cfg1.merge_with(cfg2)
    assert cfg1.a == 1
    assert cfg1.b == 3
    assert cfg1.c == 4


def test_conflgaconfig_merge_with_nested():
    cfg1 = ConflgaConfig(
        {"model": {"name": "M1", "layers": 10}, "data": {"batch_size": 32}}
    )
    cfg2 = ConflgaConfig(
        {"model": {"layers": 20, "dropout": 0.1}, "optimizer": {"lr": 0.001}}
    )
    cfg1.merge_with(cfg2)
    assert cfg1.model.name == "M1"
    assert cfg1.model.layers == 20
    assert cfg1.model.dropout == 0.1
    assert cfg1.data.batch_size == 32
    assert cfg1.optimizer.lr == 0.001


def test_conflgaconfig_merge_with_type_error():
    cfg = ConflgaConfig()
    with pytest.raises(
        TypeError, match="Can only merge with another ConflgaConfig instance."
    ):
        cfg.merge_with({"invalid": "type"})  # type: ignore


def test_conflgaconfig_attribute_error_on_missing_key():
    cfg = ConflgaConfig({"a": 1})
    with pytest.raises(AttributeError, match="Configuration key 'b' not found."):
        _ = cfg.b


def test_conflgaconfig_delattr():
    cfg = ConflgaConfig({"a": 1, "b": 2})
    del cfg.a
    with pytest.raises(AttributeError):
        _ = cfg.a
    assert cfg.b == 2


def test_conflgaconfig_repr_str():
    cfg = ConflgaConfig({"a": 1})
    assert "ConflgaConfig({'a': 1})" in repr(cfg)
    assert "{'a': 1}" in str(cfg)


# --- ConflgaManager Tests ---


def test_config_manager_init(temp_config_dir):
    manager = ConflgaManager(config_dir=temp_config_dir)
    assert str(manager.config_dir) == temp_config_dir


def test_config_manager_load_default(temp_config_dir, create_toml_file):
    create_toml_file("my_default", {"env": "dev"})
    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("my_default")
    cfg = manager.get_config()
    assert cfg.env == "dev"


def test_config_manager_load_default_file_not_found(temp_config_dir):
    manager = ConflgaManager(config_dir=temp_config_dir)
    with pytest.raises(FileNotFoundError, match="Default config file not found"):
        manager.load_default("non_existent_config")


def test_config_manager_merge_config_basic(temp_config_dir, create_toml_file):
    create_toml_file("base", {"app": {"name": "App1", "version": 1}})
    create_toml_file("override", {"app": {"version": 2, "debug": True}})

    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("base").merge_config("override")
    cfg = manager.get_config()
    assert cfg.app.name == "App1"
    assert cfg.app.version == 2
    assert cfg.app.debug is True


def test_config_manager_merge_config_multiple_files(temp_config_dir, create_toml_file):
    create_toml_file("base", {"a": 1, "b": 2})
    create_toml_file("patch1", {"b": 3, "c": 4})
    create_toml_file("patch2", {"c": 5, "d": 6})

    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("base").merge_config("patch1", "patch2")
    cfg = manager.get_config()
    assert cfg.a == 1
    assert cfg.b == 3
    assert cfg.c == 5
    assert cfg.d == 6


def test_config_manager_merge_config_no_default_loaded(
    temp_config_dir, create_toml_file
):
    create_toml_file("some_config", {"key": "value"})
    manager = ConflgaManager(config_dir=temp_config_dir)
    with pytest.raises(RuntimeError, match="Load a default configuration first"):
        manager.merge_config("some_config")


def test_config_manager_merge_config_file_not_found(temp_config_dir, create_toml_file):
    create_toml_file("base", {"key": "value"})
    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("base")
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        manager.merge_config("non_existent_merge")


def test_config_manager_get_config_before_load(temp_config_dir):
    manager = ConflgaManager(config_dir=temp_config_dir)
    with pytest.raises(RuntimeError, match="No configuration has been loaded yet."):
        manager.get_config()


def test_config_manager_override_config(temp_config_dir, create_toml_file):
    # Create base config
    create_toml_file(
        "base", {"key1": "value1", "nested": {"key2": "value2", "key3": "value3"}}
    )

    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("base")

    # Create override config
    override_config = ConflgaConfig(
        {"key1": "overridden", "nested": {"key2": "overridden_nested"}}
    )

    # Apply override
    manager.override_config(override_config)

    config = manager.get_config()
    assert config.key1 == "overridden"
    assert config.nested.key2 == "overridden_nested"
    assert config.nested.key3 == "value3"  # Should remain unchanged


def test_config_manager_override_from_dict(temp_config_dir, create_toml_file):
    # Create base config
    create_toml_file("base", {"key1": "value1", "nested": {"key2": "value2"}})

    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("base")

    # Apply override from dict
    override_dict = {"key1": "dict_override", "new_key": "new_value"}
    manager.override_from_dict(override_dict)

    config = manager.get_config()
    assert config.key1 == "dict_override"
    assert config.new_key == "new_value"
    assert config.nested.key2 == "value2"  # Should remain unchanged


def test_config_manager_override_config_no_default_loaded(temp_config_dir):
    manager = ConflgaManager(config_dir=temp_config_dir)
    override_config = ConflgaConfig({"key": "value"})

    with pytest.raises(RuntimeError, match="Load a default configuration first"):
        manager.override_config(override_config)


def test_config_manager_override_from_dict_no_default_loaded(temp_config_dir):
    manager = ConflgaManager(config_dir=temp_config_dir)
    override_dict = {"key": "value"}

    with pytest.raises(RuntimeError, match="Load a default configuration first"):
        manager.override_from_dict(override_dict)


def test_config_manager_override_config_invalid_type(temp_config_dir, create_toml_file):
    create_toml_file("base", {"key": "value"})
    manager = ConflgaManager(config_dir=temp_config_dir)
    manager.load_default("base")

    with pytest.raises(
        TypeError, match="override_config must be a ConflgaConfig instance"
    ):
        manager.override_config(
            {"key": "value"}  # type: ignore[call-arg]
        )  # Passing dict instead of ConflgaConfig


def test_config_manager_chaining_with_overrides(temp_config_dir, create_toml_file):
    # Test method chaining with override methods
    create_toml_file("base", {"key1": "value1"})
    create_toml_file("extra", {"key2": "value2"})

    manager = ConflgaManager(config_dir=temp_config_dir)

    # Chain all operations
    config = (
        manager.load_default("base")
        .merge_config("extra")
        .override_from_dict({"key3": "value3"})
        .override_config(ConflgaConfig({"key1": "final_override"}))
        .get_config()
    )

    assert config.key1 == "final_override"
    assert config.key2 == "value2"
    assert config.key3 == "value3"
