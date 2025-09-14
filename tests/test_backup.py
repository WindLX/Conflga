import pytest
import rtoml as toml
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from conflga import ConflgaConfig, conflga_func, conflga_method


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


# --- Backup Functionality Tests ---


def test_conflga_func_backup_basic(temp_config_dir, create_toml_file):
    """Test basic backup functionality with conflga_func decorator."""
    create_toml_file(
        "base_cfg",
        {
            "database": {"host": "localhost", "port": 5432, "name": "testdb"},
            "model": {"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
        },
    )

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            backup_path=backup_dir,
            auto_print=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.model.learning_rate

        result = test_func()
        assert result == 0.001

        # Check backup file was created
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert (
            len(backup_files) == 1
        ), f"Expected 1 backup file, found {len(backup_files)}"

        # Verify backup file content
        backup_file = backup_files[0]
        backup_content = backup_file.read_text(encoding="utf-8")
        restored_config = ConflgaConfig.loads(backup_content)
        assert restored_config.model.learning_rate == 0.001
        assert restored_config.database.host == "localhost"


def test_conflga_method_backup_basic(temp_config_dir, create_toml_file):
    """Test basic backup functionality with conflga_method decorator."""
    create_toml_file(
        "app_cfg",
        {
            "app": {"name": "TestApp", "version": "1.0.0", "debug": True},
            "server": {"host": "0.0.0.0", "port": 8080},
        },
    )

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        class TestApp:
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="app_cfg",
                backup_path=backup_dir,
                auto_print=False,
            )
            def run(self, cfg: ConflgaConfig):
                return cfg.app.name

        app = TestApp()
        result = app.run()
        assert result == "TestApp"

        # Check backup file was created
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert (
            len(backup_files) == 1
        ), f"Expected 1 backup file, found {len(backup_files)}"

        # Verify backup file content
        backup_file = backup_files[0]
        backup_content = backup_file.read_text(encoding="utf-8")
        restored_config = ConflgaConfig.loads(backup_content)
        assert restored_config.app.name == "TestApp"
        assert restored_config.server.port == 8080


def test_conflga_func_backup_disabled_when_none(temp_config_dir, create_toml_file):
    """Test that no backup is created when backup_path is None."""
    create_toml_file("simple_cfg", {"test": {"value": 42}})

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="simple_cfg",
            backup_path=None,  # Explicitly set to None
            auto_print=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.test.value

        result = test_func()
        assert result == 42

        # Ensure no backup files were created anywhere
        temp_path = Path(temp_config_dir)
        backup_files = list(temp_path.glob("**/config_backup_*.toml"))
        assert (
            len(backup_files) == 0
        ), f"Expected no backup files, found {len(backup_files)}"


def test_conflga_func_backup_with_merge_configs(temp_config_dir, create_toml_file):
    """Test backup functionality with merged configurations."""
    create_toml_file("base_cfg", {"model": {"lr": 0.01, "type": "CNN"}, "epochs": 10})
    create_toml_file(
        "override_cfg", {"model": {"lr": 0.001, "dropout": 0.5}, "batch_size": 64}
    )

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch("sys.argv", ["test_script.py"]):  # Mock clean command line

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            configs_to_merge=["override_cfg"],
            backup_path=backup_dir,
            auto_print=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.model.type, cfg.batch_size

        lr, model_type, batch_size = test_func()
        assert lr == 0.001  # Overridden by merge
        assert model_type == "CNN"  # From base
        assert batch_size == 64  # From merge

        # Verify backup contains merged configuration
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert len(backup_files) == 1

        backup_content = backup_files[0].read_text(encoding="utf-8")
        restored_config = ConflgaConfig.loads(backup_content)
        assert restored_config.model.lr == 0.001
        assert restored_config.model.type == "CNN"
        assert restored_config.model.dropout == 0.5
        assert restored_config.batch_size == 64


def test_conflga_func_backup_with_cli_override(temp_config_dir, create_toml_file):
    """Test backup functionality with CLI overrides."""
    create_toml_file(
        "base_cfg", {"model": {"lr": 0.01, "epochs": 10}, "data": {"batch_size": 32}}
    )

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch(
        "sys.argv",
        [
            "test_script.py",
            "-o",
            "model.lr=0.001",
            "-o",
            "model.epochs=100",
            "-o",
            "data.batch_size=64",
        ],
    ):

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="base_cfg",
            backup_path=backup_dir,
            enable_cli_override=True,
            use_namespace_prefix=False,
            auto_print=False,
            auto_print_override=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.model.lr, cfg.model.epochs, cfg.data.batch_size

        lr, epochs, batch_size = test_func()
        assert lr == 0.001  # CLI override
        assert epochs == 100  # CLI override
        assert batch_size == 64  # CLI override

        # Verify backup contains final configuration with CLI overrides
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert len(backup_files) == 1

        backup_content = backup_files[0].read_text(encoding="utf-8")
        restored_config = ConflgaConfig.loads(backup_content)
        assert restored_config.model.lr == 0.001
        assert restored_config.model.epochs == 100
        assert restored_config.data.batch_size == 64


def test_conflga_func_backup_with_preprocessor(temp_config_dir):
    """Test backup functionality with preprocessor enabled."""
    # Create config with macros and templates
    config_content = """
#let APP_NAME = "BackupTest"
#let PORT = 9000
#let DEBUG = true

[app]
name = "{{ APP_NAME }}"
debug = {{ DEBUG }}

[server]
port = {{ PORT }}
host = "localhost:{{ PORT }}"
"""

    config_path = os.path.join(temp_config_dir, "macro_config.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch("sys.argv", ["test_script.py"]):

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="macro_config",
            backup_path=backup_dir,
            enable_preprocessor=True,
            auto_print=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.app.name, cfg.server.port, cfg.server.host

        name, port, host = test_func()
        assert name == "BackupTest"
        assert port == 9000
        assert host == "localhost:9000"

        # Verify backup contains processed configuration (not raw macros)
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert len(backup_files) == 1

        backup_content = backup_files[0].read_text(encoding="utf-8")
        restored_config = ConflgaConfig.loads(backup_content)
        assert restored_config.app.name == "BackupTest"
        assert restored_config.server.port == 9000
        assert restored_config.server.host == "localhost:9000"
        # Ensure no macro syntax remains in backup
        assert "#let" not in backup_content
        assert "{{" not in backup_content


def test_conflga_func_backup_multiple_calls_different_timestamps(
    temp_config_dir, create_toml_file
):
    """Test that multiple function calls create different backup files with timestamps."""
    import time

    create_toml_file("multi_cfg", {"counter": 1})

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch("sys.argv", ["test_script.py"]):

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="multi_cfg",
            backup_path=backup_dir,
            auto_print=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.counter

        # First call
        result1 = test_func()
        assert result1 == 1

        # Add a small delay to ensure different timestamps
        time.sleep(0.01)

        # Second call (should create another backup)
        result2 = test_func()
        assert result2 == 1

        # Check that two backup files were created
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert (
            len(backup_files) == 2
        ), f"Expected 2 backup files, found {len(backup_files)}"

        # Verify both files have different names (timestamps)
        filenames = [f.name for f in backup_files]
        assert len(set(filenames)) == 2, "Backup files should have different names"


def test_conflga_method_backup_with_inheritance(temp_config_dir, create_toml_file):
    """Test backup functionality with method decorator and class inheritance."""
    import time

    create_toml_file(
        "inherit_cfg", {"base": {"value": 100}, "derived": {"multiplier": 2}}
    )

    backup_dir = os.path.join(temp_config_dir, "backup")

    with patch("sys.argv", ["test_script.py"]):

        class BaseClass:
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="inherit_cfg",
                backup_path=backup_dir,
                auto_print=False,
            )
            def process(self, cfg: ConflgaConfig):
                return cfg.base.value

        class DerivedClass(BaseClass):
            @conflga_method(
                config_dir=temp_config_dir,
                default_config="inherit_cfg",
                backup_path=backup_dir,
                auto_print=False,
            )
            def process_derived(self, cfg: ConflgaConfig):
                return cfg.base.value * cfg.derived.multiplier

        base = BaseClass()
        derived = DerivedClass()

        result_base = base.process()
        assert result_base == 100

        # Add a small delay to ensure different timestamps
        time.sleep(0.01)

        result_derived = derived.process_derived()
        assert result_derived == 200

        # Check that backup files were created for both method calls
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert (
            len(backup_files) == 2
        ), f"Expected 2 backup files, found {len(backup_files)}"


def test_conflga_func_backup_directory_creation(temp_config_dir, create_toml_file):
    """Test that backup directory is created if it doesn't exist."""
    create_toml_file("dir_test_cfg", {"test": {"nested": {"value": 123}}})

    # Use a nested backup path that doesn't exist
    backup_dir = os.path.join(temp_config_dir, "deep", "nested", "backup", "path")
    assert not os.path.exists(backup_dir), "Backup directory should not exist initially"

    with patch("sys.argv", ["test_script.py"]):

        @conflga_func(
            config_dir=temp_config_dir,
            default_config="dir_test_cfg",
            backup_path=backup_dir,
            auto_print=False,
        )
        def test_func(cfg: ConflgaConfig):
            return cfg.test.nested.value

        result = test_func()
        assert result == 123

        # Verify that the nested backup directory was created
        assert os.path.exists(backup_dir), "Backup directory should be created"
        assert os.path.isdir(backup_dir), "Backup path should be a directory"

        # Verify backup file was created in the nested directory
        backup_path = Path(backup_dir)
        backup_files = list(backup_path.glob("config_backup_*.toml"))
        assert (
            len(backup_files) == 1
        ), f"Expected 1 backup file, found {len(backup_files)}"


def test_conflga_config_to_toml_method():
    """Test the to_toml method of ConflgaConfig separately."""
    config_data = {
        "database": {"host": "localhost", "port": 5432, "ssl": True},
        "app": {
            "name": "TestApp",
            "version": "1.0",
            "features": ["auth", "logging", "metrics"],
        },
        "settings": {"timeout": 30.5, "retries": 3, "debug": False},
    }

    cfg = ConflgaConfig(config_data)
    toml_string = cfg.to_toml()

    # Verify the TOML string can be parsed back
    restored_data = toml.loads(toml_string)
    restored_cfg = ConflgaConfig(restored_data)

    assert restored_cfg.database.host == "localhost"
    assert restored_cfg.database.port == 5432
    assert restored_cfg.database.ssl is True
    assert restored_cfg.app.name == "TestApp"
    assert restored_cfg.app.features == ["auth", "logging", "metrics"]
    assert restored_cfg.settings.timeout == 30.5
    assert restored_cfg.settings.retries == 3
    assert restored_cfg.settings.debug is False
