from conflga.config import ConflgaConfig


def test_pretty_print():
    """测试美观打印功能"""

    # 创建一个包含各种数据类型的配置
    config_data = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp",
            "ssl_enabled": True,
            "timeout": None,
            "credentials": {"username": "admin", "password": "secret123"},
        },
        "servers": [
            {"name": "web1", "ip": "192.168.1.10", "active": True},
            {"name": "web2", "ip": "192.168.1.11", "active": False},
            {"name": "api", "ip": "192.168.1.20", "active": True},
        ],
        "features": ["auth", "logging", "caching"],
        "debug": False,
        "version": "1.0.0",
        "max_connections": 100,
    }

    # 创建配置对象
    config = ConflgaConfig(config_data)

    print("=== 测试基本的 pretty_print ===")
    config.pretty_print()

    print("\n=== 测试自定义标题 ===")
    config.pretty_print(title="应用程序配置")

    print("\n=== 测试嵌套配置的打印 ===")
    database_config = config.database
    database_config.pretty_print(title="数据库配置")

    print("\n=== 测试从 TOML 字符串加载的配置 ===")
    toml_string = """
    [app]
    name = "MyApp"
    version = "2.0.0"
    debug = true
    
    [app.server]
    host = "0.0.0.0"
    port = 8080
    workers = 4
    
    [[app.middlewares]]
    name = "cors"
    enabled = true
    
    [[app.middlewares]]
    name = "auth"
    enabled = false
    """

    toml_config = ConflgaConfig.from_string(toml_string)
    toml_config.pretty_print(title="TOML 配置")

    print("\n=== 测试配置来源信息显示 ===")
    # 测试显示目录和文件信息
    config.pretty_print(
        title="带来源信息的配置",
        directory="/etc/myapp",
        files=["config.toml", "secrets.toml"],
    )

    print("\n=== 测试单个配置文件来源 ===")
    config.pretty_print(title="单文件配置", directory="./conf", files=["app.toml"])

    print("\n=== 测试仅文件路径（无目录） ===")
    config.pretty_print(
        title="绝对路径配置", files=["/home/user/config.toml", "/etc/app/settings.toml"]
    )

    print("\n=== 测试现有配置文件（如果存在） ===")
    # 测试项目中实际存在的文件
    import os

    if os.path.exists("./pyproject.toml"):
        config.pretty_print(title="项目配置", directory=".", files=["pyproject.toml"])


if __name__ == "__main__":
    test_pretty_print()
