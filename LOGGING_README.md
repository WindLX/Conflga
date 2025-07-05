# Conflga - 使用 Logging API 的配置管理库

Conflga 是一个简单易用的配置管理库，支持 TOML 配置文件和命令行覆盖。现在完全支持使用 Python 标准库的 `logging` API 进行配置输出。

## 🔧 Logging 集成

### 设置自定义 Logger 名称

在导入库时，您可以设置 Conflga 使用的 logger 名称：

```python
import conflga

# 设置 Conflga 使用的 logger 名称
conflga.set_conflga_logger_name("my_app.config")
```

### 配置 Logging

使用标准的 Python logging 配置：

```python
import logging

# 基本配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 或者自定义格式
logger = logging.getLogger("my_app.config")
handler = logging.StreamHandler()
formatter = logging.Formatter('🔧 [CONFIG] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### 基本使用

```python
from conflga import ConflgaConfig

# 创建配置对象
config = ConflgaConfig({
    "model": {
        "learning_rate": 0.001,
        "dropout": 0.2
    },
    "dataset": {
        "batch_size": 32
    }
})

# 打印配置到日志（使用设置的 logger）
config.pretty_print(title="我的配置", level=logging.INFO)
```

### 使用装饰器

```python
import logging
import conflga

# 设置 logger 名称
conflga.set_conflga_logger_name("my_ml_app.config")

@conflga.conflga_main(
    config_dir="conf",
    default_config="model_config",
    enable_cli_override=True,
    auto_print=True,           # 自动打印最终配置
    auto_print_override=True,  # 自动打印命令行覆盖
    log_level=logging.INFO,    # 日志级别
    use_namespace_prefix=True
)
def train_model(cfg: conflga.ConflgaConfig):
    """训练函数"""
    # 获取配置的 logger 实例
    logger = conflga.get_conflga_logger()
    
    logger.info("开始训练...")
    logger.info(f"学习率: {cfg.model.learning_rate}")
    
    # 手动打印配置（如果需要）
    cfg.pretty_print(title="当前配置", level=logging.DEBUG)

if __name__ == "__main__":
    train_model()
```

### 命令行使用

```bash
# 基本使用
python my_script.py

# 使用命令行覆盖
python my_script.py --conflga-override model.learning_rate=0.01 --conflga-override dataset.batch_size=64

# 复杂嵌套配置
python my_script.py --conflga-override model.architecture=resnet --conflga-override optimizer.type=adam
```

### 日志输出示例

配置会以漂亮的树状格式输出到日志：

```
🔧 [CONFIG] === Final Configuration ===
Configuration Source:
  Config Directory: /path/to/conf
  Config File: model_config.toml
    Full Path: /path/to/conf/model_config.toml
    File Status: Exists

Configuration Content:
model:
  learning_rate: 0.001
  dropout: 0.2
  architecture: "resnet"
dataset:
  batch_size: 32
  shuffle: true
optimizer:
  type: "adam"
  lr_schedule: "cosine"
```

## 🔑 主要功能

- ✅ **无 Rich 依赖**: 完全使用 Python 标准库的 logging
- ✅ **自定义 Logger 名称**: 与您的应用日志系统集成
- ✅ **灵活的日志级别**: 支持所有标准日志级别
- ✅ **树状配置显示**: 配置以清晰的层次结构显示
- ✅ **配置来源信息**: 显示配置文件路径和状态信息
- ✅ **命令行覆盖**: 支持嵌套键的命令行覆盖
- ✅ **容错处理**: 缺失的配置文件不会导致错误

## 📦 安装

```bash
pip install conflga
```

## 🧪 兼容性

- Python 3.10+
- 不再依赖 `rich` 库
- 完全兼容标准 `logging` 模块

## 🔄 从 Rich 迁移

如果您之前使用的是依赖 Rich 的版本，只需要：

1. 更新到新版本
2. 设置您的 logger 名称：`conflga.set_conflga_logger_name("your_app.config")`
3. 配置 logging 格式（可选）
4. 移除装饰器中的 `console` 参数，使用 `log_level` 参数

就是这样！您的现有代码基本不需要修改。
