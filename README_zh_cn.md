# Conflga

[English README](README.md) | [中文说明](README.zh-CN.md)

一个强大、灵活且易用的Python配置管理库。Conflga 提供基于TOML的配置管理解决方案，支持配置合并、命令行覆盖、模板变量预处理等高级功能。

## ✨ 核心特性

- 🚀 **简单易用**: 一个装饰器即可实现完整的配置管理
- 📝 **TOML格式**: 支持TOML配置文件，语法简洁清晰  
- 🔄 **配置合并**: 支持多个配置文件的层级合并
- 🎯 **点访问语法**: 支持 `config.model.learning_rate` 形式的属性访问
- ⚡ **命令行覆盖**: 运行时通过命令行参数覆盖配置值
- 🎨 **美观输出**: 集成Rich库，提供美观的配置输出显示
- 📐 **模板预处理**: 支持宏定义和模板表达式，实现动态配置计算
- 🔧 **灵活集成**: 可嵌入其他项目，避免命令行参数冲突
- 🎪 **类型友好**: 自动解析字符串、数字、布尔值、列表、字典等类型

## 📦 安装

### 使用 uv (推荐)

```bash
uv add conflga
```

### 使用 pip

```bash
pip install conflga
```

## 🚀 快速开始

### 1. 创建配置文件

在项目目录下创建配置目录 `examples/awesome_config/`:

**base_config.toml**:
```toml
#let exp_name = "hopper_dppo"
#let num_envs = 4

[log]
log_dir = "runs"
experiment_prefix = "{{ exp_name }}-{{ str(num_envs) }}"
level = "INFO"

[env]
env_id = "Hopper-v5"
obs_dim = 11
act_dim = 3
num_envs = "{{ num_envs }}"
num_eval_envs = 1

[train]
total_steps = 10_000_000
batch_size = 64
buffer_size = "{{ num_envs * 256 }}"
learning_rate = 3e-4
update_epochs = 10
device = "cuda"

[ppo]
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
value_loss_coef = 0.5
max_grad_norm = 0.5
normalize_advantages = true
```

**other_config.toml**:
```toml
[[wrapper]]
type = "NormalizeObservation"
kwargs = {}

[statera]
path = "checkpoints/{{ exp_name }}"
save_every_iterations = 10
save_best_reward = true
use_safetensors = true
compression = "@None"
```

### 2. 使用装饰器

**quick_start.py**:
```python
from conflga import conflga_func, conflga_method, ConflgaConfig
from conflga.console import info


class MyAwesomeTraining:
    """
    A class representing an awesome training session.
    This class can be decorated with `conflga_method` to automatically load configurations.
    """

    @conflga_method(
        config_dir="examples/awesome_config",
        default_config="base_config",
        configs_to_merge=["other_config"],
        enable_preprocessor=True,
        enable_cli_override=True,
        use_namespace_prefix=True,
        auto_print=True,
        auto_print_override=True,
    )
    def train(self, cfg: ConflgaConfig):
        """
        Train the model using the provided configuration.
        The configuration will be printed automatically due to auto_print=True.
        """
        info("Training started with the following configuration:")
        info(f"\n{cfg.to_dict()}")


@conflga_func(
    config_dir="examples/awesome_config",
    default_config="base_config",
    configs_to_merge=["other_config"],
    enable_preprocessor=True,
    enable_cli_override=True,
    use_namespace_prefix=True,
    auto_print=True,
    auto_print_override=True,
)
def main(cfg: ConflgaConfig):
    """
    Main function that will be executed with the configuration loaded.
    The configuration will be printed automatically due to auto_print=True.
    """
    info("Configuration loaded successfully!")
    info(f"\n{cfg.to_dict()}")


if __name__ == "__main__":
    main()  # This will run the main function with the loaded configuration
    training_session = MyAwesomeTraining()
    training_session.train()  # This will run the train method with the loaded configuration
```

### 3. 运行程序

```bash
# 使用默认配置
python quick_start.py

# 通过命令行覆盖参数
python quick_start.py --conflga-override train.learning_rate=1e-3 --conflga-override ppo.gamma=0.95
```

## 📚 详细使用指南

### 基本配置类

```python
from conflga import ConflgaConfig

# 从文件加载
config = ConflgaConfig.load("config.toml")

# 从字符串加载
toml_string = """
[model]
name = "ResNet"
layers = 50
"""
config = ConflgaConfig.from_string(toml_string)

# 点访问语法
print(config.model.name)  # "ResNet"
print(config.model.layers)  # 50

# 字典访问语法
print(config["model"]["name"])  # "ResNet"

# 转换为字典
config_dict = config.to_dict()
```

### 配置管理器

```python
from conflga import ConflgaManager

# 初始化管理器
manager = ConflgaManager(config_dir="conf")

# 加载默认配置
manager.load_default("base")

# 合并其他配置文件
manager.merge_config("dev", "local")

# 从字典覆盖配置
override_dict = {"model": {"learning_rate": 0.001}}
manager.override_from_dict(override_dict)

# 获取最终配置
config = manager.get_config()
```

### 命令行接口

```python
from conflga.cli import ConflgaCLI, create_override_config_from_args

# 创建CLI实例
cli = ConflgaCLI(use_namespace_prefix=True)  # 使用 --conflga-override
cli = ConflgaCLI(use_namespace_prefix=False)  # 使用 -o/--override
cli = ConflgaCLI(custom_arg_name="--my-config")  # 自定义参数名

# 解析覆盖参数
override_config = cli.parse_overrides([
    "model.learning_rate=0.001",
    "dataset.batch_size=32",
    "training.use_gpu=true"
])

# 便捷函数
override_config = create_override_config_from_args([
    "model.learning_rate=0.001",
    "dataset.batch_size=32"
])
```

### 模板预处理功能

Conflga 支持强大的模板预处理功能，允许在配置文件中使用宏定义和模板表达式：

```toml
# 定义宏变量
#let SERVICE_NAME = "api"
#let ENVIRONMENT = "prod"
#let WORKERS = 4
#let BASE_MEMORY = 512
#let MEMORY_PER_WORKER = BASE_MEMORY * 2

[app]
name = "{{ SERVICE_NAME }}-service"
environment = "{{ ENVIRONMENT }}"

[deployment]
workers = {{ WORKERS }}
memory_per_worker = {{ MEMORY_PER_WORKER }}
total_memory = {{ MEMORY_PER_WORKER * WORKERS }}
high_memory_mode = {{ MEMORY_PER_WORKER * WORKERS > 2000 }}

[database]
connection_string = "postgresql://user:pass@{{ SERVICE_NAME }}-db:5432/{{ SERVICE_NAME }}_{{ ENVIRONMENT }}"
```

支持的表达式类型：
- 数学运算：`{{ WORKERS * 2 }}`
- 字符串连接：`{{ SERVICE_NAME + "-" + ENVIRONMENT }}`
- 比较运算：`{{ MEMORY > 1000 }}`
- 函数调用：`{{ str(PORT) }}`
- 布尔值：`{{ true }}`, `{{ false }}`
- 空值：`{{ null }}`, `{{ @None }}`

#### 内置函数

Conflga 内置了常用的数学、随机数和类型转换相关函数，方便在表达式或脚本中调用。

包含的函数及其说明如下：

数学计算:
- `abs(x)`: 返回 x 的绝对值。
- `max(*args)`: 返回参数中的最大值。
- `min(*args)`: 返回参数中的最小值。
- `pow(x, y)`: 返回 x 的 y 次幂。
- `round(x, n)`: 对 x 四舍五入，保留 n 位小数。
- `sqrt(x)`: 返回 x 的平方根。
- `log(x)`: 返回 x 的自然对数（以 e 为底）。
- `log10(x)`: 返回 x 的以 10 为底的对数。
- `exp(x)`: 返回 e 的 x 次幂。
- `sin(x)`: 返回 x 的正弦值（x 为弧度）。
- `cos(x)`: 返回 x 的余弦值（x 为弧度）。
- `tan(x)`: 返回 x 的正切值（x 为弧度）。
- `asin(x)`: 返回 x 的反正弦值（结果为弧度）。
- `acos(x)`: 返回 x 的反余弦值（结果为弧度）。
- `atan(x)`: 返回 x 的反正切值（结果为弧度）。
- `degrees(x)`: 将弧度 x 转换为角度。
- `radians(x)`: 将角度 x 转换为弧度。
- `tanh(x)`: 返回 x 的双曲正切值。
- `ceil(x)`: 返回大于等于 x 的最小整数。
- `floor(x)`: 返回小于等于 x 的最大整数。
- `gcd(x, y)`: 返回 x 和 y 的最大公约数。
- `lcm(x, y)`: 返回 x 和 y 的最小公倍数。

随机数:
- `random()`: 生成 0 到 1 之间的随机浮点数。
- `randint(a, b)`: 生成指定范围 [a, b] 内的随机整数。
- `normal(mu, sigma)`: 生成正态分布随机数，mu 为均值，sigma 为标准差。
- `uniform(a, b)`: 生成指定范围 [a, b] 内的随机浮点数。
- `choice(seq)`: 从序列 seq 中随机选择一个元素。
- `shuffle(seq)`: 打乱序列 seq 的顺序，返回打乱后的序列。

类型与数据结构:
- `len(obj)`: 返回对象 obj 的长度。
- `str(x)`: 将 x 转换为字符串。
- `int(x)`: 将 x 转换为整数。
- `float(x)`: 将 x 转换为浮点数。
- `range(*args)`: 返回一个范围对象，Python 的 range 函数。
- `sum(iterable)`: 返回可迭代对象 iterable 中所有元素的和。


### 美观输出

```python
from conflga.console import ConflgaEchoa

# 创建输出管理器
echoa = ConflgaEchoa()

# 打印配置
echoa.print_config(
    config_data=config,
    title="应用配置",
    directory="./conf",
    files=["base", "override"]
)

# 或者直接使用配置对象的方法
config.pretty_print(
    title="我的配置",
    directory="./conf",
    files=["config"]
)
```

## 🎯 装饰器参数详解

Conflga 提供了两个装饰器：`conflga_func` 和 `conflga_method`，`conflga_func` 用于函数或者静态方法装饰，`conflga_method` 用于实例方法或类方法装饰。它们的主要区别在于处理方式和适用场景。

`@conflga_func` 和 `@conflga_method` 装饰器支持以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_dir` | `str` | `"conf"` | 配置文件目录 |
| `default_config` | `str` | `"config"` | 默认配置文件名（不含.toml后缀） |
| `configs_to_merge` | `list[str]` | `None` | 要合并的额外配置文件列表 |
| `enable_preprocessor` | `bool` | `True` | 是否启用模板预处理 |
| `enable_cli_override` | `bool` | `True` | 是否启用命令行覆盖 |
| `use_namespace_prefix` | `bool` | `True` | 是否使用命名空间前缀避免冲突 |
| `auto_print` | `bool` | `True` | 是否自动打印最终配置 |
| `auto_print_override` | `bool` | `True` | 是否自动打印覆盖配置 |
| `console` | `Console` | `None` | Rich控制台对象 |
| `echoa` | `ConflgaEchoa` | `None` | 自定义输出管理器 |


## 💡 命令行覆盖语法

### 基本语法

```bash
# 简单值
--conflga-override key=value

# 嵌套键（点记法）
--conflga-override model.learning_rate=0.001
--conflga-override dataset.train.batch_size=32

# 不同数据类型
--conflga-override debug=true                    # 布尔值
--conflga-override max_epochs=100                # 整数
--conflga-override learning_rate=1e-4           # 浮点数
--conflga-override model_name=ResNet50          # 字符串
--conflga-override gpu_ids=[0,1,2,3]            # 列表
--conflga-override optimizer="{'type': 'adam'}" # 字典
--conflga-override dropout=null                 # 空值
```

### 高级示例

```bash
# 机器学习训练配置
python train.py \
  --conflga-override model.architecture=resnet50 \
  --conflga-override model.pretrained=true \
  --conflga-override optimizer.lr=0.001 \
  --conflga-override optimizer.weight_decay=1e-4 \
  --conflga-override dataset.batch_size=64 \
  --conflga-override training.epochs=100 \
  --conflga-override training.device=cuda
```

## 🔧 避免命令行冲突

Conflga 提供多种方式避免与其他工具的命令行参数冲突：

### 1. 使用命名空间前缀（推荐）

```python
@conflga_func(use_namespace_prefix=True)  # 使用 -co/--conflga-override
def main(cfg):
    pass
```

### 2. 使用短选项

```python
@conflga_func(use_namespace_prefix=False)  # 使用 -o/--override
def main(cfg):
    pass
```

### 3. 自定义参数名

```python
# 在装饰器中使用 ConflgaCLI
from conflga.cli import ConflgaCLI

cli = ConflgaCLI(custom_arg_name="--my-config-override")
override_config = cli.parse_overrides()
```

## 🏗️ 项目结构

```
conflga/
├── src/conflga/
│   ├── __init__.py         # 主要导出
│   ├── config.py           # ConflgaConfig 核心配置类
│   ├── manager.py          # ConflgaManager 配置管理器
│   ├── decorator.py        # conflga_func 和 conflga_method 装饰器
│   ├── cli.py              # 命令行接口
│   ├── console.py          # 美观输出功能
│   └── preprocessor.py     # 模板预处理器
├── examples/               # 示例代码
├── tests/                  # 测试用例
└── README.md              # 本文档
```

## 🧪 测试

项目包含完整的测试套件：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_config.py
pytest tests/test_cli.py
pytest tests/test_decorator.py

# 运行测试并查看覆盖率
pytest --cov=conflga --cov-report=html
```

## 📖 API 参考

### ConflgaConfig

配置数据的核心容器类，支持点访问和字典访问语法。

**方法：**
- `load(toml_path: str) -> ConflgaConfig`: 从TOML文件加载
- `from_string(toml_string: str) -> ConflgaConfig`: 从TOML字符串加载
- `merge_with(other: ConflgaConfig) -> ConflgaConfig`: 合并另一个配置
- `to_dict() -> dict`: 转换为普通字典
- `pretty_print(...)`: 美观打印配置

### ConflgaManager

配置管理器，提供配置加载、合并、覆盖等功能。

**方法：**
- `load_default(name: str) -> ConflgaManager`: 加载默认配置
- `merge_config(*names: str) -> ConflgaManager`: 合并配置文件
- `override_config(config: ConflgaConfig) -> ConflgaManager`: 配置覆盖
- `override_from_dict(dict: dict) -> ConflgaManager`: 从字典覆盖
- `get_config() -> ConflgaConfig`: 获取最终配置

### ConflgaCLI

命令行接口，解析命令行参数并生成覆盖配置。

**方法：**
- `parse_overrides(override_strings: list[str]) -> ConflgaConfig`: 解析覆盖参数

## 🤝 贡献

欢迎提交Issues和Pull Requests！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📝 许可证

本项目使用 MIT 许可证。查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [Rich](https://github.com/Textualize/rich) 提供美观的终端输出
- 感谢 [rtoml](https://github.com/samuelcolvin/rtoml) 提供快速的TOML解析

---

**Conflga** - 让配置管理变得简单而强大！ 🎉