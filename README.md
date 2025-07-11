# Conflga

[English README](README.md) | [中文说明](README.zh-CN.md)

A powerful, flexible, and easy-to-use Python configuration management library. Conflga provides a TOML-based configuration solution, supporting config merging, command-line overrides, template variable preprocessing, and other advanced features.

## ✨ Core Features

- 🚀 **Easy to Use**: Full configuration management with a single decorator
- 📝 **TOML Format**: Supports TOML config files with clean syntax  
- 🔄 **Config Merging**: Hierarchical merging of multiple config files
- 🎯 **Dot Access Syntax**: Access config via `config.model.learning_rate`
- ⚡ **Command-line Overrides**: Override config values at runtime via CLI
- 🎨 **Pretty Output**: Integrated Rich library for beautiful config display
- 📐 **Template Preprocessing**: Macro definitions and template expressions for dynamic config calculation
- 🔧 **Flexible Integration**: Embeddable in other projects, avoids CLI argument conflicts
- 🎪 **Type Friendly**: Auto-parsing for strings, numbers, booleans, lists, dicts, etc.

## 📦 Installation

### Using uv (Recommended)

```bash
uv add git+https://github.com/WindLX/Conflga.git
```

### Using pip

```bash
pip install git+https://github.com/WindLX/Conflga.git
```

## 🚀 Quick Start

### 1. Create Config Files

Create a config directory `examples/awesome_config/` in your project:

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

### 2. Use Decorators

**quick_start.py**:
```python
from conflga import conflga_func, conflga_method, ConflgaConfig
from conflga.console import info

class MyAwesomeTraining:
    """
    Represents an awesome training session.
    Decorate with `conflga_method` to auto-load configs.
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
        Config will be printed automatically.
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
    Main function executed with loaded configuration.
    Config will be printed automatically.
    """
    info("Configuration loaded successfully!")
    info(f"\n{cfg.to_dict()}")

if __name__ == "__main__":
    main()  # Runs main with loaded config
    training_session = MyAwesomeTraining()
    training_session.train()  # Runs train with loaded config
```

### 3. Run the Program

```bash
# Use default config
python quick_start.py

# Override parameters via CLI
python quick_start.py --conflga-override train.learning_rate=1e-3 --conflga-override ppo.gamma=0.95
```

## 📚 Detailed Usage Guide

### Basic Config Class

```python
from conflga import ConflgaConfig

# Load from file
config = ConflgaConfig.load("config.toml")

# Load from string
toml_string = """
[model]
name = "ResNet"
layers = 50
"""
config = ConflgaConfig.from_string(toml_string)

# Dot access
print(config.model.name)  # "ResNet"
print(config.model.layers)  # 50

# Dict access
print(config["model"]["name"])  # "ResNet"

# Convert to dict
config_dict = config.to_dict()
```

### Config Manager

```python
from conflga import ConflgaManager

# Initialize manager
manager = ConflgaManager(config_dir="conf")

# Load default config
manager.load_default("base")

# Merge other config files
manager.merge_config("dev", "local")

# Override from dict
override_dict = {"model": {"learning_rate": 0.001}}
manager.override_from_dict(override_dict)

# Get final config
config = manager.get_config()
```

### Command-line Interface

```python
from conflga.cli import ConflgaCLI, create_override_config_from_args

# Create CLI instance
cli = ConflgaCLI(use_namespace_prefix=True)  # Use --conflga-override
cli = ConflgaCLI(use_namespace_prefix=False)  # Use -o/--override
cli = ConflgaCLI(custom_arg_name="--my-config")  # Custom arg name

# Parse override parameters
override_config = cli.parse_overrides([
    "model.learning_rate=0.001",
    "dataset.batch_size=32",
    "training.use_gpu=true"
])

# Convenience function
override_config = create_override_config_from_args([
    "model.learning_rate=0.001",
    "dataset.batch_size=32"
])
```

### Template Preprocessing

Conflga supports powerful template preprocessing, allowing macros and template expressions in config files:

```toml
# Macro definitions
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

Supported expression types:
- Math: `{{ WORKERS * 2 }}`
- String concat: `{{ SERVICE_NAME + "-" + ENVIRONMENT }}`
- Comparison: `{{ MEMORY > 1000 }}`
- Function call: `{{ str(PORT) }}`
- Boolean: `{{ true }}`, `{{ false }}`
- Null: `{{ null }}`, `{{ @None }}`

#### Built-in Functions

Conflga includes common math, random, and type conversion functions for use in expressions/scripts.

Functions include:

Math:
- `abs(x)`: Absolute value
- `max(*args)`: Max value
- `min(*args)`: Min value
- `pow(x, y)`: x to the power of y
- `round(x, n)`: Round x to n decimals
- `sqrt(x)`: Square root
- `log(x)`: Natural log
- `log10(x)`: Log base 10
- `exp(x)`: e to the power of x
- `sin(x)`, `cos(x)`, `tan(x)`: Trigonometric functions (radians)
- `asin(x)`, `acos(x)`, `atan(x)`: Inverse trig (radians)
- `degrees(x)`, `radians(x)`: Convert between degrees/radians
- `tanh(x)`: Hyperbolic tangent
- `ceil(x)`: Ceiling
- `floor(x)`: Floor
- `gcd(x, y)`: Greatest common divisor
- `lcm(x, y)`: Least common multiple

Random:
- `random()`: Random float [0, 1)
- `randint(a, b)`: Random int in [a, b]
- `normal(mu, sigma)`: Normal distribution
- `uniform(a, b)`: Random float in [a, b]
- `choice(seq)`: Random element from sequence
- `shuffle(seq)`: Shuffle sequence

Types & Data Structures:
- `len(obj)`: Length of object
- `str(x)`: Convert to string
- `int(x)`: Convert to int
- `float(x)`: Convert to float
- `range(*args)`: Python range
- `sum(iterable)`: Sum of iterable

### Pretty Output

```python
from conflga.console import ConflgaEchoa

# Create output manager
echoa = ConflgaEchoa()

# Print config
echoa.print_config(
    config_data=config,
    title="App Config",
    directory="./conf",
    files=["base", "override"]
)

# Or use config object's method
config.pretty_print(
    title="My Config",
    directory="./conf",
    files=["config"]
)
```

## 🎯 Decorator Arguments

Conflga provides two decorators: `conflga_func` for functions/static methods, and `conflga_method` for instance/class methods. The main difference is their handling and use case.

Both support these arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config_dir` | `str` | `"conf"` | Config directory |
| `default_config` | `str` | `"config"` | Default config file name (no .toml) |
| `configs_to_merge` | `list[str]` | `None` | Extra config files to merge |
| `enable_preprocessor` | `bool` | `True` | Enable template preprocessing |
| `enable_cli_override` | `bool` | `True` | Enable CLI overrides |
| `use_namespace_prefix` | `bool` | `True` | Use namespace prefix to avoid conflicts |
| `auto_print` | `bool` | `True` | Auto print final config |
| `auto_print_override` | `bool` | `True` | Auto print override config |
| `console` | `Console` | `None` | Rich console object |
| `echoa` | `ConflgaEchoa` | `None` | Custom output manager |

## 💡 CLI Override Syntax

### Basic Syntax

```bash
# Simple value
--conflga-override key=value

# Nested key (dot notation)
--conflga-override model.learning_rate=0.001
--conflga-override dataset.train.batch_size=32

# Data types
--conflga-override debug=true                    # Boolean
--conflga-override max_epochs=100                # Integer
--conflga-override learning_rate=1e-4            # Float
--conflga-override model_name=ResNet50           # String
--conflga-override gpu_ids=[0,1,2,3]             # List
--conflga-override optimizer="{'type': 'adam'}"  # Dict
--conflga-override dropout=null                  # Null
```

### Advanced Example

```bash
# ML training config
python train.py \
  --conflga-override model.architecture=resnet50 \
  --conflga-override model.pretrained=true \
  --conflga-override optimizer.lr=0.001 \
  --conflga-override optimizer.weight_decay=1e-4 \
  --conflga-override dataset.batch_size=64 \
  --conflga-override training.epochs=100 \
  --conflga-override training.device=cuda
```

## 🔧 Avoid CLI Conflicts

Conflga offers several ways to avoid CLI argument conflicts:

### 1. Use Namespace Prefix (Recommended)

```python
@conflga_func(use_namespace_prefix=True)  # Uses -co/--conflga-override
def main(cfg):
    pass
```

### 2. Use Short Option

```python
@conflga_func(use_namespace_prefix=False)  # Uses -o/--override
def main(cfg):
    pass
```

### 3. Custom Argument Name

```python
# Use ConflgaCLI in decorator
from conflga.cli import ConflgaCLI

cli = ConflgaCLI(custom_arg_name="--my-config-override")
override_config = cli.parse_overrides()
```

## 🏗️ Project Structure

```
conflga/
├── src/conflga/
│   ├── __init__.py         # Main exports
│   ├── config.py           # ConflgaConfig core class
│   ├── manager.py          # ConflgaManager
│   ├── decorator.py        # conflga_func & conflga_method decorators
│   ├── cli.py              # CLI interface
│   ├── console.py          # Pretty output
│   └── preprocessor.py     # Template preprocessor
├── examples/               # Example code
├── tests/                  # Test cases
└── README.md               # This document
```

## 🧪 Testing

The project includes a complete test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py
pytest tests/test_cli.py
pytest tests/test_decorator.py

# Run tests with coverage report
pytest --cov=conflga --cov-report=html
```

## 📖 API Reference

### ConflgaConfig

Core container for config data, supports dot and dict access.

**Methods:**
- `load(toml_path: str) -> ConflgaConfig`: Load from TOML file
- `from_string(toml_string: str) -> ConflgaConfig`: Load from TOML string
- `merge_with(other: ConflgaConfig) -> ConflgaConfig`: Merge another config
- `to_dict() -> dict`: Convert to dict
- `pretty_print(...)`: Pretty print config

### ConflgaManager

Config manager for loading, merging, and overriding configs.

**Methods:**
- `load_default(name: str) -> ConflgaManager`: Load default config
- `merge_config(*names: str) -> ConflgaManager`: Merge config files
- `override_config(config: ConflgaConfig) -> ConflgaManager`: Override config
- `override_from_dict(dict: dict) -> ConflgaManager`: Override from dict
- `get_config() -> ConflgaConfig`: Get final config

### ConflgaCLI

CLI interface for parsing command-line overrides.

**Methods:**
- `parse_overrides(override_strings: list[str]) -> ConflgaConfig`: Parse override parameters

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project uses the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- Thanks to [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- Thanks to [rtoml](https://github.com/samuelcolvin/rtoml) for fast TOML parsing

---

**Conflga** - Make configuration management simple and powerful! 🎉
