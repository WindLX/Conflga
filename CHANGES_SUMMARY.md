# Conflga 库 Logging 集成总结

## 🎯 完成的更改

### 1. 移除 Rich 依赖
- 从 `pyproject.toml` 中移除了 `rich>=14.0.0` 依赖
- 更新了所有导入语句，移除 Rich 相关的导入

### 2. 创建新的 Logging 系统
- 新建 `src/conflga/logger.py` 模块
- 实现了 `ConflgaLogger` 类，提供与 Rich 类似的功能
- 支持设置自定义 logger 名称
- 支持不同的日志级别
- 实现树状配置格式化

### 3. 更新现有模块

#### `config.py`
- 移除 Rich 导入
- 修改 `pretty_print` 方法使用新的 logging 系统
- 移除了所有 Rich 相关的方法（`_print_config_source`, `_build_tree` 等）

#### `decorator.py`
- 移除 Rich 导入
- 将 `console` 参数替换为 `log_level` 参数
- 更新函数调用以使用新的参数
- 增强错误处理：缺失的配置文件不再抛出错误

#### `__init__.py`
- 添加了 `set_conflga_logger_name` 和 `get_conflga_logger` 导出

### 4. 增强的功能
- **容错处理**: 装饰器现在在默认配置文件不存在时创建空配置
- **跳过缺失文件**: 合并配置时会跳过不存在的文件
- **自定义 Logger**: 用户可以设置库使用的 logger 名称
- **灵活日志级别**: 支持所有标准 logging 级别

### 5. 测试更新
- 修复了两个测试以反映新的容错行为
- 所有 159 个测试都通过

## 🔧 新 API

### 设置 Logger 名称
```python
import conflga
conflga.set_conflga_logger_name("my_app.config")
```

### 获取 Logger 实例
```python
logger = conflga.get_conflga_logger()
logger.info("自定义日志消息")
```

### 装饰器新参数
```python
@conflga.conflga_main(
    # ... 其他参数 ...
    log_level=logging.INFO,  # 新参数：替代 console
    # console 参数已移除
)
```

### 配置打印新参数
```python
config.pretty_print(
    title="配置标题",
    directory="config/dir", 
    files=["config1", "config2"],
    level=logging.INFO  # 新参数：日志级别
)
```

## 🌟 优势

1. **无外部依赖**: 不再依赖 Rich，减少了包大小和依赖复杂度
2. **标准 Logging**: 与 Python 标准 logging 系统完全集成
3. **更好的集成**: 用户可以使用自己的 logger 配置
4. **向后兼容**: 大部分现有代码无需修改
5. **增强的容错性**: 处理缺失配置文件的情况更优雅

## 📝 使用示例

查看以下文件了解完整使用示例：
- `test_logging.py` - 基本功能测试
- `decorator_example.py` - 装饰器使用示例
- `example_usage.py` - 完整应用示例
- `LOGGING_README.md` - 详细文档

## ✅ 验证

所有功能都已测试并验证：
- [x] 基本配置打印
- [x] 命令行覆盖
- [x] 嵌套配置
- [x] 装饰器功能
- [x] 自定义 logger 名称
- [x] 不同日志级别
- [x] 缺失配置文件处理
- [x] 所有现有测试通过
