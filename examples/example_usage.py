#!/usr/bin/env python3
"""
示例：展示如何使用 Conflga 库的 logging 功能

这个示例展示了如何：
1. 设置自定义的 logger 名称
2. 配置 logging 输出格式
3. 使用 conflga_main 装饰器
4. 查看配置打印到日志中
"""

import logging
import conflga
from conflga import conflga_main, ConflgaConfig

# 步骤 1: 设置 Conflga 使用的 logger 名称
conflga.set_conflga_logger_name("my_app.config")

# 步骤 2: 配置 logging (可选，根据你的需求配置)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 或者使用更高级的配置
logger = logging.getLogger("my_app.config")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='🔧 [%(name)s] %(levelname)s: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@conflga_main(
    config_dir="conf",
    default_config="config", 
    enable_cli_override=True,
    auto_print=True,
    auto_print_override=True,
    log_level=logging.INFO,  # 使用 INFO 级别记录配置
    use_namespace_prefix=True
)
def main(cfg: ConflgaConfig):
    """主函数 - 配置会自动通过 logger 打印"""
    
    # 获取 conflga 使用的 logger 实例
    config_logger = conflga.get_conflga_logger()
    
    # 使用同一个 logger 记录应用程序日志
    config_logger.info("应用程序启动")
    config_logger.info(f"当前配置的某个值: {cfg.get('some_key', 'default_value')}")
    
    # 手动打印配置（如果需要的话）
    cfg.pretty_print(
        title="手动打印的配置",
        level=logging.DEBUG  # 使用不同的日志级别
    )
    
    config_logger.info("应用程序结束")


if __name__ == "__main__":
    # 如果你有配置文件夹，这会正常工作
    # 如果没有，decorator 仍然会工作，只是配置为空
    main()


# 使用方法:
#
# 1. 基本使用:
#    python example_usage.py
#
# 2. 使用命令行覆盖:
#    python example_usage.py --conflga-override model.learning_rate=0.001 --conflga-override dataset.batch_size=32
#
# 3. 设置不同的日志级别查看效果:
#    你可以修改上面的 logging.basicConfig 中的 level 参数来查看不同级别的日志输出
