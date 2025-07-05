#!/usr/bin/env python3
"""
简单测试脚本，验证 logging 功能是否正常工作
"""

import logging
import conflga

# 设置 conflga 使用的 logger 名称
conflga.set_conflga_logger("test_app")

# 配置基本的 logging
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

# 测试直接使用 ConflgaConfig
print("=== 测试 1: 直接使用 ConflgaConfig ===")
from conflga import ConflgaConfig

config = ConflgaConfig(
    {
        "model": {"learning_rate": 0.001, "dropout": 0.2, "layers": [128, 64, 32]},
        "dataset": {"batch_size": 32, "shuffle": True},
        "training": {"epochs": 100, "early_stopping": True},
    }
)

config.pretty_print(title="测试配置")

print("\n=== 测试 2: 命令行覆盖 ===")
from conflga import create_override_config_from_args

override_config = create_override_config_from_args(
    ["model.learning_rate=0.01", "dataset.batch_size=64", "new_param=hello"]
)

override_config.pretty_print(title="覆盖配置")

print("\n=== 测试 3: 合并配置 ===")
config.merge_with(override_config)
config.pretty_print(title="合并后的配置")

print("\n=== 测试 4: 获取 logger 实例 ===")
logger = conflga.get_conflga_logger()
logger.info("这是一条测试日志消息")
logger.warning("这是一条警告消息")
logger.error("这是一条错误消息")

print("\n测试完成！")
