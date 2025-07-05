#!/usr/bin/env python3
"""
完整示例：展示 conflga_main 装饰器和 logging 的集成使用
"""

import logging
import conflga

# 设置 conflga 使用的 logger 名称
conflga.set_conflga_logger("my_ml_app.config")

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# 为配置日志添加特殊格式
config_logger = logging.getLogger("my_ml_app.config")
handler = logging.StreamHandler()
formatter = logging.Formatter("🔧 [CONFIG] %(message)s")
handler.setFormatter(formatter)
config_logger.handlers = [handler]  # 替换默认处理器
config_logger.setLevel(logging.INFO)


@conflga.conflga_main(
    config_dir="example_conf",  # 假设这个目录不存在，演示空配置的情况
    default_config="model_config",
    enable_cli_override=True,
    auto_print=True,
    auto_print_override=True,
    log_level=logging.INFO,
    use_namespace_prefix=True,
)
def train_model(cfg: conflga.ConflgaConfig):
    """训练机器学习模型的示例函数"""

    # 获取 conflga 的 logger 用于应用日志
    logger = conflga.get_conflga_logger()

    logger.info("开始训练模型...")

    # 从配置中获取参数，提供默认值
    learning_rate = getattr(cfg, "learning_rate", 0.001)
    batch_size = getattr(cfg, "batch_size", 32)
    epochs = getattr(cfg, "epochs", 10)

    logger.info(f"训练参数:")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  训练轮数: {epochs}")

    # 模拟训练过程
    for epoch in range(epochs):
        logger.info(f"训练轮次 {epoch + 1}/{epochs}")
        # 这里会是实际的训练代码

    logger.info("模型训练完成!")

    # 如果有嵌套配置，也可以访问
    if hasattr(cfg, "model") and hasattr(cfg.model, "architecture"):
        logger.info(f"模型架构: {cfg.model.architecture}")


if __name__ == "__main__":
    # 运行示例
    print("=== Conflga Logging 集成示例 ===\n")

    print("1. 基本使用（没有配置文件）:")
    print("   python decorator_example.py\n")

    print("2. 使用命令行覆盖:")
    print(
        "   python decorator_example.py --conflga-override learning_rate=0.01 --conflga-override batch_size=64 --conflga-override epochs=20\n"
    )

    print("3. 复杂嵌套配置:")
    print(
        "   python decorator_example.py --conflga-override model.architecture=resnet --conflga-override model.layers=50 --conflga-override optimizer.type=adam\n"
    )

    print("现在运行训练函数...\n")
    train_model()
