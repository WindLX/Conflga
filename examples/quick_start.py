from conflga import conflga_func, conflga_method, ConflgaConfig
from conflga.console import info


class MyAwesomeTraining:
    """
    A class representing an awesome training session.
    This class can be decorated with `conflga_method` to automatically load configurations.
    """

    @conflga_method(
        config_dir="examples/awesome_config",
        default_config="00-marcos",
        configs_to_merge=["01-base_config", "02-other_config"],
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
    default_config="00-marcos",
    configs_to_merge=["01-base_config", "02-other_config"],
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
