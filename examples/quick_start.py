from conflga import conflga_entry, ConflgaConfig
from conflga.console import info


@conflga_entry(
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
