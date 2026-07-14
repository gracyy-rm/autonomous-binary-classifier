import argparse

from src.config import load_config
from src.train import run_pipeline

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        "Autonomous Driving Binary Classification Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to the configuration JSON file"

    )

    return parser.parse_args()

def main():
    """
    Entry point to the application.
    """
    args = parse_arguments()
    config = load_config(args.config)
    run_pipeline(config=config)

if __name__ == "__main__":
    main()

