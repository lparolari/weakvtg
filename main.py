import argparse
import json
import logging

from weakvtg.config import parse_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, validate, test or plot some example with `weakvtg` model.")

    parser.add_argument("--configs", dest="configs", type=str, default=None,
                        help="Model parameters as a JSON dictionary.")
    parser.add_argument("--log-level", dest="log_level", type=int, default=logging.DEBUG, help="Log verbosity")
    parser.add_argument("--log-file", dest="log_file", type=str, default=None, help="Log filename")

    args = parser.parse_args()
    configs = parse_configs(args.configs)

    logging.basicConfig(filename=args.log_file, level=args.log_level)

    logging.info(f"Model started with following parameters: {configs}")

