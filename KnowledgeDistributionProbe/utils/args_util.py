import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
        required=True,
    )
    return parser
