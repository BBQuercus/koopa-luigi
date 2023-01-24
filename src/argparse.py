import argparse
import sys

from . import __version__


def _parse_args():
    """Basic argument parser."""
    parser = argparse.ArgumentParser(
        prog="Koopa",
        description="Workflow for analysis of cellular microscopy data.",
        add_help=False,
    )

    # Basic running
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to luigi configuration file.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel luigi workers to spawn. [default: 4]",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="If GPUs should be used for processing - will set workers to 1.",
    )

    # Utils
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this message.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + str(__version__),
        help="Show %(prog)s's version number.",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(0)
    return parser.parse_args()
