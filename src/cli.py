"""Command line interface."""

import luigi

from . import util
from . import argparse
from .postprocess import Merge
from .postprocess import GPUMerge


def main():
    """Run koopa tasks."""
    args = argparse._parse_args()
    util.set_configuration(args.config)
    util.set_logging()

    tasks = [GPUMerge()] if args.gpu else [Merge()]
    workers = 1 if args.gpu else args.workers
    luigi.build(tasks, local_scheduler=True, workers=workers)
