"""Command line interface."""

import os
import sys
import luigi

from . import util
from . import cli_args
from .postprocess import Merge
from .postprocess import GPUMerge


def main():
    """Run koopa tasks."""
    args = cli_args._parse_args()
    
    # Handle --env-info flag
    if args.env_info:
        print(util.format_environment_info())
        sys.exit(0)
    
    # Set up configuration and logging
    util.set_configuration(args.config)
    util.set_logging()
    
    
    # Log environment information at startup
    import logging
    logger = logging.getLogger('koopa-luigi')
    env_info = util.get_environment_info()
    logger.info("=" * 50)
    logger.info("Starting Koopa-Luigi Pipeline")
    logger.info(f"Python: {env_info['python_version']}")
    logger.info(f"Virtual Environment: {env_info['virtual_env'] or 'None'}")
    
    
    
    logger.info("=" * 50)
    
    # Run Luigi tasks
    tasks = [GPUMerge()] if args.gpu else [Merge()]
    workers = 1 if args.gpu else args.workers
    luigi.build(tasks, local_scheduler=True, workers=workers)
