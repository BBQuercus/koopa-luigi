"""Command line interface."""

from __future__ import annotations

# Suppress noisy output BEFORE importing anything else
import os
import sys
import warnings

# ASCII art banners
BANNER_START = r"""
▄ •▄              ▄▄▄· ▄▄▄·
█▌▄▌▪▪     ▪     ▐█ ▄█▐█ ▀█
▐▀▀▄· ▄█▀▄  ▄█▀▄  ██▀·▄█▀▀█
▐█.█▌▐█▌.▐▌▐█▌.▐▌▐█▪·•▐█ ▪▐▌
·▀  ▀ ▀█▄▀▪ ▀█▄▀▪.▀    ▀  ▀
"""

BANNER_SUCCESS = r"""
.▄▄ · ▄• ▄▌ ▄▄·  ▄▄· ▄▄▄ ..▄▄ · .▄▄ ·   \(^‿^)/
▐█ ▀. █▪██▌▐█ ▌▪▐█ ▌▪▀▄.▀·▐█ ▀. ▐█ ▀.
▄▀▀▀█▄█▌▐█▌██ ▄▄██ ▄▄▐▀▀▪▄▄▀▀▀█▄▄▀▀▀█▄
▐█▄▪▐█▐█▄█▌▐███▌▐███▌▐█▄▄▌▐█▄▪▐█▐█▄▪▐█
 ▀▀▀▀  ▀▀▀ ·▀▀▀ ·▀▀▀  ▀▀▀  ▀▀▀▀  ▀▀▀▀
"""

BANNER_FAILURE = r"""
·▄▄▄ ▄▄▄· ▪  ▄▄▌  ▄▄▄ .·▄▄▄▄    (╯°□°)╯︵ ┻━┻
▐▄▄·▐█ ▀█ ██ ██•  ▀▄.▀·██▪ ██
██▪ ▄█▀▀█ ▐█·██▪  ▐▀▀▪▄▐█· ▐█▌
██▌.▐█ ▪▐▌▐█▌▐█▌▐▌▐█▄▄▌██. ██
▀▀▀  ▀  ▀ ▀▀▀.▀▀▀  ▀▀▀ ▀▀▀▀▀•
"""

# Suppress TensorFlow/Keras noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

# Prevent TensorFlow from greedily allocating all GPU memory.
# Without this, TF claims the entire VRAM after running deepblink,
# leaving no memory for cellpose (PyTorch) on subsequent files.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Suppress Keras progress bars and cellpose banner
os.environ["KERAS_BACKEND"] = os.environ.get("KERAS_BACKEND", "tensorflow")
os.environ["CELLPOSE_QUIET"] = "1"

# Suppress various warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")
warnings.filterwarnings("ignore", message=".*overflow encountered.*")

import luigi

from . import util
from . import cli_args
from .postprocess import Merge
from .postprocess import GPUMerge
from .preflight import run_preflight
from .util import file_tracker


def _suppress_keras_progress():
    """Disable Keras progress bar output."""
    try:
        # TensorFlow 2.x
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        # Disable Keras progress bars
        if hasattr(tf.keras.utils, "disable_interactive_logging"):
            tf.keras.utils.disable_interactive_logging()
    except (ImportError, AttributeError):
        pass


def main() -> None:
    """Run koopa tasks."""
    args = cli_args._parse_args()

    # Handle --env-info flag
    if args.env_info:
        print(util.format_environment_info())
        sys.exit(0)

    # Set up configuration first to get output_path
    util.set_configuration(args.config)
    config = util.get_configuration()

    # Ensure output directory exists before setting up logging
    os.makedirs(config["output_path"], exist_ok=True)

    # Set up logging with file output
    util.set_logging(output_path=config["output_path"], verbose=args.verbose)
    logger = util.get_logger("cli")

    # Initialize file tracker with output path for cross-process persistence
    file_tracker.reset()
    file_tracker.set_output_path(config["output_path"])

    # Suppress Keras/TensorFlow progress bars
    _suppress_keras_progress()

    # Log startup information
    env_info = util.get_environment_info()
    for line in BANNER_START.strip().split("\n"):
        logger.info(line)
    logger.info("")
    logger.info(f"Python: {env_info['python_version']}")
    logger.info(f"Virtual env: {env_info['virtual_env'] or 'None'}")
    logger.info(f"Input: {config['input_path']}")
    logger.info(f"Output: {config['output_path']}")
    logger.info(f"Log file: {os.path.join(config['output_path'], 'koopa.log')}")
    logger.info(
        f"Mode: {'GPU' if args.gpu else 'CPU'} with {1 if args.gpu else args.workers} worker(s)"
    )
    if args.skip_incompatible:
        logger.info("Skip incompatible models: enabled")
    logger.info("")

    # Run preflight checks before starting pipeline
    run_preflight(config)
    logger.info("")

    # Store skip_incompatible in config for tasks to access
    luigi_config = luigi.configuration.get_config()
    luigi_config.set("core", "skip_incompatible", str(args.skip_incompatible))

    # Run Luigi tasks with suppressed Luigi logging
    tasks = (
        [GPUMerge(skip_incompatible=args.skip_incompatible)]
        if args.gpu
        else [Merge(skip_incompatible=args.skip_incompatible)]
    )
    workers = 1 if args.gpu else args.workers

    # Suppress Luigi's internal logging during build
    import logging as _logging

    for name in ["luigi", "luigi-interface", "luigi.scheduler", "luigi.worker"]:
        _logging.getLogger(name).setLevel(_logging.CRITICAL)

    try:
        with util.log_timing(logger, "pipeline execution"):
            success = luigi.build(
                tasks, local_scheduler=True, workers=workers, log_level="CRITICAL"
            )
    except Exception as e:
        _handle_pipeline_error(logger, e, config)
        raise SystemExit(1)

    # Show file processing summary
    summary_lines = file_tracker.format_summary()
    for line in summary_lines:
        logger.info(line)

    # Show output file statistics
    _log_output_summary(logger, config)

    has_failures = len(file_tracker.get_summary()[0]["failed"]) > 0
    if not has_failures:
        for line in BANNER_SUCCESS.strip().split("\n"):
            logger.info(line)
    else:
        for line in BANNER_FAILURE.strip().split("\n"):
            logger.error(line)
        logger.error("")
        logger.error("The failed files listed above may have corrupted data.")
        logger.error("Try re-exporting them from your microscope software.")


def _log_output_summary(logger, config: dict) -> None:
    """Log summary of output files with row counts and sizes."""
    import pandas as pd

    output_path = config.get("output_path", "")
    summary_file = os.path.join(output_path, "summary.csv")
    cells_file = os.path.join(output_path, "summary_cells.csv")

    logger.info("")
    logger.info("-" * 50)
    logger.info("  OUTPUT")
    logger.info("-" * 50)

    for label, path in [("Spots", summary_file), ("Cells", cells_file)]:
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"  {label}: {len(df):,} rows ({size_mb:.1f} MB)")
                logger.info(f"    {path}")
            except Exception:
                logger.info(f"  {label}: {path}")
        else:
            logger.info(f"  {label}: not generated")

    log_path = os.path.join(output_path, "koopa.log")
    logger.info(f"  Log: {log_path}")
    logger.info("")


def _handle_pipeline_error(logger, error: Exception, config: dict) -> None:
    """Translate common exceptions into actionable user guidance."""
    msg = str(error)

    logger.error("")
    logger.error("=" * 50)
    logger.error("  PIPELINE ERROR")
    logger.error("=" * 50)

    if any(kw in msg for kw in ["trainable", "SpatialDropout2D", "Functional", "keras"]):
        logger.error("  Model incompatible with current Keras/TensorFlow version.")
        logger.error("")
        logger.error("  Try:")
        logger.error("    1. ./run_legacy.sh --config <cfg>   (for older models)")
        logger.error("    2. ./run_modern.sh --config <cfg>   (for newer models)")
        logger.error("    3. Add --skip-incompatible to skip problematic models")
    elif "No such file or directory" in msg or "FileNotFoundError" in type(error).__name__:
        logger.error(f"  File not found: {msg}")
        logger.error("")
        logger.error("  Check:")
        logger.error("    - Is the path absolute (starts with /)?")
        logger.error("    - Do you have read permissions?")
        logger.error("    - If on a network drive, is it mounted?")
    elif "out of memory" in msg.lower() or "OOM" in msg or "CUDA" in msg:
        logger.error("  GPU out of memory.")
        logger.error("")
        logger.error("  Try:")
        logger.error("    - Reduce --workers to 1")
        logger.error("    - Enable binning in config to reduce image size")
        logger.error("    - Use CPU mode instead of --gpu")
    elif "empty" in msg.lower() and "segmentation" in msg.lower():
        logger.error("  Segmentation produced empty masks.")
        logger.error("")
        logger.error("  Check:")
        logger.error("    - Is the channel index correct? (0-indexed)")
        logger.error("    - Try adjusting upper_clip (e.g., 0.99)")
        logger.error("    - Try reducing min_size_nuclei / min_size_cyto")
    else:
        logger.error(f"  {type(error).__name__}: {msg[:300]}")
        logger.error("")
        logger.error("  Check koopa.log for full traceback.")

    logger.error("")
    log_path = os.path.join(config.get("output_path", "."), "koopa.log")
    logger.error(f"  Full log: {log_path}")
    logger.error("=" * 50)
