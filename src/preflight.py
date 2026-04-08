"""Preflight checks that run before pipeline execution."""

import os
import shutil
from typing import Any

from .util import get_logger

logger = get_logger("preflight")


def run_preflight(config: dict[str, Any]) -> None:
    """Validate configuration before running the pipeline.

    Raises SystemExit on critical errors. Logs warnings for non-critical issues.
    """
    errors: list[str] = []
    warnings: list[str] = []

    _check_paths(config, errors)
    _check_detection(config, errors)
    _check_colocalization(config, errors, warnings)
    _check_segmentation(config, errors, warnings)
    _check_input_files(config, errors, warnings)
    _check_disk_space(config, warnings)

    # Report warnings
    for w in warnings:
        logger.warning(f"  {w}")

    # Abort on errors
    if errors:
        logger.error("")
        logger.error("=" * 50)
        logger.error("  PREFLIGHT CHECK FAILED")
        logger.error("=" * 50)
        for e in errors:
            logger.error(f"  - {e}")
        logger.error("")
        logger.error("Fix the issues above and try again.")
        raise SystemExit(1)

    logger.info("Preflight checks passed")


def _check_paths(config: dict[str, Any], errors: list[str]) -> None:
    """Verify required paths exist and are accessible."""
    input_path = config.get("input_path", "")
    if not os.path.isdir(input_path):
        errors.append(f"Input path does not exist: {input_path}")

    output_path = config.get("output_path", "")
    if output_path:
        parent = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(parent):
            errors.append(f"Output path parent does not exist: {parent}")

    if config.get("alignment_enabled"):
        alignment_path = config.get("alignment_path", "")
        if not os.path.isdir(alignment_path):
            errors.append(f"Alignment path does not exist: {alignment_path}")
        if config.get("alignment_method") == "deepblink":
            model = config.get("alignment_model", "")
            if not os.path.isfile(model):
                errors.append(f"Alignment model not found: {model}")


def _check_detection(config: dict[str, Any], errors: list[str]) -> None:
    """Verify detection channels and models match."""
    channels = config.get("detect_channels", [])
    models = config.get("detect_models", [])

    if not channels:
        return

    if len(channels) != len(models):
        errors.append(
            f"detect_channels has {len(channels)} entries but detect_models has {len(models)} "
            f"— they must match 1:1"
        )
        return

    for i, model_path in enumerate(models):
        if not model_path:
            continue
        if not os.path.isfile(model_path):
            errors.append(
                f"Detection model for channel {channels[i]} not found: {model_path}"
            )


def _check_colocalization(
    config: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    """Verify colocalization channels are valid."""
    if not config.get("coloc_enabled"):
        return

    detect_channels = set(config.get("detect_channels", []))
    coloc_channels = config.get("coloc_channels", [])

    for pair in coloc_channels:
        if not pair or len(pair) != 2:
            continue
        ref, transform = pair
        if ref not in detect_channels:
            errors.append(
                f"Colocalization reference channel {ref} is not in detect_channels {sorted(detect_channels)}"
            )
        if transform not in detect_channels:
            errors.append(
                f"Colocalization transform channel {transform} is not in detect_channels {sorted(detect_channels)}"
            )

    if config.get("do_3d") and config.get("gap_frames", 0) > 0:
        warnings.append(
            "gap_frames > 0 with do_3d=True — gap_frames should usually be 0 for 3D stacks"
        )


def _check_segmentation(
    config: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    """Verify segmentation configuration."""
    method = config.get("method_nuclei", "")
    if method == "cellpose":
        cellpose_models = config.get("cellpose_models", [])
        for model_path in cellpose_models:
            if model_path and not os.path.isfile(model_path):
                errors.append(f"Cellpose model not found: {model_path}")

    if config.get("sego_enabled"):
        channels = config.get("sego_channels", [])
        methods = config.get("sego_methods", [])
        if len(channels) != len(methods):
            errors.append(
                f"sego_channels has {len(channels)} entries but sego_methods has {len(methods)} "
                f"— they must match 1:1"
            )


def _check_input_files(
    config: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    """Check that input directory has matching files."""
    input_path = config.get("input_path", "")
    file_ext = config.get("file_ext", "")

    if not os.path.isdir(input_path) or not file_ext:
        return  # Already caught by _check_paths

    try:
        import koopa.util

        files = koopa.util.get_file_list(input_path, file_ext)
        if not files:
            errors.append(
                f"No .{file_ext} files found in {input_path}"
            )
        elif len(files) > 500:
            warnings.append(
                f"Found {len(files)} files — large batch, consider testing with a subset first"
            )
    except Exception:
        # Don't fail preflight if file listing fails — pipeline will catch it
        pass


def _check_disk_space(config: dict[str, Any], warnings: list[str]) -> None:
    """Warn if output directory has low disk space."""
    output_path = config.get("output_path", "")
    check_path = output_path if os.path.isdir(output_path) else os.path.dirname(os.path.abspath(output_path))

    if not os.path.isdir(check_path):
        return

    try:
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024**3)
        if free_gb < 5:
            warnings.append(f"Low disk space on output volume: {free_gb:.1f} GB free")
    except OSError:
        pass
