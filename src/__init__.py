"""Koopa-luigi."""

# Suppress noisy output BEFORE any other imports
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CELLPOSE_QUIET"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")


def _suppress_cellpose_banner():
    """Suppress cellpose banner by redirecting fd during import.

    This function is called lazily when cellpose might be imported.
    It sets up a context that can be used to suppress output.
    """
    pass  # Placeholder - actual suppression happens in util.suppress_stdout()


__version__ = "0.0.1"

# Don't import anything else at package level - let modules be imported on demand
# This avoids loading cellpose/tensorflow at import time
__all__ = ["cli", "cli_args", "postprocess", "preprocess", "spots", "util"]