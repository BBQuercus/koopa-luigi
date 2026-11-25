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


try:
    from importlib.metadata import version

    __version__ = version("koopa-luigi")
except Exception:
    __version__ = "0.1.0-dev"

# Don't import anything else at package level - let modules be imported on demand
# This avoids loading cellpose/tensorflow at import time
__all__ = ["cli", "cli_args", "postprocess", "preprocess", "spots", "util"]