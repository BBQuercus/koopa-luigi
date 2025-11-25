import logging
import os
import sys
import platform
import time
import warnings
from contextlib import contextmanager
from functools import wraps

# Suppress noisy warnings before importing libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*UnconsumedParameterWarning.*")
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")

# Suppress TensorFlow/Keras logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL only
os.environ["CELLPOSE_QUIET"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress stdout/stderr during koopa import (hides cellpose banner)
# Use BOTH Python-level and file descriptor level redirection
import io as _io

sys.stdout.flush()
sys.stderr.flush()
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
_saved_stdout_fd = os.dup(1)
_saved_stderr_fd = os.dup(2)
_saved_stdout = sys.stdout
_saved_stderr = sys.stderr
try:
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)
    sys.stdout = _io.StringIO()
    sys.stderr = _io.StringIO()
    import koopa.config
    import koopa.io
finally:
    sys.stdout = _saved_stdout
    sys.stderr = _saved_stderr
    os.dup2(_saved_stdout_fd, 1)
    os.dup2(_saved_stderr_fd, 2)
    os.close(_devnull_fd)
    os.close(_saved_stdout_fd)
    os.close(_saved_stderr_fd)

import luigi

# Suppress Luigi's UnconsumedParameterWarning
from luigi.task import UnconsumedParameterWarning

warnings.filterwarnings("ignore", category=UnconsumedParameterWarning)

# Consistent logger name for the entire package
LOGGER_NAME = "koopa"


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger with consistent naming.

    Args:
        name: Optional suffix for the logger (e.g., 'detect', 'segment')
              If None, returns the root koopa logger.

    Returns:
        Logger instance with name 'koopa' or 'koopa.{name}'
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def set_logging(output_path: str = None, verbose: bool = False, append: bool = False):
    """Configure logging with console and file output.

    Args:
        output_path: Directory to write koopa.log file. If None, file logging disabled.
        verbose: If True, use DEBUG level. If False (default), use INFO level.
        append: If True, append to existing log file (for worker processes).

    Console shows only koopa logs at INFO (or DEBUG if verbose).
    File captures ALL logs from ALL loggers at DEBUG level.
    """
    # Determine console log level
    console_level = logging.DEBUG if verbose else logging.INFO

    # === ROOT LOGGER: captures everything to file ===
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    root_logger.handlers.clear()

    # File handler on root logger - captures ALL logs from ALL loggers
    if output_path:
        log_file = os.path.join(output_path, "koopa.log")
        file_mode = "a" if append else "w"
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # === KOOPA LOGGER: console output for user ===
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = True  # Let logs go to root's file handler too

    # Console handler - only for koopa logger, clean output for terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # === THIRD-PARTY LOGGERS: suppress console, but allow file logging ===
    # These loggers will propagate to root (file) but not print to console
    noisy_loggers = [
        "luigi",
        "luigi-interface",
        "luigi.scheduler",
        "luigi.worker",
        "luigi.task",
        "tensorflow",
        "tf2onnx",
        "absl",
        "cellpose",
        "cellpose.core",
        "cellpose.models",
        "cellpose.io",
        "numba",
        "h5py",
        "PIL",
        "matplotlib",
        "urllib3",
        "keras",
    ]
    for logger_name in noisy_loggers:
        noisy = logging.getLogger(logger_name)
        noisy.handlers.clear()  # Remove any console handlers
        noisy.propagate = True  # Allow propagation to root for file logging
        # Don't set level - let them log at their default levels to file


@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout/stderr (for silencing noisy library imports).

    Uses BOTH Python-level (sys.stdout) and file descriptor level redirection
    to catch all output including C-level prints and cellpose banner.
    """
    import io

    # Flush before redirecting
    sys.stdout.flush()
    sys.stderr.flush()

    # Save originals
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr

    try:
        # Redirect at both levels
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore at both levels
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


@contextmanager
def log_timing(logger: logging.Logger, operation: str, file_id: str = None):
    """Context manager to log operation timing.

    Args:
        logger: Logger instance to use
        operation: Description of the operation being timed
        file_id: Optional file identifier for context

    Usage:
        with log_timing(self.logger, "spot detection", "file_001"):
            # do work
    """
    context = f"[{file_id}] " if file_id else ""
    logger.debug(f"{context}Starting {operation}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        logger.info(f"{context}{operation.capitalize()} completed in {time_str}")


def set_configuration(path: os.PathLike):
    # Parse configuration
    cfg = koopa.io.load_config(path)
    koopa.config.validate_config(cfg)
    config = koopa.config.flatten_config(cfg)

    # Save config
    cfg = koopa.config.add_versioning(cfg)
    fname_config = os.path.join(config["output_path"], "koopa.cfg")
    koopa.io.save_config(fname_config, cfg)

    # Set luigi config
    luigi_config = luigi.configuration.get_config()

    # Ensure core section exists
    if not luigi_config.has_section("core"):
        luigi_config.add_section("core")

    for key, value in config.items():
        luigi_config.set("core", key, str(value))

    # Store config path in environment variable so worker processes can reload it
    os.environ["KOOPA_CONFIG_PATH"] = str(path)

    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.join(
        "/tungstenfs/scratch/gchao/.cellpose"
    )


def get_configuration():
    config = {}
    luigi_config = luigi.configuration.get_config()

    # Check if core section exists
    if "core" not in luigi_config:
        # Worker process: try to reload config from environment variable
        config_path = os.environ.get("KOOPA_CONFIG_PATH")
        if config_path and os.path.exists(config_path):
            # Reload configuration in worker process
            cfg = koopa.io.load_config(config_path)
            koopa.config.validate_config(cfg)
            config_dict = koopa.config.flatten_config(cfg)

            # Set luigi config in this worker
            if not luigi_config.has_section("core"):
                luigi_config.add_section("core")

            for key, value in config_dict.items():
                luigi_config.set("core", key, str(value))
        else:
            raise RuntimeError(
                "Luigi configuration not initialized. "
                "set_configuration() must be called before creating tasks, "
                "or KOOPA_CONFIG_PATH environment variable must be set."
            )

    for key, value in luigi_config["core"].items():
        try:
            eval_value = eval(value)
        except (NameError, SyntaxError):
            eval_value = value
        config[key] = eval_value
    return config


class LuigiFileTask(luigi.Task):
    FileID = luigi.Parameter()
    gpu = luigi.BoolParameter(default=False, significant=False)
    skip_incompatible = luigi.BoolParameter(default=False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_configuration()
        self.config["gpu"] = self.gpu
        # Use consistent logger naming based on task class name
        task_name = self.__class__.__name__.lower()
        self.logger = get_logger(task_name)
        # Ensure logging is configured for worker processes
        self._setup_worker_logging()

    def _setup_worker_logging(self):
        """Configure logging for worker processes."""
        logger = logging.getLogger(LOGGER_NAME)
        if not logger.handlers:
            # Worker process doesn't have logging configured
            # Use append=True so workers don't overwrite main process logs
            set_logging(output_path=self.config.get("output_path"), verbose=False, append=True)


def get_environment_info():
    """Get detailed environment information."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "virtual_env": None,
        "packages": {},
    }

    # Detect virtual environment type
    if os.environ.get("VIRTUAL_ENV"):
        if "uv" in os.environ.get("VIRTUAL_ENV", ""):
            info["virtual_env"] = "uv"
        else:
            info["virtual_env"] = "venv"
    elif os.environ.get("CONDA_DEFAULT_ENV"):
        info["virtual_env"] = f"conda ({os.environ.get('CONDA_DEFAULT_ENV')})"

    # Check for ML packages and dependencies
    packages_to_check = [
        ("tensorflow", "tf"),
        ("keras", None),
        ("torch", None),
        ("cellpose", None),
        ("deepblink", "pink"),
        ("segmentation_models", "sm"),
        # Image reading/processing
        ("czifile", None),
        ("nd2reader", None),
        ("tifffile", None),
        ("bioformats", None),
        ("aicsimageio", None),
        # Data processing
        ("numpy", "np"),
        ("pandas", "pd"),
        ("scikit-image", "skimage"),
        ("scipy", None),
        ("h5py", None),
        ("pyarrow", None),
        # Koopa dependencies
        ("trackpy", None),
        ("pystackreg", None),
        ("numba", None),
    ]

    # Suppress noisy imports (like cellpose banner) during version checking
    with suppress_stdout():
        for package_name, import_name in packages_to_check:
            try:
                if import_name:
                    module = __import__(import_name)
                else:
                    module = __import__(package_name)

                if hasattr(module, "__version__"):
                    info["packages"][package_name] = module.__version__
                else:
                    info["packages"][package_name] = "installed (version unknown)"
            except ImportError:
                info["packages"][package_name] = None

        # Special handling for TensorFlow/Keras version detection
        try:
            import tensorflow as tf

            info["packages"]["tensorflow"] = tf.__version__
            if hasattr(tf, "keras"):
                info["packages"]["keras"] = (
                    tf.keras.__version__
                    if hasattr(tf.keras, "__version__")
                    else "integrated with TF"
                )
        except ImportError:
            pass

    return info


def format_environment_info(info=None):
    """Format environment information for display."""
    if info is None:
        info = get_environment_info()

    lines = [
        "=" * 50,
        "Koopa-Luigi Environment Information",
        "=" * 50,
        f"Python: {info['python_version']}",
        f"Platform: {info['platform']}",
        f"Virtual Environment: {info['virtual_env'] or 'None detected'}",
        "",
        "Installed Packages:",
    ]

    for package, version in info["packages"].items():
        if version:
            lines.append(f"  {package}: {version}")
        else:
            lines.append(f"  {package}: not installed")

    lines.append("=" * 50)
    return "\n".join(lines)


class LuigiTask(luigi.Task):
    skip_incompatible = luigi.BoolParameter(default=False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_configuration()
        # Use consistent logger naming based on task class name
        task_name = self.__class__.__name__.lower()
        self.logger = get_logger(task_name)
