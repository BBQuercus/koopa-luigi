"""Entrypoint module for the application."""

if __name__ == "__main__":
    # Suppress noisy startup output (cellpose banner, etc.)
    # This MUST happen before ANY package imports
    import os
    import sys
    import io
    import warnings

    # Set environment variables before any imports
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["CELLPOSE_QUIET"] = "1"
    os.environ["KERAS_BACKEND"] = os.environ.get("KERAS_BACKEND", "tensorflow")

    # Filter warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Suppress stdout/stderr using BOTH Python-level AND file descriptor level
    # This catches both Python prints and C-level output (cellpose banner)
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
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        from .cli import main
    finally:
        sys.stdout = _saved_stdout
        sys.stderr = _saved_stderr
        os.dup2(_saved_stdout_fd, 1)
        os.dup2(_saved_stderr_fd, 2)
        os.close(_devnull_fd)
        os.close(_saved_stdout_fd)
        os.close(_saved_stderr_fd)

    main()
