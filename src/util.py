import logging
import os
import sys
import platform

import koopa.config
import koopa.io
import luigi


def set_logging():
    """Prepare verbose logging for luigi and co."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s,%(msecs)d %(levelname)s (%(name)s) - %(message)s",
        datefmt="%H:%M:%S",
    )


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
    for key, value in config.items():
        luigi_config.set("core", key, str(value))
    

    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.join(
        "/tungstenfs/scratch/gchao/.cellpose"
    )


def get_configuration():
    config = {}
    for key, value in luigi.configuration.get_config()["core"].items():
        try:
            eval_value = eval(value)
        except (NameError, SyntaxError):
            eval_value = value
        config[key] = eval_value
    return config


class LuigiFileTask(luigi.Task):
    FileID = luigi.Parameter()
    gpu = luigi.BoolParameter(default=False, significant=False)
    logger = logging.getLogger("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_configuration()
        self.config["gpu"] = self.gpu


def get_environment_info():
    """Get detailed environment information."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "virtual_env": None,
        "packages": {}
    }
    
    # Detect virtual environment type
    if os.environ.get('VIRTUAL_ENV'):
        if 'uv' in os.environ.get('VIRTUAL_ENV', ''):
            info["virtual_env"] = "uv"
        else:
            info["virtual_env"] = "venv"
    elif os.environ.get('CONDA_DEFAULT_ENV'):
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
    
    for package_name, import_name in packages_to_check:
        try:
            if import_name:
                module = __import__(import_name)
            else:
                module = __import__(package_name)
            
            if hasattr(module, '__version__'):
                info["packages"][package_name] = module.__version__
            else:
                info["packages"][package_name] = "installed (version unknown)"
        except ImportError:
            info["packages"][package_name] = None
    
    # Special handling for TensorFlow/Keras version detection
    try:
        import tensorflow as tf
        info["packages"]["tensorflow"] = tf.__version__
        if hasattr(tf, 'keras'):
            info["packages"]["keras"] = tf.keras.__version__ if hasattr(tf.keras, '__version__') else "integrated with TF"
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
    logger = logging.getLogger("koopa")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_configuration()
