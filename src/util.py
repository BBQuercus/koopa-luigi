import logging
import os
import sys

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


class LuigiTask(luigi.Task):
    logger = logging.getLogger("koopa")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_configuration()
