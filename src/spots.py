import gc
import os
import sys
import warnings
import logging

# Suppress noisy output before importing ML libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CELLPOSE_QUIET"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress absl logging
logging.getLogger("absl").setLevel(logging.CRITICAL)

# Suppress Keras progress bars
try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    if hasattr(tf.keras.utils, "disable_interactive_logging"):
        tf.keras.utils.disable_interactive_logging()
except (ImportError, AttributeError):
    pass

import deepblink as pink
import koopa.colocalize
import koopa.detect
import koopa.io
import koopa.track
import luigi

from .util import LuigiFileTask, log_timing
from .preprocess import Preprocess


class Detect(LuigiFileTask):
    """Task to run deepblink for spot detection."""

    index_list = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_channel = self.config["detect_channels"][self.index_list]

    def requires(self):
        return Preprocess(self.FileID)

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"detection_raw_c{self.index_channel}",
            f"{self.FileID}.parq",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        import pandas as pd

        model_path = self.config["detect_models"][self.index_list]
        model_name = os.path.basename(model_path)

        self.logger.info(
            f"[{self.FileID}] Detecting spots in channel {self.index_channel}"
        )

        # Load image
        with log_timing(self.logger, "image loading", self.FileID):
            image = koopa.io.load_image(self.input().path)

        # Load model
        try:
            with log_timing(self.logger, f"model loading ({model_name})", self.FileID):
                model = pink.io.load_model(model_path)
        except (TypeError, KeyError, Exception) as e:
            error_msg = str(e)

            # Determine if it's a compatibility issue
            is_compatibility_issue = any(
                kw in error_msg
                for kw in ["trainable", "SpatialDropout2D", "Functional", "keras"]
            )

            if self.skip_incompatible:
                self.logger.warning(
                    f"[{self.FileID}] Skipping channel {self.index_channel}: "
                    f"model '{model_name}' incompatible with current environment"
                )
                df = pd.DataFrame(columns=["x", "y", "z", "probability"])
                df.insert(loc=0, column="FileID", value=self.FileID)
                koopa.io.save_parquet(self.output().path, df)
                return

            # Detailed error message
            self.logger.error("=" * 60)
            self.logger.error(f"  Model Load Failed: {model_name}")
            self.logger.error("=" * 60)
            self.logger.error(f"File: {self.FileID}")
            self.logger.error(f"Channel: {self.index_channel}")
            self.logger.error(f"Model path: {model_path}")
            self.logger.error(f"Error: {error_msg[:200]}")

            if is_compatibility_issue:
                self.logger.error("")
                self.logger.error(
                    "This appears to be a Keras/TensorFlow version mismatch."
                )
                self.logger.error("The model was likely created with an older version.")
                self.logger.error("")
                self.logger.error("Solutions:")
                self.logger.error("  1. Add --skip-incompatible to skip this model")
                self.logger.error(
                    "  2. Use ./run_legacy.sh for TensorFlow 2.13 environment"
                )
                self.logger.error("  3. Retrain the model with your current TF version")
            self.logger.error("=" * 60)
            raise

        # Run detection
        with log_timing(self.logger, "spot detection", self.FileID):
            df = koopa.detect.detect_image(
                image,
                self.index_channel,
                model,
                self.config["refinement_radius"],
            )

        df.insert(loc=0, column="FileID", value=self.FileID)
        koopa.io.save_parquet(self.output().path, df)

        self.logger.info(
            f"[{self.FileID}] Channel {self.index_channel}: detected {len(df)} spots"
        )
        del image, model
        gc.collect()


class Track(LuigiFileTask):
    """Task to track spots in 3D/2D+T."""

    index_list = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_channel = self.config["detect_channels"][self.index_list]

    def requires(self):
        return Detect(FileID=self.FileID, index_list=self.index_list)

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"detection_final_c{self.index_channel}",
            f"{self.FileID}.parq",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        self.logger.info(
            f"[{self.FileID}] Tracking spots in channel {self.index_channel}"
        )

        with log_timing(self.logger, "tracking", self.FileID):
            df = koopa.io.load_parquet(self.input().path)

            if len(df) == 0:
                self.logger.warning(
                    f"[{self.FileID}] No spots to track in channel {self.index_channel}"
                )
                koopa.io.save_parquet(self.output().path, df)
                return

            track = koopa.track.track(
                df,
                self.config["search_range"],
                self.config["gap_frames"],
                self.config["min_length"],
            )
            if self.config["do_3d"]:
                track = koopa.track.link_brightest_particles(df, track)
            if self.config["subtract_drift"]:
                track = koopa.track.subtract_drift(track)
            track = koopa.track.clean_particles(track)

        koopa.io.save_parquet(self.output().path, track)

        n_tracks = track["particle"].nunique() if "particle" in track.columns else 0
        self.logger.info(
            f"[{self.FileID}] Channel {self.index_channel}: {n_tracks} tracks from {len(df)} spots"
        )


class ColocalizeFrame(LuigiFileTask):
    """Task to colocalize spots between frames."""

    index_reference = luigi.IntParameter()
    index_transform = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # index_reference/index_transform are image channel indices from coloc_channels
        self.channel_reference = self.index_reference
        self.channel_transform = self.index_transform
        self.name = f"{self.channel_reference}-{self.channel_transform}"
        # Convert to indices into detect_channels for Detect/Track tasks
        detect_channels = self.config["detect_channels"]
        self._idx_reference = detect_channels.index(self.channel_reference)
        self._idx_transform = detect_channels.index(self.channel_transform)

    def requires(self):
        skip = self.skip_incompatible
        if self.config["do_3d"]:
            return [
                Track(
                    FileID=self.FileID,
                    index_list=self._idx_reference,
                    skip_incompatible=skip,
                ),
                Track(
                    FileID=self.FileID,
                    index_list=self._idx_transform,
                    skip_incompatible=skip,
                ),
            ]
        return [
            Detect(
                FileID=self.FileID,
                index_list=self._idx_reference,
                skip_incompatible=skip,
            ),
            Detect(
                FileID=self.FileID,
                index_list=self._idx_transform,
                skip_incompatible=skip,
            ),
        ]

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"colocalization_{self.name}",
            f"{self.FileID}.parq",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        self.logger.info(
            f"[{self.FileID}] Colocalizing channels {self.channel_reference} ↔ {self.channel_transform}"
        )

        with log_timing(self.logger, "colocalization", self.FileID):
            df_reference = koopa.io.load_parquet(self.input()[0].path)
            df_transform = koopa.io.load_parquet(self.input()[1].path)

            if len(df_reference) == 0 or len(df_transform) == 0:
                self.logger.warning(
                    f"[{self.FileID}] Empty input for colocalization "
                    f"(ref: {len(df_reference)}, transform: {len(df_transform)} spots)"
                )

            df = koopa.colocalize.colocalize_frames(
                df_reference,
                df_transform,
                self.name,
                self.config["z_distance"] if self.config["do_3d"] else 1,
                self.config["distance_cutoff"],
            )

        koopa.io.save_parquet(self.output().path, df)
        self.logger.info(
            f"[{self.FileID}] Colocalization {self.name}: {len(df)} pairs found"
        )


class ColocalizeTrack(LuigiFileTask):
    """Task to colocalize spots between tracks."""

    index_reference = luigi.IntParameter()
    index_transform = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # index_reference/index_transform are image channel indices from coloc_channels
        self.channel_reference = self.index_reference
        self.channel_transform = self.index_transform
        self.name = f"{self.channel_reference}-{self.channel_transform}"
        # Convert to indices into detect_channels for Track tasks
        detect_channels = self.config["detect_channels"]
        self._idx_reference = detect_channels.index(self.channel_reference)
        self._idx_transform = detect_channels.index(self.channel_transform)

    def requires(self):
        return [
            Track(
                FileID=self.FileID,
                index_list=self._idx_reference,
                skip_incompatible=self.skip_incompatible,
            ),
            Track(
                FileID=self.FileID,
                index_list=self._idx_transform,
                skip_incompatible=self.skip_incompatible,
            ),
        ]

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"colocalization_{self.name}",
            f"{self.FileID}.parq",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        self.logger.info(
            f"[{self.FileID}] Colocalizing tracks {self.channel_reference} ↔ {self.channel_transform}"
        )

        with log_timing(self.logger, "track colocalization", self.FileID):
            df_reference = koopa.io.load_parquet(self.input()[0].path)
            df_transform = koopa.io.load_parquet(self.input()[1].path)

            if len(df_reference) == 0 or len(df_transform) == 0:
                self.logger.warning(
                    f"[{self.FileID}] Empty input for track colocalization "
                    f"(ref: {len(df_reference)}, transform: {len(df_transform)} tracks)"
                )

            df = koopa.colocalize.colocalize_tracks(
                df_reference,
                df_transform,
                self.name,
                self.config["min_frames"],
                self.config["distance_cutoff"],
            )

        koopa.io.save_parquet(self.output().path, df)
        self.logger.info(
            f"[{self.FileID}] Track colocalization {self.name}: {len(df)} pairs found"
        )
