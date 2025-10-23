import os
import sys

import deepblink as pink
import koopa.colocalize
import koopa.detect
import koopa.io
import koopa.track
import luigi

from .util import LuigiFileTask
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
        skip_incompatible = "--skip-incompatible" in sys.argv

        image = koopa.io.load_image(self.input().path)
        model_path = self.config["detect_models"][self.index_list]

        try:
            model = pink.io.load_model(model_path)
        except (TypeError, KeyError, Exception) as e:
            error_msg = str(e)
            self.logger.warning(
                f"Failed to load model for channel {self.index_channel}: {error_msg[:100]}"
            )

            if skip_incompatible:
                self.logger.info(
                    f"Skipping incompatible model for channel {self.index_channel} (--skip-incompatible enabled)"
                )
                # Create empty DataFrame with expected columns
                import pandas as pd

                df = pd.DataFrame(columns=["x", "y", "z", "probability"])
                df.insert(loc=0, column="FileID", value=self.FileID)
                koopa.io.save_parquet(self.output().path, df)
                self.logger.info(
                    f"Created empty results file for channel {self.index_channel}"
                )
                return
            else:
                # Check if it's a known compatibility issue
                if "trainable" in error_msg or "SpatialDropout2D" in error_msg:
                    self.logger.error(
                        f"Model incompatibility detected for channel {self.index_channel}"
                    )
                    self.logger.error(
                        "This model was created with an older Keras version"
                    )
                    self.logger.error("Solutions:")
                    self.logger.error(
                        "  1. Run with --skip-incompatible flag to skip this model"
                    )
                    self.logger.error(
                        "  2. Use a virtual environment with TensorFlow 2.13 and Keras 2.x"
                    )
                    self.logger.error(
                        "  3. Update/retrain the model with current TensorFlow/Keras versions"
                    )
                raise

        df = koopa.detect.detect_image(
            image,
            self.index_channel,
            model,
            self.config["refinement_radius"],
        )
        df.insert(loc=0, column="FileID", value=self.FileID)
        koopa.io.save_parquet(self.output().path, df)


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
        df = koopa.io.load_parquet(self.input().path)
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


class ColocalizeFrame(LuigiFileTask):
    """Task to colocalize spots between frames."""

    index_reference = luigi.IntParameter()
    index_transform = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"{self.index_reference}-{self.index_transform}"

    def requires(self):
        if self.config["do_3d"]:
            return [
                Track(FileID=self.FileID, index_list=self.index_reference),
                Track(FileID=self.FileID, index_list=self.index_transform),
            ]
        return [
            Detect(FileID=self.FileID, index_list=self.index_reference),
            Detect(FileID=self.FileID, index_list=self.index_transform),
        ]

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"colocalization_{self.name}",
            f"{self.FileID}.parq",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        self.logger.info(f"Colocalizing {self.index_reference}<-{self.index_transform}")

        df_reference = koopa.io.load_parquet(self.input()[0].path)
        df_transform = koopa.io.load_parquet(self.input()[1].path)
        df = koopa.colocalize.colocalize_frames(
            df_reference,
            df_transform,
            self.name,
            self.config["z_distance"] if self.config["do_3d"] else 1,
            self.config["distance_cutoff"],
        )
        koopa.io.save_parquet(self.output().path, df)


class ColocalizeTrack(LuigiFileTask):
    """Task to colocalize spots between tracks."""

    index_reference = luigi.IntParameter()
    index_transform = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"{self.index_reference}-{self.index_transform}"

    def requires(self):
        return [
            Track(FileID=self.FileID, index_list=self.index_reference),
            Track(FileID=self.FileID, index_list=self.index_transform),
        ]

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"colocalization_{self.name}",
            f"{self.FileID}.parq",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        self.logger.info(f"Colocalizing {self.index_reference}<-{self.index_transform}")

        df_reference = koopa.io.load_parquet(self.input()[0].path)
        df_transform = koopa.io.load_parquet(self.input()[1].path)
        df = koopa.colocalize.colocalize_tracks(
            df_reference,
            df_transform,
            self.name,
            self.config["min_frames"],
            self.config["distance_cutoff"],
        )
        koopa.io.save_parquet(self.output().path, df)
