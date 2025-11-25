import os
import warnings

# Suppress noisy output before importing ML libraries
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CELLPOSE_QUIET", "1")
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")

import koopa.io
import luigi

from .util import LuigiFileTask, log_timing, suppress_stdout
from .preprocess import Preprocess


class SegmentCellsSingle(LuigiFileTask):
    """Task for segmentation of either nuclei or cyto."""

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"segmentation_{self.config['selection']}",
            f"{self.FileID}.tif",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        with suppress_stdout():
            import koopa.segment_cells

        selection = self.config["selection"]
        channel = self.config[f"channel_{selection}"]
        self.logger.info(f"[{self.FileID}] Segmenting {selection} (channel {channel})")

        with log_timing(self.logger, f"{selection} segmentation", self.FileID):
            image = koopa.io.load_image(self.input().path)
            image = image[channel]

            if not self.config["do_3d"] and self.config["do_timeseries"]:
                image = koopa.segment_cells.preprocess(image)

            if selection == "nuclei":
                segmap = koopa.segment_cells.segment_nuclei(image, self.config)
            else:
                segmap = koopa.segment_cells.segment_cyto(image, self.config)

        koopa.io.save_image(self.output().path, segmap)

        n_objects = segmap.max()
        self.logger.info(
            f"[{self.FileID}] {selection.capitalize()} segmentation: {n_objects} objects found"
        )


class SegmentCellsBoth(LuigiFileTask):
    """Task for segmentation of both nuclei and cyto."""

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        fname_nuclei = os.path.join(
            self.config["output_path"], "segmentation_nuclei", f"{self.FileID}.tif"
        )
        fname_cyto = os.path.join(
            self.config["output_path"], "segmentation_cyto", f"{self.FileID}.tif"
        )
        return luigi.LocalTarget(fname_nuclei), luigi.LocalTarget(fname_cyto)

    def run(self):
        with suppress_stdout():
            import koopa.segment_cells

        ch_nuclei = self.config["channel_nuclei"]
        ch_cyto = self.config["channel_cyto"]
        self.logger.info(
            f"[{self.FileID}] Segmenting nuclei (ch{ch_nuclei}) and cyto (ch{ch_cyto})"
        )

        with log_timing(self.logger, "dual segmentation", self.FileID):
            image = koopa.io.load_image(self.input().path)
            image_nuclei = image[ch_nuclei]
            image_cyto = image[ch_cyto]

            if not self.config["do_3d"] and self.config["do_timeseries"]:
                image_nuclei = koopa.segment_cells.preprocess(image_nuclei)
                image_cyto = koopa.segment_cells.preprocess(image_cyto)

            segmap_nuclei, segmap_cyto = koopa.segment_cells.segment_both(
                image_nuclei, image_cyto, self.config
            )

        koopa.io.save_image(self.output()[0].path, segmap_nuclei)
        koopa.io.save_image(self.output()[1].path, segmap_cyto)

        self.logger.info(
            f"[{self.FileID}] Dual segmentation: {segmap_nuclei.max()} nuclei, {segmap_cyto.max()} cyto objects"
        )


class SegmentCellsPredict(LuigiFileTask):
    """Task for cellpose GPU-based nuclei prediction."""

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            "segmentation_nuclei_prediction",
            f"{self.FileID}.tif",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        with suppress_stdout():
            import koopa.segment_flies

        channel = self.config["brains_channel"]
        self.logger.info(
            f"[{self.FileID}] Cellpose prediction (channel {channel}, GPU)"
        )

        with log_timing(self.logger, "cellpose prediction", self.FileID):
            image = koopa.io.load_image(self.input().path)
            image = koopa.segment_flies.normalize_nucleus(image[channel])
            segmap = koopa.segment_flies.cellpose_predict(
                image, self.config["batch_size"]
            )

        koopa.io.save_image(self.output().path, segmap)
        self.logger.info(f"[{self.FileID}] Cellpose prediction complete")


class SegmentCellsMerge(LuigiFileTask):
    """Task to combine cellpose predictions to segmaps."""

    def requires(self):
        return [Preprocess(FileID=self.FileID), SegmentCellsPredict(FileID=self.FileID)]

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            "segmentation_nuclei_merge",
            f"{self.FileID}.tif",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        with suppress_stdout():
            import koopa.segment_flies

        self.logger.info(f"[{self.FileID}] Merging cellpose predictions")

        with log_timing(self.logger, "cellpose merge", self.FileID):
            image = koopa.io.load_image(self.input()[0].path)[
                self.config["brains_channel"]
            ]
            yf = koopa.io.load_image(self.input()[1].path)
            segmap = koopa.segment_flies.merge_masks(yf)

            initial_count = segmap.max()
            segmap = koopa.segment_flies.remove_false_objects(
                image,
                segmap,
                min_intensity=self.config["min_intensity"],
                min_area=self.config["min_area"],
                max_area=self.config["max_area"],
            )
            final_count = segmap.max()

        koopa.io.save_image(self.output().path, segmap)
        self.logger.info(
            f"[{self.FileID}] Cellpose merge: {final_count} objects retained (filtered {initial_count - final_count})"
        )


class DilateCells(LuigiFileTask):
    """Task to dilate cells expanding them."""

    def requires(self):
        return SegmentCellsMerge(FileID=self.FileID)

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"], "segmentation_nuclei", f"{self.FileID}.tif"
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        with suppress_stdout():
            import koopa.segment_flies

        dilation = self.config["dilation"]
        self.logger.info(f"[{self.FileID}] Dilating cells (radius={dilation})")

        with log_timing(self.logger, "cell dilation", self.FileID):
            segmap = koopa.io.load_image(self.input().path)
            dilated = koopa.segment_flies.dilate_segmap(segmap, dilation=dilation)

        koopa.io.save_image(self.output().path, dilated)
        self.logger.info(f"[{self.FileID}] Cell dilation complete")


class SegmentOther(LuigiFileTask):
    """Task to segment other structures (non-cell channels)."""

    index_list = luigi.IntParameter()

    def requires(self):
        return Preprocess(FileID=self.FileID)

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"],
            f"segmentation_c{self.config['sego_channels'][self.index_list]}",
            f"{self.FileID}.tif",
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        with suppress_stdout():
            import koopa.segment_other

        channel = self.config["sego_channels"][self.index_list]
        method = self.config["sego_methods"][self.index_list]
        self.logger.info(
            f"[{self.FileID}] Segmenting channel {channel} (method: {method})"
        )

        with log_timing(self.logger, f"channel {channel} segmentation", self.FileID):
            image = koopa.io.load_image(self.input().path)
            segmap = koopa.segment_other.segment(
                image=image[channel],
                index_list=self.index_list,
                config=self.config,
            )

        koopa.io.save_image(self.output().path, segmap)
        self.logger.info(
            f"[{self.FileID}] Channel {channel} segmentation: {segmap.max()} objects found"
        )
