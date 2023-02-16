import os

import koopa.io
import koopa.segment_cells
import koopa.segment_flies
import koopa.segment_other
import luigi

from .util import LuigiFileTask
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
        image = koopa.io.load_image(self.input().path)
        image = image[self.config[f"channel_{self.config['selection']}"]]

        if not self.config["do_3d"] and self.config["do_timeseries"]:
            image = koopa.segment_cells.preprocess(image)
        if self.config["selection"] == "nuclei":
            segmap = koopa.segment_cells.segment_nuclei(image, self.config)
        else:
            segmap = koopa.segment_cells.segment_cyto(image, self.config)

        koopa.io.save_image(self.output().path, segmap)


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
        image = koopa.io.load_image(self.input().path)
        image_nuclei = image[self.config["channel_nuclei"]]
        image_cyto = image[self.config["channel_cyto"]]
        if not self.config["do_3d"] and self.config["do_timeseries"]:
            image_nuclei = koopa.segment_cells.preprocess(image_nuclei)
            image_cyto = koopa.segment_cells.preprocess(image_cyto)

        segmap_nuclei, segmap_cyto = koopa.segment_cells.segment_both(
            image_nuclei, image_cyto, self.config
        )
        koopa.io.save_image(self.output()[0].path, segmap_nuclei)
        koopa.io.save_image(self.output()[1].path, segmap_cyto)


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
        self.logger.info("Reading image for prediction")
        image = koopa.io.load_image(self.input().path)
        self.logger.info("Normalizing nucleus")
        image = koopa.segment_flies.normalize_nucleus(
            image[self.config["brains_channel"]]
        )
        self.logger.info("Starting prediction")
        segmap = koopa.segment_flies.cellpose_predict(image, self.config["batch_size"])
        self.logger.info("Finished prediction")
        koopa.io.save_image(self.output().path, segmap)


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
        image = koopa.io.load_image(self.input()[0].path)[self.config["brains_channel"]]
        yf = koopa.io.load_image(self.input()[1].path)
        segmap = koopa.segment_flies.merge_masks(yf)
        segmap = koopa.segment_flies.remove_false_objects(
            image,
            segmap,
            min_intensity=self.config["min_intensity"],
            min_area=self.config["min_area"],
            max_area=self.config["max_area"],
        )
        koopa.io.save_image(self.output().path, segmap)


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
        segmap = koopa.io.load_image(self.input().path)
        dilated = koopa.segment_flies.dilate_segmap(
            segmap, dilation=self.config["dilation"]
        )
        koopa.io.save_image(self.output().path, dilated)


class SegmentOther(LuigiFileTask):
    """Task to segment ."""

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
        image = koopa.io.load_image(self.input().path)
        segmap = koopa.segment_other.segment(
            image=image[self.config["sego_channels"][self.index_list]],
            index_list=self.index_list,
            config=self.config,
        )
        koopa.io.save_image(self.output().path, segmap)
