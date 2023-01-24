import os

import koopa
import luigi
import numpy as np

from .util import LuigiFileTask
from .util import LuigiTask


class ReferenceAlignment(LuigiTask):
    """Task to create affine matrix for two reference channels."""

    def output(self):
        fname_pre = os.path.join(self.config["output_path"], "alignment_pre.tif")
        fname_post = os.path.join(self.config["output_path"], "alignment_post.tif")
        fname_matrix = os.path.join(self.config["output_path"], "alignment.npy")
        return [
            luigi.LocalTarget(fname_pre),
            luigi.LocalTarget(fname_post),
            luigi.LocalTarget(fname_matrix),
        ]

    def run(self):
        reference, transform = koopa.align.load_alignment_images(
            self.config["alignment_path"],
            self.config["channel_reference"],
            self.config["channel_transform"],
        )
        sr = self.__run(reference, transform)
        koopa.align.visualize_alignment(
            sr, reference[0], transform[0], self.output()[0].path, self.output()[1].path
        )
        koopa.io.save_alignment(self.output()[2].path, sr)

    def __run(self, reference: np.ndarray, transform: np.ndarray):
        if self.config["alignment_method"] == "pystackreg":
            matrix = koopa.align.register_alignment_pystackreg(reference, transform)
        elif self.config["alignment_method"] == "deepblink":
            matrix = koopa.align.register_alignment_deepblink(
                self.config["alignment_model"], reference, transform
            )
        else:
            raise ValueError(
                f"Unknown alignment method: {self.config['alignment_method']}"
            )
        sr = koopa.align.get_stackreg(matrix)
        return sr


class Preprocess(LuigiFileTask):
    """Task to open, trim, and align images."""

    def requires(self):
        requirements = {}
        if self.config["alignment_enabled"]:
            requirements["alignment"] = ReferenceAlignment()
        return requirements

    def output(self):
        fname_out = os.path.join(
            self.config["output_path"], "preprocessed", f"{self.FileID}.tif"
        )
        return luigi.LocalTarget(fname_out)

    def run(self):
        fname_in = koopa.io.find_full_path(
            self.config["input_path"], self.FileID, self.config["file_ext"]
        )
        image = koopa.io.load_raw_image(fname_in, self.config["file_ext"])
        image = self.__run(image)
        self.logger.debug(f"Preprocessed {self.FileID}")
        koopa.io.save_image(self.output().path, image)

    def __run(self, image: np.ndarray):
        if image.ndim != 4:
            raise ValueError(
                f"Image {self.FileID} has {image.ndim} dimensions, expected 4."
            )
        if not self.config["do_3d"] and not self.config["do_timeseries"]:
            image = koopa.preprocess.register_3d_image(
                image, self.config["registration_method"]
            )
        if self.config["do_3d"] or self.config["do_timeseries"]:
            image = koopa.preprocess.trim_image(
                image, self.config["frame_start"], self.config["frame_end"]
            )
        if self.config["crop_start"] or self.config["crop_end"]:
            image = koopa.preprocess.crop_image(
                image, self.config["crop_start"], self.config["crop_end"]
            )
        if self.config["bin_axes"]:
            image = koopa.preprocess.bin_image(image, self.config["bin_axes"])

        # TODO add multiple transform channels?
        if self.config["alignment_enabled"]:
            sr = koopa.io.load_alignment(self.input()["alignment"][2].path)
            image = koopa.align.align_image(
                image, sr, [self.config["channel_transform"]]
            )
        return image
