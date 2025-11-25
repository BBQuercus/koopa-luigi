import os

import koopa.align
import koopa.io
import koopa.preprocess
import luigi
import numpy as np

from .util import LuigiFileTask, LuigiTask, log_timing


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
        ch_ref = self.config["channel_reference"]
        ch_transform = self.config["channel_transform"]
        method = self.config["alignment_method"]

        self.logger.info(
            f"Computing alignment matrix (ch{ch_ref} → ch{ch_transform}, method: {method})"
        )

        with log_timing(self.logger, "alignment computation"):
            reference, transform = koopa.align.load_alignment_images(
                self.config["alignment_path"],
                ch_ref,
                ch_transform,
            )
            sr = self.__run(reference, transform)
            koopa.align.visualize_alignment(
                sr,
                reference[0],
                transform[0],
                self.output()[0].path,
                self.output()[1].path,
            )

        koopa.io.save_alignment(self.output()[2].path, sr)
        self.logger.info("Alignment matrix saved")

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
        self.logger.info(f"[{self.FileID}] Preprocessing image")

        with log_timing(self.logger, "preprocessing", self.FileID):
            try:
                fname_in = koopa.io.find_full_path(
                    self.config["input_path"], self.FileID, self.config["file_ext"]
                )
            except FileNotFoundError as e:
                self.logger.error(
                    f"[{self.FileID}] Input file not found in {self.config['input_path']} "
                    f"with extension '{self.config['file_ext']}'"
                )
                raise

            try:
                image = koopa.io.load_raw_image(fname_in, self.config["file_ext"])
            except Exception as e:
                self.logger.error(
                    f"[{self.FileID}] Failed to load image from {fname_in}: {e}"
                )
                raise

            image = self.__run(image)

        koopa.io.save_image(self.output().path, image)
        self.logger.info(
            f"[{self.FileID}] Preprocessing complete (shape: {image.shape})"
        )

    def __run(self, image: np.ndarray):
        if image.ndim != 4:
            self.logger.error(
                f"[{self.FileID}] Invalid image dimensions: got {image.ndim}D (shape: {image.shape}), expected 4D (CZYX or TZYX)"
            )
            raise ValueError(
                f"Image {self.FileID} has {image.ndim} dimensions (shape: {image.shape}), expected 4D."
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
