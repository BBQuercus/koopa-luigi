from typing import Tuple
import os

import koopa
import pandas as pd
import luigi

from .segment import DilateCells
from .segment import SegmentCellsBoth
from .segment import SegmentCellsPredict
from .segment import SegmentCellsSingle
from .segment import SegmentOther
from .spots import ColocalizeFrame
from .spots import ColocalizeTrack
from .spots import Detect
from .spots import Track
from .util import LuigiTask


class Merge(LuigiTask):
    """Merge all analysis files into a single summary file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnames = koopa.util.get_file_list(
            self.config["input_path"], self.config["file_ext"]
        )
        self.logger.info(
            f"Running analysis with {len(self.fnames)} files - {self.fnames}"
        )

    def requires(self):
        requirements = {}
        for fname in self.fnames:
            segmaps = self.get_segmaps(fname)
            spots = self.get_final_spots(fname)
            requirements[fname] = (segmaps, spots)
        return requirements

    def output(self):
        fname_full = os.path.join(self.config["output_path"], "summary.csv")
        fname_cells = os.path.join(self.config["output_path"], "summary_cells.csv")
        return [luigi.LocalTarget(fname_full), luigi.LocalTarget(fname_cells)]

    def run(self):
        dfs = [self.__run_single(fname) for fname in self.fnames]
        for idx, target in enumerate(self.output()):
            df = pd.concat([i[idx] for i in dfs], ignore_index=True)
            koopa.io.save_csv(target.path, df)
        self.logger.info("Koopa finished analyzing everything!")

    def get_final_spots(self, fname: str, gpu: bool = False) -> list:
        if self.config["coloc_enabled"]:
            if self.config["do_timeseries"]:
                return [
                    ColocalizeTrack(FileID=fname, index_reference=r, index_transform=t)
                    for r, t in self.config["coloc_channels"]
                ]
            return [
                ColocalizeFrame(FileID=fname, index_reference=r, index_transform=t)
                for r, t in self.config["coloc_channels"]
            ]

        if self.config["do_3d"] or self.config["do_timeseries"]:
            return [
                Track(FileID=fname, index_list=idx)
                for idx, _ in enumerate(self.config["detect_channels"])
            ]
        return [
            Detect(FileID=fname, index_list=idx, gpu=gpu)
            for idx, _ in enumerate(self.config["detect_channels"])
        ]

    def get_segmaps(self, fname: str, gpu: bool = False) -> dict:
        segmaps = {}
        if self.config["sego_enabled"]:
            for idx, _ in enumerate(self.config["sego_channels"]):
                segmaps[f"other_{idx}"] = SegmentOther(
                    FileID=fname, index_list=idx, gpu=gpu
                )

        if self.config["brains_enabled"]:
            segmaps["nuclei"] = (
                SegmentCellsPredict(FileID=fname, gpu=gpu)
                if gpu
                else DilateCells(FileID=fname)
            )
        elif self.config["selection"] == "both":
            segmaps["both"] = SegmentCellsBoth(FileID=fname, gpu=gpu)
        else:
            segmaps[self.config["selection"]] = SegmentCellsSingle(
                FileID=fname, gpu=gpu
            )
        return segmaps

    def __run_single(self, fname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        req_segmaps, req_spots = self.requires()[fname]

        segmaps = {}
        for name, task in req_segmaps.items():
            if name == "both":
                segmaps["nuclei"] = koopa.io.load_image(task.output()[0].path)
                segmaps["cyto"] = koopa.io.load_image(task.output()[1].path)
            else:
                segmaps[name] = koopa.io.load_image(task.output().path)

        df = pd.concat([koopa.io.load_parquet(i.output().path) for i in req_spots])
        df, df_cell = koopa.postprocess.get_segmentation_data(df, segmaps, self.config)
        self.logger.debug(f"Merged files for {fname}")
        return df, df_cell


class GPUMerge(Merge):
    """Wrapper to retrieve all GPU dependent tasks."""

    def requires(self):
        requirements = []
        for fname in self.fnames:
            requirements.extend(self.get_segmaps(fname, gpu=True).values())
            requirements.extend(self.get_final_spots(fname, gpu=True))
        self.logger.info(f"requirements - {requirements}")
        return requirements

    def output(self):
        fname = os.path.join(self.config["output_path"], "gpu.tmp")
        return luigi.LocalTarget(fname)

    def run(self):
        with open(self.output().path, "w") as f:
            f.write("Misa can be deleted after running :)")
