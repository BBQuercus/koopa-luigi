from typing import Tuple
import os

import koopa.io
import koopa.postprocess
import koopa.util
import luigi
import pandas as pd

# Segment imports will be lazy-loaded when needed
from .spots import ColocalizeFrame
from .spots import ColocalizeTrack
from .spots import Detect
from .spots import Track
from .util import LuigiTask
from .util import log_timing
from .util import file_tracker


class Merge(LuigiTask):
    """Merge all analysis files into a single summary file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnames = koopa.util.get_file_list(
            self.config["input_path"], self.config["file_ext"]
        )
        self._total_files = len(self.fnames)
        self.logger.info(f"Found {self._total_files} files to process")
        self.logger.debug(f"Files: {self.fnames}")
        # Register all files with the tracker
        for fname in self.fnames:
            file_tracker.register_file(fname)

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
        self.logger.info(f"Merging results from {self._total_files} files")
        dfs = []
        skipped = 0
        for idx, fname in enumerate(self.fnames, 1):
            self.logger.info(f"[{idx}/{self._total_files}] Merging {fname}")
            with log_timing(self.logger, "merge", fname):
                result = self.__run_single(fname)
            if result is None:
                skipped += 1
            else:
                dfs.append(result)

        # Save final outputs
        for idx, target in enumerate(self.output()):
            df = pd.concat([i[idx] for i in dfs if i is not None], ignore_index=True)
            koopa.io.save_csv(target.path, df)

        # Summary statistics
        total_spots = (
            len(pd.read_csv(self.output()[0].path))
            if os.path.exists(self.output()[0].path)
            else 0
        )
        self.logger.info("=" * 40)
        self.logger.info("  Pipeline Complete")
        self.logger.info("=" * 40)
        self.logger.info(
            f"Files processed: {self._total_files - skipped}/{self._total_files}"
        )
        if skipped > 0:
            self.logger.warning(f"Files skipped (no spots): {skipped}")
        self.logger.info(f"Total spots detected: {total_spots}")
        self.logger.info(f"Output: {self.output()[0].path}")

    def get_final_spots(self, fname: str, gpu: bool = False) -> list:
        skip = self.skip_incompatible
        if self.config["coloc_enabled"]:
            if self.config["do_timeseries"]:
                return [
                    ColocalizeTrack(
                        FileID=fname,
                        index_reference=r,
                        index_transform=t,
                        skip_incompatible=skip,
                    )
                    for r, t in self.config["coloc_channels"]
                ]
            return [
                ColocalizeFrame(
                    FileID=fname,
                    index_reference=r,
                    index_transform=t,
                    skip_incompatible=skip,
                )
                for r, t in self.config["coloc_channels"]
            ]

        if self.config["do_3d"] or self.config["do_timeseries"]:
            return [
                Track(FileID=fname, index_list=idx, skip_incompatible=skip)
                for idx, _ in enumerate(self.config["detect_channels"])
            ]
        return [
            Detect(FileID=fname, index_list=idx, gpu=gpu, skip_incompatible=skip)
            for idx, _ in enumerate(self.config["detect_channels"])
        ]

    def get_segmaps(self, fname: str, gpu: bool = False) -> dict:
        skip = self.skip_incompatible
        segmaps = {}
        if self.config["sego_enabled"]:
            from .segment import SegmentOther

            for idx, _ in enumerate(self.config["sego_channels"]):
                segmaps[f"other_{idx}"] = SegmentOther(
                    FileID=fname, index_list=idx, gpu=gpu, skip_incompatible=skip
                )

        if self.config["brains_enabled"]:
            from .segment import SegmentCellsPredict, DilateCells

            segmaps["nuclei"] = (
                SegmentCellsPredict(FileID=fname, gpu=gpu, skip_incompatible=skip)
                if gpu
                else DilateCells(FileID=fname, skip_incompatible=skip)
            )
        elif self.config["selection"] == "both":
            from .segment import SegmentCellsBoth

            segmaps["both"] = SegmentCellsBoth(
                FileID=fname, gpu=gpu, skip_incompatible=skip
            )
        else:
            from .segment import SegmentCellsSingle

            segmaps[self.config["selection"]] = SegmentCellsSingle(
                FileID=fname, gpu=gpu, skip_incompatible=skip
            )
        return segmaps

    def __run_single(self, fname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.debug(f"[{fname}] Loading segmentation maps and spot data")
        req_segmaps, req_spots = self.requires()[fname]

        # Load segmentation maps
        segmaps = {}
        for name, task in req_segmaps.items():
            try:
                if name == "both":
                    segmaps["nuclei"] = koopa.io.load_image(task.output()[0].path)
                    segmaps["cyto"] = koopa.io.load_image(task.output()[1].path)
                else:
                    segmaps[name] = koopa.io.load_image(task.output().path)
            except Exception as e:
                self.logger.error(
                    f"[{fname}] Failed to load segmentation map '{name}' "
                    f"from {task.output().path if hasattr(task.output(), 'path') else task.output()[0].path}: {e}"
                )
                raise

        # Load spot detection results
        try:
            df = pd.concat([koopa.io.load_parquet(i.output().path) for i in req_spots])
        except Exception as e:
            self.logger.error(f"[{fname}] Failed to load spot detection results: {e}")
            raise

        # Combine segmentation and spot data
        try:
            df, df_cell = koopa.postprocess.get_segmentation_data(
                df, segmaps, self.config
            )
        except ValueError as e:
            if "empty" in str(e).lower() or len(df) == 0:
                self.logger.warning(
                    f"[{fname}] No spots detected - file will be skipped from final output"
                )
                return None
            self.logger.error(f"[{fname}] Failed to process segmentation data: {e}")
            raise

        spot_count = len(df)
        cell_count = len(df_cell)
        self.logger.debug(
            f"[{fname}] Merged {spot_count} spots across {cell_count} cells"
        )
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
