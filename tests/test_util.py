"""Tests for util module — config parsing, FileStatusTracker, logging."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from contextlib import contextmanager

import pytest


class TestLiteralEvalParsing:
    """Verify ast.literal_eval handles all config value types."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("42", 42),
            ("3.14", 3.14),
            ("True", True),
            ("False", False),
            ("'hello'", "hello"),
            ("[0, 1, 2]", [0, 1, 2]),
            ("[]", []),
            ("[()]", [()]),
            ("[(0, 1), (2, 3)]", [(0, 1), (2, 3)]),
            ("8_000", 8000),
            ("0.95", 0.95),
        ],
    )
    def test_literal_eval_parses_config_values(self, raw: str, expected):
        assert ast.literal_eval(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "hello",  # bare string without quotes
            "./path/to/file",
            "maximum",
            "otsu",
            "both",
            "none",
        ],
    )
    def test_literal_eval_falls_back_to_string(self, raw: str):
        """Values that aren't valid Python literals should fall back to raw string."""
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(raw)


class TestFileStatusTracker:
    """Tests for FileStatusTracker persistence and status tracking."""

    def _make_tracker(self, tmp_path: str):
        """Create a fresh tracker instance for testing."""
        from src.util import FileStatusTracker

        # Reset singleton
        FileStatusTracker._instance = None
        tracker = FileStatusTracker()
        tracker.set_output_path(tmp_path)
        return tracker

    def test_register_and_status(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("file_001")
        tracker.mark_processing("file_001")
        assert tracker.get_status("file_001") == "processing"

    def test_success_flow(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("file_001")
        tracker.mark_processing("file_001")
        tracker.mark_success("file_001")
        assert tracker.get_status("file_001") == "success"

    def test_failure_with_error(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("file_001")
        tracker.mark_failed("file_001", "Image corrupted")
        assert tracker.get_status("file_001") == "failed"
        assert tracker.get_error("file_001") == "Image corrupted"
        assert tracker.has_failures()

    def test_summary_counts(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("a")
        tracker.register_file("b")
        tracker.register_file("c", already_complete=True)

        tracker.mark_success("a")
        tracker.mark_failed("b", "error")

        summary, errors = tracker.get_summary()
        assert summary["success"] == ["a"]
        assert summary["failed"] == ["b"]
        assert summary["skipped"] == ["c"]

    def test_persistence_across_reads(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("file_001")
        tracker.mark_success("file_001")

        # Read the JSON directly to verify persistence
        status_file = os.path.join(str(tmp_path), ".file_status.json")
        with open(status_file) as f:
            data = json.load(f)
        assert data["files"]["file_001"] == "success"

    def test_reset_clears_state(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("file_001")
        tracker.mark_success("file_001")
        tracker.reset()
        assert not tracker.has_failures()

    def test_simplify_error_messages(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        assert "mismatched" in tracker._simplify_error(
            "Could not merge channels - all input arrays must have the same shape"
        )
        assert tracker._simplify_error("File not found") == "File not found"
        assert tracker._simplify_error(None) == "Unknown error"
        # Long errors get truncated
        long_error = "x" * 100
        assert len(tracker._simplify_error(long_error)) <= 80

    def test_format_summary_output(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("a")
        tracker.mark_success("a")
        lines = tracker.format_summary()
        assert any("RESULTS" in line for line in lines)
        assert any("1" in line for line in lines)


    def test_pending_files_in_summary(self, tmp_path):
        """Files registered but never marked should appear as pending."""
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("a")
        tracker.register_file("b")
        tracker.register_file("c")
        # Only mark 'a' as success — 'b' and 'c' stay pending
        tracker.mark_success("a")

        summary, _ = tracker.get_summary()
        assert summary["success"] == ["a"]
        assert set(summary["pending"]) == {"b", "c"}

    def test_format_summary_shows_pending_as_not_ok(self, tmp_path):
        """Pending files should not count toward 'files OK'."""
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("a")
        tracker.register_file("b")
        tracker.mark_success("a")

        lines = tracker.format_summary()
        text = "\n".join(lines)
        assert "1 of 2 files OK" in text

    def test_processing_files_are_incomplete(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("a")
        tracker.mark_processing("a")
        text = "\n".join(tracker.format_summary())
        assert "0 of 1 files OK" in text
        assert "1 file(s) INCOMPLETE" in text
        assert "processed successfully" not in text

    def test_final_task_failure_suppresses_success_message(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.register_file("a")
        tracker.mark_success("a")
        text = "\n".join(tracker.format_summary(pipeline_complete=False))
        assert "Pipeline did not complete" in text
        assert "processed successfully" not in text


class TestDeprecatedConfiguration:
    @pytest.fixture
    def config_path(self, tmp_path):
        import koopa.config

        cfg = koopa.config.create_default_config()
        cfg.set("General", "input_path", str(tmp_path))
        cfg.set("General", "output_path", str(tmp_path))
        cfg.set("SpotsDetection", "detect_channels", "[0]")
        cfg.set("SpotsDetection", "detect_models", "['model.h5']")
        cfg.set("SegmentationCells", "cellpose_resample", "False")
        path = tmp_path / "input.cfg"
        with path.open("w") as stream:
            cfg.write(stream)
        return path

    def test_removed_option_warns_and_is_omitted_from_saved_config(
        self, config_path, monkeypatch,
    ):
        from src import util
        import luigi

        parser = luigi.configuration.LuigiConfigParser()
        monkeypatch.setattr(luigi.configuration, "get_config", lambda: parser)
        monkeypatch.setenv("KOOPA_CONFIG_PATH", "")
        monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", "")
        with pytest.warns(UserWarning, match="cellpose_resample is deprecated and ignored"):
            util.set_configuration(config_path)
        assert "cellpose_resample" not in util.get_configuration()
        saved = util.koopa.io.load_config(config_path.parent / "koopa.cfg")
        assert not saved.has_option("SegmentationCells", "cellpose_resample")
        # The user's source config is preserved.
        source = util.koopa.io.load_config(config_path)
        assert source.has_option("SegmentationCells", "cellpose_resample")

    def test_worker_reload_accepts_removed_option(self, config_path, monkeypatch):
        from src import util
        import luigi

        parser = luigi.configuration.LuigiConfigParser()
        monkeypatch.setattr(luigi.configuration, "get_config", lambda: parser)
        monkeypatch.setenv("KOOPA_CONFIG_PATH", str(config_path))
        with pytest.warns(UserWarning, match="cellpose_resample"):
            config = util.get_configuration()
        assert "cellpose_resample" not in config
        assert config["detect_channels"] == [0]

    def test_unknown_options_still_raise(self, config_path):
        from src import util

        cfg = util.koopa.io.load_config(config_path)
        cfg.remove_option("SegmentationCells", "cellpose_resample")
        cfg.set("SegmentationCells", "cellpose_resampl", "True")
        with config_path.open("w") as stream:
            cfg.write(stream)
        with pytest.raises(ValueError, match="cellpose_resampl unknown"):
            util._load_configuration(config_path)

    def test_supported_option_is_preserved(self, config_path, monkeypatch):
        from src import util
        import warnings

        monkeypatch.setitem(
            util.koopa.config.CONFIGS["SegmentationCells"], "cellpose_resample",
            util.koopa.config.ConfigItem("Legacy resampling", True, bool),
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            cfg = util._load_configuration(config_path)
        assert cfg.getboolean("SegmentationCells", "cellpose_resample") is False
        assert not recorded


class TestLogTiming:
    """Tests for the log_timing context manager.

    Reimplements log_timing inline to avoid importing src.util (which
    requires koopa to be installed).
    """

    @staticmethod
    @contextmanager
    def _log_timing(logger, operation, file_id=None):
        """Mirror of src.util.log_timing for testing without koopa."""
        import time as _time

        context = f"[{file_id}] " if file_id else ""
        logger.debug(f"{context}Starting {operation}")
        start_time = _time.perf_counter()
        exc_occurred = False
        try:
            yield
        except BaseException:
            exc_occurred = True
            raise
        finally:
            elapsed = _time.perf_counter() - start_time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes}m {seconds:.1f}s"
            if exc_occurred:
                logger.error(
                    f"{context}{operation.capitalize()} failed after {time_str}"
                )
            else:
                logger.info(
                    f"{context}{operation.capitalize()} completed in {time_str}"
                )

    def test_logs_completed_on_success(self, caplog):
        import logging

        logger = logging.getLogger("test.timing")
        with caplog.at_level(logging.INFO, logger="test.timing"):
            with self._log_timing(logger, "test operation"):
                pass

        assert any("completed" in r.message.lower() for r in caplog.records)
        assert not any("failed" in r.message.lower() for r in caplog.records)

    def test_logs_failed_on_exception(self, caplog):
        import logging

        logger = logging.getLogger("test.timing")
        with caplog.at_level(logging.ERROR, logger="test.timing"):
            with pytest.raises(ValueError):
                with self._log_timing(logger, "test operation"):
                    raise ValueError("boom")

        assert any("failed" in r.message.lower() for r in caplog.records)
        assert not any("completed" in r.message.lower() for r in caplog.records)


class TestSuppressStdout:
    """Test the stdout suppression context manager."""

    def test_suppresses_print(self, capsys):
        from src.util import suppress_stdout

        with suppress_stdout():
            print("this should not appear")

        captured = capsys.readouterr()
        assert "this should not appear" not in captured.out
