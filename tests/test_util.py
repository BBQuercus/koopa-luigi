"""Tests for util module — config parsing, FileStatusTracker, logging."""

from __future__ import annotations

import ast
import json
import os
import tempfile

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


class TestSuppressStdout:
    """Test the stdout suppression context manager."""

    def test_suppresses_print(self, capsys):
        from src.util import suppress_stdout

        with suppress_stdout():
            print("this should not appear")

        captured = capsys.readouterr()
        assert "this should not appear" not in captured.out
