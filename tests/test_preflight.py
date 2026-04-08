"""Tests for preflight checks."""

from __future__ import annotations

import os

import pytest


class TestPreflightPaths:
    """Test path validation checks."""

    def test_missing_input_path(self, tmp_path):
        from src.preflight import _check_paths

        errors: list[str] = []
        config = {"input_path": "/nonexistent/path", "output_path": str(tmp_path)}
        _check_paths(config, errors)
        assert any("Input path" in e for e in errors)

    def test_valid_paths(self, tmp_path):
        from src.preflight import _check_paths

        errors: list[str] = []
        config = {"input_path": str(tmp_path), "output_path": str(tmp_path)}
        _check_paths(config, errors)
        assert len(errors) == 0

    def test_missing_alignment_model(self, tmp_path):
        from src.preflight import _check_paths

        errors: list[str] = []
        config = {
            "input_path": str(tmp_path),
            "output_path": str(tmp_path),
            "alignment_enabled": True,
            "alignment_path": str(tmp_path),
            "alignment_method": "deepblink",
            "alignment_model": "/nonexistent/model.h5",
        }
        _check_paths(config, errors)
        assert any("Alignment model" in e for e in errors)


class TestPreflightDetection:
    """Test detection channel/model validation."""

    def test_mismatched_channels_models(self):
        from src.preflight import _check_detection

        errors: list[str] = []
        config = {
            "detect_channels": [0, 1, 2],
            "detect_models": ["/model.h5"],
        }
        _check_detection(config, errors)
        assert any("must match" in e for e in errors)

    def test_empty_channels_ok(self):
        from src.preflight import _check_detection

        errors: list[str] = []
        _check_detection({"detect_channels": [], "detect_models": []}, errors)
        assert len(errors) == 0

    def test_missing_model_file(self, tmp_path):
        from src.preflight import _check_detection

        errors: list[str] = []
        config = {
            "detect_channels": [0],
            "detect_models": ["/nonexistent/model.h5"],
        }
        _check_detection(config, errors)
        assert any("not found" in e for e in errors)


class TestPreflightColocalization:
    """Test colocalization validation."""

    def test_coloc_channel_not_in_detect(self):
        from src.preflight import _check_colocalization

        errors: list[str] = []
        warnings: list[str] = []
        config = {
            "coloc_enabled": True,
            "detect_channels": [0, 1],
            "coloc_channels": [(0, 3)],  # channel 3 not detected
        }
        _check_colocalization(config, errors, warnings)
        assert any("channel 3" in e for e in errors)

    def test_valid_coloc(self):
        from src.preflight import _check_colocalization

        errors: list[str] = []
        warnings: list[str] = []
        config = {
            "coloc_enabled": True,
            "detect_channels": [0, 1],
            "coloc_channels": [(0, 1)],
        }
        _check_colocalization(config, errors, warnings)
        assert len(errors) == 0

    def test_disabled_coloc_skips_checks(self):
        from src.preflight import _check_colocalization

        errors: list[str] = []
        warnings: list[str] = []
        config = {"coloc_enabled": False, "detect_channels": [0], "coloc_channels": [(0, 99)]}
        _check_colocalization(config, errors, warnings)
        assert len(errors) == 0

    def test_3d_with_gap_frames_warns(self):
        from src.preflight import _check_colocalization

        errors: list[str] = []
        warnings: list[str] = []
        config = {
            "coloc_enabled": True,
            "detect_channels": [0, 1],
            "coloc_channels": [(0, 1)],
            "do_3d": True,
            "gap_frames": 3,
        }
        _check_colocalization(config, errors, warnings)
        assert any("gap_frames" in w for w in warnings)


class TestPreflightSegmentation:
    """Test segmentation validation."""

    def test_mismatched_sego_channels_methods(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        config = {
            "sego_enabled": True,
            "sego_channels": [0, 1],
            "sego_methods": ["otsu"],
        }
        _check_segmentation(config, errors, warnings)
        assert any("must match" in e for e in errors)


class TestPreflightDiskSpace:
    """Test disk space warnings."""

    def test_disk_space_check_runs(self, tmp_path):
        from src.preflight import _check_disk_space

        warnings: list[str] = []
        _check_disk_space({"output_path": str(tmp_path)}, warnings)
        # Should not crash — may or may not warn depending on actual disk space
