"""Tests for the dilated-segmentation proximity feature."""

from __future__ import annotations

import numpy as np
import pytest


class TestDilateLabels:
    """Unit tests for src.segment._dilate_labels."""

    def test_radius_zero_returns_input(self):
        from src.dilate import dilate_labels

        seg = np.zeros((10, 10), dtype=np.uint16)
        seg[4:6, 4:6] = 1
        out = dilate_labels(seg, 0)
        assert np.array_equal(out, seg)

    def test_2d_grows_by_exact_pixel_radius(self):
        from src.dilate import dilate_labels

        seg = np.zeros((30, 30), dtype=np.uint16)
        seg[15, 15] = 1
        out = dilate_labels(seg, 5)
        ys, xs = np.where(out > 0)
        assert int(max(abs(ys - 15))) == 5
        assert int(max(abs(xs - 15))) == 5

    def test_3d_grows_in_all_axes(self):
        from src.dilate import dilate_labels

        seg = np.zeros((20, 20, 20), dtype=np.uint16)
        seg[10, 10, 10] = 1
        out = dilate_labels(seg, 3)
        zs, ys, xs = np.where(out > 0)
        assert int(max(abs(zs - 10))) == 3
        assert int(max(abs(ys - 10))) == 3
        assert int(max(abs(xs - 10))) == 3

    def test_preserves_distinct_labels(self):
        from src.dilate import dilate_labels

        seg = np.zeros((20, 20), dtype=np.uint16)
        seg[5:8, 5:8] = 1
        seg[5:8, 14:17] = 2
        out = dilate_labels(seg, 2)
        assert sorted(np.unique(out).tolist()) == [0, 1, 2]

    def test_negative_radius_erodes(self):
        from src.dilate import dilate_labels

        seg = np.zeros((30, 30), dtype=np.uint16)
        seg[10:21, 10:21] = 1  # 11x11 block
        out = dilate_labels(seg, -2)
        # Eroded region is strictly smaller but non-empty and same label.
        assert 0 < int((out > 0).sum()) < int((seg > 0).sum())
        assert set(np.unique(out).tolist()) == {0, 1}

    def test_erosion_can_remove_small_objects(self):
        from src.dilate import dilate_labels

        seg = np.zeros((30, 30), dtype=np.uint16)
        seg[15, 15] = 1  # single pixel disappears under any erosion
        out = dilate_labels(seg, -1)
        assert int((out > 0).sum()) == 0

    def test_rejects_unsupported_dimensions(self):
        from src.dilate import dilate_labels

        with pytest.raises(ValueError, match="Unsupported"):
            dilate_labels(np.zeros((2, 2, 2, 2), dtype=np.uint16), 1)


class TestPreflightDilations:
    """Validation of sego_dilations in preflight."""

    def _base_config(self) -> dict:
        return {
            "sego_enabled": True,
            "sego_channels": [2, 3],
            "sego_methods": ["otsu", "otsu"],
        }

    def test_empty_dilations_pass(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        cfg = self._base_config()
        cfg["sego_dilations"] = []
        _check_segmentation(cfg, errors, warnings)
        assert errors == []

    def test_valid_per_channel_dilations(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        cfg = self._base_config()
        cfg["sego_dilations"] = [[0, 5, 10], []]
        _check_segmentation(cfg, errors, warnings)
        assert errors == []

    def test_wrong_length(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        cfg = self._base_config()
        cfg["sego_dilations"] = [[5]]  # only one entry, two channels
        _check_segmentation(cfg, errors, warnings)
        assert any("must match" in e for e in errors)

    def test_not_list_of_lists(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        cfg = self._base_config()
        cfg["sego_dilations"] = [5, 10]  # flat list, not list-of-lists
        _check_segmentation(cfg, errors, warnings)
        assert any("list of lists" in e for e in errors)

    def test_negative_radius_accepted(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        cfg = self._base_config()
        cfg["sego_dilations"] = [[-5, 0, 10], []]  # erosion, keep, dilation
        _check_segmentation(cfg, errors, warnings)
        assert errors == []

    def test_non_integer_radius_rejected(self):
        from src.preflight import _check_segmentation

        errors: list[str] = []
        warnings: list[str] = []
        cfg = self._base_config()
        cfg["sego_dilations"] = [[2.5], []]
        _check_segmentation(cfg, errors, warnings)
        assert any("must contain integers" in e for e in errors)


class TestPlanOtherSegmaps:
    """Verify plan_other_segmaps assembles the right (key, kind, params) tuples.

    This is the pure logic that drives Merge.get_segmaps when sego_enabled.
    """

    def test_no_dilations_legacy_keys(self):
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2], [])
        assert plan == [("other_0", "segment", {"index_list": 0})]

    def test_empty_per_channel_falls_back_to_legacy(self):
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2, 3], [[], []])
        keys = [p[0] for p in plan]
        assert keys == ["other_0", "other_1"]
        assert all(p[1] == "segment" for p in plan)

    def test_dilations_with_zero_keeps_original(self):
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2], [[0, 5, 10]])
        kinds = {p[0]: p[1] for p in plan}
        assert kinds == {
            "other_0": "segment",
            "other_0_d5": "dilate",
            "other_0_d10": "dilate",
        }

    def test_dilations_without_zero_drops_original(self):
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2], [[5, 10]])
        keys = [p[0] for p in plan]
        assert "other_0" not in keys
        assert "other_0_d5" in keys
        assert "other_0_d10" in keys
        # Each dilation entry carries its own (index_list, dilation) params
        for key, kind, params in plan:
            assert kind == "dilate"
            assert params["index_list"] == 0
            assert params["dilation"] in (5, 10)

    def test_per_channel_independent(self):
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2, 3], [[5, 10], []])
        keys = [p[0] for p in plan]
        assert keys == ["other_0_d5", "other_0_d10", "other_1"]
        # Channel 0 dilation tasks reference index_list=0
        for key, kind, params in plan[:2]:
            assert kind == "dilate"
            assert params["index_list"] == 0
        # Channel 1 falls back to legacy SegmentOther
        _, kind, params = plan[2]
        assert kind == "segment"
        assert params["index_list"] == 1

    def test_negative_radius_produces_erosion_key(self):
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2], [[-5, 10]])
        keys = [p[0] for p in plan]
        assert keys == ["other_0_d-5", "other_0_d10"]
        params = {p[0]: p[2]["dilation"] for p in plan}
        assert params["other_0_d-5"] == -5
        assert params["other_0_d10"] == 10

    def test_missing_per_channel_entry_treated_as_empty(self):
        """If sego_dilations is shorter than sego_channels, missing entries
        fall back to legacy behavior (one SegmentOther per missing channel)."""
        from src.dilate import plan_other_segmaps

        plan = plan_other_segmaps([2, 3, 4], [[5]])
        keys = [p[0] for p in plan]
        assert keys == ["other_0_d5", "other_1", "other_2"]
