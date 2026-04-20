"""Tests for alignment preprocessing — z-stack projection for companion.ome files."""

from __future__ import annotations

import numpy as np
import pytest


class TestAlignmentZProjection:
    """Verify that 3D z-stacks are max-projected to 2D before alignment."""

    def test_3d_images_projected_to_2d(self):
        """Companion.ome bead images are z-stacks (Z, Y, X) that must be
        projected to 2D before pystackreg registration."""
        # Simulate what load_alignment_images returns for companion.ome:
        # a list of 3D z-stack arrays
        z, y, x = 26, 64, 64
        reference = [np.random.rand(z, y, x)]
        transform = [np.random.rand(z, y, x)]

        # Apply the same projection logic used in ReferenceAlignment.run()
        reference = [img.max(axis=0) if img.ndim == 3 else img for img in reference]
        transform = [img.max(axis=0) if img.ndim == 3 else img for img in transform]

        assert reference[0].shape == (y, x)
        assert transform[0].shape == (y, x)

    def test_2d_images_unchanged(self):
        """Already-2D images (e.g. from regular .tif) should pass through."""
        y, x = 64, 64
        reference = [np.random.rand(y, x)]
        transform = [np.random.rand(y, x)]

        reference = [img.max(axis=0) if img.ndim == 3 else img for img in reference]
        transform = [img.max(axis=0) if img.ndim == 3 else img for img in transform]

        assert reference[0].shape == (y, x)
        assert transform[0].shape == (y, x)

    def test_multiple_bead_images(self):
        """Multiple bead images should each be projected independently."""
        z, y, x = 10, 32, 32
        reference = [np.random.rand(z, y, x), np.random.rand(z, y, x)]

        reference = [img.max(axis=0) if img.ndim == 3 else img for img in reference]

        assert len(reference) == 2
        assert all(img.shape == (y, x) for img in reference)

    def test_max_projection_values_correct(self):
        """Max projection should take the maximum value along the Z axis."""
        img = np.zeros((3, 2, 2))
        img[0, 0, 0] = 10.0
        img[1, 1, 1] = 20.0
        img[2, 0, 1] = 30.0

        projected = img.max(axis=0)

        assert projected[0, 0] == 10.0
        assert projected[1, 1] == 20.0
        assert projected[0, 1] == 30.0
        assert projected[1, 0] == 0.0
