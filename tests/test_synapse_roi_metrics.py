"""CPU-only exact regression tests for opt-in physical-metric ROI acceleration."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.synapse_physical_geometry import binary_metrics_mm
from training.synapse_roi_metrics import binary_metrics_mm_roi, union_roi_slices


class ROIMetricTests(unittest.TestCase):
    def assert_original_equal(self, pred, gt, spacing=(3.0, 0.7, 0.9)):
        before_pred, before_gt = pred.copy(), gt.copy()
        expected = binary_metrics_mm(pred, gt, spacing, compute_hd95_voxel=True)
        actual = binary_metrics_mm_roi(pred, gt, spacing, compute_hd95_voxel=True)
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(pred, before_pred)
        np.testing.assert_array_equal(gt, before_gt)
        self.assertEqual(set(actual), {"dice", "hd95_mm", "hd95_voxel", "status"})
        return actual

    def test_default_roi_skips_voxel_computation_and_keeps_dice_mm_exact(self):
        pred, gt = np.zeros((15, 17, 19), bool), np.zeros((15, 17, 19), bool)
        pred[4:7, 5:8, 6:9] = True
        gt[5:8, 5:8, 6:9] = True
        spacing = (3.0, 0.7, 0.9)
        from training.synapse_physical_geometry import metric
        original = metric.binary.hd95
        with patch("training.synapse_physical_geometry.metric.binary.hd95",
                   wraps=original) as hd95:
            disabled = binary_metrics_mm_roi(pred, gt, spacing)
            self.assertEqual(hd95.call_count, 1)
            self.assertEqual(hd95.call_args.kwargs, {"voxelspacing": spacing})
        enabled = binary_metrics_mm_roi(pred, gt, spacing, compute_hd95_voxel=True)
        self.assertIsNone(disabled["hd95_voxel"])
        self.assertEqual(disabled["dice"], enabled["dice"])
        self.assertEqual(disabled["hd95_mm"], enabled["hd95_mm"])
        self.assertEqual(disabled["status"], enabled["status"])

    def test_empty_branches_keep_exact_legacy_scores_and_status(self):
        for has_pred, has_gt in [(False, False), (True, False), (False, True)]:
            with self.subTest(has_pred=has_pred, has_gt=has_gt):
                pred = np.zeros((9, 11, 13), dtype=bool)
                gt = np.zeros_like(pred)
                if has_pred:
                    pred[1, 2, 3] = True
                if has_gt:
                    gt[6, 7, 8] = True
                actual = self.assert_original_equal(pred, gt)
                self.assertEqual(actual["dice"], float(has_pred and not has_gt))
                self.assertEqual(actual["hd95_mm"], 0.0)
                self.assertEqual(actual["hd95_voxel"], 0.0)

    def test_empty_branch_is_delegated_on_full_shape(self):
        for has_pred, has_gt in [(False, False), (True, False), (False, True)]:
            pred = np.full((6, 7, 8), has_pred)
            gt = np.full((6, 7, 8), has_gt)
            sentinel = object()
            with patch("training.synapse_roi_metrics.binary_metrics_mm",
                       return_value=sentinel) as original:
                result = binary_metrics_mm_roi(pred, gt, (2.1, 0.6, 0.8))
                self.assertIs(result, sentinel)
                self.assertEqual(original.call_count, 1)
                args = original.call_args.args
                self.assertEqual(args[0].shape, pred.shape)
                self.assertEqual(args[1].shape, gt.shape)

    def test_union_bounds_keep_both_foregrounds_and_halo(self):
        pred = np.zeros((30, 32, 34), bool)
        gt = np.zeros_like(pred)
        pred[3:5, 7:10, 13:15] = True
        gt[8:11, 12:15, 19:22] = True
        roi = union_roi_slices(pred, gt)
        self.assertEqual(roi, (slice(2, 12), slice(6, 16), slice(12, 23)))
        self.assertEqual(int(pred[roi].sum()), int(pred.sum()))
        self.assertEqual(int(gt[roi].sum()), int(gt.sum()))
        self.assert_original_equal(pred, gt)

    def test_large_halo_is_clipped_to_original_array(self):
        pred = np.zeros((5, 6, 7), bool)
        gt = np.zeros_like(pred)
        pred[2, 3, 4] = True
        self.assertEqual(union_roi_slices(pred, gt, halo=100),
                         (slice(0, 5), slice(0, 6), slice(0, 7)))

    def test_both_empty_roi_is_none_and_one_nonempty_has_bounds(self):
        pred = np.zeros((8, 9, 10), bool)
        gt = np.zeros_like(pred)
        self.assertIsNone(union_roi_slices(pred, gt))
        pred[3, 4, 5] = True
        self.assertEqual(union_roi_slices(pred, gt),
                         (slice(2, 5), slice(3, 6), slice(4, 7)))

    def test_actual_metric_receives_cropped_arrays_without_lost_foreground(self):
        pred = np.zeros((21, 23, 25), bool)
        gt = np.zeros_like(pred)
        pred[8:11, 9:12, 10:13] = True
        gt[9:12, 10:13, 11:14] = True
        with patch("training.synapse_roi_metrics.binary_metrics_mm",
                   wraps=binary_metrics_mm) as original:
            actual = binary_metrics_mm_roi(pred, gt, (3.0, 0.7, 0.9))
            args = original.call_args.args
            self.assertEqual(args[0].shape, (6, 6, 6))
            self.assertEqual(args[1].shape, (6, 6, 6))
            self.assertEqual(args[0].sum(), pred.sum())
            self.assertEqual(args[1].sum(), gt.sum())
            self.assertEqual(actual, binary_metrics_mm(pred, gt, (3.0, 0.7, 0.9)))

    def test_identical_masks_exact(self):
        for shape in [(1, 1, 1), (1, 7, 9), (9, 10, 11)]:
            for value in [False, True]:
                with self.subTest(shape=shape, value=value):
                    pred = np.full(shape, value)
                    self.assert_original_equal(pred, pred.copy())

    def test_single_voxels_axis_distances_keep_real_spacing(self):
        shape = (11, 13, 15)
        spacing = (3.25, 0.61, 0.92)
        for axis in range(3):
            pred, gt = np.zeros(shape, bool), np.zeros(shape, bool)
            first = [5, 6, 7]
            second = first.copy()
            second[axis] += 2
            pred[tuple(first)], gt[tuple(second)] = True, True
            with self.subTest(axis=axis):
                result = self.assert_original_equal(pred, gt, spacing)
                self.assertEqual(result["hd95_voxel"], 2.0)
                self.assertAlmostEqual(result["hd95_mm"], 2 * spacing[axis])

    def test_six_original_faces_preserve_boundary_surfaces(self):
        shape = (15, 17, 19)
        for axis in range(3):
            for high in [False, True]:
                with self.subTest(axis=axis, high=high):
                    pred, gt = np.zeros(shape, bool), np.zeros(shape, bool)
                    p_slices = [slice(5, 8)] * 3
                    g_slices = [slice(5, 8)] * 3
                    if high:
                        p_slices[axis] = slice(shape[axis]-3, shape[axis])
                        g_slices[axis] = slice(shape[axis]-4, shape[axis]-1)
                    else:
                        p_slices[axis], g_slices[axis] = slice(0, 3), slice(1, 4)
                    pred[tuple(p_slices)], gt[tuple(g_slices)] = True, True
                    roi = union_roi_slices(pred, gt)
                    if high:
                        self.assertEqual(roi[axis].stop, shape[axis])
                    else:
                        self.assertEqual(roi[axis].start, 0)
                    self.assert_original_equal(pred, gt)

    def test_all_corners_and_distant_components_remain_in_union(self):
        pred, gt = np.zeros((17, 19, 21), bool), np.zeros((17, 19, 21), bool)
        pred[4:7, 6:9, 8:11], gt[5:8, 6:9, 8:11] = True, True
        for z in [0, 16]:
            for y in [0, 18]:
                for x in [0, 20]:
                    pred[z, y, x] = True
        self.assertEqual(union_roi_slices(pred, gt),
                         (slice(0, 17), slice(0, 19), slice(0, 21)))
        self.assert_original_equal(pred, gt)

    def test_far_false_positive_cannot_be_removed(self):
        pred, gt = np.zeros((29, 31, 33), bool), np.zeros((29, 31, 33), bool)
        pred[20:23, 21:24, 22:25], gt[20:23, 21:24, 22:25] = True, True
        pred[2:5, 2:5, 2:5] = True
        result = self.assert_original_equal(pred, gt)
        self.assertGreater(result["hd95_mm"], 15.0)
        self.assertLess(result["dice"], 1.0)
        # A GT-only box would falsely remove the distant false positive.
        wrong_roi = (slice(19, 24), slice(20, 25), slice(21, 26))
        self.assertNotEqual(result, binary_metrics_mm(pred[wrong_roi], gt[wrong_roi],
                                                     (3.0, 0.7, 0.9)))

    def test_thin_volumes_and_disjoint_masks_exact(self):
        for shape in [(1, 1, 7), (1, 8, 9), (8, 1, 9), (8, 9, 1)]:
            pred, gt = np.zeros(shape, bool), np.zeros(shape, bool)
            pred[(0, 0, 0)] = True
            gt[tuple(n-1 for n in shape)] = True
            with self.subTest(shape=shape):
                self.assert_original_equal(pred, gt)

    def test_full_foreground_and_sparse_target_exact(self):
        shape = (8, 10, 12)
        pred = np.ones(shape, bool)
        gt = np.zeros(shape, bool)
        gt[3:5, 4:6, 5:7] = True
        self.assert_original_equal(pred, gt)
        self.assert_original_equal(gt, pred)

    def test_random_dense_and_sparse_masks_exact_with_anisotropy(self):
        rng = np.random.default_rng(76320)
        for shape in [(3, 5, 7), (8, 10, 12), (15, 17, 19)]:
            for threshold in [0.08, 0.5, 0.92]:
                pred = rng.random(shape) < threshold
                gt = rng.random(shape) < threshold
                for spacing in [(1.0, 1.0, 1.0), (5.0, 0.728515625, 0.9765625)]:
                    with self.subTest(shape=shape, threshold=threshold, spacing=spacing):
                        self.assert_original_equal(pred, gt, spacing)

    def test_numeric_binary_cast_matches_original(self):
        pred, gt = np.zeros((9, 11, 13), np.uint8), np.zeros((9, 11, 13), np.uint8)
        pred[2:5, 3:6, 4:7] = 7
        gt[3:6, 4:7, 5:8] = 9
        self.assert_original_equal(pred, gt)

    def test_non_contiguous_views_match_original(self):
        pred, gt = np.zeros((12, 14, 16), bool), np.zeros((12, 14, 16), bool)
        pred[4:8, 4:8, 4:8], gt[5:9, 4:8, 4:8] = True, True
        for p, g in [(pred[::-1], gt[::-1]), (pred[:, ::2, :], gt[:, ::2, :]),
                     (pred.transpose(2, 1, 0), gt.transpose(2, 1, 0))]:
            self.assert_original_equal(p, g)

    def test_invalid_shapes_fail(self):
        for pred, gt in [(np.zeros((3, 4)), np.zeros((3, 4))),
                         (np.zeros((3, 4, 5)), np.zeros((3, 4, 6)))]:
            with self.subTest(shapes=(pred.shape, gt.shape)):
                with self.assertRaises(ValueError):
                    binary_metrics_mm_roi(pred, gt, (1.0, 1.0, 1.0))
                with self.assertRaises(ValueError):
                    union_roi_slices(pred, gt)

    def test_invalid_spacing_fails_even_for_empty_masks(self):
        pred = np.zeros((3, 4, 5), bool)
        for spacing in [(1.0, 1.0), (0.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
                        (float("nan"), 1.0, 1.0), (float("inf"), 1.0, 1.0)]:
            with self.subTest(spacing=spacing):
                with self.assertRaises(ValueError):
                    binary_metrics_mm_roi(pred, pred.copy(), spacing)

    def test_invalid_halo_fails(self):
        pred = np.zeros((3, 4, 5), bool)
        for halo in [0, -1, 0.5, True, None]:
            with self.subTest(halo=halo):
                with self.assertRaises(ValueError):
                    union_roi_slices(pred, pred.copy(), halo=halo)


if __name__ == "__main__":
    unittest.main()
