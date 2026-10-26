"""Fail-closed tests of the pure ROI readiness report validator; no real state IO."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.finalize_synapse_roi_readiness as readiness_module
from scripts.finalize_synapse_roi_readiness import validate_reports


def valid_reports():
    cases = ["synthetic_case_%02d" % index for index in range(12)]
    cpu = {"cases": cases.copy(), "repeats": 2, "num_cpu_workers": 1,
           "gpu_used": False, "comparisons": [], "compared_case_classes": 192,
           "median_seconds": {"original": 12.0, "roi": 6.0}, "speedup": 2.0}
    for repeat in range(2):
        for case in cases:
            original = {"cached_case_total_seconds": 1.0, "all_organs_seconds": 0.8,
                        "organs": [
                            {"label_id": label,
                             "metrics": {"dice": 0.75, "hd95_mm": 2.5,
                                         "hd95_voxel": 1.25, "status": "nonempty"}}
                            for label in range(1, 9)]}
            roi = deepcopy(original)
            roi["cached_case_total_seconds"] = 0.5
            roi["all_organs_seconds"] = 0.3
            cpu["comparisons"].append({"repeat": repeat, "case": case,
                                       "exact": True, "original": original, "roi": roi})
    gpu = {"cases": [
        {"case": case, "tta_views": 8, "tta_scheme": "scale_hflip",
         "dice_per_class": [0.75] * 8, "hd95_mm_per_class": [2.5] * 8,
         "hd95_voxel_per_class": [1.25] * 8} for case in cases],
        "precision": "16-mixed", "compile_test": True, "compiled_forward_observed": True,
        "tta_views": 8, "num_cases": 12, "seconds": 3.0}
    fit = {"attempted_train_batches": 3, "finite_losses": [11.0, 10.0, 9.0],
           "batch_size": 16, "num_workers": 8, "img_size": [384, 384], "patch_size": 16,
           "precision": "16-mixed", "compile_disabled": False,
           "evaluation_cases": 12, "seconds": 1.0}
    return cpu, gpu, fit, cases


class ReadinessValidationTests(unittest.TestCase):
    def assert_rejected(self, change):
        cpu, gpu, fit, cases = valid_reports()
        change(cpu, gpu, fit, cases)
        with self.assertRaises((RuntimeError, KeyError, TypeError, ValueError)):
            validate_reports(cpu, gpu, fit, cases)

    def test_valid_complete_reports_pass_without_mutation_or_output(self):
        cpu, gpu, fit, cases = valid_reports()
        original = deepcopy((cpu, gpu, fit, cases))
        with patch.object(readiness_module, "write",
                          side_effect=AssertionError("Pure validation must not write")), \
             patch.object(readiness_module, "finalize",
                          side_effect=AssertionError("Pure validation must not finalize")):
            self.assertIsNone(validate_reports(cpu, gpu, fit, cases))
        self.assertEqual((cpu, gpu, fit, cases), original)

    def test_expected_case_count_and_uniqueness_are_required(self):
        self.assert_rejected(lambda c, g, f, e: e.pop())
        self.assert_rejected(lambda c, g, f, e: e.__setitem__(11, e[0]))

    def test_cpu_case_list_order_and_membership_must_match(self):
        self.assert_rejected(lambda c, g, f, e: c["cases"].reverse())
        self.assert_rejected(lambda c, g, f, e: c["cases"].__setitem__(0, "different_case"))

    def test_cpu_requires_two_repeats_and_single_cpu_worker(self):
        for field, value in [("repeats", 1), ("num_cpu_workers", 2), ("gpu_used", True)]:
            with self.subTest(field=field):
                self.assert_rejected(lambda c, g, f, e, k=field, v=value: c.__setitem__(k, v))

    def test_cpu_missing_comparison_is_rejected(self):
        self.assert_rejected(lambda c, g, f, e: c["comparisons"].pop())

    def test_cpu_duplicate_case_with_unchanged_count_is_rejected(self):
        self.assert_rejected(lambda c, g, f, e:
                             c["comparisons"][1].__setitem__("case", c["comparisons"][0]["case"]))

    def test_cpu_missing_case_in_one_repeat_is_rejected(self):
        self.assert_rejected(lambda c, g, f, e:
                             c["comparisons"][0].__setitem__("repeat", 1))

    def test_cpu_false_exact_flag_is_rejected(self):
        self.assert_rejected(lambda c, g, f, e:
                             c["comparisons"][0].__setitem__("exact", False))

    def test_cpu_missing_or_duplicate_organ_is_rejected(self):
        self.assert_rejected(lambda c, g, f, e:
                             c["comparisons"][0]["roi"]["organs"].pop())
        self.assert_rejected(lambda c, g, f, e:
                             c["comparisons"][0]["roi"]["organs"][7].__setitem__("label_id", 1))

    def test_cpu_unequal_actual_metric_values_are_rejected_despite_exact_flag(self):
        for key, value in [("dice", 0.76), ("hd95_mm", 2.6), ("hd95_voxel", 1.3),
                           ("status", "empty_gt_legacy_zero")]:
            with self.subTest(key=key):
                self.assert_rejected(lambda c, g, f, e, k=key, v=value:
                                     c["comparisons"][0]["roi"]["organs"][0]["metrics"].__setitem__(k, v))

    def test_cpu_nonfinite_negative_or_out_of_range_metrics_are_rejected(self):
        for key, value in [("dice", float("nan")), ("hd95_mm", float("inf")),
                           ("hd95_voxel", -1.0), ("dice", 1.01), ("dice", -0.01)]:
            with self.subTest(key=key, value=value):
                self.assert_rejected(lambda c, g, f, e, k=key, v=value:
                                     c["comparisons"][0]["original"]["organs"][0]["metrics"].__setitem__(k, v))

    def test_cpu_unknown_metric_status_is_rejected(self):
        self.assert_rejected(lambda c, g, f, e:
                             c["comparisons"][0]["original"]["organs"][0]["metrics"].__setitem__("status", "unknown"))

    def test_cpu_nonfinite_nonpositive_or_boolean_timings_are_rejected(self):
        for mode in ["original", "roi"]:
            for key in ["cached_case_total_seconds", "all_organs_seconds"]:
                for value in [float("nan"), float("inf"), 0.0, -0.1, True]:
                    with self.subTest(mode=mode, key=key, value=value):
                        self.assert_rejected(lambda c, g, f, e, m=mode, k=key, v=value:
                                             c["comparisons"][0][m].__setitem__(k, v))

    def test_cpu_forged_aggregate_speedup_or_comparison_count_is_rejected(self):
        for field, value in [("median_seconds", {"original": 120.0, "roi": 6.0}),
                             ("speedup", 3.0), ("speedup", float("nan")),
                             ("compared_case_classes", 191)]:
            with self.subTest(field=field, value=value):
                self.assert_rejected(lambda c, g, f, e, k=field, v=value: c.__setitem__(k, v))

    def test_cpu_no_measured_timing_benefit_is_rejected(self):
        def change(cpu, gpu, fit, cases):
            for item in cpu["comparisons"]:
                item["roi"]["cached_case_total_seconds"] = 1.0
            cpu["median_seconds"] = {"original": 12.0, "roi": 12.0}
            cpu["speedup"] = 1.0
        self.assert_rejected(change)

    def test_gpu_missing_duplicate_or_out_of_order_cases_are_rejected(self):
        self.assert_rejected(lambda c, g, f, e: g["cases"].pop())
        self.assert_rejected(lambda c, g, f, e:
                             g["cases"][1].__setitem__("case", g["cases"][0]["case"]))
        self.assert_rejected(lambda c, g, f, e: g["cases"].reverse())

    def test_gpu_precision_compile_and_tta_contract_are_required(self):
        for key, value in [("precision", "32-true"), ("compile_test", False),
                           ("compiled_forward_observed", False), ("tta_views", 4),
                           ("num_cases", 11)]:
            with self.subTest(key=key):
                self.assert_rejected(lambda c, g, f, e, k=key, v=value: g.__setitem__(k, v))
        self.assert_rejected(lambda c, g, f, e:
                             g["cases"][0].__setitem__("tta_views", 4))
        self.assert_rejected(lambda c, g, f, e:
                             g["cases"][0].__setitem__("tta_scheme", "mirror"))

    def test_gpu_incomplete_nonfinite_or_invalid_metrics_are_rejected(self):
        for key, value in [("dice_per_class", [0.75] * 7),
                           ("hd95_mm_per_class", [float("nan")] + [2.5] * 7),
                           ("hd95_voxel_per_class", [float("inf")] + [1.25] * 7),
                           ("hd95_mm_per_class", [-1.0] + [2.5] * 7),
                           ("dice_per_class", [1.01] + [0.75] * 7)]:
            with self.subTest(key=key, first=value[0]):
                self.assert_rejected(lambda c, g, f, e, k=key, v=value: g["cases"][0].__setitem__(k, v))

    def test_gpu_and_fit_total_times_must_be_finite_positive(self):
        for target in ["gpu", "fit"]:
            for value in [float("nan"), float("inf"), 0.0, -1.0, True]:
                with self.subTest(target=target, value=value):
                    self.assert_rejected(lambda c, g, f, e, t=target, v=value:
                                         (g if t == "gpu" else f).__setitem__("seconds", v))

    def test_fit_requires_three_finite_losses_and_three_attempts(self):
        for key, value in [("attempted_train_batches", 2), ("finite_losses", [11.0, 10.0]),
                           ("finite_losses", [11.0, float("nan"), 9.0]),
                           ("finite_losses", [11.0, float("inf"), 9.0])]:
            with self.subTest(key=key, value=value):
                self.assert_rejected(lambda c, g, f, e, k=key, v=value: f.__setitem__(k, v))

    def test_fit_batch_workers_size_precision_compile_and_cases_are_required(self):
        for key, value in [("batch_size", 8), ("num_workers", 0), ("img_size", [224, 224]),
                           ("patch_size", 14), ("precision", "32-true"),
                           ("compile_disabled", True), ("evaluation_cases", 11)]:
            with self.subTest(key=key, value=value):
                self.assert_rejected(lambda c, g, f, e, k=key, v=value: f.__setitem__(k, v))


if __name__ == "__main__":
    unittest.main()
