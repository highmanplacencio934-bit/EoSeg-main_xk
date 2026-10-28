"""CPU small-model checks for instance-scoped Synapse ROI evaluation."""
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.synapse_physical_geometry import binary_metrics_mm, label_fingerprint
from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.synapse_roi_evaluation import roi_metrics_scope
import training.synapse_roi_evaluation as roi_module
import training.synapse_multiscale_tta as tta_module


BINDINGS = ("calculate_metric_percase", "evaluate_synapse_case", "tta_views",
            "tta_scheme", "_synapse_roi_scope_active", "roi_evaluation_contract",
            "roi_evaluation_case_timings")
ABSENT = object()


class FixtureGeometry:
    def __init__(self, label):
        self.expected = label.copy()
        self.calls = []

    def for_case(self, case, label):
        self.calls.append((case, label.shape, label_fingerprint(label)))
        if case != "case_fixture" or not np.array_equal(label, self.expected):
            raise ValueError("Full case label differs from verified fixture")
        return SimpleNamespace(spacing_zyx_mm=(3.0, 0.7, 0.9),
                               label_sha256=label_fingerprint(label))


class TinyPhysical(SynapsePhysicalHD95Semantic):
    """Use the actual inherited 1-view evaluator with a deterministic CPU head."""
    def __init__(self, label):
        nn.Module.__init__(self)
        self.compute_hd95_voxel = False
        self.num_classes = 9
        self.img_size = (8, 8)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.physical_geometry = FixtureGeometry(label)
        self._physical_case_context = None
        self.physical_checkpoint_epoch = 7
        self.physical_checkpoint_global_step = 42
        self.forward_calls = 0
        self.last_inp = None

    def forward(self, image):
        self.forward_calls += 1
        self.last_inp = image
        return [torch.zeros((1, 1, 8, 8))], [torch.zeros((1, 1, 10))]

    def to_per_pixel_logits_semantic(self, masks, classes):
        hard = self.last_inp[:, 0].round().long().clamp(0, 8)
        return F.one_hot(hard, num_classes=9).permute(0, 3, 1, 2).float()


def fixture():
    prediction = (np.arange(128).reshape(2, 8, 8) % 9).astype(np.uint8)
    label = np.roll(prediction, 1, axis=2).copy()
    batch = {"case_name": "case_fixture", "image": torch.tensor(prediction).float(),
             "label": torch.tensor(label).long()}
    return TinyPhysical(label), batch, prediction, label


class ROIEvaluationTests(unittest.TestCase):
    def snapshot(self, model):
        return {key: model.__dict__.get(key, ABSENT) for key in BINDINGS}

    def assert_restored(self, model, original):
        for key, value in original.items():
            with self.subTest(binding=key):
                if value is ABSENT:
                    self.assertNotIn(key, model.__dict__)
                else:
                    self.assertIs(model.__dict__[key], value)

    def test_one_view_result_matches_actual_original_physical_evaluator(self):
        model, batch, _, _ = fixture()
        original = model.evaluate_synapse_case(batch)
        previous_calls = model.forward_calls
        before = self.snapshot(model)
        with roi_metrics_scope(model) as returned:
            self.assertIs(returned, model)
            actual = model.evaluate_synapse_case(batch)
            self.assertEqual(actual, original)
            self.assertEqual(model.forward_calls - previous_calls, 2)
            self.assertTrue(model.roi_evaluation_contract["cpu_workers"] == 1)
            self.assertIsNone(model._physical_case_context)
        self.assert_restored(model, before)

    def test_direct_physical_record_dictionary_matches_original(self):
        model, _, _, _ = fixture()
        pred, gt = np.zeros((7, 9, 11), bool), np.zeros((7, 9, 11), bool)
        pred[2:4, 3:5, 4:6], gt[3:5, 3:5, 4:6] = True, True
        records = []
        with roi_metrics_scope(model):
            model._physical_case_context = ((3.0, 0.7, 0.9), records)
            actual_pair = model.calculate_metric_percase(pred, gt)
        expected = binary_metrics_mm(pred, gt, (3.0, 0.7, 0.9))
        self.assertEqual(records, [expected])
        self.assertEqual(actual_pair, (expected["dice"], expected["hd95_mm"]))

    def test_scope_restores_class_methods_after_body_exception(self):
        model, _, _, _ = fixture()
        before = self.snapshot(model)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with roi_metrics_scope(model):
                raise RuntimeError("deliberate body failure")
        self.assert_restored(model, before)

    def test_scope_restores_existing_instance_bindings_and_attributes(self):
        model, batch, _, _ = fixture()
        original_eval = model.evaluate_synapse_case
        model.evaluate_synapse_case = original_eval
        model.calculate_metric_percase = model.calculate_metric_percase
        model.tta_views = 99
        model.tta_scheme = "previous"
        model._synapse_roi_scope_active = False
        model.roi_evaluation_contract = {"old": True}
        model.roi_evaluation_case_timings = ["old"]
        before = self.snapshot(model)
        with roi_metrics_scope(model):
            model.evaluate_synapse_case(batch)
        self.assert_restored(model, before)

    def test_other_model_instance_methods_and_metric_are_untouched(self):
        first, batch, _, _ = fixture()
        second, _, _, _ = fixture()
        before = self.snapshot(second)
        original_func = second.calculate_metric_percase.__func__
        with roi_metrics_scope(first):
            first.evaluate_synapse_case(batch)
            self.assertIs(second.calculate_metric_percase.__func__, original_func)
            self.assert_restored(second, before)
        self.assert_restored(second, before)

    def test_nested_scope_is_rejected_without_breaking_outer_scope(self):
        model, batch, _, _ = fixture()
        before = self.snapshot(model)
        with roi_metrics_scope(model):
            with self.assertRaises(RuntimeError):
                with roi_metrics_scope(model):
                    pass
            self.assertTrue(model._synapse_roi_scope_active)
            model.evaluate_synapse_case(batch)
        self.assert_restored(model, before)

    def test_metric_requires_verified_context(self):
        model, _, _, _ = fixture()
        pred = np.zeros((3, 4, 5), bool)
        with roi_metrics_scope(model):
            with self.assertRaises(RuntimeError):
                model.calculate_metric_percase(pred, pred)

    def test_capture_reconstructs_actual_prediction_and_full_gt(self):
        model, batch, prediction, gt = fixture()
        with tempfile.TemporaryDirectory() as directory:
            with roi_metrics_scope(model, capture_dir=directory):
                actual = model.evaluate_synapse_case(batch)
                self.assertEqual(model.forward_calls, 2)
                self.assertEqual(len(model.roi_evaluation_case_timings), 1)
            npz = Path(directory) / "case_fixture.npz"
            with np.load(npz) as stored:
                np.testing.assert_array_equal(stored["prediction"], prediction)
                np.testing.assert_array_equal(stored["gt"], gt)
            metadata = json.loads((Path(directory)/"case_fixture.json").read_text())
            self.assertEqual(metadata["shape_zyx"], [2, 8, 8])
            self.assertEqual(metadata["label_sha256"], label_fingerprint(gt))
            self.assertEqual(metadata["checkpoint_epoch"], 7)
            self.assertEqual(metadata["checkpoint_global_step"], 42)
            self.assertEqual(metadata["tta_views"], 1)
            self.assertEqual(actual["hd95_unit"], "mm")
            self.assertTrue(all(shape == gt.shape for _, shape, _ in
                                model.physical_geometry.calls))

    def test_capture_refuses_preexisting_prediction_or_metadata(self):
        for extension in ["npz", "json"]:
            with self.subTest(extension=extension):
                model, batch, _, _ = fixture()
                before = self.snapshot(model)
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory)/("case_fixture."+extension)
                    target.write_bytes(b"keep-existing")
                    with self.assertRaises(FileExistsError):
                        with roi_metrics_scope(model, capture_dir=directory):
                            model.evaluate_synapse_case(batch)
                    self.assertEqual(target.read_bytes(), b"keep-existing")
                    self.assertEqual(model.forward_calls, 0)
                self.assert_restored(model, before)

    def test_capture_rejects_changed_full_label_before_forward(self):
        model, batch, _, gt = fixture()
        batch["label"] = batch["label"].clone()
        batch["label"][0, 0, 0] = (int(gt[0, 0, 0])+1) % 9
        before = self.snapshot(model)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Full case"):
                with roi_metrics_scope(model, capture_dir=directory):
                    model.evaluate_synapse_case(batch)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(model.forward_calls, 0)
        self.assert_restored(model, before)

    def test_capture_rejects_wrong_class_gt_mask(self):
        model, batch, _, label = fixture()
        def wrong_evaluation(self, batch):
            self._physical_case_context = ((3.0, 0.7, 0.9), [])
            try:
                return self.calculate_metric_percase(label == 1, label == 2)
            finally:
                self._physical_case_context = None
        model.evaluate_synapse_case = MethodType(wrong_evaluation, model)
        before = self.snapshot(model)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "class label"):
                with roi_metrics_scope(model, capture_dir=directory):
                    model.evaluate_synapse_case(batch)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assert_restored(model, before)

    def test_capture_rejects_overlapping_class_predictions(self):
        model, batch, _, label = fixture()
        def overlap(self, batch):
            self._physical_case_context = ((3.0, 0.7, 0.9), [])
            try:
                shared = np.ones_like(label, dtype=bool)
                self.calculate_metric_percase(shared, label == 1)
                return self.calculate_metric_percase(shared, label == 2)
            finally:
                self._physical_case_context = None
        model.evaluate_synapse_case = MethodType(overlap, model)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "overlap"):
                with roi_metrics_scope(model, capture_dir=directory):
                    model.evaluate_synapse_case(batch)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_capture_rejects_incomplete_class_collection(self):
        model, batch, _, label = fixture()
        def incomplete(self, batch):
            self._physical_case_context = ((3.0, 0.7, 0.9), [])
            try:
                self.calculate_metric_percase(label == 1, label == 1)
                return {}
            finally:
                self._physical_case_context = None
        model.evaluate_synapse_case = MethodType(incomplete, model)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                with roi_metrics_scope(model, capture_dir=directory):
                    model.evaluate_synapse_case(batch)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_rejects_nonphysical_model(self):
        with self.assertRaises(TypeError):
            with roi_metrics_scope(nn.Identity()):
                pass

    def test_rejects_missing_or_invalid_physical_geometry(self):
        for value in [ABSENT, None, object()]:
            with self.subTest(value_type=type(value).__name__):
                model, _, _, _ = fixture()
                if value is ABSENT:
                    del model.physical_geometry
                else:
                    model.physical_geometry = value
                with self.assertRaises((TypeError, ValueError)):
                    with roi_metrics_scope(model):
                        pass

    def test_rejects_teacher_subclass(self):
        TinyTeacher = type("TinyTeacher", (TinyPhysical,),
                           {"__module__": "training.synapse_teacher_semantic"})
        _, _, _, label = fixture()
        teacher = TinyTeacher(label)
        with self.assertRaises(TypeError):
            with roi_metrics_scope(teacher):
                pass

    def test_rejects_invalid_views_and_tta_capture(self):
        model, _, _, _ = fixture()
        for views in [0, 3, 7, 9, True, None]:
            with self.subTest(views=views):
                with self.assertRaises(ValueError):
                    with roi_metrics_scope(model, tta_views=views):
                        pass
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                with roi_metrics_scope(model, tta_views=8, capture_dir=directory):
                    pass

    def test_tta_clone_keeps_original_code_and_module_globals(self):
        model, batch, _, _ = fixture()
        original = tta_module.evaluate_multiscale_case
        original_code = original.__code__
        original_metric = tta_module.binary_metrics_mm
        original_predict = tta_module.predict_multiscale_tta_scores
        clones = []
        def capture_clone(*args, **kwargs):
            clone = type(original)(*args, **kwargs)
            clones.append(clone)
            return clone
        before = self.snapshot(model)
        expected = original(model, batch, 8)
        with patch.object(roi_module, "FunctionType", side_effect=capture_clone):
            with roi_metrics_scope(model, tta_views=8):
                actual = model.evaluate_synapse_case(batch)
                actual.pop("tta_case_seconds")
                actual.pop("tta_views")
                actual.pop("tta_scheme")
                self.assertEqual(actual, expected)
                self.assertEqual(model.tta_views, 8)
        self.assert_restored(model, before)
        self.assertEqual(len(clones), 1)
        clone = clones[0]
        self.assertIs(clone.__code__, original_code)
        self.assertIsNot(clone.__globals__, original.__globals__)
        for key, value in original.__globals__.items():
            if key != "binary_metrics_mm":
                self.assertIs(clone.__globals__[key], value)
        self.assertIs(clone.__globals__["binary_metrics_mm"], roi_module.binary_metrics_mm_roi)
        self.assertIs(tta_module.binary_metrics_mm, original_metric)
        self.assertIs(tta_module.predict_multiscale_tta_scores, original_predict)
        self.assertIs(original.__code__, original_code)

    def test_tta_setup_failure_restores_all_instance_bindings(self):
        model, _, _, _ = fixture()
        before = self.snapshot(model)
        with patch.object(roi_module, "FunctionType", side_effect=RuntimeError("clone failed")):
            with self.assertRaisesRegex(RuntimeError, "clone failed"):
                with roi_metrics_scope(model, tta_views=8):
                    pass
        self.assert_restored(model, before)


if __name__ == "__main__":
    unittest.main()
