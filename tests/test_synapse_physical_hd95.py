"""Physical geometry/HD95 tests on independent CPU fixtures, never real training."""
import csv
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
import torch
from torch import nn
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.synapse_physical_geometry import SpacingManifest, binary_metrics_mm, label_fingerprint
from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.synapse_physical_selection import SynapsePhysicalHD95Selection
from training.mask_classification_semantic import MaskClassificationSemantic
from scripts.build_synapse_spacing import match_label_geometry, require_nifti_mm


def make_manifest(path, labels, spacing=(3.0, 2.0, 0.5)):
    payload = {"schema_version": 1, "units": "mm", "array_axis_order": "zyx",
               "verified": True, "orthogonal_grid": True, "cases": {}}
    for case, label in labels.items():
        payload["cases"][case] = {"spacing_zyx_mm": list(spacing), "shape_zyx": list(label.shape),
                                  "label_sha256": label_fingerprint(label),
                                  "source": "SYNTHETIC_TEST_FIXTURE_NOT_REAL_METADATA",
                                  "mapping": "synthetic identity grid"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class GeometryAndFormulaTests(unittest.TestCase):
    def test_axis_specific_mm_distance_and_unchanged_dice(self):
        for axis, scale in enumerate((3.0, 2.0, 0.5)):
            pred = np.zeros((9, 9, 9), bool)
            gt = pred.copy()
            point = [4, 4, 4]
            gt[tuple(point)] = True
            point[axis] += 2
            pred[tuple(point)] = True
            result = binary_metrics_mm(pred, gt, (3.0, 2.0, 0.5))
            self.assertEqual(result["hd95_mm"], 2 * scale)
            self.assertEqual(result["hd95_voxel"], 2)
            self.assertEqual(result["dice"], MaskClassificationSemantic.calculate_metric_percase(pred, gt)[0])

    def test_anisotropic_hd95_matches_independent_physical_coordinate_tree(self):
        rng = np.random.default_rng(58)
        spacing = np.array([3.25, 0.61, 0.92])
        structure = generate_binary_structure(3, 1)
        for _ in range(4):
            pred = rng.random((10, 11, 12)) > 0.9
            gt = rng.random(pred.shape) > 0.91
            p = np.argwhere(pred ^ binary_erosion(pred, structure=structure)) * spacing
            g = np.argwhere(gt ^ binary_erosion(gt, structure=structure)) * spacing
            direct = np.percentile(np.r_[cKDTree(g).query(p)[0], cKDTree(p).query(g)[0]], 95)
            self.assertAlmostEqual(binary_metrics_mm(pred, gt, list(spacing))["hd95_mm"], direct, places=12)

    def test_unit_spacing_exactly_matches_baseline(self):
        pred = np.zeros((9, 10, 11), bool)
        pred[1:6, 2:7, 3:8] = True
        gt = np.roll(pred, 2, axis=0)
        old = MaskClassificationSemantic.calculate_metric_percase(pred, gt)
        new = binary_metrics_mm(pred, gt, (1, 1, 1))
        self.assertEqual((new["dice"], new["hd95_mm"]), old)

    def test_empty_policy_preserved_and_explicitly_flagged(self):
        for a, b in [(False, False), (False, True), (True, False)]:
            pred = np.full((2, 3, 4), a, bool)
            gt = np.full(pred.shape, b, bool)
            result = binary_metrics_mm(pred, gt, (3, 2, 1))
            self.assertEqual((result["dice"], result["hd95_mm"]),
                             MaskClassificationSemantic.calculate_metric_percase(pred, gt))
            self.assertIn("legacy_zero", result["status"])

    def test_invalid_spacing_and_shape_rejected(self):
        pred = np.ones((2, 3, 4), bool)
        for spacing in [(0, 1, 1), (-1, 1, 1), (float("nan"), 1, 1),
                        (float("inf"), 1, 1), (None, 1, 1), (True, 1, 1), (1, 1)]:
            with self.assertRaises(ValueError):
                binary_metrics_mm(pred, pred, spacing)
        with self.assertRaises(ValueError):
            binary_metrics_mm(pred, pred[0], (1, 1, 1))

    def test_manifest_requires_verified_units_and_exact_case_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"geometry.json"
            labels = {"case0001": np.zeros((3, 4, 5), np.uint8)}
            labels["case0001"][1, 2, 3] = 5
            data = make_manifest(path, labels)
            manifest = SpacingManifest(path)
            self.assertEqual(manifest.for_case("case0001", labels["case0001"]).spacing_zyx_mm, (3, 2, .5))
            changed = labels["case0001"].copy(); changed[0, 0, 0] = 1
            with self.assertRaises(ValueError): manifest.for_case("case0001", changed)
            with self.assertRaises(ValueError): manifest.for_case("case0002", labels["case0001"])
            data["verified"] = False; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): SpacingManifest(path)
            data["verified"] = True; data["units"] = "voxel"; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): SpacingManifest(path)
            with self.assertRaises(FileNotFoundError): SpacingManifest(Path(tmp)/"missing.json")

    def test_full_label_mapping_recovers_axis_spacing_and_flips(self):
        raw = np.zeros((4, 5, 6), np.uint8)
        raw[1, 2, 3] = 1; raw[0, 1, 4] = 5; raw[3, 3, 1] = 8
        target = raw.transpose(2, 0, 1)[::-1, :, ::-1]
        spacing, transforms = match_label_geometry(raw, target, (3, 2, .5))
        self.assertEqual(spacing, [.5, 3, 2])
        self.assertTrue(transforms)
        target = target.copy(); target[0, 0, 0] = 2
        with self.assertRaises(ValueError): match_label_geometry(raw, target, (3, 2, .5))

    def test_nifti_requires_explicit_mm_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"label.nii.gz"
            header = bytearray(348)
            header[:4] = struct.pack("<i", 348); header[344:348] = b"n+1\x00"; header[123] = 2
            with gzip.open(path, "wb") as f: f.write(header)
            require_nifti_mm(path)
            header[123] = 0
            with gzip.open(path, "wb") as f: f.write(header)
            with self.assertRaises(ValueError): require_nifti_mm(path)

    def test_nifti_builder_end_to_end_and_validate_only(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root/"raw"; raw_dir.mkdir()
            h5_dir = root/"h5"; h5_dir.mkdir()
            raw = np.zeros((4, 5, 6), np.uint8)
            raw[1, 2, 3] = 8; raw[0, 1, 4] = 6; raw[3, 3, 1] = 7
            remapped = np.zeros(raw.shape, np.uint8)
            remapped[raw == 8] = 1; remapped[raw == 6] = 5; remapped[raw == 7] = 8
            target = remapped.transpose(2, 0, 1)[::-1, :, ::-1]
            for case in ("case0001", "case0002"):
                image = sitk.GetImageFromArray(raw); image.SetSpacing((.5, 2, 3))
                sitk.WriteImage(image, str(raw_dir/f"label{case[4:]}.nii.gz"))
                with h5py.File(h5_dir/f"{case}.npy.h5", "w") as f:
                    f["image"] = np.zeros(target.shape, np.float32); f["label"] = target
            (root/"test.txt").write_text("case0001\ncase0002\n")
            command = [sys.executable, str(ROOT/"scripts/build_synapse_spacing.py"),
                       "--raw-label-dir", str(raw_dir), "--test-base-dir", str(h5_dir),
                       "--list-file", str(root/"test.txt"), "--output", str(root/"geometry.json")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(SpacingManifest(root/"geometry.json").for_case("case0001", target).spacing_zyx_mm, (.5, 3, 2))
            validation = [sys.executable, str(ROOT/"scripts/build_synapse_spacing.py"),
                          "--validate-only", "--test-base-dir", str(h5_dir), "--list-file", str(root/"test.txt"),
                          "--output", str(root/"geometry.json")]
            result = subprocess.run(validation, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_physical_config_changes_only_evaluation_interface_and_output(self):
        base = yaml.safe_load((ROOT/"configs/synapse/vit_query_mul_scale_fusion.yaml").read_text())
        new = yaml.safe_load((ROOT/"configs/synapse/vit_query_mul_scale_fusion_hd95_mm.yaml").read_text())
        new["model"]["class_path"] = base["model"]["class_path"]
        new["model"]["init_args"].pop("spacing_manifest")
        new["trainer"]["callbacks"][0]["class_path"] = base["trainer"]["callbacks"][0]["class_path"]
        new["trainer"]["callbacks"][0]["init_args"].pop("output_dir")
        self.assertEqual(new, base)


class TinyNetwork(nn.Module):
    num_blocks = 3
    masked_attn_enabled = False

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        masks = images.new_full((images.shape[0], 9, 8, 8), -20)
        masks[:, 0] = 0
        masks[:, 1, 3, 2] = 20
        classes = images.new_full((images.shape[0], 9, 10), -20)
        for c in range(9): classes[:, c, c] = 20
        return [masks], [classes]


class PhysicalInferenceTests(unittest.TestCase):
    def _kwargs(self):
        return dict(network=TinyNetwork(), img_size=(8, 8), num_classes=9, attn_mask_annealing_enabled=False)

    def test_original_inference_dice_and_state_keys_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = np.zeros((2, 8, 8), np.uint8); label[:, 2, 2] = 1
            path = Path(tmp)/"spacing.json"; make_manifest(path, {"case0001": label})
            model = SynapsePhysicalHD95Semantic(spacing_manifest=str(path), **self._kwargs())
            baseline = MaskClassificationSemantic(**self._kwargs())
            model.eval(); baseline.eval()
            batch = {"case_name": "case0001", "image": torch.zeros((2, 8, 8)), "label": torch.from_numpy(label)}
            old = baseline.evaluate_synapse_case(batch)
            result = model.evaluate_synapse_case(batch)
            self.assertEqual(result["dice_per_class"], old["dice_per_class"])
            self.assertEqual(result["hd95_voxel_per_class"], old["hd95_per_class"])
            self.assertEqual(result["hd95_mm_per_class"][0], 2)
            self.assertEqual(old["hd95_per_class"][0], 1)
            self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
            self.assertIsNone(model._physical_case_context)
            for key in baseline.state_dict(): self.assertTrue(torch.equal(model.state_dict()[key], baseline.state_dict()[key]))

    def test_case_context_restored_after_inference_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = np.zeros((2, 8, 8), np.uint8); label[:, 2, 2] = 1
            path = Path(tmp)/"spacing.json"; make_manifest(path, {"case0001": label})
            model = SynapsePhysicalHD95Semantic(spacing_manifest=str(path), **self._kwargs())
            def broken(*args): raise RuntimeError("intentional fixture error")
            model.forward = broken
            batch = {"case_name": "case0001", "image": torch.zeros(label.shape), "label": torch.from_numpy(label)}
            with self.assertRaisesRegex(RuntimeError, "intentional fixture"):
                model.evaluate_synapse_case(batch)
            self.assertIsNone(model._physical_case_context)

    def test_fit_selection_keeps_dice_policy_and_exports_mm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label = np.zeros((2, 8, 8), np.uint8); label[:, 2, 2] = 1
            path = root/"spacing.json"; make_manifest(path, {"case0001": label})
            (root/"test_vol.txt").write_text("case0001\n")
            (root/"train.txt").write_text("case0003_slice001\n")
            with h5py.File(root/"case0001.npy.h5", "w") as f:
                f["image"] = np.zeros(label.shape, np.float32); f["label"] = label
            class Data(LightningDataModule):
                test_base_dir = str(root)
                list_dir = str(root)
                test_split = "test_vol"
                train_split = "train"
                def train_dataloader(self): return DataLoader(torch.ones((1, 1)), batch_size=1)
                def test_dataloader(self):
                    return DataLoader([{"case_name": "case0001", "image": torch.zeros(label.shape),
                                        "label": torch.from_numpy(label)}], batch_size=None)
            class Model(SynapsePhysicalHD95Semantic):
                def training_step(self, batch, batch_idx): return self.network.weight.square()
                def configure_optimizers(self): return torch.optim.SGD(self.parameters(), lr=.01)
            model = Model(spacing_manifest=str(path), **self._kwargs())
            callback = SynapsePhysicalHD95Selection(last_n_epochs=2, expected_test_cases=1, output_dir=str(root/"out"))
            trainer = Trainer(accelerator="cpu", devices=1, precision="32-true", max_epochs=3, logger=False,
                              enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0,
                              limit_val_batches=0, enable_model_summary=False, callbacks=[callback])
            trainer.fit(model, datamodule=Data())
            self.assertEqual(callback.best_epoch, 2)
            with (callback.run_dir/"test_epoch_metrics.csv").open() as f: rows = list(csv.DictReader(f))
            self.assertEqual([row["epoch"] for row in rows], ["2", "3"])
            self.assertEqual(float(rows[0]["hd95_class_1"]), 2)
            self.assertEqual(sorted(p.name for p in callback.run_dir.glob("*.ckpt")), ["best_test.ckpt", "last.ckpt"])
            with (callback.run_dir/"physical_epoch_organ_metrics.csv").open(encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 16)

    def test_two_case_lightning_test_exports_names_and_both_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label = np.zeros((2, 8, 8), np.uint8); label[:, 2, 2] = 1
            path = root/"spacing.json"; make_manifest(path, {"case0001": label, "case0002": label})
            (root/"test_vol.txt").write_text("case0001\ncase0002\n")
            batches = []
            for case in ("case0001", "case0002"):
                with h5py.File(root/f"{case}.npy.h5", "w") as f:
                    f["image"] = np.zeros(label.shape, np.float32); f["label"] = label
                batches.append({"case_name": case, "image": torch.zeros(label.shape), "label": torch.from_numpy(label)})
            class Data(LightningDataModule):
                test_base_dir = str(root)
                list_dir = str(root)
                test_split = "test_vol"
                def test_dataloader(self): return DataLoader(batches, batch_size=None)
            model = SynapsePhysicalHD95Semantic(spacing_manifest=str(path), **self._kwargs())
            callback = SynapsePhysicalHD95Selection(expected_test_cases=2, output_dir=str(root/"out"))
            trainer = Trainer(accelerator="cpu", devices=1, precision="32-true", logger=False,
                              enable_checkpointing=False, enable_progress_bar=False,
                              enable_model_summary=False, callbacks=[callback])
            import lightning
            checkpoint = root/"fixture.ckpt"
            torch.save({"state_dict": model.state_dict(), "epoch": 176, "global_step": 42,
                        "pytorch-lightning_version": lightning.__version__}, checkpoint)
            trainer.test(model, datamodule=Data(), ckpt_path=checkpoint, verbose=False)
            with (callback.run_dir/"physical_case_organ_metrics.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)
            first = rows[0]
            self.assertEqual(first["organ"], "Aorta")
            self.assertEqual(first["epoch"], "177")
            self.assertEqual(float(first["hd95_mm"]), 2)
            self.assertEqual(float(first["hd95_voxel"]), 1)
            self.assertEqual(json.loads((callback.run_dir/"metric_units.json").read_text())["hd95_unit"], "mm")
            summary = json.loads((callback.run_dir/"physical_test_summary.json").read_text())
            self.assertEqual(summary["num_cases"], 2)
            self.assertEqual(summary["checkpoint_epoch"], 177)
            self.assertEqual(summary["checkpoint_global_step"], 42)
            self.assertEqual(len(summary["checkpoint_sha256"]), 64)
            self.assertEqual(summary["mean_hd95_mm"], .25)


if __name__ == "__main__":
    unittest.main()
