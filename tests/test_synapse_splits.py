"""Run with python -m unittest discover -s tests -p test_synapse_splits.py."""
import importlib.util

HAS_TRAINING_DEPS = all(importlib.util.find_spec(name) is not None
                        for name in ("torch", "torchvision", "lightning"))
from pathlib import Path
import tempfile
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "datasets/synapse_splits.py"
sys.path.insert(0, str(SOURCE.parents[1]))
spec = importlib.util.spec_from_file_location("synapse_splits", SOURCE)
splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splits)


class SynapseSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train_npz"
        self.test = self.root / "test_vol_h5"
        self.lists = self.root / "lists"
        for folder in (self.train, self.test, self.lists):
            folder.mkdir()
        self.write_lists()
        (self.train / "case0001_slice000.npz").touch()
        (self.test / "case0002.npy.h5").touch()

    def write_lists(self, train="case0001_slice000", test="case0002"):
        (self.lists / "train.txt").write_text(train + "\n")
        (self.lists / "test_vol.txt").write_text(test + "\n")

    def check_split(self, **kwargs):
        return splits.validate_synapse_split(self.train, self.test, self.lists, **kwargs)

    def test_valid_fixed_split(self):
        summary = self.check_split()
        self.assertEqual(summary["train_slices"], 1)
        self.assertEqual(summary["train_cases"], ["case0001"])
        self.assertEqual(summary["test_cases"], ["case0002"])

    def test_case_leakage_rejected_before_file_loading(self):
        self.write_lists(train="case0001_slice999", test="case0001")
        with self.assertRaisesRegex(ValueError, "case overlap"):
            self.check_split()

    def test_duplicate_manifest_rejected(self):
        self.write_lists(train="case0001_slice000\ncase0001_slice000")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.check_split()

    def test_missing_file_rejected(self):
        (self.test / "case0002.npy.h5").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "case0002"):
            self.check_split()

    def test_invalid_entry_rejected(self):
        self.write_lists(train="../case0001_slice000")
        with self.assertRaisesRegex(ValueError, "Invalid Synapse entry"):
            self.check_split()

    def test_empty_split_rejected(self):
        self.write_lists(train="")
        with self.assertRaisesRegex(ValueError, "Empty"):
            self.check_split()

    def remove_train_data(self):
        (self.train / "case0001_slice000.npz").unlink()
        self.train.rmdir()

    def remove_test_data(self):
        (self.test / "case0002.npy.h5").unlink()
        self.test.rmdir()

    def test_test_files_without_train_directory(self):
        self.remove_train_data()
        self.assertEqual(self.check_split(check_train_files=False)["test_cases"],
                         ["case0002"])

    def test_train_files_without_test_directory(self):
        self.remove_test_data()
        self.assertEqual(self.check_split(check_test_files=False)["train_slices"], 1)

    def test_active_test_files_still_required(self):
        (self.test / "case0002.npy.h5").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "case0002"):
            self.check_split(check_train_files=False)

    def test_active_train_files_still_required(self):
        (self.train / "case0001_slice000.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "case0001_slice000"):
            self.check_split(check_test_files=False)

    def test_overlap_still_checked_without_inactive_files(self):
        self.remove_train_data()
        self.write_lists(test="case0001")
        with self.assertRaisesRegex(ValueError, "case overlap"):
            self.check_split(check_train_files=False)

    @unittest.skipUnless(HAS_TRAINING_DEPS, "PyTorch/torchvision/Lightning unavailable")
    def test_datamodule_test_without_train_directory(self):
        from datasets.synapse_semantic import SynapseSemantic
        self.remove_train_data()
        data = SynapseSemantic(str(self.train), str(self.test), str(self.lists),
                               num_workers=0)
        data.setup("test")
        self.assertEqual(len(data.test_dataset), 1)

    @unittest.skipUnless(HAS_TRAINING_DEPS, "PyTorch/torchvision/Lightning unavailable")
    def test_datamodule_fit_without_test_directory(self):
        from datasets.synapse_semantic import SynapseSemantic
        self.remove_test_data()
        data = SynapseSemantic(str(self.train), str(self.test), str(self.lists),
                               num_workers=0)
        data.setup("fit")
        self.assertEqual(len(data.train_dataset), 1)

    @unittest.skipUnless(HAS_TRAINING_DEPS, "PyTorch/torchvision/Lightning unavailable")
    def test_datamodule_fit_then_test_rechecks_files(self):
        from datasets.synapse_semantic import SynapseSemantic
        data = SynapseSemantic(str(self.train), str(self.test), str(self.lists),
                               num_workers=0)
        data.setup("fit")
        (self.test / "case0002.npy.h5").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "case0002"):
            data.setup("test")

    def test_paths_are_project_relative(self):
        self.assertEqual(splits.resolve_project_path("project_TransUNet"),
                         SOURCE.parents[1] / "project_TransUNet")
        self.assertEqual(splits.resolve_project_path(self.root), self.root)


if __name__ == "__main__":
    unittest.main()
