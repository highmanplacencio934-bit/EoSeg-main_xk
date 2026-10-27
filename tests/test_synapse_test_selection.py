"""Pure policy tests plus optional CPU Lightning integration on a tiny model."""
import csv
import json
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from contextlib import nullcontext
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.synapse_selection import should_evaluate, summarize_cases, append_csv

HAS_TRAINING_DEPS = all(importlib.util.find_spec(n) is not None
                        for n in ("torch", "lightning"))


class SelectionPolicyTests(unittest.TestCase):
    def test_exact_final_fifty_epochs(self):
        self.assertEqual([e for e in range(1, 221) if should_evaluate(e, 220, 50)],
                         list(range(171, 221)))

    def test_invalid_window(self):
        for n in (0, 221):
            with self.assertRaises(ValueError):
                should_evaluate(1, 220, n)

    def test_equal_case_weighted_organ_means(self):
        rows = [{"case": "a", "dice_per_class": [0.2, 0.6], "hd95_per_class": [2, 4]},
                {"case": "b", "dice_per_class": [0.4, 0.8], "hd95_per_class": [6, 8]}]
        result = summarize_cases(rows, 2, 2)
        self.assertAlmostEqual(result["mean_dice"], 0.5)
        self.assertEqual(result["mean_hd95"], 5)

    def test_incomplete_duplicate_and_nonfinite_cases_rejected(self):
        row = {"case": "a", "dice_per_class": [0.5], "hd95_per_class": [2]}
        for rows in ([], [row], [row, row]):
            with self.assertRaises(ValueError):
                summarize_cases(rows, 2, 1)
        row["dice_per_class"] = [float("nan")]
        with self.assertRaises(ValueError):
            summarize_cases([row], 1, 1)

    def test_csv_header_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            append_csv(path, [{"epoch": 171, "mean_dice": 0.5}])
            append_csv(path, [{"epoch": 172, "mean_dice": 0.6}])
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


@unittest.skipUnless(HAS_TRAINING_DEPS, "PyTorch/Lightning unavailable")
class LightningSelectionTests(unittest.TestCase):
    def test_small_fit_saves_only_best_and_last(self):
        import torch
        from lightning.pytorch import LightningModule, LightningDataModule, Trainer
        from torch.utils.data import DataLoader
        from training.synapse_test_selection import SynapseTestSelection

        class Data(LightningDataModule):
            def train_dataloader(self):
                return DataLoader(torch.ones(4, 2), batch_size=4)

            def test_dataloader(self):
                return DataLoader([{"case_name": "case1"}, {"case_name": "case2"}],
                                  batch_size=None)

        class Model(LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)
                self.num_classes = 3
                self.evaluated_epochs = []

            def training_step(self, batch, batch_idx):
                assert self.training and self.layer.training
                return self.layer(batch).square().mean()

            def configure_optimizers(self):
                opt = torch.optim.SGD(self.parameters(), lr=0.1)
                return {"optimizer": opt, "lr_scheduler": torch.optim.lr_scheduler.StepLR(opt, 1, 0.5)}

            def evaluate_synapse_case(self, batch):
                assert not self.training and not torch.is_grad_enabled()
                self.evaluated_epochs.append(self.current_epoch + 1)
                self.layer(torch.ones(1, 2))
                score = 0.8 if self.current_epoch == 2 else 0.6
                return {"case": batch["case_name"], "mean_dice": score, "mean_hd95": 2.0,
                        "dice_per_class": [score, score], "hd95_per_class": [2.0, 2.0]}

        with tempfile.TemporaryDirectory() as tmp:
            callback = SynapseTestSelection(last_n_epochs=2, expected_test_cases=2,
                                            output_dir=tmp)
            trainer = Trainer(accelerator="cpu", devices=1, max_epochs=4, logger=False,
                              enable_checkpointing=False, enable_progress_bar=False,
                              enable_model_summary=False, num_sanity_val_steps=0,
                              limit_val_batches=0, callbacks=[callback])
            model = Model()
            trainer.fit(model, datamodule=Data())
            self.assertEqual(model.evaluated_epochs, [3, 3, 4, 4])
            self.assertEqual(trainer.global_step, 4)
            self.assertAlmostEqual(trainer.optimizers[0].param_groups[0]["lr"], 0.00625)
            self.assertEqual(callback.best_epoch, 3)
            manifest = json.loads((callback.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["estimated_steps"], 4)
            self.assertIn("main.py", manifest["source_sha256"])
            self.assertIn("optimizer_groups", manifest)
            self.assertEqual(sorted(f.name for f in callback.run_dir.glob("*.ckpt")),
                             ["best_test.ckpt", "last.ckpt"])
            for name, epoch in (("best_test.ckpt", 2), ("last.ckpt", 3)):
                ckpt = torch.load(callback.run_dir / name, weights_only=False)
                self.assertEqual(ckpt["epoch"], epoch)
                self.assertIn("layer.weight", ckpt["state_dict"])
                self.assertNotIn("optimizer_states", ckpt)
            with (callback.run_dir / "test_epoch_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["epoch"] for r in rows], ["3", "4"])
            with (callback.run_dir / "test_case_metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_failure_restores_modes_and_rng(self):
        import torch
        import numpy as np
        from training.synapse_test_selection import SynapseTestSelection

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)
                self.device = torch.device("cpu")
                self.num_classes = 2

            def evaluate_synapse_case(self, batch):
                random.random(); np.random.rand(); torch.rand(1)
                raise RuntimeError("test failure")

        class Loader:
            dataset = [1]
            def __iter__(self):
                return iter([{}])

        model = Model(); model.layer.eval()
        modes = [m.training for m in model.modules()]
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
        callback = SynapseTestSelection(expected_test_cases=1)
        callback._test_loader = Loader()
        trainer = SimpleNamespace(precision_plugin=SimpleNamespace(forward_context=nullcontext))
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            callback._evaluate(trainer, model)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], npstate[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), ts))
        self.assertTrue(torch.is_grad_enabled())


if __name__ == "__main__":
    unittest.main()
