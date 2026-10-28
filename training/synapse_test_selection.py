"""Final-window test evaluation with only best_test.ckpt and last.ckpt per run."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import logging
import random
from uuid import uuid4

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from training.run_artifacts import write_run_manifest
from training.synapse_selection import append_csv, should_evaluate, summarize_cases


class SynapseTestSelection(Callback):
    """The test set is explicitly used for model selection, never gradient updates.

    Single-device callback; does not invoke Trainer.test or alter the fit loop.
    Each fit creates a new output subdirectory, preserving prior runs.
    """

    def __init__(self, last_n_epochs: int = 50, expected_test_cases: int = 12,
                 output_dir: Optional[str] = None):
        super().__init__()
        self.last_n_epochs = last_n_epochs
        self.expected_test_cases = expected_test_cases
        self.output_dir = output_dir
        self.run_dir = None
        self.best_score = float("-inf")
        self.best_epoch = None
        self._test_loader = None

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError("SynapseTestSelection supports a single device only")
        if trainer.checkpoint_callbacks:
            raise ValueError("Set trainer.enable_checkpointing=false to avoid extra checkpoints")
        should_evaluate(1, trainer.max_epochs, self.last_n_epochs)
        if not callable(getattr(pl_module, "evaluate_synapse_case", None)):
            raise TypeError("Model must implement evaluate_synapse_case")
        root = Path(self.output_dir or trainer.log_dir or trainer.default_root_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = root / "synapse_test_selection" / f"{stamp}_{uuid4().hex[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.best_score, self.best_epoch, self._test_loader = float("-inf"), None, None
        write_run_manifest(trainer, pl_module, self.run_dir)
        logging.info("Synapse test-selected checkpoints and CSV: %s", self.run_dir)

    def _evaluate(self, trainer, pl_module):
        # Preserve per-submodule mode (including deliberately frozen BN), and RNG
        # state so inserting evaluation does not consume the training RNG stream.
        modes = [(module, module.training) for module in pl_module.modules()]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = [pl_module.device.index] if pl_module.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                pl_module.eval()
                if self._test_loader is None:
                    trainer.datamodule.setup("test")
                    self._test_loader = trainer.datamodule.test_dataloader()
                if len(self._test_loader.dataset) != self.expected_test_cases:
                    raise ValueError(f"Expected {self.expected_test_cases} test cases")
                with trainer.precision_plugin.forward_context():
                    results = [pl_module.evaluate_synapse_case(batch) for batch in self._test_loader]
                summary = summarize_cases(results, self.expected_test_cases, pl_module.num_classes - 1)
                return results, summary
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        # Rolling last weights are retained even before the selection window.
        trainer.save_checkpoint(str(self.run_dir / "last.ckpt"), weights_only=True)
        if not should_evaluate(epoch, trainer.max_epochs, self.last_n_epochs):
            return
        results, summary = self._evaluate(trainer, pl_module)
        improved = summary["mean_dice"] > self.best_score
        if improved:
            trainer.save_checkpoint(str(self.run_dir / "best_test.ckpt"), weights_only=True)
            self.best_score, self.best_epoch = summary["mean_dice"], epoch
        common = {"epoch": epoch, "global_step": trainer.global_step}
        append_csv(self.run_dir / "test_epoch_metrics.csv", [dict(
            common, **summary, is_best=int(improved), best_epoch=self.best_epoch,
            best_mean_dice=self.best_score,
        )])
        case_rows = []
        for result in results:
            row = dict(common, case=result["case"], mean_dice=result["mean_dice"],
                       mean_hd95=result["mean_hd95"])
            for metric in ("dice", "hd95"):
                row.update({f"{metric}_class_{i}": value for i, value in
                            enumerate(result[f"{metric}_per_class"], start=1)})
            case_rows.append(row)
        append_csv(self.run_dir / "test_case_metrics.csv", case_rows)
        selection = {
            "selection_policy": "test set used for checkpoint selection; no gradient updates",
            "epoch_numbering": "one-based", "max_epochs": trainer.max_epochs,
            "first_evaluation_epoch": trainer.max_epochs - self.last_n_epochs + 1,
            "last_evaluated_epoch": epoch, "expected_test_cases": self.expected_test_cases,
            "best_epoch": self.best_epoch, "best_mean_dice": self.best_score,
            "best_checkpoint": "best_test.ckpt", "last_checkpoint": "last.ckpt",
        }
        (self.run_dir / "selection_summary.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if trainer.logger is not None:
            trainer.logger.log_metrics({f"selection/test_{key}": value for key, value in
                                        summary.items()}, step=trainer.global_step)
        logging.info("[TEST SELECTION] epoch=%d Dice=%.6f HD95=%.6f best_epoch=%d best=%.6f",
                     epoch, summary["mean_dice"], summary["mean_hd95"],
                     self.best_epoch, self.best_score)
