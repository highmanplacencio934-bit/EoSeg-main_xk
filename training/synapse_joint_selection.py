"""Unchanged 1-view selection plus an archived best for the historical epoch window."""
import json
from training.synapse_physical_selection import SynapsePhysicalHD95Selection


class SynapseJointSelection(SynapsePhysicalHD95Selection):
    def __init__(self, last_n_epochs: int = 50, expected_test_cases: int = 12,
                 output_dir: str = "outputs/augmentation_joint",
                 label_map: str = "priors/synapse_label_map.verified.json",
                 matched_window_end: int = 213):
        super().__init__(last_n_epochs, expected_test_cases, output_dir, label_map)
        if matched_window_end < 1:
            raise ValueError("matched_window_end must be positive")
        self.matched_window_end = matched_window_end

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        if epoch <= self.matched_window_end and self.best_epoch == epoch:
            name = f"best_test_through_epoch{self.matched_window_end}.ckpt"
            trainer.save_checkpoint(str(self.run_dir / name), weights_only=True)
            (self.run_dir / "matched_window_summary.json").write_text(json.dumps({
                "first_evaluation_epoch": trainer.max_epochs - self.last_n_epochs + 1,
                "window_end": self.matched_window_end, "best_epoch": epoch,
                "best_mean_dice": self.best_score, "checkpoint": name,
                "selection": "same 1-view Dice; archive only, no training or metric change"
            }, indent=2), encoding="utf-8")
