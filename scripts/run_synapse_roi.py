"""Independent opt-in ROI evaluation CLI; original main.py remains unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from main import LightningCLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.synapse_roi_evaluation import roi_metrics_scope


class SynapseROICLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--tta_views", type=int, choices=[1, 2, 4, 6, 8], default=1)
        parser.add_argument("--roi_capture_dir", type=str, default=None)

    def _parse_ckpt_path(self):
        # This explicit evaluator may wrap a checkpoint trained with an older
        # module. Trainer.test still restores its complete state_dict normally;
        # do not let checkpoint hyperparameters replace the selected wrapper.
        if self.config.get("subcommand") == "test":
            return
        return super()._parse_ckpt_path()

    def fit(self, model, **kwargs):
        options = self.config[self.config["subcommand"]]
        if options["tta_views"] != 1 or options["roi_capture_dir"] is not None:
            raise ValueError("Fit keeps original 1-view selection and forbids prediction capture")
        with roi_metrics_scope(model, tta_views=1):
            return super().fit(model, **kwargs)

    def test(self, model, **kwargs):
        options = self.config[self.config["subcommand"]]
        with roi_metrics_scope(model, tta_views=options["tta_views"],
                               capture_dir=options["roi_capture_dir"]):
            return super().test(model, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("fit", "test"):
        raise SystemExit("The independent ROI entry supports fit or test only")
    SynapseROICLI(
        LightningModule, LightningDataModule,
        subclass_mode_model=True, subclass_mode_data=True,
        save_config_callback=None, seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed", "devices": 1,
            "callbacks": [ModelSummary(max_depth=3), LearningRateMonitor(logging_interval="epoch")],
            "gradient_clip_val": 0.01, "gradient_clip_algorithm": "norm",
        },
    )
