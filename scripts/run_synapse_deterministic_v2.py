#!/usr/bin/env python3
"""Independent strict CUDA protocol entry; the original ROI/main entries are unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from scripts.run_synapse_roi import SynapseROICLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.deterministic_execution_v2 import (
    PROTOCOL_ID, backend_state, configure_strict, restore_backend, install_criterion,
    require_readiness, DeterministicProtocolRecord, execution_context, install_execution_boundary,
)

class SynapseExecutionV2CLI(SynapseROICLI):
    def __init__(self,*args,**kwargs):
        before=backend_state()
        try:
            super().__init__(*args,**kwargs)
        except BaseException:
            restore_backend(before)
            raise

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--deterministic_enabled",type=bool,default=True)
        parser.add_argument("--deterministic_protocol",type=str,default=PROTOCOL_ID)
        parser.add_argument("--deterministic_profile",type=str,choices=("baseline","asee","ccps"),default="baseline")
        parser.add_argument("--deterministic_probe_batches",type=int,default=0)
        parser.add_argument("--deterministic_output_dir",type=str,default="outputs/determinism/"+PROTOCOL_ID)
        parser.add_argument("--deterministic_readiness",type=str,default="outputs/determinism/"+PROTOCOL_ID+"/readiness.json")

    def _protocol_options(self):
        return self.config[self.config["subcommand"]] if self.config.get("subcommand") else self.config

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        options=self._protocol_options()
        if options["deterministic_protocol"] != PROTOCOL_ID:
            raise ValueError("Unknown deterministic protocol")
        batches=options["deterministic_probe_batches"]
        if not 0<=batches<=100:
            raise ValueError("Probe must be 1-100 native batches, or 0 for an independently authorized formal fit")
        self._previous_backend=configure_strict(enabled=options["deterministic_enabled"])
        if options["deterministic_enabled"] and self.config.get("subcommand")=="fit":
            if not batches:
                require_readiness(options["deterministic_readiness"],ROOT,options["deterministic_profile"])
            if options["compile_disabled"]:
                raise ValueError("Protocol fit preserves the default outer torch.compile")

    def after_instantiate_classes(self):
        super().after_instantiate_classes()
        options=self._protocol_options()
        if options["deterministic_enabled"]:
            install_criterion(self.model)
            install_execution_boundary(self.model,options["deterministic_profile"])
            if self.config.get("subcommand")=="fit":
                self.trainer.callbacks.append(DeterministicProtocolRecord(
                    options["deterministic_output_dir"],ROOT,
                    options["deterministic_probe_batches"],options["deterministic_profile"]))

    def fit(self,model,**kwargs):
        try:
            with execution_context(self._protocol_options()["deterministic_profile"],enabled=self._protocol_options()["deterministic_enabled"]):
                return super().fit(model,**kwargs)
        finally:
            if hasattr(self,"_previous_backend"):
                restore_backend(self._previous_backend)

    def test(self,model,**kwargs):
        try:
            with execution_context(self._protocol_options()["deterministic_profile"],enabled=self._protocol_options()["deterministic_enabled"]):
                return super().test(model,**kwargs)
        finally:
            if hasattr(self,"_previous_backend"):
                restore_backend(self._previous_backend)

def cli_main():
    if len(sys.argv)<2 or sys.argv[1] not in ("fit","test"):
        raise SystemExit("Supports fit or test only")
    SynapseExecutionV2CLI(
        LightningModule,LightningDataModule,subclass_mode_model=True,subclass_mode_data=True,
        save_config_callback=None,seed_everything_default=0,
        trainer_defaults={"precision":"16-mixed","devices":1,
            "callbacks":[ModelSummary(max_depth=3),LearningRateMonitor(logging_interval="epoch")],
            "gradient_clip_val":0.01,"gradient_clip_algorithm":"norm"})

if __name__=="__main__":
    cli_main()
