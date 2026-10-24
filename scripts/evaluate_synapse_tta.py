"""Independent TTA CLI; only permits test and does not edit main.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from main import LightningCLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.synapse_tta import bind_tta

class TTACLI(LightningCLI):
    def add_arguments_to_parser(self,parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--tta_views",type=int,choices=[1,2,4,8],default=1)
    def test(self,model,**kwargs):
        views=self.config[self.config["subcommand"]]["tta_views"]
        original=bind_tta(model,views)
        try:return super().test(model,**kwargs)
        finally:model.evaluate_synapse_case=original

if __name__=="__main__":
    if len(sys.argv)<2 or sys.argv[1]!="test":raise SystemExit("Only independent test is supported")
    TTACLI(LightningModule,LightningDataModule,subclass_mode_model=True,subclass_mode_data=True,
           save_config_callback=None,seed_everything_default=0,
           trainer_defaults={"precision":"16-mixed","devices":1,"enable_model_summary":False})
