"""Isolated CCPS smoke callback, never imported by the production config."""
import json
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback

class CCPSSmokeAudit(Callback):
    def __init__(self, output_dir):
        self.root=Path(output_dir); self.root.mkdir(parents=True,exist_ok=True)
        self.backward_checks=0;self.nonzero=False;self.steps=0
    def on_fit_start(self,trainer,pl_module):
        assert pl_module.device.type=="cuda"
        assert trainer.precision=="16-mixed"
        assert pl_module.img_size==(384,384) or list(pl_module.img_size)==[384,384]
        self.start=[p.detach().clone() for p in pl_module.network.asee.selector.parameters()]
        torch.cuda.reset_peak_memory_stats()
    def on_after_backward(self,trainer,pl_module):
        ps=list(pl_module.network.asee.selector.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
        self.nonzero |= any(bool(p.grad.abs().sum()>0) for p in ps)
        self.backward_checks+=1
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        assert batch[0].shape==(16,3,384,384)
        assert torch.isfinite(outputs["loss"])
        weights=pl_module.network.asee.last_selector_weights
        assert weights.shape==(16,8) and torch.isfinite(weights).all()
        self.steps+=1
    def on_train_end(self,trainer,pl_module):
        ps=list(pl_module.network.asee.selector.parameters())
        assert self.backward_checks==2 and self.steps==2 and self.nonzero
        assert any(not torch.equal(old,new) for old,new in zip(self.start,ps))
        assert all(torch.isfinite(p).all() for p in pl_module.parameters())
        trainer.save_checkpoint(str(self.root/"smoke.ckpt"),weights_only=True)
        from training.run_artifacts import write_run_manifest
        write_run_manifest(trainer,pl_module,self.root)
        report={"passed":True,"batches":self.steps,"backward_checks":self.backward_checks,
            "global_step":trainer.global_step,"selector_changed":True,"selector_finite_nonzero_gradients":True,
            "batch_shape":[16,3,384,384],"precision":trainer.precision,"default_compile":True,
            "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
            "selector_weights_min":float(pl_module.network.asee.last_selector_weights.min()),
            "selector_weights_max":float(pl_module.network.asee.last_selector_weights.max()),
            "physical_preflight":pl_module.physical_preflight}
        (self.root/"smoke_result.json").write_text(json.dumps(report,indent=2))
