"""Isolated AMP probe: observe native GradScaler skips without changing it."""
import torch
from scripts.audit_synapse_baseline import RuntimeAudit

class ASEERuntimeProbe(RuntimeAudit):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        gradients=[(name,param.grad) for name,param in pl_module.named_parameters() if param.grad is not None]
        bad=[name for name,grad in gradients if not torch.isfinite(grad).all()]
        asee=[(name,grad) for name,grad in gradients if 'network.asee.' in name]
        assert asee,'ASEE received no gradients'
        self.steps.append(dict(global_step_before_update=trainer.global_step,finite_unscaled_gradients=not bad,nonfinite_names=bad[:16],gradient_tensors=len(gradients),asee_gradient_tensors=len(asee),scaler_before_update=trainer.precision_plugin.scaler.get_scale()))
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss=outputs['loss'] if isinstance(outputs,dict) else outputs
        assert torch.isfinite(loss).all(),'nonfinite forward loss'
        row=self.steps[-1];scale=trainer.precision_plugin.scaler.get_scale()
        steps=[int(state['step']) for state in trainer.optimizers[0].state.values() if 'step' in state]
        row.update(loss=float(loss.detach()),scaler=scale,optimizer_state_entries=len(trainer.optimizers[0].state),actual_optimizer_updates=max(steps,default=0))
        if not row['finite_unscaled_gradients']:
            assert scale<row['scaler_before_update'],'Nonfinite gradients were not handled by native GradScaler'
        self._sample(f'train_batch_{batch_idx}_end')
    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer,pl_module)
        assert len(self.steps)==8,self.steps
        assert all(x['finite_unscaled_gradients'] for x in self.steps[-2:]),self.steps
        assert self.steps[-1]['actual_optimizer_updates']>=2,self.steps
