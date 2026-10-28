"""Opt-in protocol controls; never imported by the original training entry."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import torch
from lightning.pytorch import Callback
from training.deterministic_mask_loss import DeterministicMaskClassificationLoss

PROTOCOL_ID = "ce2d_stableseg_v1"
SOURCE_FILES = (
    "training/deterministic_protocol.py", "training/deterministic_mask_loss.py",
    "training/deterministic_point_sampling.py", "scripts/run_synapse_deterministic.py",
)

def source_hashes(root):
    root=Path(root)
    paths={root/name for name in SOURCE_FILES}
    for directory in ("models","datasets","training","configs/synapse"):
        paths.update(p for p in (root/directory).rglob("*") if p.suffix in (".py",".yaml",".yml"))
    for name in ("main.py","scripts/run_synapse_roi.py",
                 "project_TransUNet/TransUNet/lists/lists_Synapse/train.txt",
                 "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt"):
        paths.add(root/name)
    paths.update(p for p in (root/"priors").rglob("*") if p.is_file())
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str)+"\n")
    temporary.replace(path)

def backend_state():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_precision=torch.get_float32_matmul_precision())

def restore_backend(state):
    torch.use_deterministic_algorithms(state["deterministic"], warn_only=state["warn_only"])
    torch.backends.cudnn.benchmark=state["benchmark"]
    torch.backends.cudnn.deterministic=state["cudnn_deterministic"]
    torch.set_float32_matmul_precision(state["matmul_precision"])

def configure_strict(*, enabled=True):
    before = backend_state()
    if not enabled:
        return before
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Launch Python with CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Launch Python with PYTHONHASHSEED=0; changing it after launch is insufficient")
    if torch.cuda.is_initialized():
        raise RuntimeError("Configure the protocol before the first CUDA operation")
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("medium")
    return before

def install_criterion(model, *, enabled=True):
    if not enabled:
        return False
    if isinstance(model.criterion, DeterministicMaskClassificationLoss):
        return False
    original=model.criterion
    model.criterion=DeterministicMaskClassificationLoss.from_existing(original)
    return True

def require_readiness(path, root, profile):
    path=Path(path)
    if not path.is_file():
        raise RuntimeError("Formal fit is blocked: deterministic G0-G5 readiness is missing; use a bounded probe")
    status=json.loads(path.read_text())
    if status.get("protocol") != PROTOCOL_ID or status.get("passed") is not True:
        raise RuntimeError("Deterministic readiness is not passed")
    if profile not in status.get("validated_profiles", []):
        raise RuntimeError("This model profile has not passed the real-entry checks")
    if status.get("sources") != source_hashes(root):
        raise RuntimeError("Deterministic source hashes changed after readiness")
    if any(status.get("gates", {}).get(name) != "passed" for name in ("G0","G1","G2","G3","G4","G5")):
        raise RuntimeError("All G0-G5 gates are required for formal fit")

class DeterministicProtocolRecord(Callback):
    """Record actual context and bound probes without changing the LR schedule."""
    def __init__(self, output_dir, root, probe_batches=0, profile="baseline"):
        self.root=Path(root)
        self.output_dir=Path(output_dir)
        self.probe_batches=probe_batches
        self.profile=profile
        self.batches_seen=0
        self.run_dir=None

    def on_fit_start(self, trainer, pl_module):
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError("Strict deterministic mode was reset before fit")
        if not isinstance(pl_module.criterion, DeterministicMaskClassificationLoss):
            raise RuntimeError("Deterministic criterion was replaced before fit")
        run_id=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())+"_"+uuid.uuid4().hex[:8]
        self.run_dir=self.output_dir/run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        report={"protocol":PROTOCOL_ID,"profile":self.profile,"probe_batches":self.probe_batches,
                "scientific_max_epochs":trainer.max_epochs,
                "estimated_stepping_batches":trainer.estimated_stepping_batches,
                "backend":backend_state(),"sources":source_hashes(self.root),
                "torch":torch.__version__,"cuda_runtime":torch.version.cuda,
                "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "env":{name:os.environ.get(name) for name in (
                    "PYTHONHASHSEED","CUBLAS_WORKSPACE_CONFIG","TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR","HF_HUB_OFFLINE","TORCH_HOME","TIMM_USE_OLD_CACHE")},
                "criterion_forward_eager_boundary":True,"default_outer_compile_preserved":True,
                "compute_hd95_voxel":getattr(pl_module,"compute_hd95_voxel",None),
                "resolved_config":getattr(trainer,"baseline_resolved_config",None)}
        write_json(self.run_dir/"protocol_manifest.json",report)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches_seen+=1
        if self.probe_batches and self.batches_seen >= self.probe_batches:
            trainer.should_stop=True

    def on_fit_end(self, trainer, pl_module):
        if self.run_dir:
            from torch._dynamo.utils import counters
            write_json(self.run_dir/"probe_exit.json",dict(
                batches_seen=self.batches_seen,global_step=trainer.global_step,
                epoch=trainer.current_epoch,probe_complete=bool(self.probe_batches and self.batches_seen>=self.probe_batches),
                full_training_complete=not self.probe_batches and trainer.current_epoch>=trainer.max_epochs,
                compiler_counters={k:dict(v) for k,v in counters.items()}))
