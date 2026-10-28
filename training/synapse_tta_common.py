"""Shared score extraction, diagnostics and reports for scale/horizontal TTA."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from training.synapse_physical_selection import SynapsePhysicalHD95Selection
from training.synapse_selection import append_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_tta_run_manifest(trainer,model,run_dir,contract,checkpoint_info,root=None):
    """Inference-only provenance; never requests train loaders or optimizer state."""
    root=Path(root) if root is not None else PROJECT_ROOT
    run_dir=Path(run_dir)
    ready_path=root/"outputs/tta/multiscale_readiness.json"
    raw_ready=ready_path.read_bytes()
    ready=json.loads(raw_ready)
    if ready.get("state")!="passed":raise ValueError("TTA readiness must be passed before evaluation")
    for rel,expected in ready["sha256"].items():
        if file_sha256(root/rel)!=expected:raise ValueError("TTA readiness source changed: "+rel)
    (run_dir/"readiness_snapshot.json").write_bytes(raw_ready)
    sources=[root/name for name in ("main.py","requirements.txt","sitecustomize.py")]
    for folder in ("models","datasets","training","configs/synapse","scripts"):
        sources.extend(p for p in (root/folder).rglob("*") if p.suffix in {".py",".yaml",".yml"})
    hashes={str(p.relative_to(root)):file_sha256(p) for p in sorted(set(sources)) if p.is_file()}
    versions={}
    for name in ("torch","torchvision","lightning","torchmetrics","timm","jsonargparse",
                 "numpy","scipy","medpy","h5py","wandb"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    data=trainer.datamodule
    paths={key:str(getattr(data,key)) for key in ("train_base_dir","test_base_dir","list_dir") if hasattr(data,key)}
    lists={}
    if hasattr(data,"list_dir"):
        directory=Path(data.list_dir)
        if not directory.is_absolute():directory=root/directory
        for key in ("train_split","test_split"):
            name=directory/f"{getattr(data,key)}.txt"
            lists[str(name)]=file_sha256(name)
    gpu=None
    if model.device.type=="cuda":
        props=torch.cuda.get_device_properties(model.device)
        gpu={"name":props.name,"total_memory_bytes":props.total_memory,
             "capability":list(torch.cuda.get_device_capability(model.device))}
    report={"schema_version":1,"stage":"independent_test","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "hostname":platform.node(),"python":sys.executable,"python_version":platform.python_version(),
        "cwd":str(Path.cwd()),"argv":sys.argv,"project_root":str(root),"versions":versions,
        "torch_cuda":torch.version.cuda,"device":str(model.device),"gpu":gpu,
        "matmul_precision":torch.get_float32_matmul_precision(),"precision":trainer.precision,
        "resolved_cli_config":getattr(trainer,"baseline_resolved_config",None),
        "source_sha256":hashes,"manifest_sha256":lists,"data_paths":paths,
        "callbacks":[type(cb).__module__+"."+type(cb).__name__ for cb in trainer.callbacks],
        "checkpoint_source":checkpoint_info,"tta_contract":contract,
        "readiness_snapshot_sha256":hashlib.sha256(raw_ready).hexdigest(),
        "artifact_sha256":{name:file_sha256(run_dir/name) for name in
            ("spacing_manifest.json","label_map.json","checkpoint_source.json","tta_contract.json") if (run_dir/name).is_file()},
        "environment":{key:os.environ.get(key) for key in ("HF_HUB_OFFLINE","TORCH_HOME","TIMM_USE_OLD_CACHE","CUDA_VISIBLE_DEVICES")},
        "selection_policy":"fixed source checkpoint; TTA inference comparison only"}
    (run_dir/"run_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return report


def pixel_scores(model, image):
    masks,classes=model(image)
    scores=model.to_per_pixel_logits_semantic(masks[-1],classes[-1])
    if scores.shape[-2:]!=tuple(model.img_size):
        scores=F.interpolate(scores,size=model.img_size,mode="bilinear",align_corners=False)
    return scores


def kidney_confusion(prediction,gt):
    left=(gt==3);right=(gt==4)
    return {"pred_left_on_gt_right_voxels":int(((prediction==3)&right).sum()),
            "pred_right_on_gt_left_voxels":int(((prediction==4)&left).sum()),
            "gt_left_voxels":int(left.sum()),"gt_right_voxels":int(right.sum())}


class SynapseTTAReportSelection(SynapsePhysicalHD95Selection):
    """Common organ metrics, kidney diagnostics and timing; no view transforms."""

    def on_test_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
        super().on_test_batch_end(trainer,pl_module,outputs,batch,batch_idx,dataloader_idx)
        append_csv(self.run_dir/"kidney_confusion.csv",[{"case":outputs["case"],"views":outputs["tta_views"],**outputs["kidney_confusion"]}])
        append_csv(self.run_dir/"tta_case_timing.csv",[{"case":outputs["case"],"views":outputs["tta_views"],"seconds":outputs["tta_case_seconds"]}])

    def on_test_epoch_end(self,trainer,pl_module):
        super().on_test_epoch_end(trainer,pl_module)
        p=self.run_dir/"physical_test_summary.json";report=json.loads(p.read_text())
        report.update(tta_views=pl_module.tta_views,test_seconds=time.perf_counter()-self.start,
                      sum_case_seconds=sum(r["tta_case_seconds"] for r in self._test_results))
        p.write_text(json.dumps(report,indent=2))
