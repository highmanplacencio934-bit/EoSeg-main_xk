"""Independent 2D mirror TTA; original training and one-view inference are unchanged."""
import json
import hashlib
import importlib.metadata
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import time
from types import MethodType

import numpy as np
from scipy.ndimage import zoom
import torch
from torch.nn import functional as F

from training.synapse_physical_geometry import binary_metrics_mm
from training.synapse_physical_selection import SynapsePhysicalHD95Selection
from training.synapse_selection import append_csv

PROJECT_ROOT=Path(__file__).resolve().parents[1]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_tta_run_manifest(trainer,model,run_dir,contract,checkpoint_info,root=None):
    """Inference-only provenance; never requests train loaders or optimizer state."""
    root=Path(root) if root is not None else PROJECT_ROOT
    run_dir=Path(run_dir)
    ready_path=root/"outputs/tta/tta_readiness.json"
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


VIEW_AXES={1:((),),2:((),(-1,)),4:((),(-1,),(-2,),(-2,-1))}
VIEW_TRANSFORMS={n:tuple((0,axes) for axes in values) for n,values in VIEW_AXES.items()}
# Four mirrors plus the remaining four D4 transforms, without duplicates.
VIEW_TRANSFORMS[8]=VIEW_TRANSFORMS[4]+((1,()),(3,()),(1,(-1,)),(3,(-1,)))


def transform_view(tensor,rotation,axes):
    if rotation:tensor=torch.rot90(tensor,rotation,(-2,-1))
    if axes:tensor=torch.flip(tensor,axes)
    return tensor.contiguous()


def invert_view(tensor,rotation,axes):
    if axes:tensor=torch.flip(tensor,axes)
    if rotation:tensor=torch.rot90(tensor,-rotation,(-2,-1))
    return tensor


def pixel_scores(model, image):
    masks,classes=model(image)
    scores=model.to_per_pixel_logits_semantic(masks[-1],classes[-1])
    if scores.shape[-2:]!=tuple(model.img_size):
        scores=F.interpolate(scores,size=model.img_size,mode="bilinear",align_corners=False)
    return scores


def predict_tta_scores(model,image,views):
    if views not in VIEW_TRANSFORMS:raise ValueError("TTA views must be 1, 2, 4 or 8")
    total=None
    if views==8 and image.shape[-2]!=image.shape[-1]:
        raise ValueError("D4 views require square model input")
    for rotation,axes in VIEW_TRANSFORMS[views]:
        transformed=transform_view(image,rotation,axes) if rotation or axes else image
        scores=pixel_scores(model,transformed)
        scores=invert_view(scores,rotation,axes)
        # Preserve the old single-view dtype/path; multi-view sums are FP32.
        if views==1:return scores
        scores=scores.float()
        total=scores if total is None else total+scores
    result=total/views
    if not torch.isfinite(result).all():raise FloatingPointError("Nonfinite TTA pixel scores")
    return result



def kidney_confusion(prediction,gt):
    left=(gt==3);right=(gt==4)
    return {"pred_left_on_gt_right_voxels":int(((prediction==3)&right).sum()),
            "pred_right_on_gt_left_voxels":int(((prediction==4)&left).sum()),
            "gt_left_voxels":int(left.sum()),"gt_right_voxels":int(right.sum())}


def evaluate_tta_case(model,batch,views):
    case=batch["case_name"]
    image,label=batch["image"],batch["label"]
    if image.ndim==4 and image.shape[0]==1:image=image[0]
    if label.ndim==4 and label.shape[0]==1:label=label[0]
    if image.ndim!=3 or image.shape!=label.shape:raise ValueError("Expected one matching 3D image/label")
    image=image.detach().cpu().numpy();gt=label.detach().cpu().numpy()
    geometry=model.physical_geometry.for_case(case,gt)
    depth,height,width=image.shape
    ph,pw=model.img_size
    prediction=np.zeros((depth,height,width),dtype=np.int32)
    with torch.no_grad():
        for z,slc in enumerate(image):
            resized=zoom(slc,(ph/height,pw/width),order=3) if (height,width)!=(ph,pw) else slc
            inp=torch.from_numpy(resized).float()[None,None].repeat(1,3,1,1).to(model.device)
            scores=predict_tta_scores(model,inp,views)
            hard=scores[0].argmax(dim=0).to(torch.int32).cpu().numpy()
            prediction[z]=zoom(hard,(height/ph,width/pw),order=0) if (height,width)!=(ph,pw) else hard
    records=[binary_metrics_mm(prediction==c,gt==c,geometry.spacing_zyx_mm) for c in range(1,model.num_classes)]
    dice=[x["dice"] for x in records];mm=[x["hd95_mm"] for x in records];voxel=[x["hd95_voxel"] for x in records]
    return {"case":case,"mean_dice":float(np.mean(dice)),"dice_per_class":dice,
            "mean_hd95":float(np.mean(mm)),"hd95_per_class":mm,"hd95_unit":"mm",
            "mean_hd95_mm":float(np.mean(mm)),"hd95_mm_per_class":mm,
            "mean_hd95_voxel":float(np.mean(voxel)),"hd95_voxel_per_class":voxel,
            "spacing_zyx_mm":list(geometry.spacing_zyx_mm),
            "metric_status_per_class":[x["status"] for x in records],
            "kidney_confusion":kidney_confusion(prediction,gt)}


def bind_tta(model,views):
    if views not in VIEW_TRANSFORMS:raise ValueError("TTA views must be 1, 2, 4 or 8")
    original=model.evaluate_synapse_case
    def evaluate(self,batch):
        start=time.perf_counter()
        # One-view uses the exact original physical inference, without a rewrite.
        if views==1:
            # Capture only metric masks after inference; preserve the original
            # single-view prediction, geometry and metric computations exactly.
            gt=batch["label"].detach().cpu().numpy()
            if gt.ndim==4:gt=gt[0]
            saved_metric=self.calculate_metric_percase
            count=0;cross={"gt_left_voxels":int((gt==3).sum()),"gt_right_voxels":int((gt==4).sum()),
                           "pred_left_on_gt_right_voxels":0,"pred_right_on_gt_left_voxels":0}
            def capture(pred,target):
                nonlocal count
                count+=1
                if count==3:cross["pred_left_on_gt_right_voxels"]=int((pred&(gt==4)).sum())
                elif count==4:cross["pred_right_on_gt_left_voxels"]=int((pred&(gt==3)).sum())
                return saved_metric(pred,target)
            self.calculate_metric_percase=capture
            try:result=original(batch)
            finally:self.calculate_metric_percase=saved_metric
            result["kidney_confusion"]=cross
        else:
            result=evaluate_tta_case(self,batch,views)
        result["tta_case_seconds"]=time.perf_counter()-start
        result["tta_views"]=views
        return result
    model.tta_views=views
    model.evaluate_synapse_case=MethodType(evaluate,model)
    return original


class SynapseTTASelection(SynapsePhysicalHD95Selection):
    def on_test_start(self,trainer,pl_module):
        super().on_test_start(trainer,pl_module)
        self.start=time.perf_counter()
        contract={"tta_views":pl_module.tta_views,"view_transforms":[{"rotation_degrees":90*k,"flip_axes":axes} for k,axes in VIEW_TRANSFORMS[pl_module.tta_views]],
            "fusion":"inverse spatial transforms then FP32 mean of nine pixel scores before argmax",
            "single_view":"exact original physical evaluate_synapse_case",
            "resize":"original cubic input -> 384; bilinear scores -> 384; argmax; nearest back",
            "class_channels":"no left/right channel swap; original verified label IDs",
            "ccps":"recomputed separately for each view; no GT in gating",
            "hd95":"verified ZYX spacing mm + voxel; legacy empty-mask policy unchanged"}
        (self.run_dir/"tta_contract.json").write_text(json.dumps(contract,indent=2))
        write_tta_run_manifest(trainer,pl_module,self.run_dir,contract,self._test_checkpoint_info)
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
