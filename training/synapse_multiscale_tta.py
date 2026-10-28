"""Fixed-canvas scale/horizontal TTA for independent Synapse evaluation."""
import json
import time
from pathlib import Path
from types import MethodType
import numpy as np
from scipy.ndimage import zoom
from training.synapse_physical_geometry import binary_metrics_mm
from training.synapse_physical_selection import SynapsePhysicalHD95Selection
from training.synapse_tta_common import (SynapseTTAReportSelection, kidney_confusion,
                                 file_sha256, write_tta_run_manifest)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
READINESS_PATH = PROJECT_ROOT / "outputs/tta/multiscale_readiness.json"
SCHEME = "scale_hflip"
import math
import torch
from torch.nn import functional as F
from training.synapse_tta_common import pixel_scores

PRESETS={2:(1.,),4:(1.,.875),6:(1.,.875,1.125),8:(1.,.875,1.125,.75)}

def scaled_view(image,scale):
    if image.ndim != 4: raise ValueError("Expected NCHW image")
    h,w=image.shape[-2:]
    if h!=w: raise ValueError("Fixed-canvas scaling requires square input")
    if not math.isfinite(scale) or not .5<=scale<=2: raise ValueError("Invalid scale")
    n=max(1,round(h*scale))
    if n==h:return image,(h,n,0)
    resized=F.interpolate(image.float(),size=(n,n),mode="bilinear",align_corners=False)
    offset=abs(h-n)//2
    if n<h:
        resized=F.pad(resized,(offset,h-n-offset,offset,h-n-offset),value=0)
    else:
        resized=resized[...,offset:offset+h,offset:offset+w]
    return resized.contiguous(),(h,n,offset)

def restore_scores(scores,geometry):
    h,n,offset=geometry
    scores=scores.float()
    if scores.ndim != 4 or scores.shape[-2:] != (h,h):
        raise ValueError("Scores must match the fixed square canvas")
    coverage=torch.ones_like(scores[:,:1])
    if n==h:return scores,coverage
    if n<h:
        scores=scores[...,offset:offset+n,offset:offset+n]
        return F.interpolate(scores,size=(h,h),mode="bilinear",align_corners=False),coverage
    padding=(offset,n-h-offset,offset,n-h-offset)
    scores=F.pad(scores,padding)
    coverage=F.pad(coverage,padding)
    return (F.interpolate(scores,size=(h,h),mode="bilinear",align_corners=False),
            F.interpolate(coverage,size=(h,h),mode="bilinear",align_corners=False))

def scale_pair(model,image,scale):
    transformed,geometry=scaled_view(image,scale)
    raw=pixel_scores(model,transformed).float()
    flipped=pixel_scores(model,torch.flip(transformed,(-1,)).contiguous()).float()
    score=(raw+torch.flip(flipped,(-1,)))/2
    if not torch.isfinite(score).all():raise FloatingPointError("Nonfinite scale prediction")
    return restore_scores(score,geometry)

def fuse(pairs,scales):
    if not scales or len(set(scales))!=len(scales) or 1. not in scales:
        raise ValueError("Unique scales including identity required for full coverage")
    total=None;coverage=None
    for scale in scales:
        num,den=pairs[scale]
        total=num if total is None else total+num
        coverage=den if coverage is None else coverage+den
    if not torch.isfinite(total).all() or not torch.isfinite(coverage).all() or not (coverage>0).all():raise ValueError("Invalid fusion coverage")
    return total/coverage


def predict_multiscale_tta_scores(model, image, views=4):
    """Each scale contributes an original and horizontally flipped prediction."""
    if views not in PRESETS:
        raise ValueError("Scale/horizontal TTA views must be 2, 4, 6 or 8")
    if tuple(image.shape[-2:]) != tuple(model.img_size):
        raise ValueError("Image must match the configured fixed model canvas")
    scales = PRESETS[views]
    pairs = {scale: scale_pair(model, image, scale) for scale in scales}
    return fuse(pairs, scales)


def evaluate_multiscale_case(model, batch, views=4):
    case = batch["case_name"]
    image, label = batch["image"], batch["label"]
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if label.ndim == 4 and label.shape[0] == 1:
        label = label[0]
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError("Expected one matching 3D image/label")
    image = image.detach().cpu().numpy()
    gt = label.detach().cpu().numpy()
    geometry = model.physical_geometry.for_case(case, gt)
    depth, height, width = image.shape
    ph, pw = model.img_size
    prediction = np.zeros((depth, height, width), dtype=np.int32)
    with torch.no_grad():
        for z, slc in enumerate(image):
            resized = zoom(slc, (ph/height, pw/width), order=3) if (height, width) != (ph, pw) else slc
            inp = torch.from_numpy(resized).float()[None, None].repeat(1, 3, 1, 1).to(model.device)
            scores = predict_multiscale_tta_scores(model, inp, views)
            hard = scores[0].argmax(dim=0).to(torch.int32).cpu().numpy()
            prediction[z] = zoom(hard, (height/ph, width/pw), order=0) if (height, width) != (ph, pw) else hard
    compute_voxel = getattr(model, "compute_hd95_voxel", True)
    records = [binary_metrics_mm(prediction == c, gt == c, geometry.spacing_zyx_mm, compute_hd95_voxel=compute_voxel)
               for c in range(1, model.num_classes)]
    dice = [r["dice"] for r in records]
    mm = [r["hd95_mm"] for r in records]
    voxel = [r["hd95_voxel"] for r in records]
    return {"case": case, "mean_dice": float(np.mean(dice)), "dice_per_class": dice,
            "mean_hd95": float(np.mean(mm)), "hd95_per_class": mm, "hd95_unit": "mm",
            "mean_hd95_mm": float(np.mean(mm)), "hd95_mm_per_class": mm,
            "mean_hd95_voxel": float(np.mean(voxel)) if compute_voxel else None, "hd95_voxel_per_class": voxel,
            "hd95_voxel_enabled": compute_voxel,
            "spacing_zyx_mm": list(geometry.spacing_zyx_mm),
            "metric_status_per_class": [r["status"] for r in records],
            "kidney_confusion": kidney_confusion(prediction, gt)}


def bind_multiscale_tta(model, views=4):
    if views not in PRESETS:
        raise ValueError("Scale/horizontal TTA views must be 2, 4, 6 or 8")
    original = model.evaluate_synapse_case
    def evaluate(self, batch):
        start = time.perf_counter()
        result = evaluate_multiscale_case(self, batch, views)
        result.update(tta_case_seconds=time.perf_counter()-start,
                      tta_views=views, tta_scheme=SCHEME)
        return result
    model.tta_views = views
    model.tta_scheme = SCHEME
    model.evaluate_synapse_case = MethodType(evaluate, model)
    return original


def multiscale_contract(views, img_size, *, compute_hd95_voxel=True):
    if views not in PRESETS:
        raise ValueError("Invalid scale/horizontal view count")
    size = tuple(img_size)
    if len(size) != 2 or size[0] != size[1]:
        raise ValueError("Fixed-canvas scale TTA requires square model input")
    contract = {
        "tta_scheme": SCHEME, "tta_views": views, "fixed_canvas_hw": list(size),
        "scales": list(PRESETS[views]),
        "view_transforms": [{"scale": scale, "resized_hw": [round(size[0]*scale)]*2,
                             "horizontal_flip": flip, "rotation_degrees": 0}
                            for scale in PRESETS[views] for flip in (False, True)],
        "scale_transform": "bilinear resize; centered zero pad or center crop to fixed canvas",
        "inverse": "undo horizontal flip, discard padding or restore crop canvas, bilinear to original model grid",
        "fusion": "FP32 scores divided by valid coverage; each visible view has equal weight; argmax last",
        "resize": "original cubic input to model canvas; existing bilinear pixel score upsample; nearest hard labels back",
        "class_channels": "original verified label IDs; no left/right channel swap",
        "ccps": "recomputed per view; no GT in prediction or fusion",
        "hd95": "verified ZYX spacing mm plus voxel; unchanged legacy empty-mask policy",
    }
    if not compute_hd95_voxel:
        contract["hd95"] = "verified ZYX spacing mm only; voxel disabled; unchanged legacy empty-mask policy"
        contract["hd95_voxel_enabled"] = False
    return contract


def validated_readiness(path=READINESS_PATH):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("state") != "passed" or data.get("tta_scheme") != SCHEME:
        raise ValueError("Scale/horizontal readiness must be passed")
    if not data.get("sha256"):
        raise ValueError("Scale/horizontal readiness requires source hashes")
    for rel, expected in data["sha256"].items():
        if file_sha256(PROJECT_ROOT / rel) != expected:
            raise ValueError("Scale/horizontal readiness source changed: " + rel)
    return raw


class SynapseMultiScaleTTASelection(SynapseTTAReportSelection):
    """Physical organ reports and provenance for independent scale/horizontal test."""
    def on_fit_start(self, trainer, pl_module):
        raise ValueError("This callback supports independent test only")

    def on_test_start(self, trainer, pl_module):
        raw_ready = validated_readiness(READINESS_PATH)
        if getattr(pl_module, "tta_scheme", None) != SCHEME:
            raise ValueError("Use the independent scale/horizontal CLI")
        SynapsePhysicalHD95Selection.on_test_start(self, trainer, pl_module)
        self.start = time.perf_counter()
        contract = multiscale_contract(pl_module.tta_views, pl_module.img_size, compute_hd95_voxel=getattr(pl_module, "compute_hd95_voxel", True))
        (self.run_dir / "tta_contract.json").write_text(json.dumps(contract, indent=2))
        snapshot = self.run_dir / "multiscale_readiness_snapshot.json"
        snapshot.write_bytes(raw_ready)
        report = write_tta_run_manifest(trainer, pl_module, self.run_dir, contract,
                                        self._test_checkpoint_info)
        report["multiscale_readiness_snapshot_sha256"] = file_sha256(snapshot)
        report["artifact_sha256"][snapshot.name] = file_sha256(snapshot)
        (self.run_dir / "run_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def on_test_epoch_end(self, trainer, pl_module):
        super().on_test_epoch_end(trainer, pl_module)
        path = self.run_dir / "physical_test_summary.json"
        report = json.loads(path.read_text())
        report.update(tta_scheme=SCHEME, scales=list(PRESETS[pl_module.tta_views]))
        path.write_text(json.dumps(report, indent=2))
