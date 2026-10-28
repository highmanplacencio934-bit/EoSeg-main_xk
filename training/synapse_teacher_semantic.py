"""Actual uploaded AgileFormer test/metric functions with an EoSeg output adapter.

The reference utils.py is loaded unchanged and pinned by SHA256. Its voxel
metric remains explicit; physical HD95 is an additional computation on the
same masks. This module is for independent testing only.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from medpy import metric

from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic

ROOT = Path(__file__).resolve().parents[1]
TEACHER_UTILS = ROOT / "AgileFormer-main/AgileFormer-main/utils.py"
TEACHER_SHA256 = "6e7eceb5f6e1dd7571c38094efd85951c229989c89e6cdc8f90b8bed92309a2c"


def load_teacher_utils():
    digest = hashlib.sha256(TEACHER_UTILS.read_bytes()).hexdigest()
    if digest != TEACHER_SHA256:
        raise ValueError("Uploaded teacher utils.py changed; review it before using this adapter")
    spec = importlib.util.spec_from_file_location("_reviewed_agileformer_hd95", TEACHER_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(TEACHER_UTILS.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Teacher source changed while loading")
    return module, {"path": str(TEACHER_UTILS), "sha256": digest,
                    "prediction_function": "test_single_volume", "metric_function": "calculate_metric_percase",
                    "original_hd95_unit": "voxel_grid", "z_spacing_used_only_for_image_export": True}


class EoSegPixelAdapter(nn.Module):
    """Teacher expects one-channel inputs and a dense tensor, EoSeg uses queries."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Teacher adapter expects [B,1,H,W]")
        masks, classes = self.model(image.repeat(1, 3, 1, 1))
        scores = self.model.to_per_pixel_logits_semantic(masks[-1], classes[-1])
        if scores.shape[-2:] != tuple(self.model.img_size):
            scores = F.interpolate(scores, size=self.model.img_size, mode="bilinear", align_corners=False)
        # Teacher expects logits; EoSeg returns nonnegative pixel scores.
        # Raw tiny scores become equal after a second softmax, even in FP32.
        # log(scores) makes teacher softmax normalize these scores instead,
        # preserving their argmax. Zero scores map to the smallest normal FP32
        # value; all-zero pixels remain a tie with the original class-0 winner.
        # Model and decoder calculations keep the original AMP context.
        return scores.float().clamp_min(torch.finfo(torch.float32).tiny).log()


class SynapseTeacherHD95Semantic(SynapsePhysicalHD95Semantic):
    def setup(self, stage):
        if stage not in (None, "test"):
            raise ValueError("Teacher adapter is an independent test interface, not a training configuration")
        super().setup(stage)
        self.teacher_utils, self.teacher_source = load_teacher_utils()

    def evaluate_synapse_case(self, batch):
        if self._physical_case_context is not None:
            raise RuntimeError("Reentrant teacher evaluation is unsupported")
        case = batch["case_name"]
        if not isinstance(case, str):
            raise ValueError("Expected one case-name string")
        image, label = batch["image"], batch["label"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if label.ndim == 3:
            label = label.unsqueeze(0)
        if image.ndim != 4 or label.ndim != 4 or image.shape[0] != 1 or image.shape != label.shape:
            raise ValueError("Teacher evaluation requires a single 3D image and label of equal shape")
        geometry = self.physical_geometry.for_case(case, label[0].detach().cpu().numpy())
        records = []
        original_metric = self.teacher_utils.calculate_metric_percase

        def collect_metric(pred, gt):
            has_pred, has_gt = bool(pred.any()), bool(gt.any())
            if self.compute_hd95_voxel:
                dice, voxel = original_metric(pred.copy(), gt.copy())
            else:
                dice = float(metric.binary.dc(pred, gt)) if has_pred and has_gt else (1.0 if has_pred and not has_gt else 0.0)
                voxel = None
            if has_pred and has_gt:
                physical = float(metric.binary.hd95(pred, gt, voxelspacing=geometry.spacing_zyx_mm))
                status = "nonempty"
            else:
                physical = 0.0
                status = ("empty_gt_legacy_zero" if has_pred else
                          "empty_pred_legacy_zero" if has_gt else "both_empty_legacy_zero")
            records.append({"dice": float(dice), "hd95_voxel": float(voxel) if voxel is not None else None,
                            "hd95_mm": physical, "status": status})
            return dice, voxel if self.compute_hd95_voxel else physical  # Numeric pair required by teacher; disabled voxel is never computed.

        self._physical_case_context = (geometry.spacing_zyx_mm, records)
        self.teacher_utils.calculate_metric_percase = collect_metric
        try:
            values = self.teacher_utils.test_single_volume(
                image, label, EoSegPixelAdapter(self).eval(), classes=self.num_classes,
                patch_size=list(self.img_size), test_save_path=None, case=case,
                z_spacing=geometry.spacing_zyx_mm[0])
        finally:
            self.teacher_utils.calculate_metric_percase = original_metric
            self._physical_case_context = None
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.num_classes - 1, 2) or len(records) != self.num_classes - 1:
            raise RuntimeError("Teacher returned incomplete per-organ metrics")
        if not np.isfinite(values).all():
            raise ValueError("Teacher returned nonfinite metrics")
        physical = [row["hd95_mm"] for row in records]
        return {"case": case, "mean_dice": float(values[:, 0].mean()),
                "dice_per_class": values[:, 0].tolist(),
                "mean_hd95": float(np.mean(physical)), "hd95_per_class": physical,
                "hd95_unit": "mm", "hd95_mm_per_class": physical,
                "mean_hd95_mm": float(np.mean(physical)),
                "hd95_voxel_per_class": values[:, 1].tolist() if self.compute_hd95_voxel else [None] * (self.num_classes - 1),
                "hd95_voxel_enabled": self.compute_hd95_voxel,
                "mean_hd95_voxel": float(values[:, 1].mean()) if self.compute_hd95_voxel else None,
                "teacher_mean_hd95_voxel": float(values[:, 1].mean()) if self.compute_hd95_voxel else None,
                "spacing_zyx_mm": list(geometry.spacing_zyx_mm),
                "metric_status_per_class": [row["status"] for row in records]}
