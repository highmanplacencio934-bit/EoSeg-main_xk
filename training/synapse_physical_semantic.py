"""Opt-in physical HD95; inherit the original model, loss and inference path."""
from typing import List, Optional

import numpy as np
import torch.nn as nn

from training.mask_classification_semantic import MaskClassificationSemantic
from training.synapse_physical_geometry import SpacingManifest, binary_metrics_mm


class SynapsePhysicalHD95Semantic(MaskClassificationSemantic):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        *,
        spacing_manifest: str,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        self.spacing_manifest = spacing_manifest
        self.physical_geometry = SpacingManifest(spacing_manifest)
        self.save_hyperparameters({"spacing_manifest": spacing_manifest})
        self._physical_case_context = None
        self.physical_preflight = None
        self.physical_checkpoint_epoch = None
        self.physical_checkpoint_global_step = None

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        epoch = checkpoint.get("epoch")
        self.physical_checkpoint_epoch = epoch + 1 if isinstance(epoch, int) and epoch >= 0 else None
        self.physical_checkpoint_global_step = checkpoint.get("global_step")

    def setup(self, stage):
        super().setup(stage)
        if stage in (None, "fit", "test"):
            data = self.trainer.datamodule
            self.physical_preflight = self.physical_geometry.validate_dataset(
                data.test_base_dir, f"{data.list_dir}/{data.test_split}.txt")

    def evaluate_synapse_case(self, batch):
        if self._physical_case_context is not None:
            raise RuntimeError("Concurrent/reentrant physical case evaluation is unsupported")
        case = batch["case_name"]
        if not isinstance(case, str):
            raise ValueError("Physical evaluation expects a single case-name string")
        label = batch["label"].detach().cpu().numpy()
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]
        geometry = self.physical_geometry.for_case(case, label)
        records = []
        self._physical_case_context = (geometry.spacing_zyx_mm, records)
        try:
            result = super().evaluate_synapse_case(batch)
        finally:
            self._physical_case_context = None
        if len(records) != self.num_classes - 1:
            raise RuntimeError("Incomplete physical per-organ metric collection")
        voxel = [r["hd95_voxel"] for r in records]
        return {**result, "hd95_unit": "mm", "hd95_mm_per_class": result["hd95_per_class"],
                "mean_hd95_mm": result["mean_hd95"], "hd95_voxel_per_class": voxel,
                "mean_hd95_voxel": float(np.mean(voxel)),
                "spacing_zyx_mm": list(geometry.spacing_zyx_mm),
                "metric_status_per_class": [r["status"] for r in records]}

    def calculate_metric_percase(self, pred, gt):
        if self._physical_case_context is None:
            raise RuntimeError("Physical metric requires verified per-case geometry")
        spacing, records = self._physical_case_context
        result = binary_metrics_mm(pred, gt, spacing)
        records.append(result)
        return result["dice"], result["hd95_mm"]

    def test_step(self, batch, batch_idx):
        result = self.evaluate_synapse_case(batch)
        for c, (dice, physical, voxel) in enumerate(zip(
                result["dice_per_class"], result["hd95_mm_per_class"],
                result["hd95_voxel_per_class"]), start=1):
            self.log(f"test/dice_class_{c}", dice, sync_dist=True, batch_size=1)
            self.log(f"test/hd95_mm_class_{c}", physical, sync_dist=True, batch_size=1)
            self.log(f"test/hd95_voxel_class_{c}", voxel, sync_dist=True, batch_size=1)
        self.log("test/mean_dice", result["mean_dice"], sync_dist=True, batch_size=1)
        self.log("test/mean_hd95_mm", result["mean_hd95_mm"], sync_dist=True, batch_size=1)
        self.log("test/mean_hd95_voxel", result["mean_hd95_voxel"], sync_dist=True, batch_size=1)
        return result
