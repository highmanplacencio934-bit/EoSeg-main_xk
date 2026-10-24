"""Optional slice-level organ presence auxiliary loss; no changes to old trainer."""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.mask_classification_semantic import MaskClassificationSemantic


class MaskClassificationSemanticCCPS(MaskClassificationSemantic):
    # Explicit signature is intentional: LightningCLI/jsonargparse needs to
    # discover the inherited training settings in addition to CCPS knobs.
    def __init__(
        self, network: nn.Module, img_size: tuple[int, int], num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4, llrd: float = 0.8,
        llrd_l2_enabled: bool = True, lr_mult: float = 1.0,
        weight_decay: float = 0.05, num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: Optional[List[int]] = None,
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8, overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        presence_loss_weight: float = 0.0,
        presence_pos_weight: float = 1.0,
    ):
        super().__init__(
            network=network, img_size=img_size, num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx, lr=lr, llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled, lr_mult=lr_mult,
            weight_decay=weight_decay, num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=[500, 1000] if warmup_steps is None else warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh, overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path, delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        if presence_loss_weight < 0 or presence_pos_weight <= 0:
            raise ValueError("Invalid presence loss weights")
        self.presence_loss_weight = float(presence_loss_weight)
        self.presence_pos_weight = float(presence_pos_weight)
        if (self.presence_loss_weight > 0 and
                self.network.asee.selector_mode not in {"visual", "visual_prior"}):
            raise ValueError("Presence supervision requires a slice-conditioned selector")

    @staticmethod
    def build_presence_targets(targets, num_foreground, device):
        """targets list is AFTER training transforms/crop, never the test volume."""
        labels = torch.zeros((len(targets), num_foreground),
                             dtype=torch.float32, device=device)
        for i, target in enumerate(targets):
            cls = target["labels"].to(device=device, dtype=torch.long)
            cls = cls[(cls >= 1) & (cls <= num_foreground)].unique()
            labels[i, cls - 1] = 1.0
        return labels

    def training_step(self, batch, batch_idx):
        # Parent computes and logs the UNMODIFIED original segmentation loss.
        loss_seg = super().training_step(batch, batch_idx)
        if self.presence_loss_weight == 0:
            return loss_seg
        logits = self.network.asee.last_selector_logits
        if logits is None:
            raise RuntimeError("Selector logits were not saved in training forward")
        targets = self.build_presence_targets(
            batch[1], logits.shape[1], device=logits.device)
        pos_weight = torch.full((logits.shape[1],), self.presence_pos_weight,
                                dtype=logits.dtype, device=logits.device)
        loss_presence = F.binary_cross_entropy_with_logits(
            logits.float(), targets, pos_weight=pos_weight.float())
        loss_total = loss_seg + self.presence_loss_weight * loss_presence
        self.log("losses/train_ccps_presence", loss_presence,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("losses/train_total_with_ccps", loss_total,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_total
