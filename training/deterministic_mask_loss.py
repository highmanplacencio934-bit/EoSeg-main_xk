"""Opt-in deterministic loss, preserving the original EoSeg objective.

The matcher and mask-loss sampling follow Transformers 5.17.0 Mask2Former
(Apache-2.0) and the project's MaskClassificationLoss. Only the local point
sampler and the weighted-CE reduction layout differ. No global Transformers
functions or the original criterion are modified.
"""

from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    dice_loss,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sigmoid_cross_entropy_loss,
)

from training.deterministic_point_sampling import sample_point
from training.mask_classification_loss import MaskClassificationLoss


class DeterministicHungarianMatcher(Mask2FormerHungarianMatcher):
    """Original matching costs and random draws with a local sampling dependency."""

    @staticmethod
    def _sample_point(input_features, point_coordinates, **kwargs):
        return sample_point(input_features, point_coordinates, **kwargs)

    @torch.no_grad()
    def forward(
        self, masks_queries_logits, class_queries_logits, mask_labels, class_labels
    ):
        indices = []
        for i in range(masks_queries_logits.shape[0]):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)[:, None]
            pred_mask = pred_mask[:, None]

            # Keep one shared coordinate draw per image, before target/pred samples.
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = self._sample_point(
                target_mask, target_coordinates, align_corners=False
            ).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = self._sample_point(
                pred_mask, pred_coordinates, align_corners=False
            ).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            # Preserve the installed original's treatment of infinite/NaN costs.
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            indices.append(linear_sum_assignment(cost_matrix.cpu()))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class DeterministicMaskClassificationLoss(MaskClassificationLoss):
    """Explicit alternative; coefficients, targets, draws and normalization stay fixed.

    The sampler supports input-feature gradients needed by this criterion. Point
    selection and GT sampling retain the original no-grad boundary, so coordinate
    gradients are neither requested nor silently detached by this loss.
    """

    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        super().__init__(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_labels,
            no_object_coefficient=no_object_coefficient,
        )
        self.matcher = DeterministicHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @classmethod
    def from_existing(cls, criterion: MaskClassificationLoss):
        """Clone configuration/state without changing tensors, training mode or RNG."""
        if not isinstance(criterion, MaskClassificationLoss):
            raise TypeError("Expected the original MaskClassificationLoss criterion")
        result = cls(
            num_points=criterion.num_points,
            oversample_ratio=criterion.oversample_ratio,
            importance_sample_ratio=criterion.importance_sample_ratio,
            mask_coefficient=criterion.mask_coefficient,
            dice_coefficient=criterion.dice_coefficient,
            class_coefficient=criterion.class_coefficient,
            num_labels=criterion.num_labels,
            no_object_coefficient=criterion.eos_coef,
        )
        result.to(device=criterion.empty_weight.device, dtype=criterion.empty_weight.dtype)
        result.load_state_dict(criterion.state_dict(), strict=True)
        # Preserve matcher configuration even if explicitly customized on the source.
        for name in ("num_points", "cost_mask", "cost_dice", "cost_class"):
            setattr(result.matcher, name, getattr(criterion.matcher, name))
        result.train(criterion.training)
        result.matcher.train(criterion.matcher.training)
        return result

    @staticmethod
    def _sample_point(input_features, point_coordinates, **kwargs):
        return sample_point(input_features, point_coordinates, **kwargs)

    def loss_labels(self, class_queries_logits, class_labels, indices):
        pred_logits = class_queries_logits
        batch_size, num_queries, num_classes = pred_logits.shape
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )
        target_classes = torch.full(
            (batch_size, num_queries),
            fill_value=self.num_labels,
            dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o
        # CrossEntropy's weighted mean uses sum(weight[target]), including no-object.
        # Flatten B,Q directly; transposing before reshape would scramble classes.
        loss_ce = F.cross_entropy(
            pred_logits.reshape(-1, num_classes),
            target_classes.reshape(-1),
            weight=self.empty_weight,
            reduction="mean",
        )
        return {"loss_cross_entropy": loss_ce}

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function: Callable,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ):
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)
        point_coordinates = torch.rand(
            num_boxes, num_points_sampled, 2, device=logits.device
        )
        point_logits = self._sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(
            num_boxes, dtype=torch.long, device=logits.device
        )
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2
        )
        if num_random_points > 0:
            point_coordinates = torch.cat(
                [
                    point_coordinates,
                    torch.rand(num_boxes, num_random_points, 2, device=logits.device),
                ],
                dim=1,
            )
        return point_coordinates

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = self._sample_point(
                target_masks, point_coordinates, align_corners=False
            ).squeeze(1)
        point_logits = self._sample_point(
            pred_masks, point_coordinates, align_corners=False
        ).squeeze(1)

        # The original custom loss asks HF for num_masks=1, then normalizes once.
        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, 1),
            "loss_dice": dice_loss(point_logits, point_labels, 1),
        }
        num_masks = sum(len(tgt) for _, tgt in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)
        return {key: loss / num_masks for key, loss in losses.items()}
