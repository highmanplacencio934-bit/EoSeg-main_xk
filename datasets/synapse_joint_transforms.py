"""Opt-in joint augmentation; the author transform remains unchanged."""
import math
import torch
from torchvision.transforms import InterpolationMode, v2 as T
from datasets.synapse_transforms import Transforms


class JointTransforms(Transforms):
    def __init__(self, img_size, color_jitter_enabled=True, scale_range=(0.75, 1.50),
                 rotation_degrees: float = 10.0, rotation_p: float = 0.3):
        if (not math.isfinite(rotation_degrees) or not 0 <= rotation_degrees <= 180
                or not math.isfinite(rotation_p) or not 0 <= rotation_p <= 1):
            raise ValueError("Invalid rotation degrees or probability")
        if (len(scale_range) != 2 or not all(math.isfinite(x) for x in scale_range)
                or not 0 < scale_range[0] <= scale_range[1]):
            raise ValueError("Invalid scale range")
        super().__init__(img_size=img_size, color_jitter_enabled=color_jitter_enabled,
                         scale_range=scale_range)
        self.rotation_degrees = float(rotation_degrees)
        self.rotation_p = float(rotation_p)
        # Construction and the disabled path must not consume random numbers.
        self.random_rotation = (T.RandomApply([T.RandomRotation(
            degrees=(-rotation_degrees, rotation_degrees),
            interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)], p=rotation_p)
            if rotation_p > 0 and rotation_degrees > 0 else None)

    def forward(self, img, target):
        if self.random_rotation is None:
            return super().forward(img, target)
        img_orig, target_orig = img, target
        target = self._filter(target, ~target["is_crowd"])
        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        # torchvision v2 shares the angle; Mask dispatch uses nearest interpolation.
        img, target = self.random_rotation(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)
        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)
        return img, self._filter(target, valid)
