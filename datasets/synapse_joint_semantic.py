"""Separate Synapse DataModule selecting the joint augmentation transform."""
from typing import Optional
from datasets.synapse_semantic import SynapseSemantic
from datasets.synapse_joint_transforms import JointTransforms


class SynapseJointSemantic(SynapseSemantic):
    def __init__(self, train_base_dir: Optional[str] = None, test_base_dir: Optional[str] = None,
                 list_dir: Optional[str] = None, img_size=(384, 384), num_classes: int = 9,
                 batch_size: int = 16, num_workers: int = 8, color_jitter_enabled: bool = True,
                 scale_range=(0.75, 1.50), check_empty_targets: bool = True,
                 ignore_idx: int = 255, drop_background: bool = False, make_rgb: bool = True,
                 train_split: str = "train", test_split: str = "test_vol",
                 rotation_degrees: float = 10.0, rotation_p: float = 0.3):
        super().__init__(train_base_dir=train_base_dir, test_base_dir=test_base_dir,
                         list_dir=list_dir, img_size=img_size, num_classes=num_classes,
                         batch_size=batch_size, num_workers=num_workers,
                         color_jitter_enabled=color_jitter_enabled, scale_range=scale_range,
                         check_empty_targets=check_empty_targets, ignore_idx=ignore_idx,
                         drop_background=drop_background, make_rgb=make_rgb,
                         train_split=train_split, test_split=test_split)
        self.save_hyperparameters(ignore=["_class_path"])
        self.transforms = JointTransforms(img_size=img_size,
            color_jitter_enabled=color_jitter_enabled, scale_range=scale_range,
            rotation_degrees=rotation_degrees, rotation_p=rotation_p)
