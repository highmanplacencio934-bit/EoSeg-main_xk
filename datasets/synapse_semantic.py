from typing import Optional, Union
import logging
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.synapse_transforms import Transforms
from datasets.synapse_splits import (DEFAULT_DATA_ROOT, DEFAULT_LIST_DIR,
                                     resolve_project_path, validate_synapse_split)
from datasets.synapse_npz_dataset import SynapseNPZ2DSliceDataset, SynapseH5VolumeCaseDataset


def single_case_collate(batch):
    return batch[0]


class SynapseSemantic(LightningDataModule):
    def __init__(
        self,
        train_base_dir: Optional[str] = None,
        test_base_dir: Optional[str] = None,
        list_dir: Optional[str] = None,
        img_size=(384, 384),
        num_classes: int = 9,
        batch_size: int = 16,
        num_workers: int = 8,
        color_jitter_enabled: bool = True,
        scale_range=(0.5, 2.0),
        check_empty_targets: bool = True,
        ignore_idx: int = 255,
        drop_background: bool = False,
        make_rgb: bool = True,
        train_split: str = "train",
        test_split: str = "test_vol",
    ):
        super().__init__(
            path="",
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.train_base_dir = str(resolve_project_path(train_base_dir or f"{DEFAULT_DATA_ROOT}/train_npz"))
        self.test_base_dir = str(resolve_project_path(test_base_dir or f"{DEFAULT_DATA_ROOT}/test_vol_h5"))
        self.list_dir = str(resolve_project_path(list_dir or DEFAULT_LIST_DIR))
        self.train_split = train_split
        self.test_split = test_split
        self.split_summary = None
        self.drop_background = drop_background
        self.make_rgb = make_rgb

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None):
        # Recheck on every setup so fit -> test also checks the newly active files.
        self.split_summary = validate_synapse_split(
            self.train_base_dir, self.test_base_dir, self.list_dir,
            self.train_split, self.test_split,
            check_train_files=stage in (None, "fit"),
            check_test_files=stage in (None, "test"),
        )
        logging.info("Synapse fixed split (%s): %s", stage, self.split_summary)
        if stage == "fit" or stage is None:
            self.train_dataset = SynapseNPZ2DSliceDataset(
                base_dir=self.train_base_dir,
                list_dir=self.list_dir,
                split=self.train_split,
                transforms=self.transforms,
                ignore_index=self.ignore_idx,
                drop_background=self.drop_background,
                make_rgb=self.make_rgb,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SynapseH5VolumeCaseDataset(
                base_dir=self.test_base_dir,
                list_dir=self.list_dir,
                split=self.test_split,
                transforms=None,
                ignore_index=self.ignore_idx,
                make_rgb=self.make_rgb,
            )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        # This fixed split has no independent validation set.
        return []

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=single_case_collate,   
        )
