"""Behavioral contracts for the opt-in augmentation, never alter production defaults."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.synapse_transforms import Transforms
from datasets.synapse_joint_transforms import JointTransforms
from datasets.synapse_joint_semantic import SynapseJointSemantic
from datasets.synapse_npz_dataset import label_to_eomt_target
from training.synapse_joint_selection import SynapseJointSelection
from training.synapse_physical_selection import SynapsePhysicalHD95Selection


def fixture():
    label = torch.zeros((40, 48), dtype=torch.long)
    label[8:23, 6:15] = 3
    label[15:33, 30:43] = 4
    return tv_tensors.Image(label.float().repeat(3, 1, 1)), label_to_eomt_target(label)


def cloned(img, target):
    return img.clone(), {k: v.clone() for k, v in target.items()}


class JointAugmentationTests(unittest.TestCase):
    def test_disabled_rotation_exactly_matches_original_and_rng(self):
        img, target = fixture()
        for degrees, probability in [(0., .3), (10., 0.)]:
            old = Transforms((32, 32), True, (.5, 2.))
            new = JointTransforms((32, 32), True, (.5, 2.), degrees, probability)
            for seed in range(12):
                torch.manual_seed(seed)
                a, ta = old(*cloned(img, target))
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                b, tb = new(*cloned(img, target))
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for k in ta: self.assertTrue(torch.equal(ta[k], tb[k]), k)

    def test_constructor_preserves_rng(self):
        torch.manual_seed(34)
        state = torch.get_rng_state()
        JointTransforms((32, 32))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        SynapseJointSemantic(num_workers=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_rotation_shares_angle_and_preserves_ids_and_binary_masks(self):
        img, target = fixture()
        aug = JointTransforms((32, 32), False, (1., 1.), 10., 1.)
        for angle in [-10., 7., 10.]:
            with patch('torchvision.transforms.v2.RandomRotation.make_params', return_value={'angle': angle}):
                out, tgt = aug.random_rotation(*cloned(img, target))
            expected = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            expected_mask = F.rotate(target['masks'], angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)
            self.assertTrue(torch.equal(out, expected))
            self.assertTrue(torch.equal(tgt['masks'], expected_mask))
            self.assertEqual(tgt['masks'].dtype, torch.bool)
            self.assertIsInstance(tgt['masks'], tv_tensors.Mask)
            self.assertTrue(torch.equal(tgt['labels'], target['labels']))
            self.assertTrue(torch.equal(tgt['is_crowd'], target['is_crowd']))

    def test_rotation_corner_uses_legacy_zero_all_masks(self):
        img = tv_tensors.Image(torch.ones((3, 40, 40)))
        target = label_to_eomt_target(torch.zeros((40, 40), dtype=torch.long))
        aug = JointTransforms((40, 40), False, (1., 1.), 10., 1.)
        with patch('torchvision.transforms.v2.RandomRotation.make_params', return_value={'angle': 10.}):
            out, tgt = aug.random_rotation(img, target)
        self.assertEqual(float(out[:, 0, 0].sum()), 0.)
        self.assertFalse(tgt['masks'][:, 0, 0].any())

    def test_seeded_samples_repeat_and_angles_vary(self):
        img, target = fixture()
        aug = JointTransforms((32, 32), False, (.75, 1.5), 10., 1.)
        def sample(seed):
            torch.manual_seed(seed)
            return [aug(*cloned(img, target))[0] for _ in range(12)]
        a, b = sample(7), sample(7)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertTrue(any(not torch.equal(a[0], x) for x in a[1:]))

    def test_scale_boundaries_and_filtered_fields(self):
        img, target = fixture()
        for scale in [.75, 1.5]:
            aug = JointTransforms((32, 32), False, (scale, scale), 10., .3)
            for seed in range(5):
                torch.manual_seed(seed)
                out, tgt = aug(*cloned(img, target))
                self.assertEqual(tuple(out.shape), (3, 32, 32))
                self.assertTrue(torch.isfinite(out).all())
                self.assertEqual(tgt['masks'].dtype, torch.bool)
                self.assertTrue(set(tgt['labels'].tolist()) <= {0, 3, 4})
                self.assertEqual(tgt['labels'].numel(), len(tgt['masks']))
                self.assertEqual(tgt['labels'].numel(), len(tgt['is_crowd']))
                self.assertTrue(tgt['masks'].flatten(1).any(1).all())

    def test_empty_crop_retries_original_input(self):
        img, target = fixture()
        aug = JointTransforms((32, 32), False, (1., 1.), 10., .3)
        original = aug.random_crop
        calls = []
        class FirstEmpty(torch.nn.Module):
            def forward(self, im, tgt):
                calls.append(True)
                im, tgt = original(im, tgt)
                if len(calls) == 1:
                    tgt['masks'] = tv_tensors.Mask(torch.zeros_like(tgt['masks']))
                return im, tgt
        aug.random_crop = FirstEmpty()
        out, tgt = aug(*cloned(img, target))
        self.assertEqual(len(calls), 2)
        self.assertTrue(tgt['masks'].any())

    def test_invalid_parameters_rejected(self):
        for kw in [dict(rotation_degrees=-1), dict(rotation_degrees=float('nan')),
                   dict(rotation_p=1.1), dict(rotation_p=float('nan')),
                   dict(scale_range=(0., 1.)), dict(scale_range=(2., 1.))]:
            with self.assertRaises(ValueError): JointTransforms((32, 32), **kw)

    def test_data_module_only_selects_independent_transform(self):
        from datasets.synapse_semantic import SynapseSemantic
        original = SynapseSemantic(num_workers=0)
        new = SynapseJointSemantic(num_workers=0)
        self.assertIs(type(original.transforms), Transforms)
        self.assertIsInstance(new.transforms, JointTransforms)
        self.assertEqual(new.val_dataloader(), [])
        for key in ['train_base_dir', 'test_base_dir', 'list_dir', 'train_split', 'test_split']:
            self.assertEqual(getattr(original, key), getattr(new, key))
        self.assertEqual(new.hparams.rotation_degrees, 10.)
        self.assertEqual(new.hparams.rotation_p, .3)
        self.assertEqual(tuple(new.hparams.scale_range), (.75, 1.5))

    def test_yaml_only_changes_declared_factors_and_bookkeeping(self):
        base = yaml.safe_load((ROOT/'configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/synapse/vit_query_mul_scale_fusion_asee_ccps_joint_aug.yaml').read_text())
        new['data']['class_path'] = base['data']['class_path']
        for key in ['rotation_degrees', 'rotation_p', 'scale_range']: new['data']['init_args'].pop(key)
        new['trainer']['default_root_dir'] = base['trainer']['default_root_dir']
        new['trainer']['logger'] = base['trainer']['logger']
        new['trainer']['callbacks'] = base['trainer']['callbacks']
        self.assertEqual(new, base)

    def test_matched_window_archive_does_not_replace_main_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = SynapseJointSelection(matched_window_end=213)
            cb.run_dir = Path(tmp); cb.best_epoch = 210; cb.best_score = .8
            saved = []
            trainer = SimpleNamespace(current_epoch=209, max_epochs=220,
                save_checkpoint=lambda path, weights_only: saved.append((Path(path).name, weights_only)))
            with patch.object(SynapsePhysicalHD95Selection, 'on_train_epoch_end'):
                cb.on_train_epoch_end(trainer, None)
                trainer.current_epoch=213; cb.best_epoch=214; cb.best_score=.9
                cb.on_train_epoch_end(trainer, None)
            self.assertEqual(saved, [('best_test_through_epoch213.ckpt', True)])
            result = json.loads((Path(tmp)/'matched_window_summary.json').read_text())
            self.assertEqual(result['best_epoch'], 210)
            self.assertEqual(cb.best_epoch, 214)


if __name__ == '__main__': unittest.main()
