"""CPU fixtures execute the uploaded teacher functions; CUDA is patched only here."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.mask_classification_semantic import MaskClassificationSemantic
from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.synapse_teacher_semantic import SynapseTeacherHD95Semantic, EoSegPixelAdapter, load_teacher_utils, ROOT
from test_synapse_physical_hd95 import TinyNetwork, make_manifest
from main import LightningCLI
from lightning.pytorch.cli import LightningCLI as BaseCLI

class TeacherAdapterTests(unittest.TestCase):
    def test_exact_uploaded_metric_matches_legacy_including_empty_masks(self):
        module, meta = load_teacher_utils()
        self.assertEqual(meta['original_hd95_unit'], 'voxel_grid')
        rng=np.random.default_rng(11)
        for pred,gt in [(rng.random((5,6,7))>.8,rng.random((5,6,7))>.8)]+[(np.full((3,4,5),a,bool),np.full((3,4,5),b,bool)) for a,b in [(0,0),(0,1),(1,0)]]:
            self.assertEqual(module.calculate_metric_percase(pred.copy(),gt.copy()),MaskClassificationSemantic.calculate_metric_percase(pred.copy(),gt.copy()))

    def test_real_teacher_volume_path_matches_original_inference_and_true_mm(self):
        for shape in [(2,8,8),(3,10,12)]:
            with self.subTest(shape=shape),tempfile.TemporaryDirectory() as tmp:
                label=np.zeros(shape,np.uint8);label[:,2,2]=1
                path=Path(tmp)/'spacing.json';make_manifest(path,{'case0001':label})
                kwargs=dict(img_size=(8,8),num_classes=9,attn_mask_annealing_enabled=False,compute_hd95_voxel=True)
                model=SynapseTeacherHD95Semantic(network=TinyNetwork(),spacing_manifest=str(path),**kwargs).eval()
                model.teacher_utils,model.teacher_source=load_teacher_utils()
                old=SynapsePhysicalHD95Semantic(network=TinyNetwork(),spacing_manifest=str(path),**kwargs).eval()
                batch={'case_name':'case0001','image':torch.zeros(shape),'label':torch.from_numpy(label)}
                expected=old.evaluate_synapse_case(batch)
                original=model.teacher_utils.calculate_metric_percase
                with patch.object(torch.Tensor,'cuda',lambda tensor,*a,**k:tensor),patch.object(model.teacher_utils,'test_single_volume',wraps=model.teacher_utils.test_single_volume) as actual:
                    result=model.evaluate_synapse_case(batch)
                    actual.assert_called_once()
                for key in ['dice_per_class','hd95_voxel_per_class','hd95_mm_per_class','metric_status_per_class']:
                    self.assertEqual(result[key],expected[key])
                self.assertIs(model.teacher_utils.calculate_metric_percase,original)
                self.assertIsNone(model._physical_case_context)
                self.assertEqual(set(model.state_dict()),set(old.state_dict()))
                if shape==(2,8,8):
                    self.assertEqual(result['hd95_mm_per_class'][0],2)
                    self.assertEqual(result['hd95_voxel_per_class'][0],1)

    def test_teacher_softmax_receives_log_scores_and_keeps_tiny_winner(self):
        class ScoredModel(nn.Module):
            img_size=(1,1)
            def forward(self,image):return [torch.zeros(1)], [torch.zeros(1)]
            def to_per_pixel_logits_semantic(self,*args):
                scores=torch.zeros((1,9,1,1));scores[:,5]=2**-24
                return scores
        logits=EoSegPixelAdapter(ScoredModel())(torch.zeros((1,1,1,1)))
        probabilities=logits.softmax(1)
        expected=torch.zeros_like(probabilities);expected[:,5]=1
        torch.testing.assert_close(probabilities,expected,atol=1e-7,rtol=0)
        self.assertEqual(int(probabilities.argmax(1)),5)

    def test_metric_callback_restored_after_teacher_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            label=np.zeros((2,8,8),np.uint8);label[:,2,2]=1
            path=Path(tmp)/'spacing.json';make_manifest(path,{'case0001':label})
            model=SynapseTeacherHD95Semantic(network=TinyNetwork(),spacing_manifest=str(path),img_size=(8,8),num_classes=9,attn_mask_annealing_enabled=False)
            model.teacher_utils,model.teacher_source=load_teacher_utils();original=model.teacher_utils.calculate_metric_percase
            with patch.object(model.teacher_utils,'test_single_volume',side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                    model.evaluate_synapse_case({'case_name':'case0001','image':torch.zeros(label.shape),'label':torch.from_numpy(label)})
            self.assertIsNone(model._physical_case_context);self.assertIs(model.teacher_utils.calculate_metric_percase,original)

    def test_teacher_checkpoint_class_is_preserved_only_for_test(self):
        for sub in ['test','fit']:
            cli=object.__new__(LightningCLI);cli.config={'subcommand':sub,sub:{'model':{'class_path':'training.synapse_teacher_semantic.SynapseTeacherHD95Semantic'}}}
            with patch.object(BaseCLI,'_parse_ckpt_path') as old:
                cli._parse_ckpt_path()
                if sub=='test':old.assert_not_called()
                else:old.assert_called_once_with()

    def test_changed_teacher_source_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'utils.py';path.write_text('raise RuntimeError("must never run")')
            with patch('training.synapse_teacher_semantic.TEACHER_UTILS',path):
                with self.assertRaisesRegex(ValueError,'changed'):load_teacher_utils()

    def test_teacher_config_keeps_network_data_and_scientific_settings(self):
        base=yaml.safe_load((ROOT/'configs/synapse/vit_query_mul_scale_fusion_asee_plan_hd95_mm.yaml').read_text())
        new=yaml.safe_load((ROOT/'configs/synapse/vit_query_mul_scale_fusion_asee_teacher_hd95.yaml').read_text())
        self.assertEqual(base['data'],new['data']);self.assertEqual(base['model']['init_args'],new['model']['init_args'])
        for key in ['precision','max_epochs','gradient_clip_val']:
            self.assertEqual(base['trainer'].get(key),new['trainer'].get(key))

    def test_entry_has_no_import_time_cli_and_preserves_checkpoint_options(self):
        spec=importlib.util.spec_from_file_location('_teacher_entry_test',ROOT/'HD95_test.py');module=importlib.util.module_from_spec(spec)
        with patch('main.cli_main') as run:
            spec.loader.exec_module(module);run.assert_not_called()
            with patch.object(sys,'argv',['HD95_test.py','--ckpt_path','fixture.ckpt','--compile_test']):
                module.main();run.assert_called_once()
                self.assertEqual(sys.argv[1],'test');self.assertIn('fixture.ckpt',sys.argv);self.assertIn('--compile_test',sys.argv)

if __name__=='__main__':unittest.main()
