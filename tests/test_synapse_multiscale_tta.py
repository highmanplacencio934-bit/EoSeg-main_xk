"""Geometry, metric and provenance contracts for opt-in scale/horizontal TTA."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import torch
import yaml
from training.synapse_multiscale_tta import (
    PRESETS, scaled_view, restore_scores, scale_pair, fuse,
    predict_multiscale_tta_scores, evaluate_multiscale_case,
    bind_multiscale_tta, multiscale_contract, validated_readiness,
    SynapseMultiScaleTTASelection)
from training.synapse_tta_common import pixel_scores, file_sha256
import test_synapse_tta_common as shared_tests
Toy = shared_tests.Toy

ROOT = Path(__file__).resolve().parents[1]

class MultiScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_identity_pair_equals_manual_horizontal_score_mean(self):
        for half in (False, True):
            x=torch.rand(1,3,8,8); model=Toy(half)
            expected = (pixel_scores(model, x).float() + torch.flip(
                pixel_scores(model, torch.flip(x, (-1,))).float(), (-1,))) / 2
            self.assertTrue(torch.equal(predict_multiscale_tta_scores(model,x,2), expected))

    def test_constant_scores_preserve_coverage_normalization(self):
        image=torch.zeros(1,3,32,32); pairs={}
        for scale in PRESETS[8]:
            _,geometry=scaled_view(image,scale)
            pairs[scale]=restore_scores(torch.full((1,9,32,32),.7),geometry)
        for scales in PRESETS.values():
            torch.testing.assert_close(fuse(pairs,scales),torch.full((1,9,32,32),.7))

    def test_zoom_in_excludes_unseen_edges(self):
        _,geometry=scaled_view(torch.zeros(1,3,32,32),1.125)
        num,den=restore_scores(torch.ones(1,9,32,32),geometry)
        self.assertEqual(den[0,0,0,0],0)
        self.assertEqual(den[0,0,16,16],1)
        self.assertTrue(torch.equal(num[:,:1],den))
        identity=(torch.full_like(num,2),torch.ones_like(den))
        mixed=fuse({1.:identity,1.125:(num,den)},(1.,1.125))
        self.assertEqual(mixed[0,0,0,0],2)
        self.assertEqual(mixed[0,0,16,16],1.5)

    def test_zoom_out_discards_padding_predictions(self):
        _,geometry=scaled_view(torch.zeros(1,3,32,32),.75)
        pred=torch.full((1,9,32,32),999.);pred[...,4:28,4:28]=2
        num,den=restore_scores(pred,geometry)
        self.assertTrue(torch.all(num==2));self.assertTrue(torch.all(den==1))

    def test_horizontal_channels_preserved(self):
        class ClassToy(Toy):
            def forward(self,image):
                out=torch.zeros(1,9,8,8);out[:,3]=2
                return [out],[None]
        for views in PRESETS:
            self.assertTrue((predict_multiscale_tta_scores(ClassToy(),torch.zeros(1,3,8,8),views).argmax(1)==3).all())

    def test_invalid_geometry_and_counts_rejected(self):
        for scale in (float('nan'),0.,3.):
            with self.assertRaises(ValueError):scaled_view(torch.zeros(1,3,8,8),scale)
        with self.assertRaises(ValueError):scaled_view(torch.zeros(1,3,8,6),1.)
        with self.assertRaises(ValueError):restore_scores(torch.zeros(1,9,6,6),(8,8,0))
        with self.assertRaises(ValueError):predict_multiscale_tta_scores(Toy(),torch.zeros(1,3,8,8),3)
        with self.assertRaises(ValueError):fuse({},(.875,))

    def test_nonfinite_prediction_rejected(self):
        class Bad(Toy):
            def forward(self,image):
                masks,classes=super().forward(image);masks[-1][:,0]=float('nan')
                return masks,classes
        with self.assertRaises(FloatingPointError):
            predict_multiscale_tta_scores(Bad(),torch.zeros(1,3,8,8),4)

    def test_case_metrics_keep_spacing_and_legacy_empty_rules(self):
        image=torch.zeros(2,8,8);image[:,2:6,1:4]=1
        batch={'case_name':'fixture','image':image,'label':image.long()}
        new=evaluate_multiscale_case(Toy(),batch,2)
        self.assertEqual(new['spacing_zyx_mm'],[3.,.7,.7])
        self.assertEqual(new['dice_per_class'][0],1.)
        self.assertEqual(new['dice_per_class'][1:], [0.]*7)

    def test_bound_case_names_scheme_and_views(self):
        model=Toy();model.evaluate_synapse_case=lambda batch:None
        original=model.evaluate_synapse_case
        self.assertIs(bind_multiscale_tta(model,4),original)
        image=torch.zeros(2,8,8)
        result=model.evaluate_synapse_case({'case_name':'fixture','image':image,'label':image.long()})
        self.assertEqual(result['tta_scheme'],'scale_hflip')
        self.assertEqual(result['tta_views'],4)

    def test_contract_has_only_scales_and_horizontal_flips(self):
        c=multiscale_contract(4,(384,384))
        self.assertEqual(c['scales'],[1.,.875])
        self.assertEqual([x['resized_hw'] for x in c['view_transforms']],[[384,384],[384,384],[336,336],[336,336]])
        self.assertEqual([x['horizontal_flip'] for x in c['view_transforms']],[False,True,False,True])
        self.assertTrue(all(x['rotation_degrees']==0 for x in c['view_transforms']))

    def test_config_only_changes_test_callback_output(self):
        folder=ROOT/'configs/synapse'
        old=yaml.safe_load((folder/'vit_query_mul_scale_fusion_asee_ccps.yaml').read_text())
        new=yaml.safe_load((folder/'vit_query_mul_scale_fusion_asee_ccps_multiscale_tta.yaml').read_text())
        self.assertEqual(new['model'],old['model']);self.assertEqual(new['data'],old['data'])
        self.assertEqual(new['trainer']['precision'],'16-mixed')
        self.assertIn('SynapseMultiScaleTTASelection',new['trainer']['callbacks'][0]['class_path'])
        for data in (old,new):
            data['trainer'].pop('callbacks');data['trainer'].pop('default_root_dir')

class MultiScaleManifestTests(unittest.TestCase):
    def test_own_readiness_and_immutable_run_manifest(self):
        import training.synapse_multiscale_tta as module
        from training.synapse_physical_selection import SynapsePhysicalHD95Selection
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            trainer,model,out,_=shared_tests.TTAManifestTests().fixture(root)
            model.tta_views=4;model.tta_scheme='scale_hflip';model.img_size=(384,384)
            ready=root/'scale_ready.json'
            source=root/'scripts/evaluate_synapse_multiscale_tta.py'
            ready.write_text(json.dumps({'state':'passed','tta_scheme':'scale_hflip',
                                        'sha256':{'scripts/evaluate_synapse_multiscale_tta.py':file_sha256(source)}}))
            raw=ready.read_bytes()
            def parent_start(callback,trainer,model):
                callback.run_dir=out;callback._test_checkpoint_info={'checkpoint_sha256':'fixture'}
            with mock.patch.object(module,'PROJECT_ROOT',root), mock.patch.object(module,'READINESS_PATH',ready), \
                 mock.patch('training.synapse_tta_common.PROJECT_ROOT',root), \
                 mock.patch.object(SynapsePhysicalHD95Selection,'on_test_start',parent_start):
                SynapseMultiScaleTTASelection().on_test_start(trainer,model)
                source.write_text('changed')
                with self.assertRaisesRegex(ValueError,'source changed'):validated_readiness(ready)
            result=json.loads((out/'run_manifest.json').read_text())
            self.assertEqual(result['tta_contract']['tta_scheme'],'scale_hflip')
            self.assertEqual(result['tta_contract']['tta_views'],4)
            self.assertEqual((out/'multiscale_readiness_snapshot.json').read_bytes(),raw)
            self.assertIn('multiscale_readiness_snapshot_sha256',result)

    def test_fit_is_refused(self):
        with self.assertRaisesRegex(ValueError,'independent test'):
            SynapseMultiScaleTTASelection().on_fit_start(None,None)

if __name__=='__main__':unittest.main()
