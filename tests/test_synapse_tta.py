"""CPU contracts for inverse transforms, score fusion and independent TTA."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest import mock
import hashlib
import numpy as np
import torch
from torch import nn
import yaml
from training.synapse_tta import VIEW_AXES,VIEW_TRANSFORMS,transform_view,invert_view,predict_tta_scores,evaluate_tta_case,bind_tta,kidney_confusion
from training.synapse_physical_geometry import binary_metrics_mm

ROOT=Path(__file__).resolve().parents[1]


def fixture_tta_reports(directory):
    """Synthetic CSVs only; deliberately reorder view2 to test keyed joins."""
    import csv
    run=Path(directory);(run/"source_checkpoint.json").write_text('{"epoch":1,"fixture":true}')
    reports=[]
    for views in (1,2,4,8):
        folder=run/str(views);folder.mkdir()
        cases=[];cross=[]
        for i in range(12):
            case=f"fixture{i:02d}"
            for label in range(1,9):
                dice=100. if views==2 and label==1 else 80.
                hd=1. if views==4 else 2.
                status="nonempty"
                if views==2 and i==0 and label==3:dice=70.
                if views==2 and i==1 and label==4:hd=3.
                if views==4 and i==1 and label==3:dice=0.;hd=0.;status="empty_pred_legacy_zero"
                cases.append({"case":case,"label_id":label,"organ_zh":f"fixture_{label}",
                              "dice_percent":dice,"hd95_mm":hd,"status":status})
            cross.append({"case":case,"pred_left_on_gt_right_voxels":5 if views==2 and i==0 else 0,
                          "pred_right_on_gt_left_voxels":1,"gt_left_voxels":1000,"gt_right_voxels":900})
        means=[{"label_id":c,"organ_zh":f"fixture_{c}",
                "dice_percent":sum(x["dice_percent"] for x in cases if x["label_id"]==c)/12,
                "hd95_mm":sum(x["hd95_mm"] for x in cases if x["label_id"]==c)/12} for c in range(1,9)]
        summary={"mean_dice":sum(x["dice_percent"] for x in means)/800,
                 "mean_hd95_mm":sum(x["hd95_mm"] for x in means)/8,
                 "mean_hd95_voxel":1.,"test_seconds":10}
        (folder/"physical_test_summary.json").write_text(json.dumps(summary))
        (folder/"run_manifest.json").write_text('{"fixture":true,"tta_views":'+str(views)+'}')
        for name,data in [("physical_epoch_organ_metrics.csv",means),
                          ("physical_case_organ_metrics.csv",list(reversed(cases)) if views==2 else cases),
                          ("kidney_confusion.csv",list(reversed(cross)) if views==2 else cross)]:
            with (folder/name).open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        reports.append((views,folder,12))
    return reports


class Toy(nn.Module):
    def __init__(self,half=False):
        super().__init__();self.img_size=(8,8);self.device=torch.device("cpu");self.num_classes=9
        self.calls=0;self.half=half
        self.physical_geometry=SimpleNamespace(for_case=lambda case,gt:SimpleNamespace(spacing_zyx_mm=(3.,.7,.7)))
    def forward(self,image):
        self.calls+=1
        x=image[:,0:1]
        scores=torch.cat([1-x,x]+[torch.zeros_like(x)]*7,dim=1)
        if self.half:scores=scores.half()
        return [scores],[torch.zeros(1)]
    def to_per_pixel_logits_semantic(self,masks,classes):return masks

class TTATests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_view_sets_are_nested_unique_mirrors(self):
        self.assertEqual(VIEW_AXES[1],((),))
        self.assertEqual(VIEW_AXES[2],((),(-1,)))
        self.assertEqual(len(set(VIEW_AXES[4])),4)
        self.assertTrue(set(VIEW_AXES[2]).issubset(VIEW_AXES[4]))
    def test_inverse_flips_restore_asymmetric_predictions(self):
        x=torch.arange(64,dtype=torch.float32).view(1,1,8,8)/64
        image=x.repeat(1,3,1,1)
        for views in [1,2,4,8]:
            model=Toy();out=predict_tta_scores(model,image,views)
            self.assertEqual(model.calls,views)
            torch.testing.assert_close(out[:,1],x[:,0],rtol=0,atol=0)
    def test_multi_view_accumulator_is_fp32(self):
        image=torch.rand(1,3,8,8)
        self.assertEqual(predict_tta_scores(Toy(True),image,1).dtype,torch.float16)
        self.assertEqual(predict_tta_scores(Toy(True),image,2).dtype,torch.float32)
        self.assertEqual(predict_tta_scores(Toy(True),image,4).dtype,torch.float32)
    def test_scores_are_averaged_before_argmax(self):
        class Inconsistent(Toy):
            def forward(self,image):
                self.calls+=1
                out=torch.zeros(1,9,8,8)
                out[:,0]=.9 if self.calls==1 else .4
                out[:,1]=.1 if self.calls==1 else .6
                return [out],[torch.zeros(1)]
        out=predict_tta_scores(Inconsistent(),torch.zeros(1,3,8,8),2)
        torch.testing.assert_close(out[:,0],torch.full((1,8,8),.65))
        self.assertTrue((out.argmax(1)==0).all())
    def test_no_left_right_channel_swap(self):
        class ChannelToy(Toy):
            def forward(self,image):
                out=torch.zeros(1,9,8,8);out[:,3]=image[:,0]
                return [out],[torch.zeros(1)]
        out=predict_tta_scores(ChannelToy(),torch.ones(1,3,8,8),4)
        self.assertTrue((out.argmax(1)==3).all())
    def test_invalid_view_count_rejected(self):
        with self.assertRaises(ValueError):predict_tta_scores(Toy(),torch.zeros(1,3,8,8),3)
    def test_one_view_dispatches_original_method_exactly(self):
        m=Toy();expected={"case":"fixture","dice_per_class":[.123]}
        m.evaluate_synapse_case=lambda batch:copy.deepcopy(expected)
        m.calculate_metric_percase=lambda pred,gt:(0.,0.)
        original=bind_tta(m,1);result=m.evaluate_synapse_case({"label":torch.zeros(2,8,8,dtype=torch.long)})
        self.assertEqual(result['dice_per_class'],expected['dice_per_class'])
        self.assertEqual(m.calls,0)
        m.evaluate_synapse_case=original
    def test_mirror_case_preserves_grid_metrics_and_original_label_ids(self):
        image=torch.zeros(2,8,8);image[:,2:6,1:4]=1
        batch={"case_name":"fixture","image":image,"label":image.long()}
        for views in [1,2,4,8]:
            result=evaluate_tta_case(Toy(),batch,views)
            self.assertEqual(result['dice_per_class'][0],1)
            self.assertEqual(result['hd95_mm_per_class'][0],0)
            self.assertEqual(result['spacing_zyx_mm'],[3.,.7,.7])
            self.assertEqual(len(result['dice_per_class']),8)
    def test_real_spacing_and_legacy_empty_policy_unchanged(self):
        a=np.zeros((3,4,4),bool);b=a.copy();a[0,1,1]=True;b[1,1,1]=True
        result=binary_metrics_mm(a,b,(3.,.7,.7))
        self.assertEqual(result['hd95_mm'],3)
        self.assertEqual(result['hd95_voxel'],1)
        result=binary_metrics_mm(a,np.zeros_like(a),(3.,.7,.7))
        self.assertEqual(result['dice'],1);self.assertEqual(result['hd95_mm'],0)
    def test_original_training_config_does_not_import_tta(self):
        text=(ROOT/'configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml').read_text()
        self.assertNotIn('tta',text.lower())
        cfg=yaml.safe_load((ROOT/'configs/synapse/vit_query_mul_scale_fusion_asee_ccps_tta.yaml').read_text())
        self.assertEqual(cfg['model']['class_path'],'training.mask_classification_ccps.MaskClassificationSemanticCCPS')
        self.assertEqual(cfg['trainer']['callbacks'][0]['class_path'],'training.synapse_tta.SynapseTTASelection')

    def test_d4_contains_eight_distinct_views_and_exact_inverse(self):
        x=torch.arange(64).reshape(1,1,8,8)
        transformed=[]
        for k,axes in VIEW_TRANSFORMS[8]:
            y=transform_view(x,k,axes)
            self.assertTrue(torch.equal(invert_view(y,k,axes),x))
            transformed.append(tuple(y.flatten().tolist()))
        self.assertEqual(len(set(transformed)),8)
        self.assertEqual(VIEW_TRANSFORMS[8][:4],VIEW_TRANSFORMS[4])
    def test_d4_rejects_non_square_model_input(self):
        with self.assertRaises(ValueError):predict_tta_scores(Toy(),torch.zeros(1,3,8,6),8)

    def test_left_right_kidney_confusion_is_not_merged(self):
        gt=np.array([[[3,3,4,4]]]);pred=np.array([[[4,3,3,4]]])
        out=kidney_confusion(pred,gt)
        self.assertEqual(out["pred_left_on_gt_right_voxels"],1)
        self.assertEqual(out["pred_right_on_gt_left_voxels"],1)
        self.assertEqual(out["gt_left_voxels"],2)
        self.assertEqual(out["gt_right_voxels"],2)

    def test_original_metric_hook_captures_kidneys_and_restores(self):
        model=Toy();gt=np.zeros((1,8,8),dtype=np.int64);gt[0,:4]=3;gt[0,4:]=4
        pred=gt.copy();pred[0,0,0]=4;pred[0,7,7]=3
        metric=lambda p,g:(float(p.sum()),0.)
        model.calculate_metric_percase=metric
        def original(batch):
            pairs=[model.calculate_metric_percase(pred==c,gt==c) for c in range(1,9)]
            return {"case":"fixture","dice_per_class":[v[0] for v in pairs]}
        model.evaluate_synapse_case=original
        expected=original({})
        bind_tta(model,1)
        result=model.evaluate_synapse_case({"label":torch.from_numpy(gt)})
        self.assertEqual(result["dice_per_class"],expected["dice_per_class"])
        self.assertEqual(result["kidney_confusion"],kidney_confusion(pred,gt))
        self.assertIs(model.calculate_metric_percase,metric)
    def test_query_permutation_does_not_change_fused_pixel_channels(self):
        class QueryToy(Toy):
            def forward(self,image):
                x=image[:,0];masks=torch.stack((x,1-x),dim=1)
                classes=torch.zeros(image.shape[0],2,9);classes[:,0,3]=1;classes[:,1,4]=1
                if image[0,0,0,0]>.5:
                    masks=masks[:,[1,0]];classes=classes[:,[1,0]]
                return [masks],[classes]
            def to_per_pixel_logits_semantic(self,masks,classes):
                return torch.einsum("bqc,bqhw->bchw",classes,masks)
        x=torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)/64
        for views in [1,2,4,8]:
            result=predict_tta_scores(QueryToy(),x.repeat(1,3,1,1),views)
            torch.testing.assert_close(result[:,3],x[:,0],rtol=0,atol=0)

    def test_comparison_report_separates_dice_and_hd95_winners(self):
        from scripts.run_synapse_tta_comparison import make_report
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);reports=fixture_tta_reports(run)
            make_report(run,reports)
            result=json.loads((run/"comparison_summary.json").read_text())
            self.assertEqual(result["best_mean_dice_views"],2)
            self.assertEqual(result["best_mean_hd95_views"],4)
            self.assertGreater((run/"comparison_summary.md").read_text().count("\n"),40)
            self.assertEqual(len(result["kidneys"]),8)
            self.assertEqual(len(result["kidney_case_changes"]),96)

    def test_kidney_deltas_align_case_and_label_not_row_order(self):
        from scripts.run_synapse_tta_comparison import make_report
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);make_report(run,fixture_tta_reports(run))
            result=json.loads((run/"comparison_summary.json").read_text())
            left=next(x for x in result["kidneys"] if x["views"]==2 and x["label_id"]==3)
            self.assertAlmostEqual(left["dice_delta_pp"],-10/12)
            self.assertEqual(left["dice_degraded_cases"],1)
            self.assertEqual(left["cross_kidney_voxels_delta"],5)
            case=next(x for x in result["kidney_case_changes"] if x["views"]==2 and x["case"]=="fixture00" and x["label_id"]==3)
            self.assertEqual(case["dice_delta_pp"],-10.)
            self.assertTrue(case["dice_degraded"])
            right=next(x for x in result["kidneys"] if x["views"]==2 and x["label_id"]==4)
            self.assertAlmostEqual(right["hd95_delta_mm"],1/12)
            self.assertEqual(right["hd95_degraded_cases"],1)
            for x in result["kidney_case_changes"]:
                if x["views"]==1:
                    self.assertEqual(x["dice_delta_pp"],0.)
                    self.assertEqual(x["hd95_delta_mm"],0.)
            self.assertTrue((run/"kidney_case_comparison.csv").is_file())
            self.assertTrue((run/"kidney_degraded_cases.csv").is_file())

    def test_empty_mask_hd95_zero_is_not_counted_as_improvement(self):
        from scripts.run_synapse_tta_comparison import kidney_case_changes
        with tempfile.TemporaryDirectory() as folder:
            changes=kidney_case_changes(fixture_tta_reports(folder))
            case=next(x for x in changes if x["views"]==4 and x["case"]=="fixture01" and x["label_id"]==3)
            self.assertEqual(case["hd95_mm"],0.)
            self.assertEqual(case["hd95_delta_mm"],-2.)
            self.assertFalse(case["hd95_comparable"])
            self.assertIsNone(case["hd95_degraded"])
            self.assertTrue(case["dice_degraded"])

    def test_kidney_case_comparison_rejects_missing_case_label(self):
        from scripts.run_synapse_tta_comparison import kidney_case_changes
        with tempfile.TemporaryDirectory() as folder:
            reports=fixture_tta_reports(folder)
            p=reports[1][1]/"physical_case_organ_metrics.csv"
            import csv
            with p.open() as f:data=list(csv.DictReader(f))
            data=[x for x in data if not (x["case"]=="fixture00" and x["label_id"]=="3")]
            with p.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            with self.assertRaisesRegex(ValueError,"12 distinct cases"):
                kidney_case_changes(reports)

    def test_report_requires_per_view_provenance(self):
        from scripts.run_synapse_tta_comparison import make_report
        with tempfile.TemporaryDirectory() as folder:
            reports=fixture_tta_reports(folder);(reports[1][1]/"run_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError,"run_manifest"):
                make_report(Path(folder),reports)

    def test_nonfinite_augmented_scores_fail_instead_of_argmax(self):
        class Nonfinite(Toy):
            def forward(self,image):
                masks,classes=super().forward(image);masks[-1][:,0]=float("nan")
                return masks,classes
        with self.assertRaises(FloatingPointError):predict_tta_scores(Nonfinite(),torch.zeros(1,3,8,8),2)

class TTAManifestTests(unittest.TestCase):
    def fixture(self,root):
        source=root/"scripts/evaluate_synapse_tta.py";source.parent.mkdir(parents=True);source.write_text("fixture entry")
        (root/"scripts/run_synapse_tta_comparison.py").write_text("fixture runner")
        lists=root/"data/lists";lists.mkdir(parents=True)
        (lists/"train.txt").write_text("train_fixture\n");(lists/"test_vol.txt").write_text("test_fixture\n")
        ready=root/"outputs/tta/tta_readiness.json";ready.parent.mkdir(parents=True)
        ready.write_text(json.dumps({"state":"passed","sha256":{"scripts/evaluate_synapse_tta.py":hashlib.sha256(source.read_bytes()).hexdigest()}}))
        class EvalTrainer:
            precision="32-true";callbacks=[]
            baseline_resolved_config={"tta_views":4,"model":"fixture"}
            datamodule=SimpleNamespace(list_dir=lists,train_split="train",test_split="test_vol",train_base_dir="train",test_base_dir="test")
            @property
            def estimated_stepping_batches(self):raise AssertionError("Must not load training for test manifest")
        model=SimpleNamespace(device=torch.device("cpu"),tta_views=4)
        out=root/"run";out.mkdir()
        return EvalTrainer(),model,out,ready

    def test_manifest_captures_config_sources_environment_and_immutable_readiness(self):
        from training.synapse_tta import write_tta_run_manifest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trainer,model,out,ready=self.fixture(root)
            original=ready.read_bytes()
            with mock.patch.dict("os.environ",{"HF_HUB_OFFLINE":"1","FAKE_TTA_SECRET":"do-not-record"}):
                result=write_tta_run_manifest(trainer,model,out,{"views":4},{"checkpoint_sha256":"fixture"},root=root)
            self.assertEqual(result["resolved_cli_config"],trainer.baseline_resolved_config)
            self.assertIn("torch",result["versions"])
            self.assertIn("scripts/evaluate_synapse_tta.py",result["source_sha256"])
            self.assertIn("scripts/run_synapse_tta_comparison.py",result["source_sha256"])
            self.assertEqual(len(result["manifest_sha256"]),2)
            self.assertEqual(result["readiness_snapshot_sha256"],hashlib.sha256(original).hexdigest())
            self.assertEqual(result["environment"]["HF_HUB_OFFLINE"],"1")
            self.assertNotIn("do-not-record",(out/"run_manifest.json").read_text())
            ready.write_text("replaced later")
            self.assertEqual((out/"readiness_snapshot.json").read_bytes(),original)

    def test_manifest_rejects_code_changed_after_readiness(self):
        from training.synapse_tta import write_tta_run_manifest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trainer,model,out,_=self.fixture(root)
            (root/"scripts/evaluate_synapse_tta.py").write_text("changed")
            with self.assertRaisesRegex(ValueError,"source changed"):
                write_tta_run_manifest(trainer,model,out,{}, {},root=root)

    def test_real_tta_start_hook_writes_manifest(self):
        import training.synapse_tta as module
        from training.synapse_physical_selection import SynapsePhysicalHD95Selection
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trainer,model,out,_=self.fixture(root)
            def parent_start(callback,trainer,model):
                callback.run_dir=out;callback._test_checkpoint_info={"checkpoint_sha256":"fixture"}
            with mock.patch.object(module,"PROJECT_ROOT",root), mock.patch.object(SynapsePhysicalHD95Selection,"on_test_start",parent_start):
                callback=module.SynapseTTASelection();callback.on_test_start(trainer,model)
            self.assertTrue((out/"run_manifest.json").is_file())
            self.assertTrue((out/"readiness_snapshot.json").is_file())
            result=json.loads((out/"run_manifest.json").read_text())
            self.assertEqual(result["tta_contract"]["tta_views"],4)

if __name__=="__main__":unittest.main()
