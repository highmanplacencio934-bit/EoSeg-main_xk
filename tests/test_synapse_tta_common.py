"""CPU contracts for retained TTA scores, kidney diagnostics and provenance."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
from training.synapse_tta_common import pixel_scores, kidney_confusion, write_tta_run_manifest

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


class TTAManifestTests(unittest.TestCase):

    def fixture(self,root):
        source=root/"scripts/evaluate_synapse_multiscale_tta.py";source.parent.mkdir(parents=True);source.write_text("fixture entry")
        (root/"scripts/run_synapse_roi.py").write_text("fixture runner")
        lists=root/"data/lists";lists.mkdir(parents=True)
        (lists/"train.txt").write_text("train_fixture\n");(lists/"test_vol.txt").write_text("test_fixture\n")
        ready=root/"outputs/tta/multiscale_readiness.json";ready.parent.mkdir(parents=True)
        ready.write_text(json.dumps({"state":"passed","sha256":{"scripts/evaluate_synapse_multiscale_tta.py":hashlib.sha256(source.read_bytes()).hexdigest()}}))
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
        from training.synapse_tta_common import write_tta_run_manifest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trainer,model,out,ready=self.fixture(root)
            original=ready.read_bytes()
            with mock.patch.dict("os.environ",{"HF_HUB_OFFLINE":"1","FAKE_TTA_SECRET":"do-not-record"}):
                result=write_tta_run_manifest(trainer,model,out,{"views":4},{"checkpoint_sha256":"fixture"},root=root)
            self.assertEqual(result["resolved_cli_config"],trainer.baseline_resolved_config)
            self.assertIn("torch",result["versions"])
            self.assertIn("scripts/evaluate_synapse_multiscale_tta.py",result["source_sha256"])
            self.assertIn("scripts/run_synapse_roi.py",result["source_sha256"])
            self.assertEqual(len(result["manifest_sha256"]),2)
            self.assertEqual(result["readiness_snapshot_sha256"],hashlib.sha256(original).hexdigest())
            self.assertEqual(result["environment"]["HF_HUB_OFFLINE"],"1")
            self.assertNotIn("do-not-record",(out/"run_manifest.json").read_text())
            ready.write_text("replaced later")
            self.assertEqual((out/"readiness_snapshot.json").read_bytes(),original)

    def test_manifest_rejects_code_changed_after_readiness(self):
        from training.synapse_tta_common import write_tta_run_manifest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trainer,model,out,_=self.fixture(root)
            (root/"scripts/evaluate_synapse_multiscale_tta.py").write_text("changed")
            with self.assertRaisesRegex(ValueError,"source changed"):
                write_tta_run_manifest(trainer,model,out,{}, {},root=root)


class SharedScoreTests(unittest.TestCase):
    def test_left_right_kidney_confusion_is_not_merged(self):
        gt=np.array([[[3,3,4,4]]]);pred=np.array([[[4,3,3,4]]])
        out=kidney_confusion(pred,gt)
        self.assertEqual(out["pred_left_on_gt_right_voxels"],1)
        self.assertEqual(out["pred_right_on_gt_left_voxels"],1)
        self.assertEqual(out["gt_left_voxels"],2)
        self.assertEqual(out["gt_right_voxels"],2)

    def test_pixel_score_dtype_is_preserved(self):
        image = torch.zeros(1, 3, 8, 8)
        for half in (False, True):
            with self.subTest(half=half):
                scores = pixel_scores(Toy(half), image)
                self.assertEqual(scores.dtype, torch.float16 if half else torch.float32)
                self.assertEqual(tuple(scores.shape), (1, 9, 8, 8))


if __name__ == "__main__":
    unittest.main()
