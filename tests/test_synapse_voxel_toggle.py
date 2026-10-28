import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np,torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer,LightningDataModule
from medpy import metric
from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.synapse_physical_selection import SynapsePhysicalHD95Selection
from training.synapse_roi_evaluation import roi_metrics_scope
from training.synapse_multiscale_tta import evaluate_multiscale_case,multiscale_contract
from training.synapse_teacher_semantic import SynapseTeacherHD95Semantic,load_teacher_utils
from test_synapse_physical_hd95 import TinyNetwork,make_manifest

class VoxelToggleIntegrationTests(unittest.TestCase):
    def test_default_mm_only_lightning_export_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);label=np.zeros((2,8,8),np.uint8);label[:,2,2]=1
            spacing=root/'spacing.json';make_manifest(spacing,{'case0001':label})
            (root/'test_vol.txt').write_text('case0001\n')
            with h5py.File(root/'case0001.npy.h5','w') as f:f['image']=np.zeros_like(label,dtype=np.float32);f['label']=label
            batch={'case_name':'case0001','image':torch.zeros(label.shape),'label':torch.from_numpy(label)}
            class Data(LightningDataModule):
                test_base_dir=str(root);list_dir=str(root);test_split='test_vol'
                def test_dataloader(self):return DataLoader([batch],batch_size=None)
            model=SynapsePhysicalHD95Semantic(network=TinyNetwork(),spacing_manifest=str(spacing),img_size=(8,8),num_classes=9,attn_mask_annealing_enabled=False)
            self.assertFalse(model.compute_hd95_voxel)
            cb=SynapsePhysicalHD95Selection(expected_test_cases=1,output_dir=str(root/'out'))
            trainer=Trainer(accelerator='cpu',devices=1,precision='32-true',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,callbacks=[cb])
            with patch('training.synapse_physical_geometry.metric.binary.hd95',wraps=metric.binary.hd95) as hd:
                trainer.test(model,datamodule=Data(),verbose=False)
                self.assertTrue(hd.call_count>0)
                self.assertTrue(all('voxelspacing' in c.kwargs for c in hd.call_args_list))
            with (cb.run_dir/'physical_case_organ_metrics.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),8);self.assertTrue(all(x['hd95_voxel']=='' for x in rows))
            summary=json.loads((cb.run_dir/'physical_test_summary.json').read_text())
            self.assertFalse(summary['hd95_voxel_enabled']);self.assertIsNone(summary['mean_hd95_voxel'])
            self.assertFalse(json.loads((cb.run_dir/'metric_units.json').read_text())['hd95_voxel_enabled'])
            old=model.evaluate_synapse_case(batch)
            model.compute_hd95_voxel=True
            dual=model.evaluate_synapse_case(batch)
            self.assertEqual(old['dice_per_class'],dual['dice_per_class']);self.assertEqual(old['hd95_mm_per_class'],dual['hd95_mm_per_class'])
            self.assertTrue(all(x is None for x in old['hd95_voxel_per_class']));self.assertTrue(all(x is not None for x in dual['hd95_voxel_per_class']))

    def test_roi_and_multiscale_honor_model_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            label=np.zeros((2,8,8),np.uint8);label[:,2,2]=1
            spacing=Path(tmp)/'spacing.json';make_manifest(spacing,{'case0001':label})
            model=SynapsePhysicalHD95Semantic(network=TinyNetwork(),spacing_manifest=str(spacing),img_size=(8,8),num_classes=9,attn_mask_annealing_enabled=False).eval()
            batch={'case_name':'case0001','image':torch.zeros(label.shape),'label':torch.from_numpy(label)}
            for views in [1,2,8]:
                with self.subTest(views=views),roi_metrics_scope(model,tta_views=views):
                    result=model.evaluate_synapse_case(batch)
                    self.assertFalse(result['hd95_voxel_enabled']);self.assertIsNone(result['mean_hd95_voxel'])
                    self.assertEqual(result['hd95_voxel_per_class'],[None]*8)
            model.compute_hd95_voxel=True
            with roi_metrics_scope(model,tta_views=2):
                result=model.evaluate_synapse_case(batch)
                self.assertTrue(result['hd95_voxel_enabled']);self.assertIsNotNone(result['mean_hd95_voxel'])
            self.assertFalse(multiscale_contract(8,(384,384),compute_hd95_voxel=False)['hd95_voxel_enabled'])
            self.assertNotIn('hd95_voxel_enabled',multiscale_contract(8,(384,384),compute_hd95_voxel=True))

    def test_teacher_default_skips_original_voxel_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            label=np.zeros((2,8,8),np.uint8);label[:,2,2]=1
            spacing=Path(tmp)/'spacing.json';make_manifest(spacing,{'case0001':label})
            model=SynapseTeacherHD95Semantic(network=TinyNetwork(),spacing_manifest=str(spacing),img_size=(8,8),num_classes=9,attn_mask_annealing_enabled=False).eval()
            model.teacher_utils,model.teacher_source=load_teacher_utils()
            model.teacher_utils.calculate_metric_percase=lambda *a: (_ for _ in ()).throw(AssertionError('voxel metric called'))
            batch={'case_name':'case0001','image':torch.zeros(label.shape),'label':torch.from_numpy(label)}
            with patch.object(torch.Tensor,'cuda',lambda tensor,*a,**kw:tensor):
                result=model.evaluate_synapse_case(batch)
            self.assertIsNone(result['mean_hd95_voxel']);self.assertFalse(result['hd95_voxel_enabled'])

if __name__=='__main__':unittest.main()
