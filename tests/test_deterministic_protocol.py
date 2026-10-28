import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from training.mask_classification_loss import MaskClassificationLoss
from training.deterministic_mask_loss import DeterministicMaskClassificationLoss
from training.deterministic_protocol import (PROTOCOL_ID,backend_state,configure_strict,
    restore_backend,install_criterion,require_readiness,source_hashes)
ROOT=Path(__file__).resolve().parents[1]

class DeterministicProtocolTests(unittest.TestCase):
    def test_disabled_mode_leaves_model_and_flags(self):
        model=torch.nn.Module();model.criterion=MaskClassificationLoss(32,3,.75,5,5,2,9,.1)
        original=model.criterion;before=backend_state()
        configure_strict(enabled=False)
        self.assertFalse(install_criterion(model,enabled=False))
        self.assertIs(original,model.criterion)
        self.assertEqual(before,backend_state())

    def test_install_preserves_parameters_buffers_and_rng(self):
        model=torch.nn.Module();model.layer=torch.nn.Linear(3,4)
        model.criterion=MaskClassificationLoss(32,3,.75,5,5,2,9,.1)
        parameters=[id(p) for p in model.parameters()]
        state={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        self.assertTrue(install_criterion(model))
        self.assertFalse(install_criterion(model))
        self.assertIsInstance(model.criterion,DeterministicMaskClassificationLoss)
        self.assertEqual(parameters,[id(p) for p in model.parameters()])
        self.assertEqual(set(state),set(model.state_dict()))
        for key,value in state.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_required_launch_environment(self):
        before=backend_state()
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'CUBLAS_WORKSPACE_CONFIG'):
                configure_strict()
        self.assertEqual(before,backend_state())

    def test_strict_and_explicit_restore(self):
        before=backend_state()
        try:
            with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0'}):
                snapshot=configure_strict()
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)
            restore_backend(snapshot)
            self.assertEqual(before,backend_state())
        finally:restore_backend(before)

    def test_profiles_preserve_scientific_yaml(self):
        sources={'baseline':'baseline_2_roi.yaml','asee':'vit_query_mul_scale_fusion_asee_plan_hd95_mm.yaml','ccps':'vit_query_mul_scale_fusion_asee_ccps.yaml'}
        def normalize(value):
            v=copy.deepcopy(value)
            for key in list(v):
                if key.startswith('deterministic_'):v.pop(key)
            v['trainer'].pop('default_root_dir',None)
            logger=v['trainer'].get('logger')
            if isinstance(logger,dict):
                logger['init_args'].pop('save_dir',None);logger['init_args'].pop('name',None)
            for cb in v['trainer'].get('callbacks',[]):cb.get('init_args',{}).pop('output_dir',None)
            v['model']['init_args'].pop('compute_hd95_voxel',None)
            return v
        for profile,source in sources.items():
            a=yaml.safe_load((ROOT/'configs/synapse'/source).read_text())
            b=yaml.safe_load((ROOT/'configs/synapse/deterministic'/f'{profile}.yaml').read_text())
            self.assertEqual(normalize(a),normalize(b))
            self.assertIs(b['model']['init_args']['compute_hd95_voxel'],False)

    def test_formal_fit_requires_all_gates_current_sources_and_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'readiness.json'
            with self.assertRaises(RuntimeError):require_readiness(path,ROOT,'baseline')
            record={'protocol':PROTOCOL_ID,'passed':True,'sources':source_hashes(ROOT),
                    'validated_profiles':['baseline'],'gates':{f'G{i}':'passed' for i in range(6)}}
            path.write_text(json.dumps(record));require_readiness(path,ROOT,'baseline')
            with self.assertRaises(RuntimeError):require_readiness(path,ROOT,'ccps')
            record['gates']['G5']='pending';path.write_text(json.dumps(record))
            with self.assertRaises(RuntimeError):require_readiness(path,ROOT,'baseline')


    def test_probe_comparator_accepts_complete_g4_g5_and_rejects_difference(self):
        # Synthetic CPU fixtures test the validator itself, not training results.
        import subprocess,sys
        for batch_target in (0,20):
            with tempfile.TemporaryDirectory() as temp:
                base=Path(temp)
                count=batch_target or 2
                attempt={key:{'fixture':1} for key in (
                    'batch','rng_before_batch','forward','random_samples','sample_point_calls','matcher',
                    'loss_components','scaled_gradients','unscaled_gradients','clipped_gradients',
                    'weights_after_step','optimizer_state_after')}
                attempt.update(loss_total=1.,loss_total_scaled=2.,scaler_before={'scale':2.},
                    scaler_after={'scale':2.},optimizer_called=True,optimizer_step_lrs=[0.])
                trace={'state':'captured_native_batch_target' if batch_target else 'captured_target_optimizer_updates',
                    'actual_optimizer_updates':count,'attempts':[copy.deepcopy(attempt) for _ in range(count)],
                    'optimizer_groups':[{'fixture':1,'lr':0.}],'estimated_steps':30360,'precision':'16-mixed',
                    'native_batches':True,'target_updates':2,'native_rng_hashes':{'fixture':1},'effective_rng_hashes':{'fixture':1}}
                trace['attempts'][0]['parameter_keys_changed_from_initial']=[]
                for item in trace['attempts'][1:]:
                    item['optimizer_step_lrs']=[4e-7]
                    item['parameter_keys_changed_from_initial']=['p']
                for name in ('a','b'):
                    directory=base/('trajectory_'+name);directory.mkdir()
                    (directory/'trajectory_trace.json').write_text(json.dumps(trace))
                    (directory/'trajectory_initialization.json').write_text(json.dumps({'state':{'p':'fixture'},'rng':{'fixture':1}}))
                command=[sys.executable,str(ROOT/'scripts/probe_synapse_deterministic_entry.py'),
                    'compare','--output-dir',str(base),'--left','a','--right','b']
                if batch_target:command += ['--batch-target',str(batch_target)]
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                trace['attempts'][0]['loss_total']=1.01
                (base/'trajectory_b/trajectory_trace.json').write_text(json.dumps(trace))
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                report=json.loads((base/'trajectory_comparison_a_b.json').read_text())
                self.assertFalse(report['exact_equality'])

if __name__=='__main__':unittest.main()
