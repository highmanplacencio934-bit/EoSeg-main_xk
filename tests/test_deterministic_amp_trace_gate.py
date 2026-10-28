import copy
import unittest
from scripts.validate_synapse_deterministic_trace import compare_pair,validate_trace

class AMPTraceGateTests(unittest.TestCase):
    def fixture(self,skip=False):
        x={key:{'fixture':1} for key in ('batch','rng_before_batch','forward','random_samples','sample_point_calls','matcher','loss_components','scaled_gradients','unscaled_gradients','clipped_gradients','weights_after_step','optimizer_state_after')}
        x.update(loss_total=1.,loss_total_scaled=65536.,scaler_before={'scale':65536.},scaler_after={'scale':65536.},optimizer_called=True,optimizer_step_lrs=[0.],parameter_keys_changed_from_initial=[])
        a=copy.deepcopy(x);a.update(optimizer_step_lrs=[4e-7],parameter_keys_changed_from_initial=['p'])
        attempts=[x,a]
        if skip:
            missed=copy.deepcopy(x);missed.update(optimizer_called=False,scaler_after={'scale':32768.})
            for key in ('optimizer_step_lrs','clipped_gradients','weights_after_step','optimizer_state_after'):missed.pop(key)
            b=copy.deepcopy(a);b['optimizer_step_lrs']=[8e-7]
            a['scaler_before']=a['scaler_after']={'scale':32768.};b['scaler_before']=b['scaler_after']={'scale':32768.}
            attempts=[missed,a,b]
        return {'state':'captured_target_optimizer_updates','attempts':attempts,'actual_optimizer_updates':2,
            'optimizer_groups':[{'lr':0.}],'estimated_steps':30360,'precision':'16-mixed','native_batches':False,
            'target_updates':2,'native_rng_hashes':{'fixture':1},'effective_rng_hashes':{'fixture':1}}
    def test_original_lr0_called_case(self):
        a=self.fixture();i={'state':{'p':'x'},'rng':{'s':1}}
        self.assertTrue(compare_pair(a,copy.deepcopy(a),i,i)['exact_equality'])
    def test_first_lr0_overflow_skip_is_valid(self):
        a=self.fixture(True);i={'state':{'p':'x'},'rng':{'s':1}}
        result=compare_pair(a,copy.deepcopy(a),i,i)
        self.assertTrue(result['exact_equality']);self.assertEqual(result['semantics'][0]['first_lr0_attempt'],'overflow_skip')
    def test_difference_still_fails_exact_comparison(self):
        a=self.fixture(True);b=copy.deepcopy(a);b['attempts'][1]['scaled_gradients']['fixture']=2
        i={'state':{'p':'x'},'rng':{'s':1}}
        self.assertFalse(compare_pair(a,b,i,i)['exact_equality'])
    def test_skip_without_scale_drop_rejected(self):
        a=self.fixture(True);a['attempts'][0]['scaler_after']['scale']=65536.
        with self.assertRaises(ValueError):validate_trace(a)
    def test_no_real_positive_lr_update_rejected(self):
        a=self.fixture()
        for item in a['attempts']:item['optimizer_step_lrs']=[0.];item['parameter_keys_changed_from_initial']=[]
        with self.assertRaises(ValueError):validate_trace(a)
    def test_incomplete_native_window_rejected(self):
        a=self.fixture();a['state']='captured_native_batch_target'
        with self.assertRaises(ValueError):validate_trace(a,batch_target=20)
