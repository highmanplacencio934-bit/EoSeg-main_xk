"""CPU validation of the low-observer evidence gate, never GPU evidence."""
import copy
import unittest
from scripts.probe_synapse_deterministic_low_observer import compare_records

class LowObserverGateTests(unittest.TestCase):
    def fixture(self):
        value={key:{'fixture':1} for key in ('initial_state','initial_rng','optimizer_groups',
            'final_state','final_optimizer','final_scheduler','final_rng')}
        value.update(profile='baseline',compiled=True,strict=True,precision='16-mixed',estimated_steps=30360,
            state='captured_native_batch_target',batches=20,actual_optimizer_updates=2,
            losses=[1]*20,scaler=[1]*20,lrs=[[0],[4e-7]])
        return value
    def test_identical_complete_fixture(self):
        a=self.fixture();self.assertTrue(compare_records(a,copy.deepcopy(a))['exact_equality'])
    def test_numerical_difference_is_rejected(self):
        a=self.fixture();b=copy.deepcopy(a);b['final_optimizer']['fixture']=2
        self.assertFalse(compare_records(a,b)['exact_equality'])
    def test_incomplete_observations_rejected(self):
        for key in ('losses','scaler','lrs'):
            a=self.fixture();b=copy.deepcopy(a);b[key]=[]
            with self.assertRaises(ValueError):compare_records(a,b)
    def test_wrong_protocol_rejected(self):
        for key,value in (('strict',False),('compiled',False),('precision','32-true'),('estimated_steps',100)):
            a=self.fixture();b=copy.deepcopy(a);b[key]=value
            with self.assertRaises(ValueError):compare_records(a,b)
