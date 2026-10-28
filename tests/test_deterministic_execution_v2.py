from pathlib import Path
import importlib.util,torch,json,tempfile,unittest
from unittest.mock import patch
from training import deterministic_execution_v2 as v2
BASE=Path(__file__).resolve().parents[1]
import torch._functorch.config as cfg
class Dummy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.network=torch.nn.Module();self.network.asee=torch.nn.Module()
  self.network.asee.selector=torch.nn.Linear(2,2)
  self.network.feature_upscale=torch.nn.ModuleList([torch.nn.Identity() for _ in range(10)]+[torch.nn.BatchNorm2d(256)])
class Tests(unittest.TestCase):
 def test_baseline_and_asee_noop(self):
  for profile in ("baseline","asee"):
   model=Dummy();method=model.network.feature_upscale[10].forward
   before=cfg.backward_pass_autocast;rng=torch.get_rng_state().clone()
   self.assertFalse(v2.install_execution_boundary(model,profile))
   self.assertEqual(method,model.network.feature_upscale[10].forward)
   with v2.execution_context(profile):self.assertEqual(before,cfg.backward_pass_autocast)
   self.assertTrue(torch.equal(rng,torch.get_rng_state()))
 def test_ccps_state_rng_identity_idempotent(self):
  model=Dummy();state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();ids=[id(x) for x in model.parameters()]
  self.assertTrue(v2.install_execution_boundary(model,"ccps"));self.assertFalse(v2.install_execution_boundary(model,"ccps"))
  self.assertEqual(ids,[id(x) for x in model.parameters()]);self.assertEqual(set(state),set(model.state_dict()))
  self.assertTrue(all(torch.equal(x,model.state_dict()[k]) for k,x in state.items()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
 def test_context_restore_on_error(self):
  before=cfg.backward_pass_autocast
  with self.assertRaises(ValueError):
   with v2.execution_context("ccps"):
    self.assertEqual(cfg.backward_pass_autocast,"off");raise ValueError("exercise")
  self.assertEqual(before,cfg.backward_pass_autocast)
 def test_disabled_noops(self):
  model=Dummy();self.assertFalse(v2.install_execution_boundary(model,"ccps",enabled=False))
  before=cfg.backward_pass_autocast
  with v2.execution_context("ccps",enabled=False):self.assertEqual(before,cfg.backward_pass_autocast)
 def test_invalid_profile_or_model(self):
  with self.assertRaises(ValueError):v2.install_execution_boundary(Dummy(),"bad")
  with self.assertRaises(TypeError):v2.install_execution_boundary(torch.nn.Linear(2,2),"ccps")
  model=Dummy();model.network.feature_upscale[10]=torch.nn.LayerNorm(256)
  with self.assertRaises(ValueError):v2.install_execution_boundary(model,"ccps")
 def test_formal_gate_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"readiness.json"
   with self.assertRaises(RuntimeError):v2.require_readiness(p,BASE,"ccps")
   data={"protocol":v2.PROTOCOL_ID,"passed":True,"validated_profiles":["ccps"],"gates":{f"G{i}":"passed" for i in range(6)},"sources":{"fixture":"ok"}}
   with patch.object(v2,"source_hashes",return_value={"fixture":"ok"}):
    p.write_text(json.dumps(data));v2.require_readiness(p,BASE,"ccps")
    data["gates"]["G5"]="pending";p.write_text(json.dumps(data))
    with self.assertRaises(RuntimeError):v2.require_readiness(p,BASE,"ccps")
if __name__=="__main__":unittest.main()
