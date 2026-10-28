import unittest
"""CPU-only tests, run in repository root: pytest -q tests/test_ccps.py."""
import torch
from models.asee import AnatomicalSemanticEncoding
from models.asee_ccps import AnatomicalSemanticEncodingCCPS
from models.ccps_selector import ClassConditionalPriorSelector


def make_module(mode="visual_prior", patch_update="plan", on_patch=True, on_query=True):
    return AnatomicalSemanticEncodingCCPS(
        embed_dim=32, text_features=torch.randn(8, 12),
        conditional_matrix=torch.randn(8, 8),
        queries_per_class=2, patch_update=patch_update,
        selector_mode=mode, selector_on_patch=on_patch,
        selector_on_query=on_query, selector_hidden_dim=16)


def test_selector_shapes_and_gradients(mode):
    torch.manual_seed(7)
    m = make_module(mode=mode)
    patches = torch.randn(2, 16, 32, requires_grad=True)
    queries = torch.randn(2, 16, 32, requires_grad=True)
    out_p, out_q, attn = m(patches, queries)
    assert out_p.shape == patches.shape
    assert out_q.shape == queries.shape
    assert attn.shape == (2, 16, 8)
    assert m.last_selector_weights.shape == (2, 8)
    assert m.last_selector_weights.min() > 0
    assert m.last_selector_weights.max() < 1
    (out_p.square().mean() + out_q.square().mean()).backward()
    assert patches.grad is not None
    assert any(p.grad is not None for p in m.selector.parameters())


def test_off_matches_original_asee(update):
    torch.manual_seed(5)
    old = AnatomicalSemanticEncoding(32, torch.randn(8, 12),
                                     torch.randn(8, 8), patch_update=update)
    new = make_module(mode="off", patch_update=update)
    new.load_state_dict(old.state_dict(), strict=True)
    patches, queries = torch.randn(2, 12, 32), torch.randn(2, 16, 32)
    a = old(patches, queries)
    b = new(patches, queries)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_zero_gates_bypass_both_paths(update):
    class ZeroSelector(torch.nn.Module):
        def forward(self, patches, prior):
            weights = patches.new_zeros((patches.shape[0], prior.shape[0]))
            return weights, weights

    m = make_module(patch_update=update)
    m.selector = ZeroSelector()
    patches, queries = torch.randn(1, 9, 32), torch.randn(1, 16, 32)
    output_p, output_q, attn = m(patches, queries)
    assert torch.equal(output_p, patches)
    assert torch.equal(output_q, queries)
    assert attn.shape == (1, 9, 8)


def test_patch_only_query_is_original_query_update():
    torch.manual_seed(17)
    m = make_module(on_patch=True, on_query=False)
    old = AnatomicalSemanticEncoding(32, m.text_features, m.conditional_matrix)
    old.load_state_dict({k: v for k, v in m.state_dict().items()
                         if not k.startswith("selector.")}, strict=True)
    patches, queries = torch.randn(1, 10, 32), torch.randn(1, 16, 32)
    _, q_a, _ = m(patches, queries)
    _, q_b, _ = old(patches, queries)
    assert torch.equal(q_a, q_b)


# Integration regressions use a tiny ViT and synthetic priors only.
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from models.eoseg_asee_ccps import EoSegMultiQSegFusionASEECCPS
from models.eoseg_asee import EoSegMultiQSegFusionASEE
from training.mask_classification_ccps import MaskClassificationSemanticCCPS
from training.mask_classification_semantic import MaskClassificationSemantic
from scripts.generate_ccps_ablations import generate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("asee_test_helpers", ROOT / "tests/test_asee_innovation.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def network(tmp_path, mode="visual_prior", **kwargs):
    path = tmp_path / "synthetic_prior.pt"
    if not path.exists():
        torch.save(helpers.fixture_prior(), path)
    return EoSegMultiQSegFusionASEECCPS(helpers.tiny_encoder(), 9, str(path),
                                      selector_mode=mode, selector_hidden_dim=16, **kwargs)


def test_full_network_initialization_preserves_parent_weights_and_rng(tmp_path, mode):
    path = tmp_path / "synthetic_prior.pt"
    torch.save(helpers.fixture_prior(), path)
    torch.manual_seed(29)
    old = EoSegMultiQSegFusionASEE(helpers.tiny_encoder(), 9, str(path))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    new = network(tmp_path, mode)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, new.state_dict()[key]), key
    if mode == "off":
        x = torch.rand(2, 3, 32, 32)
        for train in (True, False):
            old.train(train); new.train(train)
            with torch.random.fork_rng(devices=[]):
                a = old(x)
            with torch.random.fork_rng(devices=[]):
                b = new(x)
            for group_a, group_b in zip(a,b):
                for v_a,v_b in zip(group_a, group_b):
                    assert torch.equal(v_a,v_b)


def test_autocast_linears_are_really_fp32(mode):
    m = ClassConditionalPriorSelector(32, mode=mode, hidden_dim=16)
    seen=[]
    handles=[layer.register_forward_hook(lambda mod, inp, out: seen.append(out.dtype))
             for layer in m.modules() if isinstance(layer,torch.nn.Linear)]
    with torch.autocast("cpu",dtype=torch.bfloat16):
        weights, logits=m(torch.randn(2,10,32),torch.randn(8,32))
    for handle in handles: handle.remove()
    assert seen and set(seen)=={torch.float32}
    assert weights.dtype==logits.dtype==torch.float32
    logits.square().mean().backward()
    for param in m.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    assert any(p.grad.abs().sum()>0 for p in m.parameters())


def test_effect_formula(effect):
    m=make_module(on_query=False)
    m.patch_gate_effect=effect
    patches=torch.randn(2,5,32);queries=torch.randn(2,16,32)
    actual,_,a=m(patches,queries)
    prior=m.text_proj(m.text_features)+m.prob_proj(m.conditional_matrix)
    weights,_=m.selector(patches,prior)
    selected=a*weights[:,None,:]
    context=(a if effect=="residual_only" else selected)@prior
    proposed=m.norm_patch_out(m.ffn_patch_out(context)+context+patches)
    expected=proposed if effect=="context_only" else patches+selected.sum(-1,keepdim=True)*(proposed-patches)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_checkpoint_metadata_roundtrip_and_reject_mismatch(tmp_path):
    m=network(tmp_path);state=copy.deepcopy(m.state_dict())
    other=network(tmp_path);other.load_state_dict(state,strict=True)
    bad=network(tmp_path,patch_gate_effect="context_only")
    with unittest.TestCase().assertRaisesRegex(ValueError,"configuration mismatch"):
        bad.load_state_dict(state)
    for key,value in state.items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,other.state_dict()[key])


def trainer_module(tmp_path, **kwargs):
    return MaskClassificationSemanticCCPS(network(tmp_path), (32,32),9,False,
        lr=2e-4,num_points=32,**kwargs)


def test_optimizer_and_real_segmentation_presence_backward(tmp_path):
    m=trainer_module(tmp_path,presence_loss_weight=.1)
    m._trainer=SimpleNamespace(estimated_stepping_batches=30000,global_step=1,barebones=False)
    m.log=lambda *a,**k:None
    opt=m.configure_optimizers()["optimizer"]
    selector_ids={id(p) for p in m.network.asee.selector.parameters()}
    groups=[g for g in opt.param_groups if g.get("name","").startswith("network.asee.selector.")]
    assert selector_ids=={id(p) for g in groups for p in g["params"]}
    assert all(g["initial_lr"]==2e-4 for g in groups)
    x=torch.rand(2,3,32,32)
    labels=torch.zeros(32,32,dtype=torch.long);labels[8:20,10:25]=6
    from datasets.synapse_npz_dataset import label_to_eomt_target
    targets=[label_to_eomt_target(labels) for _ in range(2)]
    loss=m.training_step((x,targets),0)
    assert torch.isfinite(loss)
    loss.backward()
    for p in m.network.asee.selector.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert any(p.grad.abs().sum()>0 for p in m.network.asee.selector.parameters())
    opt.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    built=m.build_presence_targets(targets,8,device="cpu")
    assert torch.equal(built[:,5],torch.ones(2)) and built.sum()==2


def test_generator_refuses_all_writes_on_collision(tmp_path):
    base=ROOT/"configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml"
    generate(base,tmp_path)
    first=tmp_path/"a0_off.yaml";first.write_text("manual edit")
    with unittest.TestCase().assertRaises(FileExistsError):generate(base,tmp_path)
    assert first.read_text()=="manual edit"
    generate(base,tmp_path,overwrite=True)
    assert "manual edit" not in first.read_text()


def test_detach_prior_only_affects_selector_branch():
    p=torch.randn(2,4,32,requires_grad=True);e=torch.randn(8,32,requires_grad=True)
    selector=ClassConditionalPriorSelector(32,hidden_dim=16,detach_prior_for_selector=True)
    weights,logits=selector(p,e);logits.sum().backward()
    assert e.grad is None and p.grad is not None


def test_physical_contract_and_nondefault_loss_options(tmp_path):
    m=trainer_module(tmp_path,mask_coefficient=7.0)
    assert m.criterion.mask_coefficient==7.0
    m._physical_case_context=((2.,1.,1.),[])
    import numpy as np
    a=np.zeros((4,5,5),dtype=np.uint8);a[1,2,2]=1
    b=np.zeros_like(a);b[2,2,2]=1
    dice,mm=m.calculate_metric_percase(a,b)
    assert dice==0 and mm==2
    assert m.compute_hd95_voxel is False
    assert m._physical_case_context[1][0]["hd95_voxel"] is None
    m.compute_hd95_voxel=True
    restored_dice,restored_mm=m.calculate_metric_percase(a,b)
    assert (restored_dice,restored_mm)==(dice,mm)
    assert m._physical_case_context[1][-1]["hd95_voxel"]==1




def test_launcher_idempotence_with_simultaneous_timer_contract(tmp_path):
    from scripts import launch_innovation2_ccps as launcher
    with mock.patch.object(launcher,"BASE",tmp_path), mock.patch.object(launcher,"ROOT",tmp_path), \
         mock.patch.object(launcher,"preflight",return_value={}), \
         mock.patch.object(launcher.subprocess,"Popen",return_value=SimpleNamespace(pid=123)) as popen:
        launcher.launch();launcher.launch()
        assert popen.call_count==1
        assert (tmp_path/"launch_receipt.json").is_file()


def test_launcher_rejects_pending_readiness(tmp_path):
    from scripts import launch_innovation2_ccps as launcher
    (tmp_path/"outputs/innovation2").mkdir(parents=True)
    (tmp_path/"outputs/innovation2/ccps_readiness.json").write_text('{"state":"pending"}')
    with mock.patch.object(launcher,"ROOT",tmp_path), mock.patch.object(launcher.socket,"gethostname",return_value="5090"), mock.patch.object(launcher.os,"chdir"):
        with unittest.TestCase().assertRaisesRegex(RuntimeError,"Readiness not passed"):
            launcher.preflight()


def test_launcher_rejects_changed_sources(tmp_path):
    from scripts import launch_innovation2_ccps as launcher
    import hashlib,json
    (tmp_path/"outputs/innovation2").mkdir(parents=True)
    (tmp_path/"source.py").write_text("changed")
    (tmp_path/"outputs/innovation2/ccps_readiness.json").write_text(json.dumps({"state":"passed","sha256":{"source.py":hashlib.sha256(b"old").hexdigest()}}))
    with mock.patch.object(launcher,"ROOT",tmp_path), mock.patch.object(launcher.socket,"gethostname",return_value="5090"), mock.patch.object(launcher.os,"chdir"):
        with unittest.TestCase().assertRaisesRegex(RuntimeError,"source/artifact changed"):
            launcher.preflight()

class CCPSContracts(unittest.TestCase):
    pass


def _bind_case(function, arguments, values):
    def run(self):
        import tempfile
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(values)
            if "tmp_path" in arguments:
                kwargs["tmp_path"] = Path(folder)
            function(**kwargs)
    return run

setattr(CCPSContracts, "test_selector_shapes_and_gradients_0", _bind_case(test_selector_shapes_and_gradients, ['mode'], {'mode': 'global'}))
setattr(CCPSContracts, "test_selector_shapes_and_gradients_1", _bind_case(test_selector_shapes_and_gradients, ['mode'], {'mode': 'visual'}))
setattr(CCPSContracts, "test_selector_shapes_and_gradients_2", _bind_case(test_selector_shapes_and_gradients, ['mode'], {'mode': 'visual_prior'}))
setattr(CCPSContracts, "test_off_matches_original_asee_0", _bind_case(test_off_matches_original_asee, ['update'], {'update': 'plan'}))
setattr(CCPSContracts, "test_off_matches_original_asee_1", _bind_case(test_off_matches_original_asee, ['update'], {'update': 'reference_gated'}))
setattr(CCPSContracts, "test_zero_gates_bypass_both_paths_0", _bind_case(test_zero_gates_bypass_both_paths, ['update'], {'update': 'plan'}))
setattr(CCPSContracts, "test_zero_gates_bypass_both_paths_1", _bind_case(test_zero_gates_bypass_both_paths, ['update'], {'update': 'reference_gated'}))
setattr(CCPSContracts, "test_patch_only_query_is_original_query_update_0", _bind_case(test_patch_only_query_is_original_query_update, [], {}))
setattr(CCPSContracts, "test_full_network_initialization_preserves_parent_weights_and_rng_0", _bind_case(test_full_network_initialization_preserves_parent_weights_and_rng, ['tmp_path', 'mode'], {'mode': 'off'}))
setattr(CCPSContracts, "test_full_network_initialization_preserves_parent_weights_and_rng_1", _bind_case(test_full_network_initialization_preserves_parent_weights_and_rng, ['tmp_path', 'mode'], {'mode': 'global'}))
setattr(CCPSContracts, "test_full_network_initialization_preserves_parent_weights_and_rng_2", _bind_case(test_full_network_initialization_preserves_parent_weights_and_rng, ['tmp_path', 'mode'], {'mode': 'visual'}))
setattr(CCPSContracts, "test_full_network_initialization_preserves_parent_weights_and_rng_3", _bind_case(test_full_network_initialization_preserves_parent_weights_and_rng, ['tmp_path', 'mode'], {'mode': 'visual_prior'}))
setattr(CCPSContracts, "test_autocast_linears_are_really_fp32_0", _bind_case(test_autocast_linears_are_really_fp32, ['mode'], {'mode': 'visual'}))
setattr(CCPSContracts, "test_autocast_linears_are_really_fp32_1", _bind_case(test_autocast_linears_are_really_fp32, ['mode'], {'mode': 'visual_prior'}))
setattr(CCPSContracts, "test_effect_formula_0", _bind_case(test_effect_formula, ['effect'], {'effect': 'both'}))
setattr(CCPSContracts, "test_effect_formula_1", _bind_case(test_effect_formula, ['effect'], {'effect': 'context_only'}))
setattr(CCPSContracts, "test_effect_formula_2", _bind_case(test_effect_formula, ['effect'], {'effect': 'residual_only'}))
setattr(CCPSContracts, "test_checkpoint_metadata_roundtrip_and_reject_mismatch_0", _bind_case(test_checkpoint_metadata_roundtrip_and_reject_mismatch, ['tmp_path'], {}))
setattr(CCPSContracts, "test_optimizer_and_real_segmentation_presence_backward_0", _bind_case(test_optimizer_and_real_segmentation_presence_backward, ['tmp_path'], {}))
setattr(CCPSContracts, "test_generator_refuses_all_writes_on_collision_0", _bind_case(test_generator_refuses_all_writes_on_collision, ['tmp_path'], {}))
setattr(CCPSContracts, "test_detach_prior_only_affects_selector_branch_0", _bind_case(test_detach_prior_only_affects_selector_branch, [], {}))
setattr(CCPSContracts, "test_physical_contract_and_nondefault_loss_options_0", _bind_case(test_physical_contract_and_nondefault_loss_options, ['tmp_path'], {}))

setattr(CCPSContracts, "test_launcher_idempotence_with_simultaneous_timer_contract", _bind_case(test_launcher_idempotence_with_simultaneous_timer_contract, ["tmp_path"], {}))
setattr(CCPSContracts, "test_launcher_rejects_pending_readiness", _bind_case(test_launcher_rejects_pending_readiness, ["tmp_path"], {}))
setattr(CCPSContracts, "test_launcher_rejects_changed_sources", _bind_case(test_launcher_rejects_changed_sources, ["tmp_path"], {}))
if __name__ == "__main__":
    unittest.main()
