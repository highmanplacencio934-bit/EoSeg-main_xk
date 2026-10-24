"""CPU-only tests, run in repository root: pytest -q tests/test_ccps.py."""
import pytest
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


@pytest.mark.parametrize("mode", ["global", "visual", "visual_prior"])
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


@pytest.mark.parametrize("update", ["plan", "reference_gated"])
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


@pytest.mark.parametrize("update", ["plan", "reference_gated"])
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
