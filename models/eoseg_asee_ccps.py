"""CCPS network model: inheriting EoSeg-ASEE without editing the original."""
from __future__ import annotations
import copy
import torch
from models.eoseg_asee import EoSegMultiQSegFusionASEE
from models.asee_ccps import AnatomicalSemanticEncodingCCPS


class EoSegMultiQSegFusionASEECCPS(EoSegMultiQSegFusionASEE):
    def __init__(self, encoder, num_classes, prior_path, num_q=16,
                 queries_per_class=2, patch_update="plan",
                 freeze_text_projection=False, enhance_patch=True,
                 enhance_query=True, use_text=True, use_probability=True,
                 selector_mode="visual_prior", selector_on_patch=True,
                 selector_on_query=False, selector_hidden_dim=256,
                 selector_init_probability=0.95,
                 detach_prior_for_selector=False, patch_gate_effect="both"):
        super().__init__(
            encoder=encoder, num_classes=num_classes, prior_path=prior_path,
            num_q=num_q, queries_per_class=queries_per_class,
            patch_update=patch_update,
            freeze_text_projection=freeze_text_projection,
            enhance_patch=enhance_patch, enhance_query=enhance_query,
            use_text=use_text, use_probability=use_probability,
        )
        # Existing parent loads/validates the prior and creates old ASEE.
        # Reconstruct the subclass from its validated raw buffers; no extra prior reads.
        old = self.asee
        with torch.random.fork_rng(devices=[]):
            self.asee = AnatomicalSemanticEncodingCCPS(
                embed_dim=old.embed_dim,
                text_features=old.text_features,
                conditional_matrix=old.conditional_matrix,
                queries_per_class=queries_per_class,
                patch_update=patch_update,
                freeze_text_projection=freeze_text_projection,
                enhance_patch=enhance_patch, enhance_query=enhance_query,
                use_text=use_text, use_probability=use_probability,
                patch_gate_effect=patch_gate_effect,
                selector_mode=selector_mode,
                selector_on_patch=selector_on_patch,
                selector_on_query=selector_on_query,
                selector_hidden_dim=selector_hidden_dim,
                selector_init_probability=selector_init_probability,
                detach_prior_for_selector=detach_prior_for_selector,
            )
        # Critical for controlled experiments: retain exactly the parent ASEE
        # initial weights. Only the newly added selector gets a new initialization.
        old_state = old.state_dict()
        missing, unexpected = self.asee.load_state_dict(old_state, strict=False)
        if unexpected or any(not k.startswith("selector.") for k in missing):
            raise RuntimeError(f"Unexpected ASEE initialization drift: {missing}, {unexpected}")
        self.ccps_configuration = {
            "patch_gate_effect": patch_gate_effect,
            "selector_mode": selector_mode,
            "selector_on_patch": selector_on_patch,
            "selector_on_query": selector_on_query,
            "selector_hidden_dim": selector_hidden_dim,
            "selector_init_probability": selector_init_probability,
            "detach_prior_for_selector": detach_prior_for_selector,
        }
        self.prior_metadata["ccps_configuration"] = copy.deepcopy(self.ccps_configuration)

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("ccps_configuration") != self.ccps_configuration:
            raise ValueError("CCPS checkpoint configuration mismatch")
        return super().set_extra_state(state)
