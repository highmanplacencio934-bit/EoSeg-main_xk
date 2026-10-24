"""Non-destructive CCPS extension of the existing models/asee.py module."""
from __future__ import annotations
import math
import torch
from models.asee import AnatomicalSemanticEncoding
from models.ccps_selector import ClassConditionalPriorSelector


class AnatomicalSemanticEncodingCCPS(AnatomicalSemanticEncoding):
    def __init__(self, *args, selector_mode: str = "visual_prior",
                 selector_on_patch: bool = True, selector_on_query: bool = False,
                 selector_hidden_dim: int = 256,
                 selector_init_probability: float = 0.95,
                 detach_prior_for_selector: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if selector_mode not in {"off", *ClassConditionalPriorSelector.MODES}:
            raise ValueError("Unknown CCPS mode")
        if selector_mode != "off" and not (selector_on_patch or selector_on_query):
            raise ValueError("Enabled CCPS must affect patch and/or query")
        self.selector_mode = selector_mode
        self.selector_on_patch = selector_on_patch
        self.selector_on_query = selector_on_query
        self.selector = None if selector_mode == "off" else ClassConditionalPriorSelector(
            embed_dim=self.embed_dim,
            num_classes=self.num_foreground,
            mode=selector_mode,
            hidden_dim=selector_hidden_dim,
            init_probability=selector_init_probability,
            detach_prior_for_selector=detach_prior_for_selector,
        )
        self.last_selector_logits = None
        self.last_selector_weights = None
        self.last_patch_strength = None

    def forward(self, patch_tokens, query_tokens):
        self.last_selector_logits = None
        self.last_selector_weights = None
        self.last_patch_strength = None
        if self.selector is None:
            # Exact inherited ASEE behavior, including plan / reference_gated.
            return super().forward(patch_tokens, query_tokens)
        if patch_tokens.ndim != 3 or query_tokens.ndim != 3:
            raise ValueError("Expected [B,N,D] patch/query tensors")
        if patch_tokens.shape[0] != query_tokens.shape[0] or patch_tokens.shape[-1] != self.embed_dim:
            raise ValueError("Token batch/channel mismatch")
        if query_tokens.shape[1:] != (self.num_foreground * self.queries_per_class, self.embed_dim):
            raise ValueError("Unexpected Query count/dimension")
        if not self.enhance_patch and not self.enhance_query:
            return patch_tokens, query_tokens, None

        # Keep projection and the relative contributions of both priors identical to ASEE.
        prior = 0
        if self.use_text:
            prior = self.text_proj(self.text_features)
        if self.use_probability:
            prior = prior + self.prob_proj(self.conditional_matrix)

        weights, logits = self.selector(patch_tokens, prior)  # [B,8], [B,8]
        # Auxiliary loss needs the graph; visual logs should not retain it.
        self.last_selector_logits = logits if self.training else None
        self.last_selector_weights = weights.detach()
        attention = None

        if self.enhance_patch:
            local = self.norm_patch_in(self.ffn_patch_in(patch_tokens) + patch_tokens)
            logits_patch = (local @ prior.T) / math.sqrt(self.embed_dim)
            attention = logits_patch.softmax(dim=-1)
            alpha = (attention * weights[:, None, :].to(attention.dtype)
                     if self.selector_on_patch else attention)
            context = alpha @ prior
            update = self.ffn_patch_out(context) + context
            if self.patch_update == "plan":
                proposed = self.norm_patch_out(update + patch_tokens)
                if self.selector_on_patch:
                    strength = alpha.sum(dim=-1, keepdim=True)  # [B,N,1], 0..1
                    patch_tokens = patch_tokens + strength * (proposed - patch_tokens)
                    self.last_patch_strength = strength.detach()
                else:
                    patch_tokens = proposed
            else:
                delta = self.gamma_patch * self.norm_patch_out(update)
                if self.selector_on_patch:
                    strength = alpha.sum(dim=-1, keepdim=True)
                    patch_tokens = patch_tokens + strength * delta
                    self.last_patch_strength = strength.detach()
                else:
                    patch_tokens = patch_tokens + delta

        if self.enhance_query:
            expanded = prior.repeat_interleave(self.queries_per_class, dim=0)
            bias = self.query_prior_proj(expanded).unsqueeze(0)
            if self.selector_on_query:
                q_weights = weights.repeat_interleave(self.queries_per_class, dim=1)
                bias = bias * q_weights[:, :, None].to(bias.dtype)
            if self.patch_update == "reference_gated":
                bias = self.gamma_query * bias
            query_tokens = query_tokens + bias

        return patch_tokens, query_tokens, attention
