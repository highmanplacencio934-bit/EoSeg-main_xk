# Forward adapted from models/eoseg.py, MIT license.
# Copyright 2025 Mobile Perception Systems Lab at TU/e.
"""ASEE extension of the current fusion baseline; baseline source is untouched."""
import copy

import torch
from torch import nn

from models.eoseg import EoSegMultiQSegFusion
from models.asee import AnatomicalSemanticEncoding, load_prior


class EoSegMultiQSegFusionASEE(EoSegMultiQSegFusion):
    def __init__(self, encoder: nn.Module, num_classes: int, prior_path: str,
                 num_q: int = 16, queries_per_class: int = 2, patch_update: str = "plan",
                 freeze_text_projection: bool = False, enhance_patch: bool = True,
                 enhance_query: bool = True, use_text: bool = True, use_probability: bool = True):
        if num_classes != 9 or num_q != 8 * queries_per_class:
            raise ValueError("Synapse ASEE expects 9 labels including background and 2C-compatible queries")
        prior, digest = load_prior(prior_path)
        super().__init__(encoder=encoder, num_classes=num_classes, num_q=num_q)
        backbone = self.encoder.backbone
        if len(backbone.blocks) < self.num_blocks:
            raise ValueError("Backbone must have at least three blocks")
        if not all(hasattr(backbone.patch_embed, k) for k in ("num_patches", "grid_size")):
            raise ValueError("ASEE requires an explicit fixed patch grid")
        if not hasattr(backbone, "num_prefix_tokens"):
            raise ValueError("Backbone must expose CLS/register prefix token count")
        self.prior_metadata = {k: v for k, v in prior.items() if not isinstance(v, torch.Tensor)}
        self.prior_metadata["artifact_sha256"] = digest
        self.prior_metadata["asee_configuration"] = {
            "embed_dim": backbone.embed_dim, "num_classes": num_classes, "num_q": num_q,
            "queries_per_class": queries_per_class, "patch_update": patch_update,
            "freeze_text_projection": freeze_text_projection, "enhance_patch": enhance_patch,
            "enhance_query": enhance_query, "use_text": use_text, "use_probability": use_probability,
        }
        self.asee = AnatomicalSemanticEncoding(
            backbone.embed_dim, prior["text_features"], prior["conditional_matrix"],
            queries_per_class, patch_update, freeze_text_projection,
            enhance_patch, enhance_query, use_text, use_probability)

    def get_extra_state(self):
        # Provenance follows weights into both best_test/last checkpoints.
        return copy.deepcopy(self.prior_metadata)

    def set_extra_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("Checkpoint must contain ASEE prior provenance")
        if state.get("artifact_sha256") != self.prior_metadata["artifact_sha256"]:
            raise ValueError("Checkpoint prior differs from the configured prior artifact")
        if state.get("asee_configuration") != self.prior_metadata["asee_configuration"]:
            raise ValueError("Checkpoint ASEE configuration differs from the configured experiment")
        self.prior_metadata = copy.deepcopy(state)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # strict=False must not silently turn an ASEE checkpoint into a new experiment.
        # Baseline-only partial initialization remains available (no ASEE keys).
        if (any(key.startswith(prefix + "asee.") for key in state_dict)
                and prefix + "_extra_state" not in state_dict):
            raise ValueError("ASEE checkpoint weights must include prior/configuration provenance")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        B, _, H, W = x.shape
        x_norm = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # 1. Build initial tokens
        tokens = self.encoder.backbone.patch_embed(x_norm)
        if hasattr(self.encoder.backbone, "_pos_embed"):
            tokens = self.encoder.backbone._pos_embed(tokens)

        # 2. Prepare RoPE for DINOv3 if available
        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x_norm)

        # 3. Late query injection + collect outputs from the last 3 layers
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        inject_layer_idx = len(self.encoder.backbone.blocks) - self.num_blocks

        collected_tokens = []
        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == inject_layer_idx:
                n_patches = self.encoder.backbone.patch_embed.num_patches
                n_prefix = self.encoder.backbone.num_prefix_tokens
                if tokens.shape[1] != n_prefix + n_patches:
                    raise ValueError("Unexpected CLS/register/patch token layout")
                prefix = tokens[:, :n_prefix, :]
                patches, queries, _ = self.asee(tokens[:, n_prefix:, :], queries)
                tokens = torch.cat((queries, prefix, patches), dim=1)

            tokens = self._run_block(block, tokens, rope=rope)

            if i >= inject_layer_idx:
                collected_tokens.append(tokens)

        if len(collected_tokens) != self.num_blocks:
            raise RuntimeError(
                f"Expected {self.num_blocks} collected layers, got {len(collected_tokens)}."
            )

        # 4. Predict masks, class logits, and query features from each of the last 3 layers
        masks_per_layer = []
        class_logits_per_layer = []
        query_feats_per_layer = []

        for layer_tokens in collected_tokens:
            masks_i, class_logits_i, query_feats_i = self._predict_from_tokens(layer_tokens, H, W)
            masks_per_layer.append(masks_i)
            class_logits_per_layer.append(class_logits_i)
            query_feats_per_layer.append(query_feats_i)

        # 5. Query-conditioned fusion:
        # Use the last layer query feature to generate fusion weights.
        # Shape: [B, Q, 3]
        query_feat = query_feats_per_layer[-1]
        layer_scores = self.layer_gate(query_feat)
        layer_scores = layer_scores + self.layer_fusion_bias.view(1, 1, -1)
        fusion_weights = torch.softmax(layer_scores, dim=-1)

        # 6. Fuse masks and class logits with query-wise weights
        fused_masks = 0.0
        fused_class_logits = 0.0

        for i in range(self.num_blocks):
            w_mask = fusion_weights[..., i].unsqueeze(-1).unsqueeze(-1)  # [B, Q, 1, 1]
            fused_masks = fused_masks + w_mask * masks_per_layer[i]

            w_cls = fusion_weights[..., i].unsqueeze(-1)  # [B, Q, 1]
            fused_class_logits = fused_class_logits + w_cls * class_logits_per_layer[i]

        # Return query-level outputs for MaskClassificationLoss
        return [fused_masks], [fused_class_logits]
