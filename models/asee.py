"""Independent ASEE implementation; see docs/innovation1_asee.md for Plan differences."""
from pathlib import Path
import hashlib
import io
import math

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_LIST_SHA256 = "2a3afc3b126f560d023c09e9a943e31c9ab95cea544afa69bf6b39f80ade59f5"
TEST_LIST_SHA256 = "7a1cdb883ad2f76f79cb922cee782eb8618ee7a72e75b640bbcee1367d66c2a6"


def validate_prior(prior):
    if not isinstance(prior, dict) or prior.get("schema_version") != 1:
        raise ValueError("Expected ASEE prior schema_version=1, not legacy bare tensors")
    if prior.get("dataset") != "synapse" or prior.get("class_ids") != list(range(1, 9)):
        raise ValueError("Prior rows must be Synapse foreground label IDs 1..8")
    if prior.get("label_mapping_verified") is not True or not prior.get("label_mapping_source", "").strip():
        raise ValueError("Verify label ID/organ mapping and record its source before constructing priors")
    if len(prior.get("class_names", [])) != 8:
        raise ValueError("Expected eight ordered class names")
    stats = prior.get("statistics", {})
    if stats.get("unit") != "train_npz_slice" or stats.get("train_list_sha256") != TRAIN_LIST_SHA256:
        raise ValueError("Prior must use the fixed training slice manifest")
    if stats.get("test_list_sha256") != TEST_LIST_SHA256 or stats.get("num_samples") != 2211:
        raise ValueError("Prior does not describe the fixed 2211-slice, 18/12 split")
    for key, shape in (("text_features", None), ("conditional_matrix", (8, 8))):
        value = prior.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 8:
            raise ValueError(f"Invalid {key} shape")
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(f"Expected {key} shape {shape}")
        if value.shape[1] == 0 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid/nonfinite {key}")
    return prior


def load_prior(path):
    path = Path(path).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    # Deserialize and hash the same immutable snapshot, even if the file is replaced.
    raw = path.read_bytes()
    prior = validate_prior(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    return prior, hashlib.sha256(raw).hexdigest()


class AnatomicalSemanticEncoding(nn.Module):
    """One patch cross-attention and one query bias, before the last three blocks.

    `plan` follows the outer-LN formula in both Plans. `reference_gated` follows
    asee_module.py, including its two trainable gates initialized to 0.1.
    Cached raw features/M are frozen; projected E is recomputed with gradients.
    """
    def __init__(self, embed_dim, text_features, conditional_matrix,
                 queries_per_class=2, patch_update="plan", freeze_text_projection=False,
                 enhance_patch=True, enhance_query=True, use_text=True, use_probability=True):
        super().__init__()
        if patch_update not in {"plan", "reference_gated"}:
            raise ValueError("patch_update must be plan or reference_gated")
        if embed_dim < 4 or queries_per_class < 1:
            raise ValueError("Invalid embedding/query dimensions")
        if text_features.ndim != 2 or conditional_matrix.shape != (text_features.shape[0],) * 2:
            raise ValueError("Text and conditional prior dimensions disagree")
        if not all(torch.isfinite(t).all() for t in (text_features, conditional_matrix)):
            raise ValueError("Priors must be finite")
        if not (use_text or use_probability):
            raise ValueError("At least one knowledge source must be enabled")
        self.embed_dim = embed_dim
        self.num_foreground = text_features.shape[0]
        self.queries_per_class = queries_per_class
        self.patch_update = patch_update
        self.enhance_patch = enhance_patch
        self.enhance_query = enhance_query
        self.use_text = use_text
        self.use_probability = use_probability
        self.register_buffer("text_features", text_features.detach().clone().float())
        self.register_buffer("conditional_matrix", conditional_matrix.detach().clone().float())
        self.text_proj = nn.Sequential(nn.Linear(text_features.shape[1], embed_dim),
                                      nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.prob_proj = nn.Sequential(nn.Linear(self.num_foreground, embed_dim // 4), nn.GELU(),
                                      nn.Linear(embed_dim // 4, embed_dim), nn.LayerNorm(embed_dim))
        self.ffn_patch_in = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.norm_patch_in = nn.LayerNorm(embed_dim)
        self.ffn_patch_out = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.norm_patch_out = nn.LayerNorm(embed_dim)
        self.query_prior_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        # No hidden gate in the Plan formula.
        if patch_update == "reference_gated":
            self.gamma_patch = nn.Parameter(torch.full((1,), 0.1))
            self.gamma_query = nn.Parameter(torch.full((1,), 0.1))
        self.text_proj.requires_grad_(use_text and not freeze_text_projection and (enhance_patch or enhance_query))
        self.prob_proj.requires_grad_(use_probability and (enhance_patch or enhance_query))
        for module in (self.ffn_patch_in, self.norm_patch_in, self.ffn_patch_out, self.norm_patch_out):
            module.requires_grad_(enhance_patch)
        self.query_prior_proj.requires_grad_(enhance_query)
        if patch_update == "reference_gated":
            self.gamma_patch.requires_grad_(enhance_patch)
            self.gamma_query.requires_grad_(enhance_query)

    def forward(self, patch_tokens, query_tokens):
        if patch_tokens.ndim != 3 or query_tokens.ndim != 3:
            raise ValueError("Expected [batch, tokens, channels]")
        if patch_tokens.shape[0] != query_tokens.shape[0] or patch_tokens.shape[-1] != self.embed_dim:
            raise ValueError("Token batch/channel mismatch")
        if query_tokens.shape[1:] != (self.num_foreground * self.queries_per_class, self.embed_dim):
            raise ValueError("Query count must equal foreground classes * queries_per_class")
        if not self.enhance_patch and not self.enhance_query:
            return patch_tokens, query_tokens, None
        prior = 0
        if self.use_text:
            prior = self.text_proj(self.text_features)
        if self.use_probability:
            prior = prior + self.prob_proj(self.conditional_matrix)
        attention = None
        if self.enhance_patch:
            local = self.norm_patch_in(self.ffn_patch_in(patch_tokens) + patch_tokens)
            logits = (local @ prior.T) / math.sqrt(self.embed_dim)
            attention = logits.softmax(dim=-1)
            context = attention @ prior
            update = self.ffn_patch_out(context) + context
            if self.patch_update == "plan":
                patch_tokens = self.norm_patch_out(update + patch_tokens)
            else:
                patch_tokens = patch_tokens + self.gamma_patch * self.norm_patch_out(update)
        if self.enhance_query:
            expanded = prior.repeat_interleave(self.queries_per_class, dim=0)
            bias = self.query_prior_proj(expanded).unsqueeze(0)
            if self.patch_update == "reference_gated":
                bias = self.gamma_query * bias
            query_tokens = query_tokens + bias
        return patch_tokens, query_tokens, attention
