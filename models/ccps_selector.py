"""Class-Conditional Prior Selector (CCPS).

Per-slice [B, C] soft gates; independent of test IDs, labels and scores.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class ClassConditionalPriorSelector(nn.Module):
    MODES = {"global", "visual", "visual_prior"}

    def __init__(self, embed_dim: int, num_classes: int = 8,
                 mode: str = "visual_prior", hidden_dim: int = 256,
                 init_probability: float = 0.95,
                 detach_prior_for_selector: bool = False):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"selector mode must be one of {sorted(self.MODES)}")
        if not 0 < init_probability < 1:
            raise ValueError("init_probability must be in (0, 1)")
        if embed_dim < 4 or num_classes < 1 or hidden_dim < 4:
            raise ValueError("Invalid selector dimensions")
        self.mode = mode
        self.num_classes = num_classes
        self.detach_prior_for_selector = detach_prior_for_selector
        initial_logit = math.log(init_probability / (1 - init_probability))

        if mode == "global":
            self.global_logits = nn.Parameter(torch.full((num_classes,), initial_logit))
        elif mode == "visual":
            self.visual_head = nn.Sequential(nn.LayerNorm(embed_dim),
                                             nn.Linear(embed_dim, num_classes))
            nn.init.normal_(self.visual_head[-1].weight, std=1e-3)
            nn.init.constant_(self.visual_head[-1].bias, initial_logit)
        else:
            self.visual_proj = nn.Sequential(nn.LayerNorm(embed_dim),
                                             nn.Linear(embed_dim, hidden_dim), nn.GELU())
            self.prior_proj = nn.Sequential(nn.LayerNorm(embed_dim),
                                            nn.Linear(embed_dim, hidden_dim), nn.GELU())
            self.scorer = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.scorer[-1].weight, std=1e-3)
            nn.init.constant_(self.scorer[-1].bias, initial_logit)

    def forward(self, patch_tokens: torch.Tensor, prior: torch.Tensor):
        """Return (weights, logits), each shape [B, C]."""
        if patch_tokens.ndim != 3 or prior.ndim != 2:
            raise ValueError("Expected tokens [B,N,D] and prior [C,D]")
        b, _, d = patch_tokens.shape
        if prior.shape != (self.num_classes, d):
            raise ValueError("Prior shape mismatch")
        # Float32 logits/sigmoid are more stable with mixed-precision training.
        with torch.autocast(device_type=patch_tokens.device.type, enabled=False):
            pooled = patch_tokens.float().mean(dim=1)
            if self.mode == "global":
                logits = self.global_logits.float().unsqueeze(0).expand(b, -1)
            elif self.mode == "visual":
                logits = self.visual_head(pooled)
            else:
                e = prior.detach() if self.detach_prior_for_selector else prior
                v = self.visual_proj(pooled)[:, None, :]  # [B,1,H]
                p = self.prior_proj(e.float())[None, :, :]  # [1,C,H]
                v = v.expand(-1, self.num_classes, -1)
                p = p.expand(b, -1, -1)
                features = torch.cat((v, p, (v - p).abs(), v * p), dim=-1)
                logits = self.scorer(features).squeeze(-1)
        logits = logits.float()  # logits and sigmoid in FP32 under autocast
        return logits.sigmoid(), logits
