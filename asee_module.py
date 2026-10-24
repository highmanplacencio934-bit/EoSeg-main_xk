# ==========================================
# File: asee_module.py
# 描述：通用 ASEE (Anatomical Semantic Enhanced Encoding) 模块
# ==========================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASEEModule(nn.Module):
    """
    通用解剖语义增强编码模块 (改编自 MGTN 的 SEPE 层)[cite: 2, 3]
    适用于多数据集切换，支持动态配置类别数与特征通道数。
    """
    def __init__(self, embed_dim=1024, text_dim=768, num_classes=8, queries_per_class=2, 
                 text_emb_path="priors/synapse_text_emb.pt", 
                 cond_prob_path="priors/synapse_cond_prob.pt"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class

        # 1. 动态加载对应数据集的离线先验张量并注册为 buffer
        self.register_buffer("text_emb", torch.load(text_emb_path))
        self.register_buffer("cond_prob_matrix", torch.load(cond_prob_path))

        # 2. 语义与概率投影层
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.prob_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 3. 对应 MGTN 公式 (10) 的 Patch 特征变换与归一化[cite: 3]
        self.ffn_patch_in = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm_patch_in = nn.LayerNorm(embed_dim)

        # 4. 对应 MGTN 公式 (12) 的输出特征变换与归一化[cite: 3]
        self.ffn_patch_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm_patch_out = nn.LayerNorm(embed_dim)
        
        # 残差门控缩放因子（保证训练初期不破坏预训练模型的强大表征）
        self.gamma_patch = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma_query = nn.Parameter(torch.ones(1) * 0.1)

        # 5. 先验引导 Query 初始化的投影层
        self.query_prior_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, patch_tokens, query_tokens):
        """
        参数:
            patch_tokens: T^(0), 形状 [B, N_p, d]
            query_tokens: Q^(0), 形状 [B, N_q, d] (其中 N_q = num_classes * queries_per_class)
        返回:
            enhanced_patches: 增强后的图像 Patch 特征 [B, N_p, d]
            enhanced_queries: 增强后的可学习 Query [B, N_q, d]
            attn_weights: 空间-语义注意力权重 [B, N_p, C] (可用于可视化)
        """
        B, N_p, d = patch_tokens.shape

        # 动态获取当前加载的文本与概率先验
        S = self.text_proj(self.text_emb)                         # [C, d]
        CP = self.prob_proj(self.cond_prob_matrix)                # [C, d]
        E_prior = S + CP                                          # [C, d]
        E_prior_batch = E_prior.unsqueeze(0).expand(B, -1, -1)    # [B, C, d]

        # 步骤 B: 增强图像 Patch 空间特征 (对应 MGTN 公式 10-12)[cite: 3]
        L = self.norm_patch_in(self.ffn_patch_in(patch_tokens) + patch_tokens) # [B, N_p, d]

        # 计算交叉注意力分数 A_{i,j}
        attn_logits = torch.bmm(L, E_prior_batch.transpose(1, 2)) / math.sqrt(d) # [B, N_p, C]
        attn_weights = F.softmax(attn_logits, dim=-1)                            # [B, N_p, C]

        # 聚合上下文语义特征
        context_feat = torch.bmm(attn_weights, E_prior_batch)                    # [B, N_p, d]
        R = self.norm_patch_out(self.ffn_patch_out(context_feat) + context_feat) # [B, N_p, d]

        # 残差连接
        enhanced_patches = patch_tokens + self.gamma_patch * R                   # [B, N_p, d]

        # 步骤 C: 解剖先验引导的 Query 增强
        E_prior_expanded = E_prior.repeat_interleave(self.queries_per_class, dim=0) # [N_q, d]
        query_prior = self.query_prior_proj(E_prior_expanded).unsqueeze(0).expand(B, -1, -1) # [B, N_q, d]

        enhanced_queries = query_tokens + self.gamma_query * query_prior         # [B, N_q, d]

        return enhanced_patches, enhanced_queries, attn_weights