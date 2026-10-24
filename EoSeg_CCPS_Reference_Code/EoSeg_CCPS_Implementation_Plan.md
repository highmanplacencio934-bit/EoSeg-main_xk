# EoSeg / Synapse — Class-Conditional Prior Selector (CCPS) 具体实现 Plan

> 版本：v1.0（2026-10-23）  
> 目标：在现有 Innovation1（ASEE）上添加独立、可开关、可消融的**类别条件语义先验选择器**。  
> 适用仓库：`highmanplacencio934-bit/EoSeg-main_xk`，现有 `models/asee.py` / `models/eoseg_asee.py`。  
> 交付：完整原理、文件改动清单、训练接口、消融表、CPU 测试和可复制的代码。  
> 状态：参考代码已通过独立 CPU 模块测试 8 项、Python 语法和 YAML 解析；**尚未在你的服务器上运行完整 EoSeg、Synapse 或 GPU 训练**。

## 0. 先给结论：本 Plan 选什么、不改什么

**第一轮主方案：** `selector_mode: visual_prior`、`selector_on_patch: true`、`selector_on_query: false`、`presence_loss_weight: 0.0`、`selector_init_probability: 0.95`。

这意味着先只研究“**根据当前二维 CT 切片，动态调节 8 类器官先验对 Patch 分支的贡献**”，不同时改 Query，不先加辅助监督，更容易隔离变量。

**明确保持不变：** 官方 Baseline、已完成的 Innovation1 文件与 Checkpoint；DINOv2 ViT-L 24 Blocks；现有 21+3 插入点；384×384 输入、实际 patch=16、24×24=576 Patch Tokens、1024 维；9 标签（含背景）、8 前景先验、16 Queries；后 3 Blocks、共享 Head、Query-wise 三层融合；原训练划分与先验文件；AdamW/LR/LLRD/调度/Batch16/220 Epoch/seed；原 Dice/HD95 口径与 Test 最佳 Checkpoint 策略。**本文不调整前面已决定暂不处理的评估协议。**

新增文件独立放置，绝不覆盖 `models/asee.py`、`models/eoseg_asee.py`、`training/lightning_module.py` 或现有 Innovation1 配置。

---

## 1. 数据流与设计定义

### 1.1 模块位置

```text
CT slice [B,3,384,384]
        │
        ▼
DINOv2 ViT-L 前 21 Blocks
        │
        ▼
Patch Tokens T [B,576,1024] ───────────────────────────────┐
        │                                                   │
        ├───────────────┐                                   │
        ▼               ▼                                   │
原 ASEE 先验 E       CCPS 类别选择器                          │
8×1024              S = sigmoid(f(T,E)) [B,8]               │
        │               │                                   │
        └───────┬───────┘                                   │
                ▼                                           │
        ASEE Patch-to-Class 注意力 A [B,576,8]               │
                │                                           │
                ▼                                           │
        C_selected = (A ⊙ S) @ E                             │
                │                                           │
                ▼                                           │
        Patch 更新（带可旁路语义强度） ←──────────────────────┘
                │
     可选：把 S 也用于 Query 语义偏置
                │
                ▼
     16 Queries + Prefix Tokens + 576 Patches
                │
                ▼
      ViT 最后三层 + EoSeg 原有分割头
                │
                ▼
             9 类预测
```

**重要定义：这是“每二维切片 8 个类别门控”，而不是“一整个 3D 病例 8 个门控”。** 你当前在测试时将每个 3D Volume 切片逐张送入 2D 模型，因此同一病例的不同切片可以有不同的门控。也不是 Patch-wise 8 类门控：`A` 已经是 `[B,576,8]` 的 Patch-to-Class 注意力；`S` 是 `[B,8]` 的全切片类别适用性。

### 1.2 可选选择器

| `selector_mode` | 生成方式 | 目的 | 是否依赖当前切片 |
|---|---|---|---|
| `off` | **直接调用原 ASEE 的 forward** | 严格控制组、关闭新增方法 | 否 |
| `global` | 8 个可学习全局 Logits | 测试“仅全局调节类别强度”是否足够 | 否 |
| `visual` | `Linear(MeanPool(T))` | 测试仅依据当前视觉内容进行类选择 | 是 |
| `visual_prior` | `MLP([v,e,|v-e|,v⊙e])` | 测试视觉与类别先验联合选择 | 是 |

`visual_prior` 的类别共享打分网络具有 `[B,8]` 输出，而不是为每个病例记忆一个参数表。

### 1.3 数学定义

令 `T:[B,576,D]`，`E:[8,D]`，`D=1024`，原 ASEE：

\[
A_{i,c}=\operatorname{Softmax}_c\left(\frac{\langle\phi(T_i),E_c\rangle}{\sqrt D}\right)
\]

新增类别选择权重（默认以全局池化 Patch 特征为输入）：

\[
v_b=\operatorname{MLP}_v(\operatorname{MeanPool}(T_b)),\quad
 e_c=\operatorname{MLP}_e(E_c)
\]

\[
S_{b,c}=\sigma\left(\operatorname{MLP}_{score}([v_b,e_c,|v_b-e_c|,v_b\odot e_c])\right)
\]

应用到原注意力**之后**，不在 Softmax 内部硬屏蔽：

\[
\tilde A_{b,i,c}=A_{b,i,c}S_{b,c},\quad
C_{b,i}=\sum_{c=1}^{8}\tilde A_{b,i,c}E_c
\]

\[
m_{b,i}=\sum_{c=1}^{8}\tilde A_{b,i,c}\in(0,1)
\]

`plan` 版 Patch 路径：原提议结果 `P=LN(FFN(C)+C+T)`，新更新采用：

\[
\boxed{T'=T+m\odot(P-T)}
\]

- 当 `S` 全部接近 1 时，`m` 接近 1，近似原 ASEE 的 `plan` 更新；
- 当 `S` 全部为 0（单测注入的硬零）时，`m=0`，**Patch 严格旁路**；
- 不额外将 `A⊙S` 重新归一化到 1，否则会丢失“所有先验都弱”的表达能力；
- `reference_gated` 版则是原残差增量再乘 `m`；
- 注意该结构在 `selector_on_patch=true` 时除了类别加权，也增加残差强度控制，所以消融必须包含 `global`、`visual`、`visual_prior`，必要时进一步拆出 `context-only` 与 `residual-only`，不能把增益全部归功于某个因素。

可选 Query 路径：

\[
Q'_{c,j}=Q_{c,j}+S_c\operatorname{Proj}(E_c),\quad j\in\{1,2\}
\]

这里的 `(c,j)` 只是当前代码的**语义偏置槽位**，不是对匈牙利匹配输出类别的硬绑定；严禁把“第 c 类的两个 Query”强行限定只能预测 c 类。

---

## 2. 文件组织与接入原则

新增文件：

```text
models/ccps_selector.py                        # 纯选择器；三种可学习 mode
models/asee_ccps.py                           # 继承已有 ASEE；Patch/Query 可独立开关
models/eoseg_asee_ccps.py                     # 继承已有 EoSeg+ASEE；不复制整个 forward
training/mask_classification_ccps.py          # 可选 presence 辅助损失；默认 0
configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml  # 完整可运行参考配置
scripts/generate_ccps_ablations.py            # 自动生成互不覆盖的实验配置
# tests/test_ccps.py                           # CPU 单元测试，复制进 repo 中 tests/
```

**父类复用策略：** `EoSegMultiQSegFusionASEECCPS(EoSegMultiQSegFusionASEE)` 先运行原父类构造函数（包括先验合法性与 SHA 核对），再用 CCPS 子类替换 `self.asee`；原网络 forward / 融合 / 预测头全部继承。替换前将**原 ASEE 初始化权重完整复制**进 CCPS ASEE，使 `off` / 新门控实验的初始原模块相同，避免模块重建导致随机权重漂移。

`get_extra_state()` 由父类保留先验摘要；子类 `set_extra_state()` 增加 CCPS 配置核对，避免错误加载不匹配的 CCPS Checkpoint。新模型应从原 DINOv2 预训练初始化进行公平重训；不能默认把已经训练好的 Innovation1 Checkpoint 直接拿来做不同选项的公平性比较。

---

## 3. 可选辅助监督：从训练标签生成“器官存在性”

默认先**不开**：`presence_loss_weight=0.0`。这样 CCPS 仅被原分割目标端到端训练，最容易比较方法本身。

后续可以开启：

\[
y^{presence}_{b,c}=\mathbf 1[\text{当前训练裁剪的 GT 中出现前景类 }c]
\]

\[
L=L_{EoSeg}+\lambda L_{presence},\quad
L_{presence}=\operatorname{BCEWithLogits}(logits_{CCPS},y^{presence})
\]

- GT 仅在**训练**使用，推理时 CCPS 仅看图像和固定的训练先验。
- 从当前 `target['labels']` 产生 8 位向量，并且标签已经经过训练随机裁剪/增强；无需改动 NPZ 文件。
- 背景类 0 不进入 presence 标签；每类只判断该类是否出现（没有额外标注）。
- **Presence ≠ prior reliability。** 这个辅助任务是在学习器官“出现概率”，不证明该先验对 Dice 一定有帮助。胰腺等难例会有假阴性门控风险。
- `global` 是对所有切片固定的参数，**不允许**开启切片级 presence 辅助 BCE；参考训练类对此直接报错。
- 强烈推荐按 `λ=0`、`0.1`、`0.3` 顺序试，不要一开始就给强监督。

---

## 4. 消融设计（单项可开关，避免绑死）

### 4.1 第一批最低成本实验

| 实验 ID | Mode | Patch gate | Query gate | Presence λ | 检验目标 |
|---|---|---:|---:|---:|---|
| A0 | `off` | — | — | 0 | 新代码是否复现原 ASEE |
| A1 | `global` | ✅ | ❌ | 0 | 全局固定类别权重是否足够 |
| A2 | `visual` | ✅ | ❌ | 0 | 加入切片内容后是否改善 |
| **A3 主方案** | **`visual_prior`** | ✅ | ❌ | **0** | **视觉 + 先验联合选择是否更有价值** |

**优先比较 A0→A1→A2→A3**，它们围绕同一个 Patch 分支，能较清晰隔离选择器机制。

### 4.2 第二批组件消融

| 实验 ID | Mode | Patch gate | Query gate | Presence λ | 检验目标 |
|---|---|---:|---:|---:|---|
| A4 | `visual_prior` | ❌ | ✅ | 0 | 只控制 Query 语义偏置 |
| A5 | `visual_prior` | ✅ | ✅ | 0 | Patch + Query 联合控制 |
| A6 | `visual_prior` | ✅ | ❌ | 0.1 | 辅助存在性监督的贡献 |
| A7 | `visual_prior` | ✅ | ✅ | 0.1 | 双分支 + presence 的协同效果 |
| A8 | `visual_prior` | ✅ | ❌ | 0 | 只使用文本先验：`use_text=true, use_probability=false` |
| A9 | `visual_prior` | ✅ | ❌ | 0 | 只使用共现先验：`use_text=false, use_probability=true` |

可追加消融（**不要首轮一次做完**）：

- `patch_update=plan/reference_gated`；
- `selector_hidden_dim=64/128/256`；
- `selector_init_probability=0.8/0.9/0.95`；
- `detach_prior_for_selector=true/false`（是否允许选择器梯度反向影响原语义投影）；
- 类别权重只作用于 `context`、只作用于 Patch 残差强度、二者均作用，以验证机制贡献；
- 限制选择器输出（soft gate）与硬阈值选择的比较；后者可能产生不可导、早期抑制等问题，**不作为默认方案**。

**所有实验要求**：相同 train/test 清单与先验 SHA、输入、优化器、种子、轮数、同一原项目指标和 checkpoint 选择口径；每个实验使用不同 `output_dir` 和 W&B run 名称；新结果不得覆盖 `outputs/innovation1`。

---

## 5. 实施步骤（建议交给 Codex 按顺序完成）

1. 新建功能分支 / 单独目录，先记录当前 commit SHA。把代码包中的 7 个文件按相同相对路径加入仓库，保留现有 Baseline/Innovation1 原样。
2. 运行 `python -m py_compile`、`pytest -q tests/test_ccps.py`。**必须保证 `off` 对原 ASEE forward 在同权重条件下逐值一致**。
3. 建立小模型 Smoke Test，确认 24×24 Patch、16 Query、原 3 层输出维度、匈牙利匹配损失反向传播均可运行；确认新选择器参数进入 AdamW 非主干组。
4. 在服务器使用 1 个真实训练 Batch 做 `forward + backward + optimizer.step`；用 `precision=16-mixed` 检查 NaN / Inf / 显存；先不用测试集做调试。
5. `python scripts/generate_ccps_ablations.py` 自动生成全部独立配置。先启动 A0、A1、A2、A3，检查 loss 与 selector `[B,8]` 权重不是全恒定或迅速塌到 0。
6. 第一批稳定后扩展 A4～A9；每轮结束后生成 Dice 总表、8 类器官 Dice、12 病例 Dice 升降表以及选择器权重分布。现阶段不要求 HD95 展示。
7. 如果默认 `visual_prior` 有收益，才继续增加辅助监督、门控细粒度或跨层深度；每次只改一个因素。

### 5.1 服务器运行命令

以你之前的 EoSeg 服务器目录为例（若目录不同按实际修改）：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
WANDB_MODE=offline \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python \
  main.py fit --config configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml
```

上面这条是**训练命令模板**；使用前仍要核对服务器真实路径、预训练缓存以及本地 `priors/synapse_asee_v1.pt`，不是已实际启动训练的记录。

生成消融配置：

```bash
python scripts/generate_ccps_ablations.py
python main.py fit --config configs/synapse/ccps_ablations/a3_visual_prior_patch.yaml
```

脚本仅**生成配置**，绝不自动启动 10 组大型训练。

---

## 6. 验收标准与必须做的代码检查

1. **维度**：`S:[B,8]`，`A:[B,576,8]`，`m:[B,576,1]`，`T':[B,576,1024]`，`Q':[B,16,1024]`；前缀 CLS/register 不变。
2. **off 回归**：相同先验、同一权重、同一输入的 `selector_mode=off` 与原 ASEE `plan` / `reference_gated` 输出严格一致。
3. **零门控边界**：人为将 `S=0` 时，若 `selector_on_patch=true` 且 `selector_on_query=true`，Patch 和 Query 完整旁路语义更新。它只验证更新式的数学性质，**不代表回到独立训练的 Baseline**。
4. **梯度**：`global/visual/visual_prior` 选择器均应收到分割 loss 梯度；presence λ>0 时 BCE Logits 梯度完整；`detach_prior` 只阻断“来自 selector 支路”的 prior 梯度，而不阻断原 ASEE 自身的梯度。
5. **可控性**：关闭 Patch gate 不改变 Query 原增强；关闭 Query gate 不改变 Patch 原增强。保留 `enhance_patch/enhance_query` 原开关。
6. **不泄漏**：forward 绝不读 `case_id`、测试 GT、Dice；标签仅在训练 step 构建辅助监督。
7. **Checkpoint 元数据**：包含先验哈希、ASEE 配置和 CCPS 配置，错配时拒绝静默加载。
8. **实验一致性**：相同训练设置；新增参数按现有非 backbone 参数组训练；每组输出目录隔离；测量额外参数、训练速度和显存。
9. **病例分析**：除了平均 Dice，按 12 病例和 8 类器官比较，包括 case0003/0038 的增益是否保持、case0008/0035 的退化是否恢复，不能人工对特定病例改门控。

### 6.1 风险与边界（必须理解）

- **这是参考实现，不是已被验证有效的研究结论**。需训练后才知道能否提高 Dice。
- **不会自动把所有背景 Patch 的更新减到 0**：类别选择是切片级；存在器官的切片中，背景 Patch 仍可能接受器官先验影响。若这成为主要问题，未来才考虑 `null/background prior` 或 Patch×Class 选择。
- 训练切片存在性监督可能对**小器官/低对比度器官**造成过度抑制。强烈建议先做 `λ=0` 的纯分割监督消融。
- 混合精度下门控用 FP32 计算 logits，然后转换到原注意力 dtype；仍需验证生产 GPU 的数值稳定性。
- `Case-wise` 与 `Slice-wise` 不同。当前具体方案是 **slice-wise class gates**，不应在论文中写成“自动选择整例是否使用 ASEE”。
- `visual_prior` 属于常见跨模态门控思想的**任务化实现**，其论文新颖性需根据更全面的相关工作与消融证据判断。引入 RELiF 等相关方法时应规范引用，不应把“可学习门控”本身包装成首创。

---

## 7. 完整参考代码（可逐文件复制）

> **注意：** 附带 ZIP 包里已按相同路径整理好全部文件。以下代码与 ZIP 同步；生产集成时按该路径复制，不要覆盖原文件。

### 7.1 `models/ccps_selector.py`

```python
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
```

### 7.2 `models/asee_ccps.py`

```python
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
```

### 7.3 `models/eoseg_asee_ccps.py`

```python
"""CCPS network model: inheriting EoSeg-ASEE without editing the original."""
from __future__ import annotations
import copy
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
                 detach_prior_for_selector=False):
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
        self.asee = AnatomicalSemanticEncodingCCPS(
            embed_dim=old.embed_dim,
            text_features=old.text_features,
            conditional_matrix=old.conditional_matrix,
            queries_per_class=queries_per_class,
            patch_update=patch_update,
            freeze_text_projection=freeze_text_projection,
            enhance_patch=enhance_patch, enhance_query=enhance_query,
            use_text=use_text, use_probability=use_probability,
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
```

### 7.4 `training/mask_classification_ccps.py`

```python
"""Optional slice-level organ presence auxiliary loss; no changes to old trainer."""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.mask_classification_semantic import MaskClassificationSemantic


class MaskClassificationSemanticCCPS(MaskClassificationSemantic):
    # Explicit signature is intentional: LightningCLI/jsonargparse needs to
    # discover the inherited training settings in addition to CCPS knobs.
    def __init__(
        self, network: nn.Module, img_size: tuple[int, int], num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4, llrd: float = 0.8,
        llrd_l2_enabled: bool = True, lr_mult: float = 1.0,
        weight_decay: float = 0.05, num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: Optional[List[int]] = None,
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8, overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        presence_loss_weight: float = 0.0,
        presence_pos_weight: float = 1.0,
    ):
        super().__init__(
            network=network, img_size=img_size, num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx, lr=lr, llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled, lr_mult=lr_mult,
            weight_decay=weight_decay, num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=[500, 1000] if warmup_steps is None else warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh, overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path, delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        if presence_loss_weight < 0 or presence_pos_weight <= 0:
            raise ValueError("Invalid presence loss weights")
        self.presence_loss_weight = float(presence_loss_weight)
        self.presence_pos_weight = float(presence_pos_weight)
        if (self.presence_loss_weight > 0 and
                self.network.asee.selector_mode not in {"visual", "visual_prior"}):
            raise ValueError("Presence supervision requires a slice-conditioned selector")

    @staticmethod
    def build_presence_targets(targets, num_foreground, device):
        """targets list is AFTER training transforms/crop, never the test volume."""
        labels = torch.zeros((len(targets), num_foreground),
                             dtype=torch.float32, device=device)
        for i, target in enumerate(targets):
            cls = target["labels"].to(device=device, dtype=torch.long)
            cls = cls[(cls >= 1) & (cls <= num_foreground)].unique()
            labels[i, cls - 1] = 1.0
        return labels

    def training_step(self, batch, batch_idx):
        # Parent computes and logs the UNMODIFIED original segmentation loss.
        loss_seg = super().training_step(batch, batch_idx)
        if self.presence_loss_weight == 0:
            return loss_seg
        logits = self.network.asee.last_selector_logits
        if logits is None:
            raise RuntimeError("Selector logits were not saved in training forward")
        targets = self.build_presence_targets(
            batch[1], logits.shape[1], device=logits.device)
        pos_weight = torch.full((logits.shape[1],), self.presence_pos_weight,
                                dtype=logits.dtype, device=logits.device)
        loss_presence = F.binary_cross_entropy_with_logits(
            logits.float(), targets, pos_weight=pos_weight.float())
        loss_total = loss_seg + self.presence_loss_weight * loss_presence
        self.log("losses/train_ccps_presence", loss_presence,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("losses/train_total_with_ccps", loss_total,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_total
```

### 7.5 `configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml`

```yaml
# Complete config copied from Innovation1; only module/trainer output changes.
trainer:
  default_root_dir: "outputs/innovation2/ccps_visual_prior_patch"
  max_epochs: 220
  limit_val_batches: 0
  check_val_every_n_epoch: 1000000
  precision: 16-mixed
  enable_checkpointing: false
  callbacks:
    - class_path: training.synapse_test_selection.SynapseTestSelection
      init_args:
        last_n_epochs: 50
        expected_test_cases: 12
        output_dir: "outputs/innovation2/ccps_visual_prior_patch"
  logger:
    class_path: lightning.pytorch.loggers.wandb.WandbLogger
    init_args:
      save_dir: "outputs/innovation2"
      resume: allow
      project: eomt
      name: vit_query_mul_scale_fusion_asee_ccps_visual_prior_patch

model:
  class_path: training.mask_classification_ccps.MaskClassificationSemanticCCPS
  init_args:
    num_classes: 9
    img_size: [384, 384]
    attn_mask_annealing_enabled: false
    ignore_idx: 255
    lr: 2e-4
    presence_loss_weight: 0.0
    presence_pos_weight: 1.0
    network:
      class_path: models.eoseg_asee_ccps.EoSegMultiQSegFusionASEECCPS
      init_args:
        num_classes: 9
        num_q: 16
        prior_path: priors/synapse_asee_v1.pt
        queries_per_class: 2
        patch_update: plan
        freeze_text_projection: false
        enhance_patch: true
        enhance_query: true
        use_text: true
        use_probability: true
        selector_mode: visual_prior
        selector_on_patch: true
        selector_on_query: false
        selector_hidden_dim: 256
        selector_init_probability: 0.95
        detach_prior_for_selector: false
        encoder:
          class_path: models.vit.ViT

data:
  class_path: datasets.synapse_semantic.SynapseSemantic
  init_args:
    train_base_dir: project_TransUNet/data/Synapse/train_npz
    test_base_dir: project_TransUNet/data/Synapse/test_vol_h5
    list_dir: project_TransUNet/TransUNet/lists/lists_Synapse
    train_split: train
    test_split: test_vol
    img_size: [384, 384]
    num_classes: 9
    batch_size: 16
    num_workers: 8
    ignore_idx: 255
    drop_background: false
    make_rgb: true
```

### 7.6 `scripts/generate_ccps_ablations.py`

```python
"""Generate independent CCPS configs without touching the original ASEE config.
Run in repo root: python scripts/generate_ccps_ablations.py
Requires PyYAML already used by project configs.
"""
from pathlib import Path
import copy
import yaml

BASE = Path("configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml")
DEST = Path("configs/synapse/ccps_ablations")

EXPERIMENTS = {
    "a0_off": {"selector_mode": "off"},
    "a1_global_patch": {"selector_mode": "global"},
    "a2_visual_patch": {"selector_mode": "visual"},
    "a3_visual_prior_patch": {"selector_mode": "visual_prior"},
    "a4_visual_prior_query": {"selector_mode": "visual_prior", "selector_on_patch": False, "selector_on_query": True},
    "a5_visual_prior_both": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": True},
    "a6_visual_prior_patch_presence": {"selector_mode": "visual_prior", "presence_loss_weight": 0.1},
    "a7_visual_prior_both_presence": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": True, "presence_loss_weight": 0.1},
    "a8_visual_prior_text_only": {"selector_mode": "visual_prior", "use_text": True, "use_probability": False},
    "a9_visual_prior_prob_only": {"selector_mode": "visual_prior", "use_text": False, "use_probability": True},
}


def main():
    data = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    for name, overrides in EXPERIMENTS.items():
        cfg = copy.deepcopy(data)
        net = cfg["model"]["init_args"]["network"]["init_args"]
        mdl = cfg["model"]["init_args"]
        for k, v in overrides.items():
            if k in {"presence_loss_weight", "presence_pos_weight"}:
                mdl[k] = v
            else:
                net[k] = v
        out = f"outputs/innovation2/{name}"
        cfg["trainer"]["default_root_dir"] = out
        cfg["trainer"]["callbacks"][0]["init_args"]["output_dir"] = out
        cfg["trainer"]["logger"]["init_args"]["name"] = "ccps_" + name
        (DEST / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"wrote {DEST / (name + '.yaml')}")


if __name__ == "__main__":
    main()
```

### 7.7 `tests/test_ccps.py`

```python
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
```

---

## 8. 运行状态与交付说明

- **本次未修改 GitHub 仓库，也未启动服务器训练。**
- 已执行参考代码 `py_compile` 检查；已通过隔离 CPU 单测 **8/8**（借助忠实复现的现有 ASEE API/forward 运行，不等同于在完整仓库安装了全部依赖）。
- 已检查完整 YAML 可被 PyYAML 正确解析。建议在目标仓库运行全套原测试、真实 Batch 和 GPU/FP16 集成测试后再开始正式消融。
- 训练辅助损失启用后，原日志 `losses/train_loss_total` 仍记录**原分割损失**；新日志 `losses/train_total_with_ccps` 才是最终反传的总损失，避免误读。
- 代码参考和 Plan 对应本文所述 GitHub 当前 `main` 的接口；未来仓库若修改，需要重新校验继承与 Checkpoint 兼容性。
