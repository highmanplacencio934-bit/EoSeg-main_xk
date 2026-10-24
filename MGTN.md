# 基于解剖语义增强编码（ASEE）改进 EoSeg 的详细规划文档

## 一、 核心动机与背景 (Motivation)

1. **Baseline 的局限性**：EoSeg 采用纯编码器架构，摆脱了传统的 U-Net 解码器。它在最后 $K=3$ 个 Transformer Block 中引入随机初始化的可学习查询（Learnable Queries $Q^{(0)}$）与图像 Patch Tokens $T^{(0)}$ 进行自注意力交互。然而，这种纯视觉驱动的 Query 缺乏对医学图像**解剖学先验知识（Anatomical Priors）**的显式感知。
2. **医学图像的天然属性**：与自然图像或点云场景不同，医学图像（如腹部 CT 的 Synapse 数据集、心脏 MRI 的 ACDC 数据集）中的器官和结构具有极强的**空间拓扑规律和类别共现关系**（例如：胆囊紧邻肝脏下方，左心室被心肌包裹，主动脉位于脊椎前方）。
3. **引入 MGTN 思想的切入点**：MGTN证明了通过语义描述模板和条件概率编码将语义上下文融入空间位置编码（即 SEPE 层），能显著提升模型对小目标和弱势类别（Minor Categories）的识别能力。在医学分割中，**胆囊（Gallbladder）和胰腺（Pancreas）**正是典型的难分割弱势小器官（在 EoSeg 的 Synapse 实验中，肝脏 Dice 高达 94.87%，而胆囊和胰腺仅为 74.99% 和 75.89%）。因此，设计**解剖语义增强编码模块（ASEE，Anatomical Semantic Enhanced Encoding）**，能有效引导 Query 精准捕捉小器官与模糊边界。

---

## 二、 模块详细设计与数学推导 (Methodology)

我们将改进模块命名为 **ASEE，无缝嵌入到 EoSeg 的多层级查询建模（Multi-level Query Modeling）阶段之前。整体分为三个步骤：

### 步骤 1：构建解剖学语义描述模板与文本嵌入
模仿 MGTN 为室内物体设计空间关系模板的做法，针对目标医学数据集（含 $C$ 个解剖类别）构建**解剖空间关系描述模板**。
* **特征提取**：使用预训练的医学文本编码器（如 BioClinicalBERT）提取这 $C$ 个类别的语义描述特征，并通过线性层映射到与 DINOv2 特征维度相同的 $d$ 维空间，得到**解剖语义嵌入矩阵**：
  $$S \in \mathbb{R}^{C \times d}$$

### 步骤 2：构建类别共现与空间邻接条件概率编码
在医学切片中，不同器官的出现具有很强的条件依赖性。在训练集上统计各类别共同出现或空间相邻的先验概率 $P(K_j)$ 和条件概率 $P(K_j \vert{} S_i)$[cite: 3]：
* 计算条件概率矩阵的对数比，并通过轻量级 MLP 映射为 $d$ 维的**条件概率嵌入（Conditional Probability Embedding）** $CP \in \mathbb{R}^{C \times d}$[cite: 3]：
  $$M_{i,j} = \log\left(\frac{P(K_j \vert{} S_i) + \epsilon}{P(K_j) + \epsilon}\right), \quad CP = \text{MLP}(M) \in \mathbb{R}^{C \times d}$$
* 将语义嵌入与概率嵌入相加融合，得到最终的**先验知识库表示**：
  $$E_{prior} = S \oplus CP \in \mathbb{R}^{C \times d}$$

### 步骤 3：双向注入 EoSeg 的 Multi-level Query Modeling
在进入倒数第 $K$ 个 Transformer Block（$K=3$）之前，利用 $E_{prior}$ 对图像 Patch Tokens $T^{(0)}$ 和 Learnable Queries $Q^{(0)}$ 进行双向增强：
1. **增强图像 Patch 空间特征（对应 MGTN 公式 10-12）**[cite: 3]：
   让每个图像 Patch Token 作为 Query，去查询解剖先验知识库 $E_{prior}$（作为 Key 和 Value），使图像特征提前感知上下文语义与位置关联[cite: 3]：
   $$L = \text{LayerNorm}(\text{FFN}(T^{(0)}) + T^{(0)})$$
   $$A = \text{Softmax}\left(\frac{L \cdot (E_{prior})^T}{\sqrt{d}}\right)$$
   $$\tilde{T}^{(0)} = \text{LayerNorm}(\text{FFN}(A \cdot E_{prior}) + A \cdot E_{prior} + T^{(0)})$$
2. **解剖先验引导的 Query 初始化**：
   EoSeg 中每个类别有 2 个 Query（$N_q = 2C$）[cite: 12]。将 $E_{prior}$ 扩展后与原本随机初始化的可学习参数结合，生成具有明确解剖指向性的初始 Query：
   $$\tilde{Q}^{(0)} = Q_{init} + \text{MLP}_{proj}(\hat{E}_{prior})$$

---

## 三、 代码架构与实现路径

修改 EoSeg 主干网络前向传播 (`forward`) 的局部节点：

```text
[DINOv2 Backbone 前 L-3 层]
       │
       ▼ 提取出中间特征 T^(0) [B, N_p, d]
┌──────────────────────────────────────────────────────────┐
│ ASEE 模块 (改编自 MGTN 的 SEPE Layer)                      │
│  1. 离线提取好文本嵌入 S 与概率矩阵 M                    │
│  2. 计算 E_prior = S + MLP(M)                            │
│  3. Cross-Attention: T^(0) 查询 E_prior ──► 得到 T̃^(0)   │
│  4. 先验引导: Q_init + MLP(E_prior)     ──► 得到 Q̃^(0)   │
└──────────────────────────────────────────────────────────┘
       │
       ▼ 拼接 Z^(0) = Concat(Q̃^(0), T̃^(0))
[DINOv2 最后 3 层: Block L-2, Block L-1, Block L] (保持 EoSeg 原逻辑)
       │
       ▼
[Learnable Block Fusion & Prediction Head] (保持 EoSeg 原逻辑)