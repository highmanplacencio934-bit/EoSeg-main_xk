# ==========================================
# File: build_priors.py
# 描述：通用解剖/语义先验离线生成脚本（适配多数据集）
# ==========================================

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 1. 定义不同数据集的默认文本描述模板（可按需扩展 ACDC、BraTS 等数据集）
DATASET_TEMPLATES = {
    "synapse": {
        0: "Aorta: The main trunk of the systemic arterial system, located in front of the spine.",
        1: "Gallbladder: A small pear-shaped organ beneath the right lobe of the liver.",
        2: "Left Kidney: A bean-shaped retroperitoneal organ on the left side of the spine.",
        3: "Right Kidney: A bean-shaped retroperitoneal organ on the right side of the spine.",
        4: "Liver: The largest solid organ in the upper right quadrant of the abdomen.",
        5: "Pancreas: An elongated organ located horizontally behind the stomach.",
        6: "Spleen: An organ in the upper far left quadrant, to the left of the stomach.",
        7: "Stomach: A muscular hollow organ located in the upper abdomen."
    },
    "acdc": {
        0: "Right Ventricle (RV): The chamber of the heart that receives deoxygenated blood.",
        1: "Myocardium (Myo): The muscular tissue of the heart surrounding the ventricles.",
        2: "Left Ventricle (LV): The lower left chamber of the heart that pumps oxygenated blood."
    }
}

def extract_text_embeddings(templates_dict, save_path="priors/text_emb.pt", model_name="emilyalsentzer/Bio_ClinicalBERT"):
    """使用预训练语言模型（如 BioClinicalBERT）提取各类别的语义描述向量 S_k"""
    print(f"正在加载文本编码器 ({model_name}) 提取语义特征...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda().eval()

    # 按照类别索引顺序整理文本
    sorted_classes = sorted(templates_dict.keys())
    texts = [templates_dict[c] for c in sorted_classes]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        # 取 [CLS] token 特征作为全局语义嵌入: [C, hidden_dim]
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(cls_embeddings, save_path)
    print(f"[成功] 语义嵌入已保存至 {save_path}, 维度: {cls_embeddings.shape}")

def compute_conditional_probability(label_masks_list, num_classes, save_path="priors/cond_prob.pt"):
    """
    通用条件共现概率矩阵统计（改编自 MGTN 公式 9）[cite: 3]
    参数:
        label_masks_list: 包含训练集所有 2D/3D Ground Truth 掩码数组的列表 (背景为 0, 前景为 1~C)
        num_classes: 前景类别总数 C
    """
    print(f"正在统计 {num_classes} 个类别的条件共现概率矩阵...")
    eps = 1e-5
    total_samples = len(label_masks_list)
    class_counts = np.zeros(num_classes, dtype=np.float64)       # N(i)
    co_occur_counts = np.zeros((num_classes, num_classes), dtype=np.float64) # N(i, j)

    for mask in label_masks_list:
        present_classes = np.unique(mask)
        # 将掩码中的类别标签映射为 0 到 C-1 的索引
        fg_classes = [int(c) - 1 for c in present_classes if 0 < c <= num_classes]
        
        for c_i in fg_classes:
            class_counts[c_i] += 1.0
            for c_j in fg_classes:
                co_occur_counts[c_i, c_j] += 1.0

    # 先验概率 P(K_j): [C]
    P_Kj = (class_counts + eps) / (total_samples + eps)
    # 条件概率 P(K_j | S_i) = N(i, j) / N(i): [C, C]
    P_Kj_given_Si = (co_occur_counts + eps) / (class_counts[:, None] + eps)

    # 计算对数条件概率比: log( P(K_j | S_i) / P(K_j) )
    cp_matrix = np.log((P_Kj_given_Si + eps) / (P_Kj[None, :] + eps))
    cp_tensor = torch.tensor(cp_matrix, dtype=torch.float32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(cp_tensor, save_path)
    print(f"[成功] 条件概率矩阵已保存至 {save_path}, 维度: {cp_tensor.shape}")

if __name__ == "__main__":
    # 以 Synapse 数据集为例生成先验文件
    dataset_name = "synapse"
    templates = DATASET_TEMPLATES[dataset_name]
    num_classes = len(templates)
    
    extract_text_embeddings(templates, save_path=f"priors/{dataset_name}_text_emb.pt")
    
    # 模拟 Synapse 训练集标签掩码列表（实际实验时替换为从你的 DataLoader 或 h5/nii 文件中读取的真实 mask）
    dummy_masks = [np.random.choice(range(num_classes + 1), size=(224, 224), p=[0.4]+[0.6/num_classes]*num_classes) for _ in range(300)]
    compute_conditional_probability(dummy_masks, num_classes=num_classes, save_path=f"priors/{dataset_name}_cond_prob.pt")