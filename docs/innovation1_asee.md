> 最新训练授权与状态见 `.codex/innovation1_training_run.md`：用户已要求审核后启动主Plan训练；真实官方文本编码器和仅训练集先验已补齐。下文“等待/缺少编码器/未生成先验”为该次审查或首次准备时的历史状态。

> 2026-10-21更新：baseline已完成220轮；当前默认原Dice口径84.718466%，单分支修正复核83.676799%，均对应177轮，见 `.codex/synapse_baseline_results.md`。ASEE最新代码审查/修复见 `.codex/asee_code_review.md`；66项CPU检查通过。下文44项及“baseline仍运行”属于首次准备时的历史记录。器官映射已提供，正式先验及医学文本编码器仍待补齐。

# 创新点1：ASEE 代码准备说明

状态：2026-10-21，已准备独立代码及必要CPU验证；未启动创新训练，未生成正式训练先验，未下载或安装依赖。baseline仍独立运行。本说明不代表创新已有精度收益。

## 材料核对与实现依据

按顺序阅读了根目录 `MGTN.md`、`ASEE算法改进方案_v2.pptx`、`asee_module.py`、`build_prior.py`；四份原文件未改。PPT共14页，已检查全部文字、关键方法图（第5/8/10/11/13页），并核对压缩包：没有notesSlides备注文件。抽取、渲染与备注清单在 `outputs/innovation1/preparation_20261021`。

两份Plan共同要求：训练标签共现矩阵M与冻结文本编码器的原始特征，投影后形成E；在最后3个block之前执行一次patch与先验的交互，并向16个query加先验偏置；后3层、共享预测头及融合方式沿用baseline。

|材料差异/风险|准备代码的处理|
|---|---|
|两份Plan的patch公式为LN(FFN(AE)+AE+T)；参考实现为T+γpatch·LN(FFN(AE)+AE)|显式提供plan与reference_gated模式，不能当成同一公式。query在plan中无门控，reference_gated中γquery初始0.1，两门控均可训练。|
|PPT第9页写映射可训练，第13页写“先冻结”但未给解冻时点|主准备配置映射可训练；另给整程冻结text_proj配置。没有编造解冻epoch。冻结版本的映射是随机初始化后冻结，不是已有预训练投影；正式采用前需决定这是否符合意图。|
|PPT建议离线缓存最终E，同时要求训练概率MLP/映射|仅缓存原始CLS特征与M；投影后的E每次forward计算，保留梯度。冻结text_proj时概率投影仍训练，因此E仍不能固定缓存。|
|参考build_prior主程序使用300张随机模拟mask，并强制CUDA/可能联网|独立构建器从固定训练清单流式读取真实NPZ；CPU、本地编码器、local_files_only；从未执行原参考main。|
|PPT共现热图是示意，包含合并肾与腔静脉；参考文本顺序未证明等于实际标签ID|示例JSON明确verified=false，构建器拒绝使用；须核验实际预处理映射来源并按真实ID1..8填写，不据表格顺序猜映射。|
|Plan提到“邻接”，参考实现只有共现统计|实现明确限定二维训练切片级共现，不声称测量空间邻接、器官距离或病例级概率。|
|Plan称原模型每类2个query|实际baseline是16个自由query，没有固定类别身份。创新按先验行重复2次加偏置，但不修改匈牙利匹配，不强制query对应某个类。|

Plan中关于MGTN、瓶颈及预期收益的表述属于待验证设计；文件中的[cite:…]没有可追溯原论文信息，本次不将其作为已核实证据，也不编造引用或预期提升。

## 独立接口及变化范围

- `models/asee.py`：`AnatomicalSemanticEncoding`，默认plan公式；可分别关闭patch/query、text/probability分支。原始先验为buffer，保持不训练；禁用分支参数冻结。两增强分支都关闭时直接返回输入。
- `models/eoseg_asee.py`：继承原 `EoSegMultiQSegFusion`。空间patch tokens送入ASEE；CLS和4个register tokens保持原样，再按query、prefix、patch顺序拼接；在最后3个block之前注入一次。原共享预测头、末层query融合权重和输出列表结构不变。固定网格与prefix数不符合预期时显式报错。
- `scripts/build_synapse_asee_prior.py`：通过 `python -m scripts.build_synapse_asee_prior` 使用；导入无构建副作用。新产物为一个带元数据的PT，不直接接受旧版两份无来源的裸Tensor文件。
- 新配置均位于 `configs/synapse`，不覆盖原五份配置。`vit_query_mul_scale_fusion_asee_plan.yaml` 对应两份Plan的公式+参考投影结构；`..._asee_reference.yaml` 对应参考代码门控公式；`..._asee_plan_frozen_text.yaml` 仅另加整程冻结文本投影。这是待选择的实现分歧，不表示授权开展三轮训练。
- 默认raw文本特征8×768；实际backbone embed_dim驱动投影维度，目标主干为1024。新增参数实算：plan为7,360,768，reference_gated为7,360,770，约为baseline309M的2.4%；没有照搬“增量极小”的定性结论。

创新改变的变量是语义/共现先验、注入模块及其可训练参数，和显式变体中的门控/投影冻结。其余仍为220 epochs、batch16、384输入/patch16、9类、16query、seed0、16-mixed、LR2e-4、AdamW/WD0.05、LLRD0.8、原预热和poly、原损失/采样/增强/划分/评估。ASEE参数由现有优化器归入非主干组，沿用基础LR与500 step预热；主干LLRD不变。没有增加新损失或辅助监督。

日志名称、logger.save_dir、trainer.default_root_dir、测试选择回调output_dir改到独立 `outputs/innovation1` 路径，只改变产物位置。每轮运行仍创建带时间戳/随机标识的子目录。必须继续使用原入口，不直接另写一套训练循环。

## 训练先验的约束和产物

只接受当前固定train.txt与test_vol.txt的SHA256；18训练病例/2211切片，12测试病例且无病例交叉。只打开train.txt列出的NPZ，不读取测试H5、增强后mask或模型预测。当前原始标签保存为float32但数值为整数，先检查有限、整数值、0..8，再统计；不截断小数标签。

对每张原始切片记录出现的前景类别，统计N(i)、N(i,j)，包含对角线。平滑严格沿用参考实现eps=1e-5：

```text
P(j) = (N(j)+eps)/(N+eps)
P(j|i) = (N(i,j)+eps)/(N(i)+eps)
M(i,j) = log((P(j|i)+eps)/(P(j)+eps))
```

某前景类从未出现时直接报错，不以平滑值冒充有效统计。18病例没有等权重重采样，因此是切片级频率；切片更多的病例贡献更多统计样本，这是所提供参考实现的自然统计单位，不是病例级结果。

文本使用本地BioClinicalBERT的 `last_hidden_state[:,0,:]`（CLS），冻结编码器并eval，CPU离线提取。构建器核对8×768形状，记录本地编码器全部文件哈希；哈希与维度本身不能证明模型确实来自BioClinicalBERT，仍需核验本地检查点来源。

产物记录：真实类ID/名字/文本及映射来源和哈希、两份清单哈希、逐NPZ哈希、共现计数、统计单位/eps、编码器文件哈希、时间及schema版本。读取使用CPU与weights_only=True，验证形状/有限性/固定划分/已确认映射。构建器拒绝覆盖旧文件，遇到输入变化中断。

best/last的state_dict包含原始先验buffer以及模型extra_state中的来源元数据和artifact哈希。加载创新checkpoint时，配置的prior文件必须仍可读且哈希一致；不允许换先验后静默加载旧权重。保留同一PT产物及其原路径，或使用内容完全一致的新路径。当前公共delta_weights路径不是本次支持的加载方式；创新从原离线timm预训练流程初始化，未接入用户指定本地dinov2_vitl14_reg4_pretrain.pth，也不默认从最终baseline ckpt微调。

## 待用户指示后的命令模板（本次未执行）

以下先验构建需要先有可信的映射JSON和已核验的本地BioClinicalBERT目录。example文件不能直接用；不要只将verified改成true而未核验。指定本地目录、来源信息与映射是实际前置条件。

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python -m scripts.build_synapse_asee_prior \
  --label-map-json priors/synapse_label_map.verified.json \
  --encoder-path /absolute/path/to/verified/local/BioClinicalBERT \
  --output priors/synapse_asee_v1.pt
```

选择Plan变体后，未来独立训练入口模板：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
WANDB_MODE=offline \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python main.py fit \
  --config configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml
```

在任何正式命令前仍应确认服务器GPU/环境/磁盘，执行新模型真实FP16/compile冒烟；不能把本次CPU检查当成5090D显存或完整训练验证。若OOM不默认改batch或其他baseline变量。

## 已完成验证与边界

服务器CPU，CUDA_VISIBLE_DEVICES空，OMP/MKL线程2；原31项回归和新增13项共44项通过。覆盖Plan公式/梯度、参考门控数值对照、关闭ASEE后eval与train模式逐值等于原模型、保留CLS/register、单次注入、冻结与消融、原匈牙利损失反传、优化器分组、权重和来源加载、拒绝非法先验/标签/清单、真实一张训练NPZ读取。集成测试使用32×32、dim32、depth4、4 register小模型；生产配置不受这些测试缩小值影响。

三份配置通过现有main.py fit --print_config，检查除新模块选项/名称/产物路径外均等于选定baseline。目标1024维模块只在CPU计算参数量，没有构建大主干训练。未生成正式M、未提取正式文本特征，未做GPU、FP16、真实compile、创新训练或分数比较。测试仅使用临时合成先验；不把合成fixture发布到priors或用于训练。

baseline受保护文件与四份原材料哈希核对结果、CPU日志、配置解析和PPT证据均在准备输出目录。八小时baseline检查报告另在 `outputs/baseline/20261021T132635Z/followup_8h_summary.md`，其中192/220与最佳177轮是检查时部分结果，不是最终结果。

待决定/补充：实际label-ID映射来源、本地文本编码器来源及路径、Plan公式与参考门控选择、投影训练/冻结/是否需要另定解冻计划。等待用户指示后再接入真实先验及GPU冒烟；本次到此停止，不启动创新或后续监控，不停止尚未完成的baseline。

## 本轮加载与构建契约补充

先验相对路径现在按项目根目录解析；加载和hash使用同一份字节。checkpoint同时保存并强制核对ASEE公式/消融/冻结等配置；具有ASEE参数但缺元数据的checkpoint在strict=False下也报错。原baseline权重和baseline-only部分初始化接口不受此保护条件影响。

共现统计增加有限整数、对称、对角和上界校验，有效数据的eps与公式不变。文本提取显式限制到本地编码器/分词器的位置长度上限，提取前后hash必须一致；不接受提取中变动的来源。后续评分继续采用原Dice空掩码逻辑，不自动采用83.676799%的复核口径。
