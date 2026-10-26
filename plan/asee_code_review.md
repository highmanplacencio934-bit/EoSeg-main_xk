> 最新训练授权与状态见 `.codex/innovation1_training_run.md`：用户已要求审核后启动主Plan训练；真实官方文本编码器和仅训练集先验已补齐。下文“等待/缺少编码器/未生成先验”为该次审查或首次准备时的历史状态。

# ASEE代码审查与修复（2026-10-21）

用户要求记录原口径/修正口径baseline，后续保持原空掩码逻辑；审查创新点1的ASEE代码并直接修复实现问题。本轮没有启动创新正式训练、生成正式先验、使用GPU或继续物理HD95工作。

## Baseline评分记录

两种口径的最终结果、8个器官名称/标签/Dice、原体素HD95及同一177轮权重的说明见 `.codex/synapse_baseline_results.md` 和本目录 `baseline_two_dice_conventions.csv`。

- 当前默认：原空掩码逻辑，平均Dice84.718466%。
- 单独保留的修正分支复核：83.676799%。不等于新训练、未改原CSV或checkpoint。
- baseline完整220轮、正常退出；两个口径最终最佳均177轮。后续创新继续以原逻辑比较，已写入AGENTS和回归。

## 审查依据与范围

已核对MGTN.md、PPT首次准备时保存的文本/方法公式、两个根目录参考Python、当前ASEE模块/集成模型、训练先验构建脚本、3个变体配置、已有测试和baseline的注入/融合调用。原参考文件未改。这里的ASSE按项目实际名称ASEE理解。

实现核对：patch在最后3个block前增强一次，CLS/register前缀保留，query仍16个，原预测头和融合、匈牙利匹配/损失/采样均沿用。Plan的外层LN公式与参考门控残差不是同一算法，作为显式变体保留，没有在本轮改公式或切换默认参数。

## 已复现并修复的问题

|级别|问题和触发方式|修复|
|---|---|---|
|P1|相同prior文件但Plan/门控、patch/query、text/probability或冻结选项不同的checkpoint，在strict=False下仍可加载，可能静默改变训练或推理实验。|extra_state同时记录并核验ASEE配置；出现ASEE权重却缺少来源元数据时也拒绝，避免strict=False绕过。显式baseline-only部分初始化仍可用。|
|P2|先验相对路径按终端cwd解析，与项目数据路径规则不同，从其他目录启动会找不到正确prior。|统一相对项目根目录解析，绝对路径继续支持。|
|P2|先torch.load文件再单独读文件计算hash，两次读取间文件被替换时，加载的特征与记录的artifact hash可不同。|只读取一次原始字节，反序列化和SHA256使用同一快照。|
|P2|共现计数接受NaN、N非整数、频次大于样本数、不对称矩阵或错误对角线，可能给出非法或看似有限但不成立的先验。|校验有限整数计数、1..N频次、N为正整数、矩阵对称/对角/上界、eps有限正数；有效输入的原平滑和公式保持不变。|
|P2|文本tokenizer只写truncation没有明确编码器位置上限，tokenizer未声明上限时长描述可越过BERT位置嵌入。|从encoder配置读取max_position_embeddings，并尊重tokenizer更小上限，显式传max_length；仍是原CLS提取。|
|P2|只在文本提取前记录本地编码器文件hash，过程中源文件变化时仍可发布过期的来源记录。|提取后再次核对本地文件集合/hash，变化则中止，不写正式先验。|

回归首先在原实现上复现失败，再修复转绿。缺少元数据绕过另做了单独失败复现。证据为regressions_before_fix.log、missing_metadata_before_fix.log和修复后CPU日志。没有把正常公式或研究设计取舍标成实现bug。

## 验证

- 66/66项CPU回归通过，ASEE21项（原13+新增8）；无跳过。包括禁用ASEE时与baseline在train/eval逐值一致、Plan公式/梯度、参考门控数值对照、前缀保留、单次注入、原匈牙利损失反传和优化器归组。
- CPU混合精度使用bfloat16，小模型的两个公式均完成torch.compile(backend=eager, fullgraph=True)前向/反向、梯度有限。这只验证图捕获/接口，不等于GPU FP16或默认Inductor已经验证。
- 3份ASEE配置均通过main.py fit --print_config；对照测试确认研究变量外的设置与第三套baseline一致。
- 审查前66份代码/配置/参考材料哈希中，只有models/asee.py、models/eoseg_asee.py、scripts/build_synapse_asee_prior.py、tests/test_asee_innovation.py四份源码改变。baseline原模型、训练模块、5份原配置、3份创新配置、数据接口、评估规则、四份原材料均未改。

新checkpoint配置元数据是加载契约的强化；缺失配置的旧ASEE权重不能再被静默当成已核验实验。当前没有正式ASEE训练checkpoint。原baseline checkpoint不受影响。

## 未代替用户决定的事项和实际缺口

- 本批器官名称映射已经有priors/synapse_label_map.verified.json，builder可用；核验范围仍是此前真实图像/标签的语义交叉检查，不虚构原始预处理出处。
- 真实BioClinicalBERT本地目录和来源尚未提供；正式priors/synapse_asee_v1.pt仍未生成。文本编码测试用明确的mock，不冒充真实医学编码器验证。
- Plan、参考门控和整程冻结投影3个显式变体继续保留。冻结版本仍是随机初始化投影后冻结；未编造解冻epoch或改为预训练映射。主Plan配置为可训练投影。正式选用哪个变体须在训练前明确。
- GPU FP16/384/patch16/batch16/default compile创新冒烟及正式训练未执行；没有声称CPU测试能证明实际显存或精度收益。

本轮改变的是创新的加载/校验/来源记录和输入边界；未改变ASEE数学公式、参数量、默认分支选择或baseline科学参数，也未修正用户要求保留的原Dice分支。

证据目录：outputs/audit/asee_review_20261022T041619Z。
