# 服务器终端执行约定

## 直接终端执行权限（2026-10-21 用户更新）

用户已明确授权代理直接控制终端和 SSH，取消“终端命令必须由用户执行”的限制。后续任务范围内的检查、测试与修复由代理直接执行，不默认要求用户代运行命令。正式长训是否启动取决于当前任务授权，不因终端可用而自行启动。

2026-10-21 已直接通过 SSH 核验：hostname=5090，项目路径为下文目录，GPU=NVIDIA GeForce RTX 5090 D；可用解释器为 /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python，PyTorch=2.13.0+cu132、CUDA runtime=13.2，CUDA 可用。SSH 非交互 shell 当时未提供默认 python，应显式使用已核验的解释器或激活对应环境，不假定 base。环境改变后重新核验。

当前 PowerShell 仍在本地，SSHFS 映射不等于远程执行。服务器任务应使用实际 SSH/远程终端；保留以下三个环境变量。连接问题按实际错误处理，不重新施加用户代执行的长期限制。


2026-10-20 用户指定：运行 Python/训练/检查命令时，前置以下三个环境变量：

```bash
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
python ...
```

当前项目对应的服务器目录：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
```

不要使用先前误给的 /Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point。此路径依据当前 SSHFS 工作区映射，执行时仍以服务器 pwd/项目文件核验为准。

先激活实际训练 Conda 环境，不假定 base；正式训练使用服务器 RTX 5090D（32GB），不使用本机 GPU。

冒烟测试使用独立临时 YAML/病例列表/输出目录，仅缩短测试时长和样本范围；原 baseline 的 YAML、模型、patch16、输入、batch、LR、LLRD、损失与增强不得被覆盖。冒烟输出不能作为正式 baseline 成绩。

每次回复结束列出新增/修改/删除文件；没有文件变更则明确说明。
