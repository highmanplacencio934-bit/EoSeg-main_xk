# EoSeg: Does Your ViT Still Need U-Net for Segmentation?

[![Code](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/Retinal-Research/EoSeg)
[![Paper](https://img.shields.io/badge/arXiv-2607.00223-b31b1b)](https://arxiv.org/abs/2607.00223)

---

## Overview

This repository contains the official codebase for **EoSeg** and the paper:

**Does Your ViT Still Need U-Net for Segmentation?**

The project is organized around a Lightning CLI training pipeline with dataset-specific configurations for Synapse, ACDC, GlaS, ISIC, Kvasir, and MoNuSeg.

---

## Repository Structure

```text
EoSeg/
├── configs/
│   ├── acdc/
│   ├── glas/
│   ├── isic2016/
│   ├── isic2017/
│   ├── kvasir/
│   ├── monuseg/
│   └── synapse/
├── datasets/
├── models/
├── scripts/
├── training/
├── main.py
├── requirements.txt
└── README.md
```

---

## Main Entry

Training and evaluation are driven through `main.py` with Lightning CLI configs.

The primary experiment configuration is:

- `configs/synapse/vit_query_mul_scale_fusion.yaml`

The remaining Synapse configs are kept as ablation settings.

---

## Environment

This repository currently provides a single environment dependency file:

- `requirements.txt`

There is no `environment.yml` or `pyproject.toml` at the moment.

Recommended setup:

```bash
conda create -n eoseg python=3.10 -y
conda activate eoseg
pip install --upgrade pip
pip install -r requirements.txt
```

If you are running on a cluster, it is often better to install a CUDA-matched `torch` and `torchvision` first, then install the remaining packages from `requirements.txt`.

The current codebase depends on:

- `lightning` and `jsonargparse` for the CLI training pipeline
- `torch`, `torchvision`, `torchmetrics`, and `timm` for modeling and training
- `transformers` for backbone integrations
- `numpy`, `scipy`, `h5py`, `Pillow`, `matplotlib`, and `medpy` for data handling and evaluation
- `wandb` for experiment logging
- `mmengine` and `mmsegmentation` for selected backbone/util layers

---

## Training

Example training command:

```bash
python3 main.py fit --config configs/synapse/vit_query_mul_scale_fusion.yaml
```

Example evaluation command:

```bash
python3 main.py test --config configs/synapse/vit_query_mul_scale_fusion.yaml --ckpt_path /path/to/checkpoint.ckpt
```

Independent testing uses eager execution by default. To reproduce case metrics from
compiled training, add `--compile_test` to the evaluation command. This opt-in flag
compiles only the model forward; checkpoint loading and metric calculations stay
unchanged. Leave it off for runs trained with `--compile_disabled`; the two flags
cannot be combined. Training settings and Synapse YAML defaults are unchanged.

Please update dataset paths and checkpoint paths in the YAML configs to match your local or cluster environment.

---

## Notes

- The repository keeps a compact set of visualization utilities under `scripts/`.
- The YAML files under `configs/synapse/` include the main model configuration plus retained ablation settings.

---

## Acknowledgement

This codebase is developed with code reference to:

- EOMT: [https://github.com/tue-mps/EoMT](https://github.com/tue-mps/EoMT)
- TransUNet: [https://github.com/Beckschen/TransUNet](https://github.com/Beckschen/TransUNet)

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@article{eoseg,
  title   = {Does Your ViT Still Need U-Net for Segmentation?},
  author  = {Li, Xin and Zhu, Wenhui and Dong, Xuanzhao and Chen, Xiwen and Chen, Yanxi and Xiong, Yujian and Wang, Hao and Dumitrascu, Oana M and Wang, Yalin},
  journal = {arXiv preprint arXiv:2607.00223},
  year    = {2026}
}
```


## Local Synapse data interface

Synapse configs retain their original epoch counts: the main fusion config uses
220 epochs; the other four use 200. All use the existing fixed TransUNet split:
18 training cases (2211 slices) and 12 test cases. The manifests are
`project_TransUNet/TransUNet/lists/lists_Synapse/train.txt` and `test_vol.txt`.
No random resplitting or validation subset is created. Validation is disabled;
`val_dataloader()` returns an empty list rather than reusing training data.

Data paths in Synapse configs are resolved relative to this repository, not the
shell working directory. Absolute server paths are also accepted through
`train_base_dir`, `test_base_dir`, and `list_dir`. The manifest stems are exposed
as `train_split` (default `train`) and `test_split` (default `test_vol`).
At each setup, both manifests are checked for duplicate/invalid entries and
overlapping case IDs; `data.split_summary` records the resulting split. Both
manifest files must be available. File checks follow the active stage: `fit`
requires only training slices, `test` requires only test volumes, and
`setup(None)` checks both data directories. Switching from fit to test rechecks
the test files; the training data directory need not be mounted for testing.
The checks do not read every image array or regenerate the manifests.

Run data-interface checks without PyTorch:
`python -m unittest discover -s tests -p test_synapse_splits.py`.
Backbone checkpoint/patch-size compatibility still requires separate setup.


## Main Synapse evaluation and checkpoint selection

The third config (`vit_query_mul_scale_fusion.yaml`) now uses our explicitly
chosen test-based model-selection policy. This is a project adaptation, not a
claim about an undocumented author evaluation procedure. The other four configs
retain their checkpoint settings.

Training still runs for 220 epochs. After epochs **171 through 220** (one-based),
`SynapseTestSelection` evaluates every one of the 12 test cases using the current
in-memory model. Evaluation uses eval mode, no gradients and the configured
precision context; it restores module modes and Python/NumPy/PyTorch RNG states.
No optimizer or scheduler step is performed by evaluation. The test set is used
for checkpoint selection, but never for gradient updates or early stopping.

Each fit creates a unique `synapse_test_selection/<timestamp>_<id>/` folder under
`trainer.log_dir` (or the default root if no logger is used). The callback's
optional `output_dir` overrides that parent directory. Each run retains only:

- `best_test.ckpt`: highest test mean Dice in the final 50 candidate epochs;
  ties keep the earlier candidate.
- `last.ckpt`: rolling weights after every completed epoch, including epochs
  before the selection window.
- `test_epoch_metrics.csv`: per-epoch case-averaged Dice and HD95 for classes
  1..8, overall means, epoch, step and best-so-far information.
- `test_case_metrics.csv`: corresponding metrics for every individual case.
- `selection_summary.json`: policy, window and selected epoch/score.

Both checkpoints contain the full segmentation model including its backbone,
but omit optimizer/scheduler state (`weights_only=True`). They are evaluation
weights, not full-state training-resumption checkpoints. Old runs are preserved;
the callback does not delete existing checkpoints. Standard automatic
ModelCheckpoint saving is disabled in this config to prevent extra copies.
Use an explicit checkpoint path for a later standalone test command.

Metrics reuse the existing Synapse test case inference, resizing and empty-mask
conventions. HD95 still uses voxel coordinates without physical spacing.
Results selected this way must be described as **test-selected**, not as
independent held-out test estimates. No evaluation results are available until
an actual server run has completed.
